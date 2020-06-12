import codecs
import csv
import re
import numpy as np
from builtins import range

from fonduer.learning.utils import confusion_matrix
from fonduer.supervision.models import GoldLabel, GoldLabelKey
from fonduer.candidates.models import Candidate 
from fonduer.utils.data_model_utils import *
from fonduer.supervision import Labeler
from fonduer import Meta

from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

from fonduer_utils import get_applied_lfs, get_neighbor_cell_ngrams_own, _min_range_diff, min_row_diff, min_col_diff, schema_match_filter
from my_subclasses import stations_mapping_dict

import emmental
from emmental.modules.embedding_module import EmbeddingModule
from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.learner import EmmentalLearner

from fonduer.learning.utils import collect_word_counter
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task

from my_subclasses import stations_list, stations_mapping_dict

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


# Define labels
ABSTAIN = -1
FALSE = 0
TRUE = 1


def get_gold_dict(
    filename, doc_on=True, station_on=True, price_on=True, attribute=None, docs=None, stations_mapping_dict=None
):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        headers = next(gold_reader, None)  # skip the headers
        for row in gold_reader:
            if (row == headers):
                continue
            (folder, subfolder, doc, sheet, station, date, designation, price, volume) = row
            if (docs is None or re.sub("\.xls", "", doc).upper() in docs) and designation == 'On Peak': # filter off peak labels
                key = []
                if doc_on:
                    key.append(re.sub("\.xls", "", doc).upper())
                if station_on:
                    key.append(station.upper())
                if price_on:
                    key.append(re.sub("[^0-9.]", "", price).upper())
                gold_dict.add(tuple(key))
    return gold_dict



def get_gold_func(
    gold_file, attribute, stations_mapping_dict=None
): 
    gold_dict = get_gold_dict(
        gold_file, 
        attribute=attribute, 
        stations_mapping_dict=stations_mapping_dict
    )
    def gold(c: Candidate) -> int:
        doc = (c[0].context.sentence.document.name).upper()
        station = (c[0].context.get_span()).upper()
        price = ("".join(c[1].context.get_span().split())).upper()

        stations = stations_mapping_dict[station.lower()] if stations_mapping_dict != None else [station]

        # account for all station abbrevations, as we do not consider the entity-linking problem (same entity with multiple identity descriptors)
        for station_abbr in stations:
            if (doc, station_abbr.upper(), price) in gold_dict:
                return TRUE
        return FALSE
    return gold

def entity_level_f1(
    candidates, gold_file, attribute=None, corpus=None, stations_mapping_dict=None
):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from electricity_utils import entity_level_f1
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = 'tutorials/tables/data/electricity/electricity_gold.csv'
        entity_level_f1(candidates, gold_file, 'elec_price_vol')
    """
    docs = [(re.sub("Document ", "", doc.name)).upper() for doc in corpus] if corpus else None
    price_on = attribute is not None
    gold_set = get_gold_dict(
        gold_file,
        docs=docs,
        doc_on=True,
        station_on=True,
        price_on=price_on,
        attribute=attribute,
        stations_mapping_dict=stations_mapping_dict
    )
    if len(gold_set) == 0:
        print(f"Gold File: {gold_file}\n Attribute: {attribute}")
        print("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    print("Preparing candidates...")
    entities = set()
    for i, c in enumerate(tqdm(candidates)):
        station = c[0].context.get_span().upper()
        doc = c[0].context.sentence.document.name.upper()
        price = c[1].context.get_span()

        # Account for all station abbrevations, as we do not consider the entity-linking problem (same entity with multiple identity descriptors)
        # We only take the entity by the name how it is represented in the gold_dict
        stations = stations_mapping_dict[station.lower()] if stations_mapping_dict != None else [station]
        added_any = False
        for station_abbr in stations:
            if (doc, station_abbr.upper(), price) in gold_set:
                entities.add((doc, station_abbr.upper(), price))
                added_any = True
        if (not added_any):
            entities.add((doc, station, price))
    
    (TP_set, FP_set, FN_set) = confusion_matrix(entities, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    print("========================================")
    print("Scoring on Entity-Level Gold Data")
    print("========================================")
    print(f"Corpus Precision {prec:.3}")
    print(f"Corpus Recall    {rec:.3}")
    print(f"Corpus F1        {f1:.3}")
    print("----------------------------------------")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("========================================\n")
    return [sorted(list(x)) for x in [TP_set, FP_set, FN_set]]

def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple(
            [c[0].context.sentence.document.name.upper()]
            + [c[i].context.get_span().upper() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches


def eval_LFs(train_marginals, train_cands, gold):
    LF_positive = np.where(train_marginals[:, TRUE] > 0.6)
    LF_pos_preds = np.zeros(len(train_marginals))
    LF_pos_preds[LF_positive] = 1

    LF_TP = [x for i,x in enumerate(train_cands[0]) if gold(x) and LF_pos_preds[i]]
    LF_FP = [x for i,x in enumerate(train_cands[0]) if not gold(x) and LF_pos_preds[i]]
    LF_FN = [x for i,x in enumerate(train_cands[0]) if gold(x) and not LF_pos_preds[i]]

    L_TP = len(LF_TP)
    L_FP = len(LF_FP)
    L_FN = len(LF_FN)

    prec = L_TP / (L_TP + L_FP) if L_TP + L_FP > 0 else float("nan")
    rec = L_TP / (L_TP + L_FN) if L_TP + L_FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    print("========================================")
    print("Scoring on Entity-Level Gold Data")
    print("========================================")
    print(f"Corpus Precision {prec:.3}")
    print(f"Corpus Recall    {rec:.3}")
    print(f"Corpus F1        {f1:.3}")
    print("----------------------------------------")
    print(f"TP: {L_TP} | FP: {L_FP} | FN: {L_FN}")
    print("========================================\n")


def summarize_results(results):
    results_train = results[0]
    results_dev = results[1]
    results_test = results[2]

    pos_total = (
        len(results_train[0]) + 
        len(results_dev[0]) + 
        len(results_test[0])
    )
    prec_total = pos_total / (
        pos_total + 
        len(results_train[1]) + 
        len(results_dev[1]) + 
        len(results_test[1])
    )
    rec_total = pos_total / (
        pos_total + 
        len(results_train[2]) + 
        len(results_dev[2]) + 
        len(results_test[2]) 
    )
    f1_total = 2 * (prec_total * rec_total) / (prec_total + rec_total)
    
    return (prec_total, rec_total, f1_total)


def get_model_methods(session, ATTRIBUTE, gold, gold_file, all_docs, StationPrice, PARALLEL):
    train_docs = all_docs[0]
    dev_docs = all_docs[1]
    test_docs = all_docs[2]

    def run_labeling_functions(cands):
        ABSTAIN = -1
        FALSE = 0
        TRUE = 1
        # Extract candidates
        train_cands = cands[0]
        dev_cands = cands[1]
        test_cands = cands[2] 

        @labeling_function()
        def LF_other_station_table(c):
            station_span = c.station.context.get_span().lower()
            neighbour_cells = get_neighbor_cell_ngrams_own(c.price, dist=100, directions=True, n_max = 4, absolute = True)
            up_cells = [x for x in neighbour_cells if len(x) > 1 and x[1] == 'DOWN' and x[0] in stations_list]
            # No station name in upper cells
            if (len(up_cells) == 0):
                return ABSTAIN
            # Check if the next upper aligned station-span corresponds to the candidate span (or equivalents)
            closest_header = up_cells[len(up_cells)-1]
            return TRUE if closest_header[0] in stations_mapping_dict[station_span] else FALSE

        @labeling_function()
        def LF_station_non_meta_tag(c):
            html_tags = get_ancestor_tag_names(c.station)
            return FALSE if ('head' in html_tags and 'title' in html_tags) else ABSTAIN

        # Basic constraint for the price LFs to be true -> no wrong station (increase accuracy)
        def base(c):
            return (
                LF_station_non_meta_tag(c) != 0 and 
                LF_other_station_table(c) != 0 and 
                LF_off_peak_head(c) != 0 and
                LF_purchases(c)
            )

        # 2.) Create labeling functions 
        @labeling_function()
        def LF_on_peak_head(c):
            return TRUE if 'on peak' in get_aligned_ngrams(c.price, n_min=2, n_max=2)  and base(c) else ABSTAIN

        @labeling_function()
        def LF_off_peak_head(c):
            return FALSE if 'off peak' in get_aligned_ngrams(c.price, n_min=2, n_max=2) else ABSTAIN

        @labeling_function()
        def LF_price_range(c):
            price = float(c.price.context.get_span())
            return TRUE if price > 0 and price < 1000 and base(c) else FALSE

        @labeling_function()
        def LF_price_head(c):
            return TRUE if 'price' in get_aligned_ngrams(c.price) and base(c) else ABSTAIN

        @labeling_function()
        def LF_firm_head(c):
            return TRUE if 'firm' in get_aligned_ngrams(c.price)and base(c) else ABSTAIN

        @labeling_function()
        def LF_dollar_to_left(c):
            return TRUE if '$' in get_left_ngrams(c.price, window=2) and base(c) else ABSTAIN

        @labeling_function()
        def LF_purchases(c):
            return FALSE if 'purchases' in get_aligned_ngrams(c.price, n_min=1) else ABSTAIN

        station_price_lfs = [
            LF_other_station_table,
            LF_station_non_meta_tag,

            # indicator
            LF_price_range,

            # negative indicators
            LF_off_peak_head,
            LF_purchases,

            # positive indicators
            LF_on_peak_head,    
            LF_price_head,
            LF_firm_head,
            LF_dollar_to_left,
        ]

        # 3.) Apply the LFs on the training set
        labeler = Labeler(session, [StationPrice])
        labeler.apply(split=0, lfs=[station_price_lfs], train=True, clear=True, parallelism=PARALLEL)
        L_train = labeler.get_label_matrices(train_cands)

        # Check that LFs are all applied (avoid crash)
        applied_lfs = L_train[0].shape[1]
        has_non_applied = applied_lfs != len(station_price_lfs)
        print(f"Labeling functions on train_cands not ABSTAIN: {applied_lfs} (/{len(station_price_lfs)})")

        if (has_non_applied):
            applied_lfs = get_applied_lfs(session)
            non_applied_lfs = [l.name for l in station_price_lfs if l.name not in applied_lfs]
            print(f"Labling functions {non_applied_lfs} are not applied.")
            station_price_lfs = [l for l in station_price_lfs if l.name in applied_lfs]

        # 4.) Evaluate their accuracy
        L_gold_train = labeler.get_gold_labels(train_cands, annotator='gold')
        # Sort LFs for LFAnalysis because LFAnalysis does not sort LFs,
        # while columns of L_train are sorted alphabetically already.
        sorted_lfs = sorted(station_price_lfs, key=lambda lf: lf.name)
        LFAnalysis(L=L_train[0], lfs=sorted_lfs).lf_summary(Y=L_gold_train[0].reshape(-1))

        # 5.) Build generative model
        gen_model = LabelModel(cardinality=2)
        gen_model.fit(L_train[0], n_epochs=500, log_freq=100)

        train_marginals_lfs = gen_model.predict_proba(L_train[0])

        # Apply on dev-set
        labeler.apply(split=1, lfs=[station_price_lfs], clear=True, parallelism=PARALLEL)
        L_dev = labeler.get_label_matrices(dev_cands)

        L_gold_dev = labeler.get_gold_labels(dev_cands, annotator='gold')
        LFAnalysis(L=L_dev[0], lfs=sorted_lfs).lf_summary(Y=L_gold_dev[0].reshape(-1))
        return (gen_model, train_marginals_lfs)

    def train_model(cands, F, train_marginals, model_type="LogisticRegression"):
        # Extract candidates and features
        train_cands = cands[0]
        F_train = F[0]
        
        # 1.) Setup training config
        config = {
            "meta_config": {"verbose": True},
            "model_config": {"model_path": None, "device": 0, "dataparallel": False},
            "learner_config": {
                "n_epochs": 50,
                "optimizer_config": {"lr": 0.001, "l2": 0.0},
                "task_scheduler": "round_robin",
            },
            "logging_config": {
                "evaluation_freq": 1,
                "counter_unit": "epoch",
                "checkpointing": False,
                "checkpointer_config": {
                    "checkpoint_metric": {f"{ATTRIBUTE}/{ATTRIBUTE}/train/loss": "min"},
                    "checkpoint_freq": 1,
                    "checkpoint_runway": 2,
                    "clear_intermediate_checkpoints": True,
                    "clear_all_checkpoints": True,
                },
            },
        }

        emmental.init(Meta.log_path)
        emmental.Meta.update_config(config=config)
        
        # 2.) Collect word counter from training data
        word_counter = collect_word_counter(train_cands)
        
        # 3.) Generate word embedding module for LSTM model
        # (in Logistic Regression, we generate it since Fonduer dataset requires word2id dict)
        # Geneate special tokens
        arity = 2
        specials = []
        for i in range(arity):
            specials += [f"~~[[{i}", f"{i}]]~~"]

        emb_layer = EmbeddingModule(
            word_counter=word_counter, word_dim=300, specials=specials
        )
        
        # 4.) Generate dataloader for training set
        # Filter out noise samples
        diffs = train_marginals.max(axis=1) - train_marginals.min(axis=1)
        train_idxs = np.where(diffs > 1e-6)[0]

        train_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE,
                train_cands[0],
                F_train[0],
                emb_layer.word2id,
                train_marginals,
                train_idxs,
            ),
            split="train",
            batch_size=100,
            shuffle=True,
        )
        
        # 5.) Training 
        tasks = create_task(
            ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model=model_type # "LSTM" 
        )

        model = EmmentalModel(name=f"{ATTRIBUTE}_task")

        for task in tasks:
            model.add_task(task)

        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, [train_dataloader])
        
        return (model, emb_layer)
        
    price_col_keywords = ["price", "weighted avg."]  
    DEBUG = False

    def eval_model(model, emb_layer, cands, F, schema_filter=False):
        # Extract candidates and features 
        train_cands = cands[0]
        dev_cands = cands[1]
        test_cands = cands[2] 
        F_train = F[0]
        F_dev = F[1]
        F_test = F[2]
        
        # apply schema filter
        def apply(cands):
            return schema_match_filter(
                cands, 
                "station", 
                "price", 
                price_col_keywords, 
                stations_mapping_dict, 
                0.05,
                DEBUG,
            )  
        
        # Generate dataloader for test data
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands[0], F_test[0], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )
        
        test_preds = model.predict(test_dataloader, return_preds=True)
        positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
        true_pred = [test_cands[0][_] for _ in positive[0]]
        true_pred = apply(true_pred) if schema_filter else true_pred        
        test_results = entity_level_f1(true_pred, gold_file, ATTRIBUTE, test_docs, stations_mapping_dict=stations_mapping_dict)

        # Run on dev and train set for validation
        # We run the predictions also on our training and dev set, to validate that everything seems to work smoothly
        
        # Generate dataloader for dev data
        dev_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, dev_cands[0], F_dev[0], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        dev_preds = model.predict(dev_dataloader, return_preds=True)
        positive_dev = np.where(np.array(dev_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
        true_dev_pred = [dev_cands[0][_] for _ in positive_dev[0]]
        true_dev_pred = apply(true_dev_pred) if schema_filter else true_dev_pred        
        dev_results = entity_level_f1(true_dev_pred, gold_file, ATTRIBUTE, dev_docs, stations_mapping_dict=stations_mapping_dict)

        # Generate dataloader for train data
        train_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, train_cands[0], F_train[0], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        train_preds = model.predict(train_dataloader, return_preds=True)
        positive_train = np.where(np.array(train_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
        true_train_pred = [train_cands[0][_] for _ in positive_train[0]]
        true_train_pred = apply(true_train_pred) if schema_filter else true_train_pred        
        train_results = entity_level_f1(true_train_pred, gold_file, ATTRIBUTE, train_docs, stations_mapping_dict=stations_mapping_dict)
    
        return [train_results, dev_results, test_results]

    return (train_model, eval_model, run_labeling_functions)