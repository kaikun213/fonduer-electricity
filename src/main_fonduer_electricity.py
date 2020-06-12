# 1st Part
import argparse

# 2nd Part
import os
import sys
from fonduer import Meta, init_logging
import logging

# 3rd Part
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.models import Document, Sentence
from fonduer.parser import Parser
from pprint import pprint

# 4th Path
from my_subclasses import mention_classes, mention_spaces, matchers, candidate_classes, throttlers
from fonduer.candidates import MentionExtractor 
from fonduer.candidates.models import Mention
from fonduer_utils import prune_duplicate_mentions

# 5th Part
from fonduer.candidates import CandidateExtractor

# 6th Part
from fonduer.features import Featurizer
from fonduer.features.models import Feature
from fonduer.features.feature_extractors import FeatureExtractor

# 7th Part
from fonduer.supervision.models import GoldLabel
from fonduer.supervision import Labeler
from electricity_utils import get_gold_func
from my_subclasses import stations_mapping_dict

# 8-10 Part
from electricity_utils import eval_LFs, get_model_methods, summarize_results
import numpy as np

# 1.) PARAMETERS
if __name__=='__main__':
    arg_parser = argparse.ArgumentParser(description='processing inputs.')
    arg_parser.add_argument('--docs', type=int, default=114,
                      help='the number of docs to load.')
    arg_parser.add_argument('--exp', type=str, default="gold",
                    help='the experiment to run. Any option of "gold", "full", "gold_pred", "full_pred" which includes ' +
                    'with predicted cell annoations, manually labelled cell annoations or no cell annotations respectively.')
    arg_parser.add_argument('--cls_methods', type=str, default="rule-based, logistic-regression, lstm",
                    help='delimited list of the classification methods. Defaults to "rule-based, logistic-regression, lstm"')
    arg_parser.add_argument('--clear_db', type=int, default=1,
                    help='if the database should be cleared (recalculate mentions, candidates and features). 1 = True, 0 = False, Default = 1')
    args = arg_parser.parse_args()

    print("\n#1 Start configure parameters")
    max_docs = args.docs # 50 # 200
    experiment = args.exp # ["gold", "full", "pred"]
    cls_methods = [m for m in args.cls_methods.split(', ')]
    clear_db = True if args.clear_db == 1 else False
                            
    PARALLEL = 8 # 4  # assuming a quad-core machine
    ATTRIBUTE = f"elec_price_vol_{experiment}"

    DB_USERNAME = 'user'
    DB_PASSWORD = 'venron'
    conn_string = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@postgres:5432/{ATTRIBUTE}'

    folder = f"{experiment}_annotated" if "pred" in experiment else experiment    
    docs_path = f'data/{folder}/html/'
    pdf_path = f"data/pdf/"
    gold_file = 'data/electricity_gold.csv'


    # 2.) Cleanup and initialize
    print("\n#2 Cleanup and initialize")

    # Clear database
    if (clear_db):
        os.system(f"dropdb -h postgres --if-exists {ATTRIBUTE}")
    os.system(f"createdb -h postgres {ATTRIBUTE}")

    # Configure logging for Fonduer
    init_logging(log_dir=f"logs_{ATTRIBUTE}", level=logging.INFO) # DEBUG LOGGING
    session = Meta.init(conn_string).Session()

    # Initialize NLP library for vector similarities
    os.system(f"python3 -m spacy download en_core_web_lg")


    # 3.) Process documents into train,dev,test
    print("\n#3 Process Document into train, dev, test sets")

    # parse documents
    has_documents = session.query(Document).count() > 0
    corpus_parser = Parser(session, structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    if (not has_documents): 
        doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
        corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
        
    print(f"Documents: {session.query(Document).count()}")
    print(f"Sentences: {session.query(Sentence).count()}")

    # split documents
    docs = session.query(Document).order_by(Document.name).all()
    ld   = len(docs)
    train_docs = set()
    dev_docs   = set()
    test_docs  = set()
    splits = (0.8, 0.9)
    data = [(doc.name, doc) for doc in docs]
    data.sort(key=lambda x: x[0])
    for i, (doc_name, doc) in enumerate(data):
        if i < splits[0] * ld:
            train_docs.add(doc)
        elif i < splits[1] * ld:
            dev_docs.add(doc)
        else:
            test_docs.add(doc)
    all_docs = [train_docs, dev_docs, test_docs]

    # 4.) Mention Extraction
    print("\n#4 Mention extraction")

    hasMentions = session.query(Mention).count() > 0
    if (not hasMentions):
        mention_extractor = MentionExtractor(
            session, mention_classes,  mention_spaces, matchers
        )
        docs = session.query(Document).order_by(Document.name).all()
        mention_extractor.apply(docs, parallelism=PARALLEL)
    mentions = session.query(Mention).all()
    print(f"Total Mentions: {len(mentions)}")

    # Performance increase (reduce quadratic candidates combination by deleting duplicate mentions)
    Station = mention_classes[0]
    mentions = prune_duplicate_mentions(session, mentions, Station)

    # 5.) Candidate Extraction
    print("\n#5 Candidate extraction")
    StationPrice = candidate_classes[0]
    has_candidates = session.query(StationPrice).filter(StationPrice.split == 0).count() > 0

    candidate_extractor = CandidateExtractor(session, [StationPrice], throttlers=throttlers)
    for i, docs in enumerate([train_docs, dev_docs, test_docs]):
        if (not has_candidates):
            candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)
        print(f"Number of Candidates in split={i}: {session.query(StationPrice).filter(StationPrice.split == i).count()}")

    train_cands = candidate_extractor.get_candidates(split = 0)
    dev_cands = candidate_extractor.get_candidates(split = 1)
    test_cands = candidate_extractor.get_candidates(split = 2)
    cands = [train_cands, dev_cands, test_cands]

    # 6.) Featurize candidates
    has_features = session.query(Feature).count() > 0
    print(f"\n#6 Candidate featurization ({not has_features})")

    featurizer = Featurizer(
        session, 
        [StationPrice], 
        feature_extractors=FeatureExtractor(["textual", "structural", "tabular", "visual"])
    )

    if (not has_features):
        # Training set
        featurizer.apply(split=0, train=True, parallelism=PARALLEL)
        F_train = featurizer.get_feature_matrices(train_cands)
        print(F_train[0].shape)

        # Dev set
        featurizer.apply(split=1, parallelism=PARALLEL)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        print(F_dev[0].shape)

        # Test set
        featurizer.apply(split=2, parallelism=PARALLEL)
        F_test = featurizer.get_feature_matrices(test_cands)
        print(F_test[0].shape)
    else:
        F_train = featurizer.get_feature_matrices(train_cands)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        F_test = featurizer.get_feature_matrices(test_cands)
        
    F = [F_train, F_dev, F_test]

    # 7.) Load gold data
    print("\n#8 Load Gold Data")
    gold = get_gold_func(gold_file, attribute=ATTRIBUTE, stations_mapping_dict=stations_mapping_dict)
    docs = corpus_parser.get_documents()
    labeler = Labeler(session, [StationPrice])
    labeler.apply(docs=docs, lfs=[[gold]], table=GoldLabel, train=True, parallelism=PARALLEL)

    # 8.) Rule-based evaluation (Generative Model)
    (train_model, eval_model, run_labeling_functions) = get_model_methods(session, ATTRIBUTE, gold, gold_file, all_docs, StationPrice, PARALLEL)
    if ("rule-based" in cls_methods):
        print("\n#6 Rule-based classification")
        (gen_model, train_marginals_lfs) = run_labeling_functions(cands)
        eval_LFs(train_marginals_lfs, train_cands, gold)

    # 9.) Supervised classification (Logistic Regression)
    train_marginals_gold = np.array([[0,1] if gold(x) else [1,0] for x in train_cands[0]])
    train_marginals = train_marginals_gold if 'gold' in experiment else train_marginals_lfs
    if ("logistic-regression" in cls_methods):
        print("\n#9 Train and classify with Logistic Regression")

        # Build model and evaluate for Logistic Regression
        (lr_model, lr_emb_layer) = train_model(cands, F, train_marginals, "LogisticRegression" )

        print("Evaluate Logistic Regression method")
        lr_results = eval_model(lr_model, lr_emb_layer, cands, F)
        (prec_total, rec_total, f1_total) = summarize_results(lr_results)
        print(f"TOTAL DOCS PAIRWISE (LogisticRegression): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")

        print("Evaluate Logistic Regression method with schema matching")
        lr_results = eval_model(lr_model, lr_emb_layer, cands, F, True)
        (prec_total, rec_total, f1_total) = summarize_results(lr_results)
        print(f"TOTAL DOCS PAIRWISE (LogisticRegression): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")

    # 10.) Supervised classification (LSTM)
    if ("lstm" in cls_methods):
        print("\n#10 Train and classify with LSTM")

        # Build model and evaluate for LSTM
        (lstm_model, lstm_emb_layer) = train_model(cands, F, train_marginals, "LSTM" )

        print("Evaluate LSTM method")
        lstm_results = eval_model(lstm_model, lstm_emb_layer, cands, F)
        (prec_total, rec_total, f1_total) = summarize_results(lstm_results)
        print(f"TOTAL DOCS PAIRWISE (LSTM): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")


        print("Evaluate LSTM method with schema matching")
        lstm_results = eval_model(lstm_model, lstm_emb_layer, cands, F, True)
        (prec_total, rec_total, f1_total) = summarize_results(lstm_results)
        print(f"TOTAL DOCS PAIRWISE (LSTM): Precision={prec_total}, Recall={rec_total}, F1={f1_total}")
