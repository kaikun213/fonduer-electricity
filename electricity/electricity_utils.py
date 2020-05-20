import codecs
import csv
import re
from builtins import range

from fonduer.learning.utils import confusion_matrix
from fonduer.supervision.models import GoldLabel, GoldLabelKey
from fonduer.candidates.models import Candidate 

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
