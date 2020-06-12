from fonduer.candidates.models import mention_subclass
from fonduer.candidates.matchers import RegexMatchSpan, DictionaryMatch
from fonduer.candidates import MentionNgrams
from fonduer_utils import RegexMatchFull, StationMentionSpace
from fonduer.candidates.models import candidate_subclass
from fonduer.utils.data_model_utils import *


### Dictionary of known stations ###
stations = [
    [
        "cob",
        "california-oregon border",
        "california-oregon border (cob)",
    ],
    [
        "palo verde",
        "palo",
    ],
    [
        "mid columbia",
        "mid-columbia",
        "midc",
        "mid-c",
    ],
    [
        "mead",
        "meadmktplace",
        "mead/marketplace",
    ],
    [
        "np-15",
        "np 15",
        "np%2015",
        "california northern zone",
        "california northern zone (np-15)",
    ],
    [
        "sp-15",
        "sp 15",
        "sp%2015",
        "california southern zone",
        "california southern zone (sp-15)",
    ],
    [
        "pjm – western hub",
        "pjm",
    ],
]

stations_mapping_dict = { k:station_list for station_list in stations for k in station_list }
stations_list = [s for station_list in stations for s in station_list]
station_rgx = '|'.join(stations_list)

# 1.) Mention classes
Station = mention_subclass("Station")
Price = mention_subclass("Price")

# 2.) Mention spaces
station_ngrams = MentionNgrams(n_max=4, split_tokens=[" ", "_", "\.", "%"]) # StationMentionSpace(n_max=4) # 
price_ngrams = MentionNgrams(n_max=1)

# 3.) Matcher functions
station_matcher = RegexMatchFull(
    rgx=station_rgx, 
    ignore_case=True, 
    # search=True,
    # full_match=False,
    # longest_match_only=False,
)  # DictionaryMatch(d=stations_list)
price_matcher = RegexMatchSpan(rgx=r"\d{1,4}(\.\d{1,5})", longest_match_only=True)

# 4.) Candidate classes
StationPrice = candidate_subclass("StationPrice", [Station, Price])

# 5.) Throttlers
def my_throttler(c):
    (station, price) = c
    if 'volume' in get_aligned_ngrams(price, lower=True):
        return False
    if 'date' in get_aligned_ngrams(price, lower=True):
        return False 
    if 'non' in get_aligned_ngrams(price, lower=True):
        return False 
    html_tags = get_ancestor_tag_names(station)
    if ('head' in html_tags and 'title' in html_tags):
        return False
    return True

mention_classes = [Station, Price]
mention_spaces = [station_ngrams, price_ngrams]
matchers = [station_matcher, price_matcher]
candidate_classes = [StationPrice]
throttlers = [my_throttler]
