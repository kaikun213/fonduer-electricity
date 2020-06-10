from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.candidates import MentionNgrams, MentionDocuments, MentionFigures
from typing import Collection, Optional, Iterator
from fonduer.parser.models import Document

from fonduer.candidates.models.document_mention import TemporaryDocumentMention
from fonduer.candidates.models.figure_mention import TemporaryFigureMention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.candidates.matchers import RegexMatchSpan
from fonduer.utils.data_model_utils import *

from typing import List, Optional, Union

from fonduer.parser.models.sentence import Sentence
from fonduer.parser.models.table import Cell
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.candidates.models import Candidate, Mention
from typing import Iterator, Tuple, Union
from fonduer.utils.data_model_utils.tabular import _get_aligned_sentences
from fonduer.utils.utils_table import is_axis_aligned # min_col_diff, min_row_diff, 
from fonduer.utils.utils import tokens_to_ngrams
from fonduer.utils.data_model_utils.utils import _to_span, _to_spans
from itertools import chain, tee, groupby, product
from fonduer.supervision.models import Label
from sqlalchemy import func

# Additional functionality to Fonduer in order to process spreadsheets (some is specifically for the venron corpus)

import operator
from pprint import pprint
from fonduer.utils.data_model_utils.tabular import _get_aligned_sentences
import spacy 

nlp = spacy.load("en_core_web_lg")

# Delete duplicate mentions (same document, same station-string) in order to increase performance
# For each additional station-span X price-mentions are combined to X candidates, so it quickly scales
def prune_duplicate_mentions(session, all_mentions, field):
    mentions = [m for m in all_mentions if isinstance(m, field)]
    mentions.sort(key=lambda m: f"{m.document.name} {m.context.get_span()}")
    for span, doc_mentions in groupby(mentions, lambda m: f"{m.document.name} {m.context.get_span()}"):
        doc_mentions = list(doc_mentions)
        # Keep only 1 entry
        duplicates = doc_mentions[1:]
        # Delete duplicates
        if (len(doc_mentions) > 1):
            print(f"Delete {len(duplicates)} Duplicates for {span}")
            pprint(duplicates)
            print()
            session.query(Mention).filter(Mention.id.in_([m.id for m in duplicates])).delete(synchronize_session="fetch")
    # Refetch updated mentions
    mentions = session.query(Mention).all()
    print(f"Total remaining Mentions: {len(mentions)}")
    return mentions

def get_col(m):
    s = m.context.sentence
    if (not s.is_tabular()):
        return -1
    if (s.cell.col_start != s.cell.col_end):
        return -1
    return s.cell.col_start

def get_headers(mentions_col):
    m_sentences = [m.context.sentence for m in mentions_col]
    min_row = min([x.cell.row_start for x in m_sentences])
    s = m_sentences[0]
    aligned = [x.text for x in _get_aligned_sentences(s, axis=1) if x not in m_sentences and x.cell.row_end < min_row]
    # TODO: HEADER cell-annotation condition
    return aligned

def get_sim(mentions_col_it, fid, pos_keyw, id_dict):
    headers = " , ".join(get_headers(list(mentions_col_it)))
    pos_keyw_vec = nlp(" , ".join(pos_keyw + id_dict[fid.context.get_span().lower()]))
    headers_vec = nlp(headers)

    # vectorize with word2vec and measure the similarity to positive/negative schema column keywords
    return pos_keyw_vec.similarity(headers_vec)


def schema_match_filter(cands, id_field, filter_field, pos_keyw = [], id_dict = {}, variance=0.05, DEBUG=False):
    filtered_cands = []
    
    # group them by document, itertools requires sorting    
    cands.sort(key=lambda c: c.document.name)
    for doc, doc_it in groupby(cands, lambda c: c.document.name):
        
        # group them by the candidate id field (e.g. all prices for one station-id)
        doc_cands = list(doc_it)
        doc_cands.sort(key=lambda c: getattr(c, id_field))
        for fid, doc_cand_it in groupby(doc_cands, lambda c: getattr(c, id_field)):
        
            it1, it2, it3 = tee(doc_cand_it, 3)
            # group by col
            doc_ms = [getattr(c, filter_field) for c in iter(it1)]
            doc_ms.sort(key=lambda m: get_col(m))
            ms_by_cols = { col:list(it) for col, it in groupby(doc_ms, lambda m: get_col(m)) }

            # ignore non tabular or multi-col/row 
            if (-1 in ms_by_cols.keys()):
                filtered_cands += [c for c in iter(it2) if getattr(c, filter_field) in ms_by_cols[-1]]

            # Compare headers of each column based on semantic similarity (word vectors)
            similarities = { col:get_sim(it, fid, pos_keyw, id_dict) for col, it in ms_by_cols.items() if col != -1 }
            sim_sorted = [(col, sim) for col, sim in sorted(similarities.items(), key=lambda i: i[1], reverse=True)]
            maximum = sim_sorted[0]
            
            # If there is a conflict (multiple assigned columns)
            # only take the maximum similarity as true for this candidate match
            if (len(sim_sorted) > 1 and DEBUG):
                print("#####################################")
                print(f"Similarity for {fid.context.get_span()} in doc {doc}")
                print(similarities)
                print(f"The maximum similarity is for entries in column {maximum}")
                print()
                for col, it in ms_by_cols.items():
                    print(f"Col {col} with {len(list(it))} entries and headers:")
                    pprint(get_headers(list(it)))
                print()
             
            # Filter only the k maximal similar column candidates based on variance
            for i in sim_sorted:
                if (i[1] >= maximum[1]-variance):
                    if (len(sim_sorted) > 1 and DEBUG):
                        print("KEEP", i)
                    filtered_cands += [c for c in iter(it3) if getattr(c, filter_field) in ms_by_cols[i[0]]]
            
            
          # only max column
#         counts = { col:len(list(it)) for col, it in ms_by_cols.items() if col != -1 }
#         maximum = max(counts.items(), key=operator.itemgetter(1))[0]
#         if (len(counts) > 1):
#             print("max and all", doc, maximum, counts, get_header(ms_by_cols[maximum][0]))
#             pprint(ms_by_cols)
#             print()
            
    return filtered_cands
        

# Code to check applied labelling functions
def get_applied_lfs(session): 
    labels = session.query(Label).all()
    ls = [x for x in labels if len(x.values) > 0]
    lfs = [lf for l in ls for lf in l.keys]
    return set(lfs)

def get_neighbor_cell_ngrams_own(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    dist: int = 1,
    directions: bool = False,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
    absolute: bool = False,
) -> Iterator[Union[str, Tuple[str, str]]]:
    """
    Get the ngrams from all Cells that are within a given Cell distance in one
    direction from the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be
    searched. If `directions=True``, each ngram will be returned with a
    direction in {'UP', 'DOWN', 'LEFT', 'RIGHT'}.

    :param mention: The Mention whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be
        considered
    :param directions: A Boolean expressing whether or not to return the
        direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams (or (ngram, direction) tuples if directions=True)
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = _to_spans(mention)
    for span in spans:
        for ngram in get_sentence_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
        if span.sentence.is_tabular():
            root_cell = span.sentence.cell
            for sentence in chain.from_iterable(
                [
                    _get_aligned_sentences(root_cell, "row"),
                    _get_aligned_sentences(root_cell, "col"),
                ]
            ):
                
                #####################################################################

                # Fix absolute bug. Because the underlying min_range_diff function 
                # will take negative numbers (e.g. -2), even though 0 would be closer
                row_diff = min_row_diff(sentence, root_cell, absolute=False)
                col_diff = min_col_diff(sentence, root_cell, absolute=False)
                    
                if (
                    (row_diff or col_diff)
                    and not (row_diff and col_diff)
                    and abs(row_diff) + abs(col_diff) <= dist
                ):
                    if directions:
                        direction = ""
                        if col_diff == 0:
                            if 0 < row_diff and row_diff <= dist:
                                direction = "UP"
                            elif 0 > row_diff and row_diff >= -dist:
                                direction = "DOWN"
                        elif row_diff == 0:
                            if 0 < col_diff and col_diff <= dist:
                                direction = "RIGHT"
                            elif 0 > col_diff and col_diff >= -dist:
                                direction = "LEFT"
                        for ngram in tokens_to_ngrams(
                            getattr(sentence, attrib),
                            n_min=n_min,
                            n_max=n_max,
                            lower=lower,
                        ):
                            yield (ngram, direction)
                    else:
                        for ngram in tokens_to_ngrams(
                            getattr(sentence, attrib),
                            n_min=n_min,
                            n_max=n_max,
                            lower=lower,
                        ):
                            yield ngram


# Fixed key=abs in order to process negative distances correctly
def _min_range_diff(
    a_start: int, a_end: int, b_start: int, b_end: int, absolute: bool = True
) -> int:
    # if absolute=True, return the absolute value of minimum magnitude difference
    # if absolute=False, return the raw value of minimum magnitude difference
    # TODO: move back to efficient implementation once it sees that
    # min_range_diff(3,3,2,3) = 0 return max(0, max(a_end - b_start, b_end -
    # a_start))
    f = lambda x: (abs(x) if absolute else x)
    return min(
        [
            f(ii[0] - ii[1])
            for ii in product(
                list(range(a_start, a_end + 1)), list(range(b_start, b_end + 1))
            )
        ],
        key=abs
    )

def min_row_diff(
    a: Union[Cell, Sentence], b: Union[Cell, Sentence], absolute: bool = True
) -> int:
    return _min_range_diff(
        a.row_start, a.row_end, b.row_start, b.row_end, absolute=absolute
    )


def min_col_diff(
    a: Union[Cell, Sentence], b: Union[Sentence, Cell], absolute: bool = True
) -> int:
    return _min_range_diff(
        a.col_start, a.col_end, b.col_start, b.col_end, absolute=absolute
    )

# Additional mentionspace class
class StationMentionSpace(MentionNgrams):
    """Defines the **space** of Mentions.

    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a
    Document *x*, divided into Sentences inside of html elements (such as table
    cells).

    :param n_min: Lower limit for the generated n_grams.
    :type n_min: int
    :param n_max: Upper limit for the generated n_grams.
    :type n_max: int
    :param split_tokens: Tokens, on which unigrams are split into two separate
        unigrams.
    :type split_tokens: tuple, list of str.
    :param types: If specified, only yield TemporaryFigureMentions whose url ends in
        one of the specified types. Example: types=["png", "jpg", "jpeg"].
    :type types: list, tuple of str
    """

    def __init__(
        self, n_min: int = 1, n_max: int = 5, split_tokens: Collection[str] = [], types: Optional[str] = None
    ) -> None:
        """Initialize MentionNgrams."""
        MentionNgrams.__init__(self, n_min=n_min, n_max=n_max, split_tokens=split_tokens)
        if types is not None:
            self.types = [t.strip().lower() for t in types]
        else:
            self.types = None

    def apply(self, doc: Document) -> Iterator[TemporaryContext]:
        """Generate MentionNgrams from a Document by parsing all of its Sentences.

        :param doc: The ``Document`` to parse.
        :type doc: ``Document``
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionNgrams.apply() must be of type Document"
            )

        for ts in MentionNgrams.apply(self, doc):
            yield ts
            
        for ts in MentionDocuments.apply(self, doc):
            yield ts
            
        for ts in MentionFigures.apply(self, doc):
            yield ts

# Additional matcher function for the new mentionspace class
class RegexMatchFull(RegexMatchSpan):
    """Matches regex pattern on **full concatenated span, document-name and figure-urls**.

    :param rgx: The RegEx pattern to use.
    :type rgx: str
    :param ignore_case: Whether or not to ignore case in the RegEx. Default
        True.
    :type ignore_case: bool
    :param search: If True, *search* the regex pattern through the concatenated span.
        If False, try to *match* the regex patten only at its beginning. Default False.
    :type search: bool
    :param full_match: If True, wrap the provided rgx with ``(<rgx>)$``.
        Default True.
    :type full_match: bool
    :param longest_match_only: If True, only return the longest match. Default True.
        Will be overridden by the parent matcher like :class:`Union` when it is wrapped
        by :class:`Union`, :class:`Intersect`, or :class:`Inverse`.
    :type longest_match_only: bool
    """

    def _f(self, m: TemporaryContext) -> bool:
        """The internal (non-composed) version of filter function f"""
        
        def apply_rgx(attrib_span):
            # search for string as e.g. "_" split operator is used
            return (
                True
                if self.r.search(attrib_span)
                is not None
                else False
            )
        if isinstance(m, TemporarySpanMention):
            return RegexMatchSpan._f(self, m)
        if isinstance(m, TemporaryFigureMention):
            return apply_rgx(m.figure.url)
        if isinstance(m, TemporaryDocumentMention):
            return apply_rgx(m.document.name)
        
        raise ValueError(
            f"""
            {self.__class__.__name__} only supports 
            TemporarySpanMention, TemporaryFigureMention and TemporaryDocumentMention
            """
        )