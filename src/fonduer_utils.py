from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.candidates import MentionNgrams, MentionDocuments, MentionFigures
from typing import Collection, Optional, Iterator
from fonduer.parser.models import Document

from fonduer.candidates.models.document_mention import TemporaryDocumentMention
from fonduer.candidates.models.figure_mention import TemporaryFigureMention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.candidates.matchers import RegexMatchSpan
from fonduer.utils.data_model_utils import *

import itertools
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
from itertools import chain
from fonduer.supervision.models import Label
from sqlalchemy import func

# Additional functionality to Fonduer in order to process spreadsheets (some is specifically for the venron corpus)

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
            for ii in itertools.product(
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