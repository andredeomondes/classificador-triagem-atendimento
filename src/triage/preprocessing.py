"""Text preprocessing strategies for Portuguese support requests."""

from __future__ import annotations

import functools
from typing import Protocol, runtime_checkable

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

_LANGUAGE = "portuguese"

_REQUIRED_CORPORA = (
    ("stopwords", "corpora/stopwords"),
    ("rslp", "stemmers/rslp"),
    ("punkt_tab", "tokenizers/punkt_tab"),
)


@runtime_checkable
class Preprocessor(Protocol):
    """Transforms a raw document into the string fed to the vectorizer."""

    name: str

    def __call__(self, document: str) -> str:
        """Returns the normalized form of ``document``."""


class NullPreprocessor:
    """Passes documents through unchanged.

    Serves as the control condition when measuring whether linguistic
    normalization actually improves downstream classification.
    """

    name = "raw"

    def __call__(self, document: str) -> str:
        """Returns ``document`` unchanged."""
        return document


class PortugueseStemmer:
    """Tokenizes, drops stopwords and reduces Portuguese words to their stem.

    TF-IDF treats ``cancelado``, ``cancelamento`` and ``cancelar`` as unrelated
    terms, which splits the signal across sparse dimensions. The RSLP algorithm
    is designed for Portuguese morphology and maps all three onto ``cancel``.

    The reduction is not exhaustive. Nouns derived with suffixes such as
    ``-anca`` keep their own stem: ``cobrar`` becomes ``cobr`` while
    ``cobranca`` becomes ``cobranc``. Those families stay split, which is a
    known limitation of stemming compared to full lemmatization.
    """

    name = "nltk_rslp"

    def __init__(self) -> None:
        _ensure_corpora()
        self._stopwords = frozenset(stopwords.words(_LANGUAGE))
        self._stemmer = RSLPStemmer()

    def __call__(self, document: str) -> str:
        """Returns the stemmed, stopword-free form of ``document``."""
        tokens = word_tokenize(document.lower(), language=_LANGUAGE)
        return " ".join(
            self._stemmer.stem(token)
            for token in tokens
            if token.isalpha() and token not in self._stopwords
        )


@functools.cache
def _ensure_corpora() -> None:
    """Downloads the NLTK resources required for Portuguese, once per process."""
    for package, resource in _REQUIRED_CORPORA:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)
