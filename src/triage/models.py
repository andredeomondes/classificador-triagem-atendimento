"""Model construction for the triage classifier."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocessing import NullPreprocessor, PortugueseStemmer, Preprocessor

RANDOM_STATE = 42

BASELINE_NAME = "baseline_most_frequent"

_CLASSIFIER_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "multinomial_nb": MultinomialNB,
}


def build_baseline() -> BaseEstimator:
    """Builds the naive reference model.

    The baseline always predicts the most frequent class. Any model that fails
    to beat it has learned nothing from the text.

    Returns:
        An unfitted estimator.
    """
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)


def build_pipeline(classifier_name: str, preprocessor: Preprocessor) -> Pipeline:
    """Builds a TF-IDF pipeline for one classifier and one preprocessor.

    Args:
        classifier_name: Key of a registered classifier factory.
        preprocessor: Strategy applied to each document before vectorization.

    Returns:
        An unfitted scikit-learn pipeline.

    Raises:
        KeyError: If ``classifier_name`` is not registered.
    """
    if classifier_name not in _CLASSIFIER_FACTORIES:
        raise KeyError(
            f"Unknown classifier {classifier_name!r}; "
            f"available: {sorted(_CLASSIFIER_FACTORIES)}"
        )

    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    return Pipeline(
        [("tfidf", vectorizer), ("classifier", _CLASSIFIER_FACTORIES[classifier_name]())]
    )


def build_candidates() -> Mapping[str, BaseEstimator]:
    """Builds every model to be compared, including the baseline.

    Each classifier is paired with each preprocessing strategy, so the effect
    of linguistic normalization can be isolated from the choice of classifier.

    Returns:
        Mapping from model name to unfitted estimator.
    """
    candidates: dict[str, BaseEstimator] = {BASELINE_NAME: build_baseline()}
    for preprocessor in _preprocessors():
        for classifier_name in _CLASSIFIER_FACTORIES:
            name = f"{preprocessor.name}__{classifier_name}"
            candidates[name] = build_pipeline(classifier_name, preprocessor)
    return candidates


def _preprocessors() -> Iterator[Preprocessor]:
    yield NullPreprocessor()
    yield PortugueseStemmer()
