"""Cross-validated evaluation of classification models."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .dataset import Dataset
from .models import BASELINE_NAME, RANDOM_STATE

DEFAULT_FOLDS = 5


@dataclasses.dataclass(frozen=True)
class ModelScore:
    """Cross-validated performance of a single model.

    Attributes:
        name: Identifier of the evaluated model.
        macro_f1: Macro-averaged F1 over all out-of-fold predictions.
        fold_std: Standard deviation of macro F1 across folds.
        predictions: Out-of-fold prediction for every document.
    """

    name: str
    macro_f1: float
    fold_std: float
    predictions: np.ndarray

    @property
    def is_baseline(self) -> bool:
        """Whether this score belongs to the naive reference model."""
        return self.name == BASELINE_NAME


@dataclasses.dataclass(frozen=True)
class EvaluationReport:
    """Outcome of comparing every candidate model on the same partitions."""

    scores: tuple[ModelScore, ...]

    @property
    def baseline(self) -> ModelScore:
        """The naive reference model."""
        return next(score for score in self.scores if score.is_baseline)

    @property
    def best(self) -> ModelScore:
        """Highest scoring model, excluding the baseline."""
        learned = [score for score in self.scores if not score.is_baseline]
        return max(learned, key=lambda score: score.macro_f1)

    @property
    def lift_over_baseline(self) -> float:
        """Macro F1 gained by the best model over the naive reference."""
        return self.best.macro_f1 - self.baseline.macro_f1

    def to_frame(self) -> pd.DataFrame:
        """Returns the scores as a table sorted by performance."""
        return pd.DataFrame(
            [
                {
                    "model": score.name,
                    "macro_f1": score.macro_f1,
                    "fold_std": score.fold_std,
                }
                for score in self.scores
            ]
        ).sort_values("macro_f1", ascending=False, ignore_index=True)


class CrossValidatedEvaluator:
    """Scores models with stratified k-fold cross-validation.

    A single train/test split would leave too few documents in the test set for
    the resulting metric to distinguish between models. Cross-validation
    predicts every document exactly once, while it is held out of training.
    """

    def __init__(self, n_splits: int = DEFAULT_FOLDS) -> None:
        """Initializes the evaluator.

        Args:
            n_splits: Number of stratified folds.

        Raises:
            ValueError: If ``n_splits`` is smaller than two.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self._splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )

    def evaluate(
        self, candidates: Mapping[str, BaseEstimator], dataset: Dataset
    ) -> EvaluationReport:
        """Evaluates every candidate on identical partitions.

        Args:
            candidates: Mapping from model name to unfitted estimator.
            dataset: Corpus to evaluate against.

        Returns:
            An ``EvaluationReport`` holding one score per candidate.
        """
        scores = tuple(
            self._score(name, model, dataset) for name, model in candidates.items()
        )
        return EvaluationReport(scores=scores)

    def _score(
        self, name: str, model: BaseEstimator, dataset: Dataset
    ) -> ModelScore:
        predictions = cross_val_predict(
            model, dataset.texts, dataset.labels, cv=self._splitter
        )
        per_fold = [
            f1_score(
                dataset.labels.iloc[test_index],
                predictions[test_index],
                average="macro",
            )
            for _, test_index in self._splitter.split(dataset.texts, dataset.labels)
        ]
        return ModelScore(
            name=name,
            macro_f1=float(f1_score(dataset.labels, predictions, average="macro")),
            fold_std=float(np.std(per_fold)),
            predictions=predictions,
        )
