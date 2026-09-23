"""Console and figure output for evaluation results."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from .audit import AuditReport
from .dataset import Dataset
from .evaluation import EvaluationReport, ModelScore

_MAX_LISTED_ERRORS = 10
_FIGURE_DPI = 120


def describe_dataset(dataset: Dataset) -> None:
    """Prints the shape and composition of the corpus."""
    print("Corpus")
    print(f"  documents           {len(dataset)}")
    print(f"  classes             {len(dataset.class_names)}")
    print(f"  duplicates          {dataset.duplicate_count}")
    print(f"  imbalance ratio     {dataset.imbalance_ratio:.2f}")
    for label, count in dataset.class_counts.items():
        print(f"    {label:<20} {count}")


def describe_scores(report: EvaluationReport) -> None:
    """Prints every model score, ordered by macro F1."""
    print("\nCross-validated comparison")
    for row in report.to_frame().itertuples(index=False):
        marker = " <- best" if row.model == report.best.name else ""
        print(f"  {row.model:<34} {row.macro_f1:.3f}  +/-{row.fold_std:.3f}{marker}")
    print(f"\n  lift over baseline  {report.lift_over_baseline:+.3f} macro F1")


def describe_best_model(report: EvaluationReport, dataset: Dataset) -> None:
    """Prints the per-class report of the best performing model."""
    print(f"\nBest model: {report.best.name}")
    print(classification_report(dataset.labels, report.best.predictions, digits=3))


def describe_errors(score: ModelScore, dataset: Dataset) -> None:
    """Prints misclassified documents for manual inspection."""
    errors = list(_iter_errors(score, dataset))
    print(f"Misclassified: {len(errors)} of {len(dataset)}")
    for text, expected, predicted in errors[:_MAX_LISTED_ERRORS]:
        print(f"\n  expected {expected} / predicted {predicted}")
        print(f"  {text}")


def describe_audit(report: AuditReport) -> None:
    """Prints the personal data and class balance findings."""
    print("\nData audit")
    for name, hits in report.pii_hits.items():
        print(f"  {name:<20} {hits}")
    print(f"  anonymization needed {report.requires_anonymization}")
    print(f"  severe imbalance     {report.is_severely_imbalanced}")


def save_confusion_matrix(
    score: ModelScore, dataset: Dataset, destination: Path
) -> Path:
    """Writes the confusion matrix of a model to disk.

    Args:
        score: Model whose out-of-fold predictions are plotted.
        dataset: Corpus holding the true labels.
        destination: Directory where the figure is written.

    Returns:
        Path of the written figure.
    """
    labels = dataset.class_names
    matrix = confusion_matrix(dataset.labels, score.predictions, labels=labels)

    figure, axes = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axes, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    axes.set_title(f"Confusion matrix - {score.name}")
    figure.tight_layout()

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "confusion_matrix.png"
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path


def _iter_errors(
    score: ModelScore, dataset: Dataset
) -> Iterator[tuple[str, str, str]]:
    for text, expected, predicted in zip(
        dataset.texts, dataset.labels, score.predictions, strict=True
    ):
        if expected != predicted:
            yield text, expected, predicted
