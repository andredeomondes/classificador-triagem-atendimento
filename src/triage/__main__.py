"""Command line entry point for the triage classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import audit, build_candidates, load_dataset, reporting
from .evaluation import DEFAULT_FOLDS, CrossValidatedEvaluator


def build_parser() -> argparse.ArgumentParser:
    """Builds the command line parser."""
    parser = argparse.ArgumentParser(
        prog="triage", description="Compare triage classifiers on the same partitions."
    )
    parser.add_argument(
        "--corpus", type=Path, help="CSV file to load instead of the bundled corpus."
    )
    parser.add_argument(
        "--folds", type=int, default=DEFAULT_FOLDS, help="Number of stratified folds."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Directory for generated figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the full evaluation.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    dataset = load_dataset(args.corpus)
    reporting.describe_dataset(dataset)

    report = CrossValidatedEvaluator(n_splits=args.folds).evaluate(
        build_candidates(), dataset
    )
    reporting.describe_scores(report)
    reporting.describe_best_model(report, dataset)
    reporting.describe_errors(report.best, dataset)
    reporting.describe_audit(audit(dataset))

    figure = reporting.save_confusion_matrix(report.best, dataset, args.output)
    print(f"\nFigure written to {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
