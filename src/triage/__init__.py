"""Text classification for support request triage."""

from .audit import AuditReport, audit, redact
from .dataset import Dataset, load_dataset
from .evaluation import CrossValidatedEvaluator, EvaluationReport, ModelScore
from .models import build_candidates

__all__ = [
    "AuditReport",
    "CrossValidatedEvaluator",
    "Dataset",
    "EvaluationReport",
    "ModelScore",
    "audit",
    "build_candidates",
    "load_dataset",
    "redact",
]
