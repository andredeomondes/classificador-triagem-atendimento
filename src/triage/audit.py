"""Personal data and class balance auditing."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping

import pandas as pd

from .dataset import Dataset

_PII_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "cpf": re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
    "email": re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"),
    "phone": re.compile(r"\(?\d{2}\)?\s?9?\d{4}-?\d{4}"),
    "payment_card": re.compile(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"),
}

_SEVERE_IMBALANCE = 3.0


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Findings that must be resolved before a model reaches production.

    Attributes:
        pii_hits: Number of documents matching each personal data pattern.
        imbalance_ratio: Ratio between the largest and the smallest class.
        duplicate_count: Number of exactly repeated documents.
    """

    pii_hits: Mapping[str, int]
    imbalance_ratio: float
    duplicate_count: int

    @property
    def total_pii_hits(self) -> int:
        """Documents matching at least one personal data pattern."""
        return sum(self.pii_hits.values())

    @property
    def requires_anonymization(self) -> bool:
        """Whether personal data was found anywhere in the corpus."""
        return self.total_pii_hits > 0

    @property
    def is_severely_imbalanced(self) -> bool:
        """Whether class imbalance makes plain accuracy misleading."""
        return self.imbalance_ratio >= _SEVERE_IMBALANCE


def audit(dataset: Dataset) -> AuditReport:
    """Scans a corpus for personal data and distribution problems.

    The scan runs before any training. Personal data that reaches a model, a
    log or an external API cannot be recalled afterwards.

    Args:
        dataset: Corpus to inspect.

    Returns:
        An ``AuditReport`` summarizing the findings.
    """
    return AuditReport(
        pii_hits={name: _count_matches(dataset.texts, pattern)
                  for name, pattern in _PII_PATTERNS.items()},
        imbalance_ratio=dataset.imbalance_ratio,
        duplicate_count=dataset.duplicate_count,
    )


def redact(texts: pd.Series) -> pd.Series:
    """Replaces personal data with type placeholders.

    Args:
        texts: Documents to redact.

    Returns:
        A new series where every match is replaced by a token such as
        ``[CPF]``, preserving the surrounding text.
    """
    redacted = texts
    for name, pattern in _PII_PATTERNS.items():
        redacted = redacted.str.replace(pattern, f"[{name.upper()}]", regex=True)
    return redacted


def _count_matches(texts: pd.Series, pattern: re.Pattern[str]) -> int:
    return int(texts.str.contains(pattern, regex=True, na=False).sum())
