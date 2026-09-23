"""Tests for the triage classification package."""

from __future__ import annotations

import pandas as pd
import pytest

from triage import audit, load_dataset, redact
from triage.dataset import Dataset
from triage.evaluation import CrossValidatedEvaluator
from triage.models import BASELINE_NAME, build_baseline, build_candidates, build_pipeline
from triage.preprocessing import NullPreprocessor, PortugueseStemmer


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset()


class TestDataset:
    def test_loads_bundled_corpus(self, dataset: Dataset) -> None:
        assert len(dataset) > 0
        assert len(dataset.texts) == len(dataset.labels)

    def test_rejects_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "absent.csv")

    def test_rejects_missing_columns(self, tmp_path) -> None:
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"body": ["a"], "tag": ["b"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_dataset(path)

    def test_imbalance_ratio_is_one_when_balanced(self) -> None:
        balanced = Dataset(
            texts=pd.Series(["a", "b", "c", "d"]),
            labels=pd.Series(["x", "x", "y", "y"]),
        )
        assert balanced.imbalance_ratio == pytest.approx(1.0)


class TestPreprocessing:
    def test_null_preprocessor_is_identity(self) -> None:
        document = "Fui cobrado duas vezes"
        assert NullPreprocessor()(document) == document

    def test_stemmer_drops_stopwords(self) -> None:
        result = PortugueseStemmer()("Fui cobrado duas vezes no cartao")
        assert "no" not in result.split()

    def test_stemmer_unifies_word_family(self) -> None:
        stemmer = PortugueseStemmer()
        stems = {stemmer(word) for word in ("cancelado", "cancelamento", "cancelar")}
        assert stems == {"cancel"}

    def test_stemmer_does_not_unify_derived_noun(self) -> None:
        """Documents a known limitation: ``-anca`` nouns keep their own stem."""
        stemmer = PortugueseStemmer()
        assert stemmer("cobrar") != stemmer("cobranca")


class TestModels:
    def test_candidates_include_baseline(self) -> None:
        assert BASELINE_NAME in build_candidates()

    def test_candidates_cover_both_preprocessors(self) -> None:
        names = build_candidates().keys()
        assert any(name.startswith("raw__") for name in names)
        assert any(name.startswith("nltk_rslp__") for name in names)

    def test_unknown_classifier_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="Unknown classifier"):
            build_pipeline("nonexistent", NullPreprocessor())


class TestEvaluation:
    def test_rejects_single_fold(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            CrossValidatedEvaluator(n_splits=1)

    def test_best_model_beats_baseline(self, dataset: Dataset) -> None:
        report = CrossValidatedEvaluator(n_splits=3).evaluate(
            {BASELINE_NAME: build_baseline(),
             "raw__logistic_regression": build_pipeline(
                 "logistic_regression", NullPreprocessor())},
            dataset,
        )
        assert report.lift_over_baseline > 0

    def test_predictions_cover_every_document(self, dataset: Dataset) -> None:
        report = CrossValidatedEvaluator(n_splits=3).evaluate(
            {BASELINE_NAME: build_baseline()}, dataset
        )
        assert len(report.baseline.predictions) == len(dataset)


class TestAudit:
    def test_detects_personal_data(self) -> None:
        corpus = Dataset(
            texts=pd.Series(["meu cpf e 123.456.789-00", "sem dado pessoal"]),
            labels=pd.Series(["x", "y"]),
        )
        assert audit(corpus).requires_anonymization

    def test_clean_corpus_needs_no_anonymization(self, dataset: Dataset) -> None:
        assert not audit(dataset).requires_anonymization

    def test_redaction_removes_the_value(self) -> None:
        texts = pd.Series(["contato joao@exemplo.com por favor"])
        assert "joao@exemplo.com" not in redact(texts).iloc[0]
        assert "[EMAIL]" in redact(texts).iloc[0]
