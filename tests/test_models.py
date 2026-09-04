"""Tests for src.models: the ClassifierModel interface and the TF-IDF+LR arm."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.features import full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.models.tfidf_lr import TfidfLRModel

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    train_df = records_to_dataframe(ingest_directory(ROOT / config["paths"]["train_dir"]))
    labels_df = load_train_labels(ROOT / config["paths"]["train_labels"])
    return attach_labels(train_df, labels_df)


def test_fit_returns_self_for_chaining(config: dict, labelled_df: pd.DataFrame) -> None:
    model = TfidfLRModel(config)
    result = model.fit(full_text(labelled_df), labelled_df["true_category"])
    assert result is model


def test_predict_proba_rows_sum_to_one_over_known_classes(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])

    proba = model.predict_proba(full_text(labelled_df))

    assert proba.shape == (len(labelled_df), len(config["categories"]))
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert set(model.classes_) == set(config["categories"])


def test_predict_with_confidence_matches_max_probability(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])

    proba = model.predict_proba(full_text(labelled_df))
    predicted, confidence = model.predict_with_confidence(full_text(labelled_df))

    assert confidence == pytest.approx(proba.max(axis=1))
    assert set(predicted) <= set(config["categories"])
    assert ((confidence > 0) & (confidence <= 1.0)).all()


def test_two_freshly_constructed_models_are_independent(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """A model_factory() pattern (used by the CV harness) must not leak fit state."""
    model_a = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])
    model_b = TfidfLRModel(config)

    with pytest.raises(Exception):
        model_b.predict_proba(full_text(labelled_df))
    assert model_a.predict_proba(full_text(labelled_df)) is not None
