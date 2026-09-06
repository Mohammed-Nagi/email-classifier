"""Tests for src.model: both arms' contract, and token attribution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.features import full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import (
    AbstainOtherModel,
    CalibratedTfidfLRModel,
    TfidfLRModel,
    abstain_predict,
    explain_predictions,
    predict_with_confidence,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    train_df = records_to_dataframe(ingest_directory(ROOT / config["paths"]["train_dir"]))
    labels_df = load_train_labels(ROOT / config["paths"]["train_labels"])
    return attach_labels(train_df, labels_df)


@pytest.mark.parametrize("model_class", [TfidfLRModel, CalibratedTfidfLRModel])
def test_predict_proba_contract(model_class, config: dict, labelled_df: pd.DataFrame) -> None:
    """Both arms must satisfy what the CV harness and routing assume."""
    texts = full_text(labelled_df)
    model = model_class(config).fit(texts, labelled_df["true_category"])

    proba = model.predict_proba(texts)

    assert proba.shape == (len(labelled_df), len(config["categories"]))
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert list(model.classes_) == sorted(config["categories"])


def test_model_factory_returns_independent_unfitted_models(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """The CV harness refits per fold; a leaked fit would leak across folds."""
    fitted = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])
    fresh = TfidfLRModel(config)

    with pytest.raises(Exception):
        fresh.predict_proba(full_text(labelled_df))
    assert fitted.predict_proba(full_text(labelled_df)) is not None


def test_predict_with_confidence_matches_max_probability(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    texts = full_text(labelled_df)
    model = TfidfLRModel(config).fit(texts, labelled_df["true_category"])

    predicted, confidence = predict_with_confidence(model, texts)

    assert confidence == pytest.approx(model.predict_proba(texts).max(axis=1))
    assert set(predicted) <= set(config["categories"])


def test_top_contributing_tokens_are_positive_ranked_vocabulary_words(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    texts = full_text(labelled_df)
    model = TfidfLRModel(config).fit(texts, labelled_df["true_category"])
    vocabulary = set(model._pipeline.named_steps["tfidf"].get_feature_names_out())

    rows = model.top_contributing_tokens(texts, top_k=5)

    assert len(rows) == len(labelled_df)
    for row in rows:
        assert len(row) <= 5
        assert all(token in vocabulary and weight > 0 for token, weight in row)
        weights = [weight for _, weight in row]
        assert weights == sorted(weights, reverse=True)


def test_tied_contributions_break_on_token_string_not_float_order(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """Tied coefficients previously sorted by float order, so the top_features
    column differed across platforms. Ties must order alphabetically."""
    texts = full_text(labelled_df)
    model = TfidfLRModel(config).fit(texts, labelled_df["true_category"])

    for row in model.top_contributing_tokens(texts, top_k=10):
        for (token_a, weight_a), (token_b, weight_b) in zip(row, row[1:]):
            if weight_a == weight_b:
                assert token_a < token_b


def test_abstain_model_never_trains_on_other(config: dict, labelled_df: pd.DataFrame) -> None:
    texts = full_text(labelled_df)
    model = AbstainOtherModel(config).fit(texts, labelled_df["true_category"])

    assert "Other" not in set(model.classes_)
    assert len(model.classes_) == 4


def test_abstain_predict_maps_low_confidence_to_other(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    texts = full_text(labelled_df)
    model = AbstainOtherModel(config).fit(texts, labelled_df["true_category"])

    low = abstain_predict(model, texts, threshold=1.01)  # nothing clears this
    high = abstain_predict(model, texts, threshold=0.0)  # everything clears this

    assert set(low) == {"Other"}
    assert "Other" not in set(high)


def test_explain_predictions_formats_one_string_per_row(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    texts = full_text(labelled_df)
    model = TfidfLRModel(config).fit(texts, labelled_df["true_category"])

    explanations = explain_predictions(model, texts, top_k=3)

    assert len(explanations) == len(labelled_df)
    assert all(isinstance(text, str) for text in explanations)
    assert "(+" in explanations[0]
