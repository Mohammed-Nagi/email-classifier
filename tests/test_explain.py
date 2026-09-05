"""Tests for src.explain: top-token attribution and its formatting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.explain import explain_predictions, format_top_features
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


def test_format_top_features_renders_token_and_weight() -> None:
    result = format_top_features([("loan", 0.421), ("officer", 0.313)])

    assert result == "loan (+0.42), officer (+0.31)"


def test_format_top_features_handles_empty_list() -> None:
    assert format_top_features([]) == ""


def test_top_contributing_tokens_returns_real_vocabulary_words(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])
    vocabulary = set(model._pipeline.named_steps["tfidf"].get_feature_names_out())

    contributions = model.top_contributing_tokens(full_text(labelled_df), top_k=5)

    assert len(contributions) == len(labelled_df)
    for row in contributions:
        assert len(row) <= 5
        for token, weight in row:
            assert token in vocabulary
            assert weight > 0
        # Descending order.
        weights = [weight for _, weight in row]
        assert weights == sorted(weights, reverse=True)


def test_explain_predictions_returns_one_formatted_string_per_row(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model = TfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])

    result = explain_predictions(model, full_text(labelled_df), top_k=3)

    assert len(result) == len(labelled_df)
    assert result.apply(lambda s: isinstance(s, str)).all()
