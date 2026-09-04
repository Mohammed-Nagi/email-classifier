"""Tests for src.model and the src.run end-to-end output contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import build_pipeline, build_text_input, predict_with_confidence
from src.run import OUTPUT_COLUMNS, build_predictions

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    train_df = records_to_dataframe(ingest_directory(ROOT / config["paths"]["train_dir"]))
    labels_df = load_train_labels(ROOT / config["paths"]["train_labels"])
    return attach_labels(train_df, labels_df)


def test_pipeline_predicts_known_categories(config: dict, labelled_df: pd.DataFrame) -> None:
    pipeline = build_pipeline(config)
    pipeline.fit(build_text_input(labelled_df), labelled_df["true_category"])

    predicted, confidence = predict_with_confidence(pipeline, build_text_input(labelled_df))

    assert set(predicted) <= set(config["categories"])
    assert ((confidence > 0) & (confidence <= 1.0)).all()


def test_predict_with_confidence_matches_max_probability(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    pipeline = build_pipeline(config)
    pipeline.fit(build_text_input(labelled_df), labelled_df["true_category"])

    proba = pipeline.predict_proba(build_text_input(labelled_df))
    _, confidence = predict_with_confidence(pipeline, build_text_input(labelled_df))

    assert confidence == pytest.approx(proba.max(axis=1))


def test_build_predictions_output_contract(config: dict) -> None:
    predictions = build_predictions(config)

    assert list(predictions.columns) == OUTPUT_COLUMNS
    assert list(predictions.columns[:3]) == ["email_id", "predicted_category", "confidence_score"]
    assert len(predictions) == 12
    assert set(predictions["predicted_category"]) <= set(config["categories"])
    assert predictions["confidence_score"].between(0, 1).all()
    assert predictions["email_id"].notna().all()
    assert predictions["email_id"].is_unique


def test_run_is_deterministic(config: dict) -> None:
    first = build_predictions(config)
    second = build_predictions(config)

    pd.testing.assert_frame_equal(first, second)
