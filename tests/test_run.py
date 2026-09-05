"""Tests for src.run: the end-to-end output contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.run import OUTPUT_COLUMNS, build_predictions


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def test_build_predictions_output_contract(config: dict) -> None:
    predictions = build_predictions(config)

    assert list(predictions.columns) == OUTPUT_COLUMNS
    assert list(predictions.columns[:3]) == ["email_id", "predicted_category", "confidence_score"]
    assert len(predictions) == 12
    assert set(predictions["predicted_category"]) <= set(config["categories"])
    assert predictions["confidence_score"].between(0, 1).all()
    assert predictions["email_id"].notna().all()
    assert predictions["email_id"].is_unique
    assert predictions["margin"].between(0, 1).all()
    assert set(predictions["runner_up_category"]) <= set(config["categories"])
    assert (predictions["runner_up_category"] != predictions["predicted_category"]).all()
    assert predictions["needs_review"].dtype == bool
    assert predictions["top_features"].apply(lambda s: isinstance(s, str)).all()


def test_probabilities_are_rounded_so_the_csv_is_platform_stable(config: dict) -> None:
    """Unrounded, these differ in the last ULPs across BLAS implementations,
    which would break the reproducibility claim the submission rests on."""
    predictions = build_predictions(config)

    for column in ("confidence_score", "margin"):
        assert (predictions[column].round(6) == predictions[column]).all()


def test_needs_review_agrees_with_the_confidence_column_as_written(config: dict) -> None:
    """A reader must be able to re-derive the flag from the printed number."""
    predictions = build_predictions(config)
    threshold = config["routing"]["auto_route_threshold"]

    expected = predictions["confidence_score"] < threshold
    assert (predictions["needs_review"] == expected).all()


def test_run_is_deterministic(config: dict) -> None:
    first = build_predictions(config)
    second = build_predictions(config)

    pd.testing.assert_frame_equal(first, second)
