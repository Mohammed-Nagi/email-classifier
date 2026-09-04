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


def test_run_is_deterministic(config: dict) -> None:
    first = build_predictions(config)
    second = build_predictions(config)

    pd.testing.assert_frame_equal(first, second)
