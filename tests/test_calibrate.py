"""Tests for src.calibrate: metric correctness and the paired comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibrate import (
    calibration_cv,
    expected_calibration_error,
    multiclass_brier_score,
    paired_calibration_comparison,
)
from src.config import load_config
from src.evaluate import load_labelled_data
from src.features import full_text
from src.model import TfidfLRModel


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    return load_labelled_data(config)


def test_brier_score_spans_perfect_to_confidently_wrong() -> None:
    classes = np.array(["A", "B"])
    true_labels = np.array(["A", "B"])

    perfect = multiclass_brier_score(np.array([[1.0, 0.0], [0.0, 1.0]]), true_labels, classes)
    wrong = multiclass_brier_score(np.array([[0.0, 1.0], [1.0, 0.0]]), true_labels, classes)

    assert perfect == pytest.approx(0.0)
    assert wrong == pytest.approx(2.0)


def test_expected_calibration_error_is_the_confidence_accuracy_gap() -> None:
    confidences = np.full(10, 0.9)

    calibrated = expected_calibration_error(confidences, np.array([True] * 9 + [False]))
    overconfident = expected_calibration_error(confidences, np.array([True] * 5 + [False] * 5))

    assert calibrated == pytest.approx(0.0, abs=1e-9)
    assert overconfident == pytest.approx(0.4, abs=1e-9)


def test_calibration_cv_pools_every_fold(config: dict, labelled_df: pd.DataFrame) -> None:
    """Metrics must come from held-out predictions, one per row per repeat —
    computing them in-sample would report a flatteringly wrong number."""
    result = calibration_cv(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_repeats=2,
    )

    assert len(result.confidences) == len(labelled_df) * 2
    assert len(result.correct) == len(result.confidences)
    assert 0.0 <= result.brier <= 2.0
    assert 0.0 <= result.ece <= 1.0


def test_paired_comparison_pairs_folds_and_counts_flips(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    result = paired_calibration_comparison(
        config,
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_repeats=2,
    )

    assert result.n_predictions == len(labelled_df) * 2
    assert 0 <= result.n_flipped <= result.n_predictions
    assert len(result.raw_macro_f1) == len(result.calibrated_macro_f1) == 10
    np.testing.assert_array_equal(
        result.macro_f1_delta, result.calibrated_macro_f1 - result.raw_macro_f1
    )
