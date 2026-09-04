"""Tests for src.calibrate: the calibrated model, its metrics, and the
paired raw-vs-calibrated comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibrate import (
    CalibratedTfidfLRModel,
    calibration_cv,
    expected_calibration_error,
    multiclass_brier_score,
    paired_calibration_comparison,
    reliability_curve_data,
)
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


# --- CalibratedTfidfLRModel: same interface contract as TfidfLRModel ---


def test_fit_returns_self_for_chaining(config: dict, labelled_df: pd.DataFrame) -> None:
    model = CalibratedTfidfLRModel(config)
    result = model.fit(full_text(labelled_df), labelled_df["true_category"])
    assert result is model


def test_predict_proba_rows_sum_to_one_over_known_classes(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model = CalibratedTfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])

    proba = model.predict_proba(full_text(labelled_df))

    assert proba.shape == (len(labelled_df), len(config["categories"]))
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert set(model.classes_) == set(config["categories"])


def test_two_freshly_constructed_models_are_independent(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    model_a = CalibratedTfidfLRModel(config).fit(full_text(labelled_df), labelled_df["true_category"])
    model_b = CalibratedTfidfLRModel(config)

    with pytest.raises(Exception):
        model_b.predict_proba(full_text(labelled_df))
    assert model_a.predict_proba(full_text(labelled_df)) is not None


# --- Metric correctness, on hand-built arrays (not fitted models) ---


def test_multiclass_brier_score_zero_for_perfect_confident_predictions() -> None:
    classes = np.array(["A", "B"])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    true_labels = np.array(["A", "B"])

    assert multiclass_brier_score(proba, true_labels, classes) == pytest.approx(0.0)


def test_multiclass_brier_score_two_for_fully_confident_wrong_predictions() -> None:
    classes = np.array(["A", "B"])
    proba = np.array([[0.0, 1.0], [1.0, 0.0]])
    true_labels = np.array(["A", "B"])

    assert multiclass_brier_score(proba, true_labels, classes) == pytest.approx(2.0)


def test_expected_calibration_error_zero_when_confidence_matches_accuracy_per_bin() -> None:
    # 10 predictions all at confidence 0.9, 9 correct -> bin accuracy == bin confidence.
    confidences = np.full(10, 0.9)
    correct = np.array([True] * 9 + [False])

    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_positive_when_overconfident() -> None:
    # All predictions claim 0.9 confidence but only half are correct.
    confidences = np.full(10, 0.9)
    correct = np.array([True] * 5 + [False] * 5)

    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_reliability_curve_data_counts_sum_to_sample_size() -> None:
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0, 1, size=50)
    correct = rng.random(50) < confidences

    curve = reliability_curve_data(confidences, correct, n_bins=10)

    assert curve["count"].sum() == 50
    assert (curve["mean_confidence"].between(0, 1)).all()
    assert (curve["empirical_accuracy"].between(0, 1)).all()


# --- CV-level checks, against the real pipeline ---


def test_calibration_cv_is_deterministic_given_a_seed(config: dict, labelled_df: pd.DataFrame) -> None:
    kwargs = dict(
        model_factory=lambda: CalibratedTfidfLRModel(config),
        texts=full_text(labelled_df),
        labels=labelled_df["true_category"],
        seed=config["seed"],
        n_splits=5,
        n_repeats=2,
    )
    first = calibration_cv(**kwargs)
    second = calibration_cv(**kwargs)

    np.testing.assert_array_equal(first.confidences, second.confidences)
    assert first.brier == pytest.approx(second.brier)
    assert first.ece == pytest.approx(second.ece)


def test_calibration_cv_pools_every_fold(config: dict, labelled_df: pd.DataFrame) -> None:
    result = calibration_cv(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_splits=5,
        n_repeats=2,
    )
    assert len(result.confidences) == len(labelled_df) * 2
    assert 0.0 <= result.brier <= 2.0
    assert 0.0 <= result.ece <= 1.0


def test_paired_calibration_comparison_flip_count_is_bounded(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    result = paired_calibration_comparison(
        config,
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_splits=5,
        n_repeats=2,
    )

    assert result.n_predictions == len(labelled_df) * 2
    assert 0 <= result.n_flipped <= result.n_predictions
    assert len(result.raw_macro_f1) == 10
    assert len(result.calibrated_macro_f1) == 10
    # Each fold's delta is a valid paired difference (identical train/test membership).
    np.testing.assert_array_equal(
        result.macro_f1_delta, result.calibrated_macro_f1 - result.raw_macro_f1
    )
