"""Tests for src.evaluate: the repeated stratified CV harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import (
    CVResult,
    _DummyModel,
    count_perfect_single_fold_runs,
    load_labelled_data,
    macro_f1_excluding_class,
    per_class_f1_cv,
    repeated_stratified_cv,
    run_ablation,
    run_ablation_per_class,
)
from src.features import full_text, subject_only
from src.models.tfidf_lr import TfidfLRModel


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    return load_labelled_data(config)


def test_repeated_cv_produces_n_splits_times_n_repeats_folds(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    result = repeated_stratified_cv(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_splits=5,
        n_repeats=2,
    )

    assert isinstance(result, CVResult)
    assert len(result.macro_f1_scores) == 10
    assert len(result.accuracy_scores) == 10
    assert ((result.macro_f1_scores >= 0) & (result.macro_f1_scores <= 1)).all()
    assert ((result.accuracy_scores >= 0) & (result.accuracy_scores <= 1)).all()


def test_repeated_cv_is_deterministic_given_a_seed(config: dict, labelled_df: pd.DataFrame) -> None:
    kwargs = dict(
        model_factory=lambda: TfidfLRModel(config),
        texts=full_text(labelled_df),
        labels=labelled_df["true_category"],
        seed=config["seed"],
        n_splits=5,
        n_repeats=2,
    )
    first = repeated_stratified_cv(**kwargs)
    second = repeated_stratified_cv(**kwargs)

    np.testing.assert_array_equal(first.macro_f1_scores, second.macro_f1_scores)


def test_dummy_baseline_scores_far_below_the_real_model(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """Sanity check on the harness itself: a most-frequent-class dummy should
    score far below the real pipeline on the same data, reproducing PLAN.md
    §2's ~0.09 dummy macro-F1 figure at least directionally."""
    texts = full_text(labelled_df)
    labels = labelled_df["true_category"]
    seed = config["seed"]

    dummy_result = repeated_stratified_cv(_DummyModel, texts, labels, seed=seed, n_repeats=2)
    real_result = repeated_stratified_cv(
        lambda: TfidfLRModel(config), texts, labels, seed=seed, n_repeats=2
    )

    assert dummy_result.macro_f1_mean < 0.2
    assert real_result.macro_f1_mean > dummy_result.macro_f1_mean


def test_run_ablation_returns_one_row_per_variant_sorted_descending(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    variants = {"full_text": full_text}
    result = run_ablation(config, labelled_df, variants)

    assert list(result["variant"]) == ["full_text"]
    assert {"macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std"} <= set(
        result.columns
    )


def test_per_class_f1_cv_returns_one_row_per_class_sorted_descending(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    result = per_class_f1_cv(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
        n_repeats=2,
    )

    assert set(result["category"]) == set(config["categories"])
    assert (result["f1_mean"].diff().dropna() <= 0).all()
    assert ((result["f1_mean"] >= 0) & (result["f1_mean"] <= 1)).all()


def test_count_perfect_single_fold_runs_is_bounded_by_n_seeds(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    perfect = count_perfect_single_fold_runs(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        n_seeds=5,
    )
    assert 0 <= perfect <= 5


def test_dummy_model_never_scores_a_perfect_single_fold_run(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    perfect = count_perfect_single_fold_runs(
        _DummyModel, full_text(labelled_df), labelled_df["true_category"], n_seeds=5
    )
    assert perfect == 0


def test_run_ablation_per_class_returns_one_row_per_variant_per_category(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    variants = {"full_text": full_text, "subject_only": subject_only}
    result = run_ablation_per_class(config, labelled_df, variants)

    assert set(result["variant"]) == set(variants)
    assert set(result["category"]) == set(config["categories"])
    assert len(result) == len(variants) * len(config["categories"])
    assert ((result["f1_mean"] >= 0) & (result["f1_mean"] <= 1)).all()


def test_macro_f1_excluding_class_drops_the_named_category() -> None:
    per_class_df = pd.DataFrame(
        {
            "variant": ["v1", "v1", "v1", "v2", "v2", "v2"],
            "category": ["A", "B", "Other", "A", "B", "Other"],
            "f1_mean": [1.0, 0.5, 0.0, 0.8, 0.6, 0.0],
        }
    )

    result = macro_f1_excluding_class(per_class_df, exclude="Other")

    assert set(result["variant"]) == {"v1", "v2"}
    v1 = result.loc[result["variant"] == "v1", "macro_f1_excl_other"].item()
    v2 = result.loc[result["variant"] == "v2", "macro_f1_excl_other"].item()
    assert v1 == pytest.approx(0.75)  # mean(1.0, 0.5), Other dropped
    assert v2 == pytest.approx(0.7)  # mean(0.8, 0.6), Other dropped


def test_cv_result_summary_reports_mean_and_spread() -> None:
    result = CVResult(
        macro_f1_scores=np.array([0.8, 0.9, 1.0]),
        accuracy_scores=np.array([0.8, 0.9, 1.0]),
        n_splits=5,
        n_repeats=1,
    )
    summary = result.summary()

    assert "±" in summary
    assert "min=" in summary
    assert "max=" in summary
