"""Tests for src.evaluate: the CV harness and the ablation diagnostics."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import (
    build_confusion_matrix,
    count_perfect_single_fold_runs,
    load_labelled_data,
    repeated_stratified_cv,
    run_ablation,
    sweep_abstain_thresholds,
)
from src.features import full_text, subject_only
from src.model import AbstainOtherModel, TfidfLRModel, abstain_predict


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def labelled_df(config: dict) -> pd.DataFrame:
    return load_labelled_data(config)


def test_repeated_cv_scores_every_fold_and_is_seed_deterministic(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """Determinism is the whole basis for treating same-seed runs as paired."""
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

    assert len(first.macro_f1_scores) == 10
    assert ((first.macro_f1_scores >= 0) & (first.macro_f1_scores <= 1)).all()
    np.testing.assert_array_equal(first.macro_f1_scores, second.macro_f1_scores)


def test_same_seed_gives_identical_folds_across_text_variants(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """Stratification depends on labels only, so two variants at one seed are a
    paired comparison — the basis of every variant-vs-variant claim."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    labels = labelled_df["true_category"]
    splits_a = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=config["seed"]).split(
            full_text(labelled_df), labels
        )
    )
    splits_b = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=config["seed"]).split(
            subject_only(labelled_df), labels
        )
    )

    for (train_a, test_a), (train_b, test_b) in zip(splits_a, splits_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_run_ablation_returns_summary_and_per_class_from_one_pass(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    variants = {"full_text": full_text, "subject_only": subject_only}
    summary, per_class = run_ablation(config, labelled_df, variants)

    assert set(summary["variant"]) == set(variants)
    assert len(per_class) == len(variants) * len(config["categories"])
    assert {"macro_f1_mean", "macro_f1_std", "macro_f1_excl_other"} <= set(summary.columns)


def test_ablation_macro_f1_equals_mean_of_its_own_per_class_scores(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """The single-pass derivation depends on macro-F1 being the unweighted mean
    of per-class F1s. If that ever stops holding, both columns are wrong."""
    summary, per_class = run_ablation(config, labelled_df, {"full_text": full_text})

    assert summary["macro_f1_mean"].item() == pytest.approx(per_class["f1_mean"].mean())
    excl = per_class[per_class["category"] != "Other"]["f1_mean"].mean()
    assert summary["macro_f1_excl_other"].item() == pytest.approx(excl)


def test_count_perfect_single_fold_runs_is_bounded(config: dict, labelled_df: pd.DataFrame) -> None:
    perfect = count_perfect_single_fold_runs(
        lambda: TfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        n_seeds=5,
    )
    assert 0 <= perfect <= 5


def test_run_ablation_accepts_a_different_model_and_predict_fn(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """The abstain-as-Other comparison depends on this: same harness, same
    folds, a different model/predict_fn pair."""
    variants = {"full_text": full_text}
    summary, per_class = run_ablation(
        config,
        labelled_df,
        variants,
        model_factory=lambda: AbstainOtherModel(config),
        predict_fn=partial(abstain_predict, threshold=config["abstain"]["threshold"]),
    )

    assert set(summary["variant"]) == set(variants)
    assert set(per_class["category"]) == set(config["categories"])
    # Abstain can still predict "Other" via the threshold gate, even though
    # the underlying model never saw it as a training label.
    assert (per_class["f1_mean"] >= 0).all()


def test_build_confusion_matrix_rows_sum_to_class_counts_times_repeats(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    """Pooled out-of-fold: 10 repeats over 44 examples, so each row sums to
    10x that category's training count."""
    confusion = build_confusion_matrix(config, labelled_df)

    assert set(confusion.index) == set(config["categories"])
    assert set(confusion.columns) == set(config["categories"])
    class_counts = labelled_df["true_category"].value_counts()
    for category in config["categories"]:
        assert confusion.loc[category].sum() == class_counts[category] * 10


def test_sweep_abstain_thresholds_reports_other_f1_and_macro_f1(
    config: dict, labelled_df: pd.DataFrame
) -> None:
    sweep = sweep_abstain_thresholds(config, labelled_df, thresholds=np.array([0.35, 0.45]))

    assert list(sweep["threshold"]) == [0.35, 0.45]
    assert {"other_f1_mean", "other_f1_std", "macro_f1_mean", "macro_f1_std"} <= set(sweep.columns)
    assert sweep["other_f1_mean"].between(0, 1).all()
