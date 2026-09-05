"""Tests for src.evaluate: the CV harness and the ablation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import (
    count_perfect_single_fold_runs,
    load_labelled_data,
    repeated_stratified_cv,
    run_ablation,
)
from src.features import full_text, subject_only
from src.model import TfidfLRModel


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
