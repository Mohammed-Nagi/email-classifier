"""Repeated stratified CV harness + the ablation/perturbation runner.

The provided 44-label training set is too small to hold out a test split
(see PLAN.md §7), so every reported number in this project comes from
repeated stratified k-fold CV instead. This module is written once against
:class:`~src.models.base.ClassifierModel` so any arm — TF-IDF+LR today,
embeddings or zero-shot later — reuses the same evaluation code.

Run directly to reproduce PLAN.md §2's headline CV score, the §2 dummy
baseline, and the §4 ablation/perturbation degradation table against this
codebase's actual pipeline and text-assembly variants:

    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import load_config
from src.features import ABLATION_VARIANTS, FEATURE_VARIANTS
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.models.base import ClassifierModel
from src.models.tfidf_lr import TfidfLRModel

REPO_ROOT = Path(__file__).resolve().parent.parent

ModelFactory = Callable[[], ClassifierModel]


@dataclass(frozen=True)
class CVResult:
    """Per-fold scores from repeated stratified k-fold CV.

    Holds every fold's score, not just the mean — at n=44, fold-to-fold
    variance is as important to report as the mean (PLAN.md §7/§10).
    """

    macro_f1_scores: np.ndarray
    accuracy_scores: np.ndarray
    n_splits: int
    n_repeats: int

    @property
    def macro_f1_mean(self) -> float:
        return float(self.macro_f1_scores.mean())

    @property
    def macro_f1_std(self) -> float:
        return float(self.macro_f1_scores.std(ddof=1))

    @property
    def accuracy_mean(self) -> float:
        return float(self.accuracy_scores.mean())

    @property
    def accuracy_std(self) -> float:
        return float(self.accuracy_scores.std(ddof=1))

    def summary(self) -> str:
        return (
            f"macro-F1 {self.macro_f1_mean:.3f} ± {self.macro_f1_std:.3f} "
            f"(min={self.macro_f1_scores.min():.3f}, max={self.macro_f1_scores.max():.3f}; "
            f"accuracy {self.accuracy_mean:.3f} ± {self.accuracy_std:.3f}; "
            f"{self.n_splits}-fold × {self.n_repeats} repeats, n={len(self.macro_f1_scores)} folds)"
        )


def repeated_stratified_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> CVResult:
    """Repeated stratified k-fold CV, fitting a fresh model per fold.

    ``model_factory`` must return an *unfitted* model each call — reusing one
    fitted instance across folds would leak vocabulary/coefficients from
    earlier folds into later ones.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    macro_f1_scores = []
    accuracy_scores = []
    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        proba = model.predict_proba(texts.iloc[test_idx])
        predicted = model.classes_[proba.argmax(axis=1)]
        true = labels.iloc[test_idx].to_numpy()

        macro_f1_scores.append(f1_score(true, predicted, average="macro", zero_division=0))
        accuracy_scores.append(accuracy_score(true, predicted))

    return CVResult(
        macro_f1_scores=np.array(macro_f1_scores),
        accuracy_scores=np.array(accuracy_scores),
        n_splits=n_splits,
        n_repeats=n_repeats,
    )


class _DummyModel(ClassifierModel):
    """Most-frequent-class baseline, wrapped as a ClassifierModel for the CV harness."""

    def __init__(self) -> None:
        self._dummy = DummyClassifier(strategy="most_frequent")

    def fit(self, texts: pd.Series, labels: pd.Series) -> "_DummyModel":
        self._dummy.fit(texts, labels)
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        return self._dummy.predict_proba(texts)

    @property
    def classes_(self) -> np.ndarray:
        return self._dummy.classes_


def load_labelled_data(config: dict[str, Any]) -> pd.DataFrame:
    """Ingest and label the training set, as every evaluation needs it."""
    train_dir = REPO_ROOT / config["paths"]["train_dir"]
    labels_csv = REPO_ROOT / config["paths"]["train_labels"]
    train_df = records_to_dataframe(ingest_directory(train_dir))
    labels_df = load_train_labels(labels_csv)
    return attach_labels(train_df, labels_df)


def run_ablation(
    config: dict[str, Any], labelled_df: pd.DataFrame, variants: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Run repeated stratified CV for every registered text variant.

    Returns a DataFrame with one row per variant, sorted by mean macro-F1
    descending — the degradation table PLAN.md §4 asks for.
    """
    variants = FEATURE_VARIANTS if variants is None else variants
    labels = labelled_df["true_category"]
    seed = config["seed"]

    rows = []
    for name, builder in variants.items():
        texts = builder(labelled_df)
        result = repeated_stratified_cv(lambda: TfidfLRModel(config), texts, labels, seed=seed)
        rows.append(
            {
                "variant": name,
                "macro_f1_mean": result.macro_f1_mean,
                "macro_f1_std": result.macro_f1_std,
                "macro_f1_min": result.macro_f1_scores.min(),
                "macro_f1_max": result.macro_f1_scores.max(),
                "accuracy_mean": result.accuracy_mean,
                "accuracy_std": result.accuracy_std,
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def per_class_f1_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Per-class F1 across repeated stratified CV folds — diagnoses *which*
    class drives instability in a headline macro-F1, rather than only
    reporting the aggregate."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    classes = sorted(labels.unique())

    per_class_scores: dict[str, list[float]] = {c: [] for c in classes}
    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        proba = model.predict_proba(texts.iloc[test_idx])
        predicted = model.classes_[proba.argmax(axis=1)]
        true = labels.iloc[test_idx].to_numpy()
        f1s = f1_score(true, predicted, average=None, labels=classes, zero_division=0)
        for cls, f1 in zip(classes, f1s):
            per_class_scores[cls].append(f1)

    rows = [
        {
            "category": cls,
            "f1_mean": float(np.mean(scores)),
            "f1_std": float(np.std(scores, ddof=1)),
            "f1_min": float(np.min(scores)),
        }
        for cls, scores in per_class_scores.items()
    ]
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def run_ablation_per_class(
    config: dict[str, Any], labelled_df: pd.DataFrame, variants: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Per-class F1 across the ablation ladder, long format (one row per
    variant × category).

    Flat macro-F1 on the ablation ladder is contaminated once stripping is
    heavy: `Other` and, at the two most-stripped rungs, Loan Processing
    (both n=6) collapse toward zero and drag the 5-class average down with
    them, independent of whether the four saturated categories are actually
    degrading (PLAN.md §4). This is what :func:`macro_f1_excluding_class`
    is derived from to get the decontaminated figure.
    """
    variants = ABLATION_VARIANTS if variants is None else variants
    labels = labelled_df["true_category"]
    seed = config["seed"]

    frames = []
    for name, builder in variants.items():
        texts = builder(labelled_df)
        per_class = per_class_f1_cv(lambda: TfidfLRModel(config), texts, labels, seed=seed)
        per_class.insert(0, "variant", name)
        frames.append(per_class)
    return pd.concat(frames, ignore_index=True)


def macro_f1_excluding_class(per_class_df: pd.DataFrame, exclude: str = "Other") -> pd.DataFrame:
    """Mean per-class F1 across categories other than ``exclude``, per variant.

    The ablation metric with `Other`'s collapse removed — see PLAN.md §4 for
    why flat macro-F1 is not a safe headline figure once stripping is heavy
    enough to also destabilise Loan Processing (the other n=6 class).
    """
    included = per_class_df[per_class_df["category"] != exclude]
    return (
        included.groupby("variant")["f1_mean"]
        .mean()
        .rename("macro_f1_excl_other")
        .reset_index()
    )


def count_perfect_single_fold_runs(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    n_seeds: int = 50,
    n_splits: int = 5,
) -> int:
    """Across ``n_seeds`` independent single stratified 5-fold splits, count how
    many give a perfect (macro-F1 == 1.0) run.

    A single non-repeated CV run is what a "5-fold stratified CV: macro-F1
    1.000" headline claim would come from. Repeating it across many seeds
    tests whether that headline is typical or a lucky split — the single
    number this project uses to make the case that reporting one CV run
    at n=44 is a methodological trap, not just a hunch.
    """
    perfect = 0
    for seed in range(n_seeds):
        result = repeated_stratified_cv(
            model_factory, texts, labels, seed=seed, n_splits=n_splits, n_repeats=1
        )
        if result.macro_f1_mean == 1.0:
            perfect += 1
    return perfect


def _print_table(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        print(
            f"  {row['variant']:<24} macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} "
            f"(min={row['macro_f1_min']:.3f}, max={row['macro_f1_max']:.3f})  "
            f"accuracy {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}"
        )


def main() -> None:
    config = load_config()
    labelled_df = load_labelled_data(config)
    labels = labelled_df["true_category"]
    seed = config["seed"]

    print("=" * 10, "Baseline: full_text (PLAN.md §2 headline)", "=" * 10)
    from src.features import full_text

    baseline = repeated_stratified_cv(
        lambda: TfidfLRModel(config), full_text(labelled_df), labels, seed=seed
    )
    print(f"  {baseline.summary()}")

    print()
    print("=" * 10, "Per-class F1 on full_text (diagnoses what drives the spread)", "=" * 10)
    per_class = per_class_f1_cv(lambda: TfidfLRModel(config), full_text(labelled_df), labels, seed=seed)
    for _, row in per_class.iterrows():
        print(f"  {row['category']:<22} F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f} (min={row['f1_min']:.3f})")

    print()
    print("=" * 10, "Reproducibility: single 5-fold runs giving a perfect score", "=" * 10)
    n_seeds = 50
    perfect = count_perfect_single_fold_runs(
        lambda: TfidfLRModel(config), full_text(labelled_df), labels, n_seeds=n_seeds
    )
    print(f"  {perfect}/{n_seeds} single stratified 5-fold splits scored macro-F1 == 1.000")

    print()
    print("=" * 10, "Dummy baseline: most-frequent class (PLAN.md §2)", "=" * 10)
    dummy = repeated_stratified_cv(_DummyModel, full_text(labelled_df), labels, seed=seed)
    print(f"  {dummy.summary()}")

    print()
    print("=" * 10, "Ablation: progressive artifact stripping (PLAN.md §4)", "=" * 10)
    from src.features import ABLATION_VARIANTS

    ablation_df = run_ablation(config, labelled_df, ABLATION_VARIANTS)
    _print_table(ablation_df)

    print()
    print(
        "=" * 10,
        "Ablation, Other excluded: is flat macro-F1 above contaminated? (PLAN.md §4)",
        "=" * 10,
    )
    per_class_ablation = run_ablation_per_class(config, labelled_df, ABLATION_VARIANTS)
    excl_other = macro_f1_excluding_class(per_class_ablation)
    for variant in ABLATION_VARIANTS:
        excl_val = excl_other.loc[excl_other["variant"] == variant, "macro_f1_excl_other"].item()
        flat_val = ablation_df.loc[ablation_df["variant"] == variant, "macro_f1_mean"].item()
        print(f"  {variant:<20} flat macro-F1 {flat_val:.3f}   excl. Other {excl_val:.3f}")

    print()
    print("=" * 10, "Perturbations: simulated inbox noise (beyond PLAN.md §4)", "=" * 10)
    from src.features import PERTURBATION_VARIANTS

    perturbation_df = run_ablation(config, labelled_df, PERTURBATION_VARIANTS)
    _print_table(perturbation_df)

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    full_results = pd.concat([ablation_df, perturbation_df], ignore_index=True)
    full_results.to_csv(output_dir / "ablation_results.csv", index=False)
    print(f"\nWrote {output_dir / 'ablation_results.csv'}")

    per_class_ablation.to_csv(output_dir / "ablation_per_class.csv", index=False)
    print(f"Wrote {output_dir / 'ablation_per_class.csv'}")


if __name__ == "__main__":
    main()
