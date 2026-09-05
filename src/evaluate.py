"""Repeated stratified CV, the ablation ladder, and the per-class diagnostics.

The 44 labelled emails are too few to hold out a test split, so every reported
number comes from repeated stratified k-fold CV instead. Run directly to
reproduce the README's headline CV score, per-class breakdown, 0/50
reproducibility result, and ablation table:

    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import load_config
from src.features import FEATURE_VARIANTS, full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import TfidfLRModel

REPO_ROOT = Path(__file__).resolve().parent.parent

ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class CVResult:
    """Every fold's macro-F1, not just the mean — at n=44 the spread matters
    as much as the average."""

    macro_f1_scores: np.ndarray
    n_splits: int
    n_repeats: int

    @property
    def macro_f1_mean(self) -> float:
        return float(self.macro_f1_scores.mean())

    @property
    def macro_f1_std(self) -> float:
        return float(self.macro_f1_scores.std(ddof=1))

    def summary(self) -> str:
        return (
            f"macro-F1 {self.macro_f1_mean:.3f} ± {self.macro_f1_std:.3f} "
            f"(min={self.macro_f1_scores.min():.3f}, max={self.macro_f1_scores.max():.3f}; "
            f"{self.n_splits}-fold × {self.n_repeats} repeats, n={len(self.macro_f1_scores)} folds)"
        )


def _fold_predictions(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int,
    n_repeats: int,
):
    """Yield ``(true, predicted)`` per fold, fitting a fresh model each time.

    ``model_factory`` must return an *unfitted* model — reusing one fitted
    instance would leak vocabulary and coefficients across folds.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        proba = model.predict_proba(texts.iloc[test_idx])
        yield labels.iloc[test_idx].to_numpy(), model.classes_[proba.argmax(axis=1)]


def repeated_stratified_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> CVResult:
    """Repeated stratified k-fold CV, macro-F1 per fold."""
    scores = [
        f1_score(true, predicted, average="macro", zero_division=0)
        for true, predicted in _fold_predictions(
            model_factory, texts, labels, seed, n_splits, n_repeats
        )
    ]
    return CVResult(np.array(scores), n_splits=n_splits, n_repeats=n_repeats)


def count_perfect_single_fold_runs(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    n_seeds: int = 50,
    n_splits: int = 5,
) -> int:
    """How many of ``n_seeds`` single (non-repeated) 5-fold runs score a perfect
    macro-F1 — i.e. how often the "5-fold CV: 1.000" headline reproduces."""
    return sum(
        repeated_stratified_cv(
            model_factory, texts, labels, seed=seed, n_splits=n_splits, n_repeats=1
        ).macro_f1_mean
        == 1.0
        for seed in range(n_seeds)
    )


def mean_nearest_neighbour_similarity(texts: pd.Series) -> float:
    """Mean cosine similarity between each email and its closest *other* email.

    Rules out the obvious challenge to the saturation finding — that the
    categories separate because the corpus near-duplicates itself. A low value
    means the task is keyword-separable, not leaked.
    """
    similarity = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))
    np.fill_diagonal(similarity, 0.0)
    return float(similarity.max(axis=1).mean())


def load_labelled_data(config: dict[str, Any]) -> pd.DataFrame:
    """Ingest and label the training set."""
    train_df = records_to_dataframe(ingest_directory(REPO_ROOT / config["paths"]["train_dir"]))
    labels_df = load_train_labels(REPO_ROOT / config["paths"]["train_labels"])
    return attach_labels(train_df, labels_df)


def run_ablation(
    config: dict[str, Any],
    labelled_df: pd.DataFrame,
    variants: dict[str, Any] | None = None,
    exclude: str = "Other",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One CV pass per text variant, returning ``(summary, per_class)``.

    Macro-F1 is by definition the unweighted mean of per-class F1s, so both
    tables come from the same per-fold per-class scores — no second pass.
    The ``macro_f1_excl_other`` column is why: flat macro-F1 is contaminated
    once stripping is heavy, because `Other` and, at the most-stripped rungs,
    Loan Processing (both n=6) collapse toward zero and drag the 5-class
    average down regardless of what the four saturated categories do.
    """
    variants = FEATURE_VARIANTS if variants is None else variants
    labels = labelled_df["true_category"]
    classes = sorted(labels.unique())
    kept = [i for i, cls in enumerate(classes) if cls != exclude]

    summary_rows, per_class_rows = [], []
    for name, builder in variants.items():
        fold_scores = np.array(
            [
                f1_score(true, predicted, average=None, labels=classes, zero_division=0)
                for true, predicted in _fold_predictions(
                    lambda: TfidfLRModel(config),
                    builder(labelled_df),
                    labels,
                    config["seed"],
                    n_splits=5,
                    n_repeats=10,
                )
            ]
        )
        macro_per_fold = fold_scores.mean(axis=1)
        summary_rows.append(
            {
                "variant": name,
                "macro_f1_mean": float(macro_per_fold.mean()),
                "macro_f1_std": float(macro_per_fold.std(ddof=1)),
                "macro_f1_min": float(macro_per_fold.min()),
                "macro_f1_max": float(macro_per_fold.max()),
                "macro_f1_excl_other": float(fold_scores[:, kept].mean()),
            }
        )
        per_class_rows.extend(
            {
                "variant": name,
                "category": cls,
                "f1_mean": float(fold_scores[:, i].mean()),
                "f1_std": float(fold_scores[:, i].std(ddof=1)),
                "f1_min": float(fold_scores[:, i].min()),
            }
            for i, cls in enumerate(classes)
        )

    summary = (
        pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    )
    return summary, pd.DataFrame(per_class_rows)


def main() -> None:
    config = load_config()
    labelled_df = load_labelled_data(config)
    texts, labels, seed = full_text(labelled_df), labelled_df["true_category"], config["seed"]

    def factory() -> TfidfLRModel:
        return TfidfLRModel(config)

    summary, per_class = run_ablation(config, labelled_df)

    print("=" * 10, "Headline CV: full_text", "=" * 10)
    print(f"  {repeated_stratified_cv(factory, texts, labels, seed=seed).summary()}")

    print("\n" + "=" * 10, "Per-class F1 on full_text (what drives the spread)", "=" * 10)
    full_text_rows = per_class[per_class["variant"] == "full_text"].sort_values(
        "f1_mean", ascending=False
    )
    for _, row in full_text_rows.iterrows():
        print(
            f"  {row['category']:<22} F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f} "
            f"(min={row['f1_min']:.3f})"
        )

    print("\n" + "=" * 10, "Reproducibility of a single 5-fold run", "=" * 10)
    perfect = count_perfect_single_fold_runs(factory, texts, labels, n_seeds=50)
    print(f"  {perfect}/50 single stratified 5-fold splits scored macro-F1 == 1.000")

    print("\n" + "=" * 10, "Is the saturation just near-duplicate leakage?", "=" * 10)
    similarity = mean_nearest_neighbour_similarity(labelled_df["body_text"])
    print(f"  mean nearest-neighbour cosine similarity within train: {similarity:.3f}")

    print("\n" + "=" * 10, "Ablation: progressive artifact stripping", "=" * 10)
    for _, row in summary.iterrows():
        print(
            f"  {row['variant']:<20} macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} "
            f"(min={row['macro_f1_min']:.3f}, max={row['macro_f1_max']:.3f})   "
            f"excl. Other {row['macro_f1_excl_other']:.3f}"
        )

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "ablation_results.csv", index=False)
    per_class.to_csv(output_dir / "ablation_per_class.csv", index=False)
    print(f"\nWrote {output_dir / 'ablation_results.csv'}, {output_dir / 'ablation_per_class.csv'}")


if __name__ == "__main__":
    main()
