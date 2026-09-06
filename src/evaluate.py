"""Repeated stratified CV, the ablation ladder, and the per-class diagnostics.

The 44 labelled emails are too few to hold out a test split, so every reported
number comes from repeated stratified k-fold CV instead. Run directly to
reproduce the README's headline CV score, per-class breakdown, 0/50
reproducibility result, and ablation table:

    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import load_config
from src.features import FEATURE_VARIANTS, full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import AbstainOtherModel, TfidfLRModel, abstain_predict

REPO_ROOT = Path(__file__).resolve().parent.parent

ModelFactory = Callable[[], Any]
PredictFn = Callable[[Any, pd.Series], np.ndarray]

DEFAULT_ABSTAIN_THRESHOLDS: np.ndarray = np.round(np.arange(0.30, 0.80, 0.05), 2)


def _argmax_predict(model: Any, test_texts: pd.Series) -> np.ndarray:
    proba = model.predict_proba(test_texts)
    return model.classes_[proba.argmax(axis=1)]


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
    predict_fn: PredictFn = _argmax_predict,
):
    """Yield ``(true, predicted)`` per fold, fitting a fresh model each time.

    ``model_factory`` must return an *unfitted* model — reusing one fitted
    instance would leak vocabulary and coefficients across folds. ``predict_fn``
    defaults to plain argmax over ``model.classes_``; abstain-as-Other passes a
    threshold-aware one instead so the same harness scores both models.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        yield labels.iloc[test_idx].to_numpy(), predict_fn(model, texts.iloc[test_idx])


def repeated_stratified_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> CVResult:
    """Repeated stratified k-fold CV, macro-F1 per fold.

    Always plain argmax — this is the flat model's own headline number, never
    scored against abstain-as-Other's threshold-aware labels. That comparison
    runs through ``run_ablation`` and ``sweep_abstain_thresholds`` instead,
    which is where ``_fold_predictions``'s ``predict_fn`` seam is load-bearing.
    """
    scores = [
        f1_score(true, predicted, average="macro", zero_division=0)
        for true, predicted in _fold_predictions(model_factory, texts, labels, seed, n_splits, n_repeats)
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
    model_factory: ModelFactory | None = None,
    predict_fn: PredictFn = _argmax_predict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One CV pass per text variant, returning ``(summary, per_class)``.

    Macro-F1 is by definition the unweighted mean of per-class F1s, so both
    tables come from the same per-fold per-class scores — no second pass.
    The ``macro_f1_excl_other`` column is why: flat macro-F1 is contaminated
    once stripping is heavy, because `Other` and, at the most-stripped rungs,
    Loan Processing (both n=6) collapse toward zero and drag the 5-class
    average down regardless of what the four saturated categories do.

    ``model_factory``/``predict_fn`` default to the plain flat model; passing
    :class:`~src.model.AbstainOtherModel` and :func:`~src.model.abstain_predict`
    runs the identical ladder for abstain-as-Other, on the same folds/seed.
    """
    variants = FEATURE_VARIANTS if variants is None else variants
    labels = labelled_df["true_category"]
    classes = sorted(labels.unique())
    kept = [i for i, cls in enumerate(classes) if cls != exclude]
    factory = (lambda: TfidfLRModel(config)) if model_factory is None else model_factory

    summary_rows, per_class_rows = [], []
    for name, builder in variants.items():
        fold_scores = np.array(
            [
                f1_score(true, predicted, average=None, labels=classes, zero_division=0)
                for true, predicted in _fold_predictions(
                    factory,
                    builder(labelled_df),
                    labels,
                    config["seed"],
                    n_splits=5,
                    n_repeats=10,
                    predict_fn=predict_fn,
                )
            ]
        )
        macro_per_fold = fold_scores.mean(axis=1)
        macro_excl_other_per_fold = fold_scores[:, kept].mean(axis=1)
        summary_rows.append(
            {
                "variant": name,
                "macro_f1_mean": float(macro_per_fold.mean()),
                "macro_f1_std": float(macro_per_fold.std(ddof=1)),
                "macro_f1_min": float(macro_per_fold.min()),
                "macro_f1_max": float(macro_per_fold.max()),
                "macro_f1_excl_other": float(macro_excl_other_per_fold.mean()),
                "macro_f1_excl_other_std": float(macro_excl_other_per_fold.std(ddof=1)),
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


def pooled_out_of_fold_predictions(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted labels concatenated across every CV fold.

    Same 5-fold × 10-repeat pooling as the calibration curve (440 points from
    44 examples) — here to build a confusion matrix instead of a scalar.
    """
    trues, predicteds = [], []
    for true, predicted in _fold_predictions(model_factory, texts, labels, seed, n_splits, n_repeats):
        trues.append(true)
        predicteds.append(predicted)
    return np.concatenate(trues), np.concatenate(predicteds)


def build_confusion_matrix(
    config: dict[str, Any], labelled_df: pd.DataFrame
) -> pd.DataFrame:
    """Pooled out-of-fold confusion matrix, rows=true, columns=predicted."""
    labels = labelled_df["true_category"]
    classes = sorted(labels.unique())
    true, predicted = pooled_out_of_fold_predictions(
        lambda: TfidfLRModel(config), full_text(labelled_df), labels, config["seed"]
    )
    matrix = confusion_matrix(true, predicted, labels=classes)
    result = pd.DataFrame(matrix, index=classes, columns=classes)
    result.index.name = "true_category"
    return result


def sweep_abstain_thresholds(
    config: dict[str, Any],
    labelled_df: pd.DataFrame,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-threshold Other-F1 and flat macro-F1 for abstain-as-Other on
    ``full_text``, same folds/seed as the flat model.

    This is what the abstain threshold should be read off — the same
    empirical-curve discipline as ``routing.auto_route_threshold`` — rather
    than reused from the routing config, which answers a different question
    (see README: abstain decides the bucket, routing decides review).
    """
    thresholds = DEFAULT_ABSTAIN_THRESHOLDS if thresholds is None else thresholds
    labels = labelled_df["true_category"]
    classes = sorted(labels.unique())
    other_idx = classes.index("Other")
    texts = full_text(labelled_df)

    rows = []
    for threshold in thresholds:
        fold_scores = np.array(
            [
                f1_score(true, predicted, average=None, labels=classes, zero_division=0)
                for true, predicted in _fold_predictions(
                    lambda: AbstainOtherModel(config),
                    texts,
                    labels,
                    config["seed"],
                    n_splits=5,
                    n_repeats=10,
                    predict_fn=partial(abstain_predict, threshold=threshold),
                )
            ]
        )
        macro_per_fold = fold_scores.mean(axis=1)
        rows.append(
            {
                "threshold": float(threshold),
                "other_f1_mean": float(fold_scores[:, other_idx].mean()),
                "other_f1_std": float(fold_scores[:, other_idx].std(ddof=1)),
                "macro_f1_mean": float(macro_per_fold.mean()),
                "macro_f1_std": float(macro_per_fold.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


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
    full_text_summary = summary[summary["variant"] == "full_text"].iloc[0]
    print(
        f"  {'All 4, Other excluded':<22} macro-F1 {full_text_summary['macro_f1_excl_other']:.3f} "
        f"± {full_text_summary['macro_f1_excl_other_std']:.3f}"
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

    print("\n" + "=" * 10, "Confusion matrix: pooled out-of-fold predictions (rows=true)", "=" * 10)
    confusion = build_confusion_matrix(config, labelled_df)
    print(confusion.to_string())

    print("\n" + "=" * 10, "Abstain-as-Other: threshold sweep on full_text", "=" * 10)
    sweep = sweep_abstain_thresholds(config, labelled_df)
    for _, row in sweep.iterrows():
        print(
            f"  threshold={row['threshold']:.2f}  "
            f"Other F1 {row['other_f1_mean']:.3f} ± {row['other_f1_std']:.3f}   "
            f"flat macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}"
        )
    best_swept = float(sweep.loc[sweep["other_f1_mean"].idxmax(), "threshold"])
    print(f"  best threshold by Other F1 in this sweep: {best_swept:.2f}")
    chosen_threshold = float(config["abstain"]["threshold"])
    print(f"  config.yaml abstain.threshold = {chosen_threshold:.2f} (used below)")

    print(
        "\n" + "=" * 10,
        f"Abstain-as-Other @ threshold={chosen_threshold:.2f}: same ablation ladder",
        "=" * 10,
    )
    abstain_summary, abstain_per_class = run_ablation(
        config,
        labelled_df,
        model_factory=lambda: AbstainOtherModel(config),
        predict_fn=partial(abstain_predict, threshold=chosen_threshold),
    )
    for _, row in abstain_summary.iterrows():
        print(
            f"  {row['variant']:<20} macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} "
            f"(min={row['macro_f1_min']:.3f}, max={row['macro_f1_max']:.3f})   "
            f"excl. Other {row['macro_f1_excl_other']:.3f}"
        )

    print("\n" + "=" * 10, "Abstain-as-Other: per-class F1 on full_text", "=" * 10)
    abstain_full_text_rows = abstain_per_class[
        abstain_per_class["variant"] == "full_text"
    ].sort_values("f1_mean", ascending=False)
    for _, row in abstain_full_text_rows.iterrows():
        print(
            f"  {row['category']:<22} F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f} "
            f"(min={row['f1_min']:.3f})"
        )

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "ablation_results.csv", index=False)
    per_class.to_csv(output_dir / "ablation_per_class.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    sweep.to_csv(output_dir / "abstain_threshold_sweep.csv", index=False)
    abstain_summary.to_csv(output_dir / "abstain_ablation_results.csv", index=False)
    abstain_per_class.to_csv(output_dir / "abstain_ablation_per_class.csv", index=False)
    print(
        f"\nWrote {output_dir / 'ablation_results.csv'}, {output_dir / 'ablation_per_class.csv'}, "
        f"{output_dir / 'confusion_matrix.csv'}, {output_dir / 'abstain_threshold_sweep.csv'}, "
        f"{output_dir / 'abstain_ablation_results.csv'}, {output_dir / 'abstain_ablation_per_class.csv'}"
    )


if __name__ == "__main__":
    main()
