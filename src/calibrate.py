"""Cross-validated probability calibration for the shipped TF-IDF+LR arm.

PLAN.md §6: raw LR probabilities on 44 samples are overconfident, and a
routing/abstain threshold (§6, step 10) is only as meaningful as the
probability scale it operates on. This module wraps the shipped pipeline in
sigmoid (Platt) calibration, measures whether it actually helps (Brier/ECE,
pooled out-of-fold so the metric isn't computed on training-fit
probabilities), and checks — with a fold-paired comparison, not a comparison
of two means — whether calibration changes what gets predicted, not just how
confident the model claims to be.

Run directly to reproduce the before/after numbers:

    python -m src.calibrate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import load_config
from src.evaluate import REPO_ROOT, ModelFactory, load_labelled_data
from src.features import full_text
from src.models.base import ClassifierModel
from src.models.tfidf_lr import TfidfLRModel, build_pipeline


class CalibratedTfidfLRModel(ClassifierModel):
    """TF-IDF + LR with cross-validated sigmoid (Platt) calibration on top.

    Wraps the same pipeline ``TfidfLRModel`` builds (``build_pipeline``) in
    ``CalibratedClassifierCV``, which internally cross-validates on whatever
    training fold it's given so the calibration curve isn't fit on the same
    examples the base classifier was fit on.

    ``cv=3``, not 5: this model is evaluated inside the same 5-fold outer CV
    harness the raw model is (``evaluate.py``), which already strips ~20% of
    the 44 examples into its own held-out test fold. That leaves as few as
    4-5 examples of the smallest class (Other/Loan Processing, n=6 each) in
    the *training* fold ``CalibratedClassifierCV`` then splits again
    internally. ``cv=5`` there needs at least 5 examples of every class in
    that inner split and fails outright on some outer folds
    ("n_splits=5 cannot be greater than the number of members in each
    class."); ``cv=3`` is the largest inner split that survives every outer
    fold this model is actually evaluated with.
    """

    def __init__(self, config: dict[str, Any], cv: int = 3) -> None:
        base_pipeline = build_pipeline(config)
        self._model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=cv)

    def fit(self, texts: pd.Series, labels: pd.Series) -> "CalibratedTfidfLRModel":
        self._model.fit(texts, labels)
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        return self._model.predict_proba(texts)

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_


def multiclass_brier_score(proba: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> float:
    """Mean squared error between predicted probability vectors and one-hot
    true labels, averaged over samples (summed over classes per sample —
    the standard multiclass generalisation of the binary Brier score).
    Range [0, 2]: 0 is a perfect, fully-confident correct prediction; 2 is a
    fully-confident *wrong* prediction.
    """
    class_index = {c: i for i, c in enumerate(classes)}
    one_hot = np.zeros_like(proba)
    for row, label in enumerate(true_labels):
        one_hot[row, class_index[label]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _confidence_bins(
    confidences: np.ndarray, n_bins: int
) -> list[tuple[float, float, np.ndarray]]:
    """Fixed-width bins over [0, 1], each with a boolean mask into ``confidences``.

    Shared by ``expected_calibration_error`` and ``reliability_curve_data`` so
    the two can't silently disagree on bin edges.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        bins.append((float(lo), float(hi), mask))
    return bins


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: bin predictions by confidence, weight each bin's
    |accuracy - mean confidence| gap by the bin's share of samples. Range
    [0, 1]; 0 is perfectly calibrated.
    """
    n = len(confidences)
    ece = 0.0
    for _lo, _hi, mask in _confidence_bins(confidences, n_bins):
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_curve_data(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin (mean confidence, empirical accuracy, count) — the data a
    reliability diagram plots. Empty bins are dropped, not zero-filled.
    """
    rows = []
    for lo, hi, mask in _confidence_bins(confidences, n_bins):
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "mean_confidence": float(confidences[mask].mean()),
                "empirical_accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CalibrationResult:
    """Pooled out-of-fold confidence/correctness from a repeated stratified
    CV run, plus the Brier/ECE derived from them.

    "Pooled out-of-fold" matters twice over: computing Brier/ECE on
    training-fit probabilities would be badly optimistic (the model has seen
    those labels), and a single fold's ~9 held-out examples is too few for
    ECE's 10 bins to mean anything — pooling every repeat's held-out
    predictions (44 examples × 10 repeats = 440 points) gives each bin
    enough to be more than noise.
    """

    confidences: np.ndarray
    correct: np.ndarray
    brier: float
    ece: float

    def summary(self) -> str:
        return f"Brier {self.brier:.3f}, ECE {self.ece:.3f} (n={len(self.confidences)} pooled out-of-fold predictions)"


def calibration_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_bins: int = 10,
) -> CalibrationResult:
    """Repeated stratified CV, pooling out-of-fold confidence/correctness
    and per-fold Brier score across every fold — the calibration analogue of
    ``evaluate.repeated_stratified_cv``.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    all_confidences = []
    all_correct = []
    all_brier = []
    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        proba = model.predict_proba(texts.iloc[test_idx])
        classes = model.classes_
        pred_idx = proba.argmax(axis=1)
        predicted = classes[pred_idx]
        confidence = proba[np.arange(len(proba)), pred_idx]
        true = labels.iloc[test_idx].to_numpy()

        all_confidences.append(confidence)
        all_correct.append(predicted == true)
        all_brier.append(multiclass_brier_score(proba, true, classes))

    confidences = np.concatenate(all_confidences)
    correct = np.concatenate(all_correct)
    return CalibrationResult(
        confidences=confidences,
        correct=correct,
        brier=float(np.mean(all_brier)),
        ece=expected_calibration_error(confidences, correct, n_bins=n_bins),
    )


@dataclass(frozen=True)
class PairedCalibrationResult:
    """Fold-by-fold paired comparison of raw vs. calibrated TF-IDF+LR: same
    fold, same train/test membership, both models fit and scored inside the
    same loop iteration.

    Answers "does calibration change *predictions*, not just claimed
    confidence" with the fold-level macro-F1 deltas and an explicit count of
    individual predictions that flipped label — a mean-vs-mean comparison
    can't rule out cancelling flips (some examples flip correct→wrong, others
    wrong→correct, means look unchanged), and per-class sigmoid
    calibration + renormalisation can reorder the argmax even though it
    doesn't touch the base classifier's decision function directly.
    """

    raw_macro_f1: np.ndarray
    calibrated_macro_f1: np.ndarray
    n_flipped: int
    n_predictions: int

    @property
    def macro_f1_delta(self) -> np.ndarray:
        """Per-fold (calibrated - raw) macro-F1. A valid paired difference:
        every fold has identical train/test membership for both models,
        constructed from the same ``splitter.split()`` call, not merely the
        same seed passed to two separate CV runs."""
        return self.calibrated_macro_f1 - self.raw_macro_f1

    @property
    def flip_rate(self) -> float:
        return self.n_flipped / self.n_predictions

    def summary(self) -> str:
        delta = self.macro_f1_delta
        return (
            f"macro-F1 delta (calibrated - raw): {delta.mean():+.3f} ± {delta.std(ddof=1):.3f} "
            f"(min={delta.min():+.3f}, max={delta.max():+.3f}); "
            f"{self.n_flipped}/{self.n_predictions} predictions flipped label ({self.flip_rate:.1%})"
        )


def paired_calibration_comparison(
    config: dict[str, Any],
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> PairedCalibrationResult:
    """Fit raw and calibrated TF-IDF+LR on identical folds, in the same loop
    iteration, so the comparison is paired by construction rather than by
    the "same seed ⇒ same folds" property NOTES.md documents for two
    independent CV calls.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    raw_scores = []
    calibrated_scores = []
    n_flipped = 0
    n_predictions = 0
    for train_idx, test_idx in splitter.split(texts, labels):
        train_texts, train_labels = texts.iloc[train_idx], labels.iloc[train_idx]
        test_texts, true = texts.iloc[test_idx], labels.iloc[test_idx].to_numpy()

        raw_model = TfidfLRModel(config).fit(train_texts, train_labels)
        calibrated_model = CalibratedTfidfLRModel(config).fit(train_texts, train_labels)

        raw_pred = raw_model.classes_[raw_model.predict_proba(test_texts).argmax(axis=1)]
        calibrated_pred = calibrated_model.classes_[
            calibrated_model.predict_proba(test_texts).argmax(axis=1)
        ]

        raw_scores.append(f1_score(true, raw_pred, average="macro", zero_division=0))
        calibrated_scores.append(f1_score(true, calibrated_pred, average="macro", zero_division=0))
        n_flipped += int((raw_pred != calibrated_pred).sum())
        n_predictions += len(test_idx)

    return PairedCalibrationResult(
        raw_macro_f1=np.array(raw_scores),
        calibrated_macro_f1=np.array(calibrated_scores),
        n_flipped=n_flipped,
        n_predictions=n_predictions,
    )


def main() -> None:
    config = load_config()
    labelled_df = load_labelled_data(config)
    texts = full_text(labelled_df)
    labels = labelled_df["true_category"]
    seed = config["seed"]

    print("=" * 10, "Calibration: raw vs. sigmoid-calibrated TF-IDF+LR (PLAN.md §6)", "=" * 10)
    raw_result = calibration_cv(lambda: TfidfLRModel(config), texts, labels, seed=seed)
    calibrated_result = calibration_cv(lambda: CalibratedTfidfLRModel(config), texts, labels, seed=seed)
    print(f"  raw:        {raw_result.summary()}")
    print(f"  calibrated: {calibrated_result.summary()}")

    print()
    print("=" * 10, "Paired check: does calibration change predictions, not just confidence?", "=" * 10)
    paired = paired_calibration_comparison(config, texts, labels, seed=seed)
    print(f"  {paired.summary()}")

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_curve = reliability_curve_data(raw_result.confidences, raw_result.correct)
    raw_curve.insert(0, "condition", "raw")
    calibrated_curve = reliability_curve_data(calibrated_result.confidences, calibrated_result.correct)
    calibrated_curve.insert(0, "condition", "calibrated")
    reliability_df = pd.concat([raw_curve, calibrated_curve], ignore_index=True)
    reliability_df.to_csv(output_dir / "calibration_reliability.csv", index=False)
    print(f"\nWrote {output_dir / 'calibration_reliability.csv'}")


if __name__ == "__main__":
    main()
