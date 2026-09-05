"""Does calibration help, and does it change what gets predicted?

Run directly to reproduce the README's raw-vs-calibrated Brier/ECE table and
the paired flip-rate check:

    python -m src.calibrate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import load_config
from src.evaluate import ModelFactory, load_labelled_data
from src.features import core_only, full_text
from src.model import CalibratedTfidfLRModel, TfidfLRModel


def multiclass_brier_score(proba: np.ndarray, true_labels: np.ndarray, classes: np.ndarray) -> float:
    """Squared error between predicted probability vectors and one-hot labels,
    summed over classes and averaged over samples. Range [0, 2]."""
    class_index = {c: i for i, c in enumerate(classes)}
    one_hot = np.zeros_like(proba)
    for row, label in enumerate(true_labels):
        one_hot[row, class_index[label]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Bin by confidence, weight each bin's |accuracy - mean confidence| gap by
    its share of samples. Range [0, 1]; 0 is perfectly calibrated."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high) if low > 0 else (confidences <= high)
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / len(confidences)) * abs(
            correct[in_bin].mean() - confidences[in_bin].mean()
        )
    return float(ece)


@dataclass(frozen=True)
class CalibrationResult:
    """Pooled out-of-fold confidence/correctness, plus the metrics from them.

    Pooling matters twice: Brier/ECE on training-fit probabilities would be
    optimistic, and one fold's ~9 held-out rows is too few for ECE's bins to
    mean anything. Pooling every repeat gives 44 × 10 = 440 points.
    """

    confidences: np.ndarray
    correct: np.ndarray
    brier: float
    ece: float

    def summary(self) -> str:
        return (
            f"Brier {self.brier:.3f}, ECE {self.ece:.3f}, mean confidence "
            f"{self.confidences.mean():.3f}, accuracy {self.correct.mean():.3f} "
            f"(n={len(self.confidences)} pooled out-of-fold predictions)"
        )


def calibration_cv(
    model_factory: ModelFactory,
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
    eval_texts: pd.Series | None = None,
) -> CalibrationResult:
    """Repeated stratified CV, pooling out-of-fold confidence and correctness.

    ``eval_texts`` (default: ``texts``) lets the held-out fold be scored on a
    *different* text variant than the model trained on — a domain-shift check:
    train on clean full text, score on stripped text, and see whether
    confidence keeps pace with the accuracy drop.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    eval_texts = texts if eval_texts is None else eval_texts.reset_index(drop=True)

    confidences, correct, briers = [], [], []
    for train_idx, test_idx in splitter.split(texts, labels):
        model = model_factory()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        proba = model.predict_proba(eval_texts.iloc[test_idx])
        true = labels.iloc[test_idx].to_numpy()
        predicted_idx = proba.argmax(axis=1)

        confidences.append(proba[np.arange(len(proba)), predicted_idx])
        correct.append(model.classes_[predicted_idx] == true)
        briers.append(multiclass_brier_score(proba, true, model.classes_))

    pooled_confidences = np.concatenate(confidences)
    pooled_correct = np.concatenate(correct)
    return CalibrationResult(
        confidences=pooled_confidences,
        correct=pooled_correct,
        brier=float(np.mean(briers)),
        ece=expected_calibration_error(pooled_confidences, pooled_correct),
    )


@dataclass(frozen=True)
class PairedCalibrationResult:
    """Raw vs. calibrated on identical folds, both fit in the same iteration.

    Paired by construction, not by trusting that two separate CV runs at the
    same seed drew the same folds. A mean-vs-mean comparison also can't rule
    out cancelling flips, hence the explicit flip count.
    """

    raw_macro_f1: np.ndarray
    calibrated_macro_f1: np.ndarray
    n_flipped: int
    n_predictions: int

    @property
    def macro_f1_delta(self) -> np.ndarray:
        return self.calibrated_macro_f1 - self.raw_macro_f1

    def summary(self) -> str:
        delta = self.macro_f1_delta
        return (
            f"macro-F1 delta (calibrated - raw): {delta.mean():+.3f} ± {delta.std(ddof=1):.3f} "
            f"(min={delta.min():+.3f}, max={delta.max():+.3f}); {self.n_flipped}/"
            f"{self.n_predictions} predictions flipped label "
            f"({self.n_flipped / self.n_predictions:.1%})"
        )


def paired_calibration_comparison(
    config: dict[str, Any],
    texts: pd.Series,
    labels: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> PairedCalibrationResult:
    """Fit raw and calibrated models on identical folds, in the same iteration."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    raw_scores, calibrated_scores, n_flipped, n_predictions = [], [], 0, 0
    for train_idx, test_idx in splitter.split(texts, labels):
        train_texts, train_labels = texts.iloc[train_idx], labels.iloc[train_idx]
        test_texts, true = texts.iloc[test_idx], labels.iloc[test_idx].to_numpy()

        raw = TfidfLRModel(config).fit(train_texts, train_labels)
        calibrated = CalibratedTfidfLRModel(config).fit(train_texts, train_labels)
        raw_pred = raw.classes_[raw.predict_proba(test_texts).argmax(axis=1)]
        calibrated_pred = calibrated.classes_[calibrated.predict_proba(test_texts).argmax(axis=1)]

        raw_scores.append(f1_score(true, raw_pred, average="macro", zero_division=0))
        calibrated_scores.append(
            f1_score(true, calibrated_pred, average="macro", zero_division=0)
        )
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
    texts, labels, seed = full_text(labelled_df), labelled_df["true_category"], config["seed"]

    print("=" * 10, "Raw vs. sigmoid-calibrated TF-IDF+LR", "=" * 10)
    print(f"  raw:        {calibration_cv(lambda: TfidfLRModel(config), texts, labels, seed=seed).summary()}")
    print(
        f"  calibrated: "
        f"{calibration_cv(lambda: CalibratedTfidfLRModel(config), texts, labels, seed=seed).summary()}"
    )

    print("\n" + "=" * 10, "Paired: does calibration change predictions?", "=" * 10)
    print(f"  {paired_calibration_comparison(config, texts, labels, seed=seed).summary()}")

    print("\n" + "=" * 10, "Domain shift: trained on full text, scored on stripped", "=" * 10)
    shifted = calibration_cv(
        lambda: CalibratedTfidfLRModel(config),
        texts,
        labels,
        seed=seed,
        eval_texts=core_only(labelled_df),
    )
    print(f"  calibrated, scored on core_only: {shifted.summary()}")


if __name__ == "__main__":
    main()
