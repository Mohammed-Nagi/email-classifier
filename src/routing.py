"""Routing policy: margin, runner-up, and the auto-route / needs_review gate.

The gate runs on calibrated confidence, which is a good *relative ranking* but
not a probability precise enough for cost arithmetic (ECE 0.359 after
calibration). The threshold is therefore read off the empirical
coverage/accuracy curve below rather than derived from an expected-cost model.

    python -m src.routing

reproduces that curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibrate import calibration_cv
from src.config import load_config
from src.evaluate import load_labelled_data
from src.features import full_text
from src.model import CalibratedTfidfLRModel


def margin(proba: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability: which two departments it's torn between."""
    ordered = np.sort(proba, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def runner_up_category(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Second-highest-probability class label, per row."""
    return classes[np.argsort(proba, axis=1)[:, -2]]


def needs_review(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """True where confidence falls below the auto-route threshold."""
    return confidence < threshold


def coverage_accuracy_curve(
    confidences: np.ndarray, correct: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Coverage and accuracy either side of the gate, per candidate threshold.

    What the threshold should be read off — rather than picked by eye against
    the 12 test predictions, which would overfit it to the batch it has to
    generalise beyond.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.85, 0.05), 2)

    rows = []
    for threshold in thresholds:
        auto_routed = confidences >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(auto_routed.mean()),
                "accuracy_auto_routed": float(correct[auto_routed].mean())
                if auto_routed.any()
                else float("nan"),
                "accuracy_reviewed": float(correct[~auto_routed].mean())
                if (~auto_routed).any()
                else float("nan"),
                "n_reviewed": int((~auto_routed).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    labelled_df = load_labelled_data(config)
    result = calibration_cv(
        lambda: CalibratedTfidfLRModel(config),
        full_text(labelled_df),
        labelled_df["true_category"],
        seed=config["seed"],
    )

    print("=" * 10, "Coverage vs. accuracy per candidate threshold", "=" * 10)
    for _, row in coverage_accuracy_curve(result.confidences, result.correct).iterrows():
        print(
            f"  threshold={row['threshold']:.2f}  coverage={row['coverage']:.3f}  "
            f"acc(auto-routed)={row['accuracy_auto_routed']:.3f}  "
            f"acc(reviewed)={row['accuracy_reviewed']:.3f}  n_reviewed={int(row['n_reviewed'])}"
        )
    print(f"\n  config.yaml routing.auto_route_threshold = {config['routing']['auto_route_threshold']}")


if __name__ == "__main__":
    main()
