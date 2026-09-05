"""Routing policy: margin, runner-up category, and the auto-route /
needs_review threshold gate (PLAN.md §6).

The gate operates on calibrated confidence (``src/calibrate.py``), which is
a much better relative ranking than the raw model's but still measurably
imperfect — ECE 0.359 after calibration (PLAN.md §6). It is a useful
ordering of "how sure was the model," not a probability precise enough to
run cost arithmetic against (e.g. "expected cost = P(wrong) × $cost" is not
a claim this project makes). The threshold below is chosen empirically from
the coverage/accuracy curve on pooled out-of-fold calibrated predictions,
not derived from a formal cost model, for exactly that reason.

Run directly to reproduce the curve that justified the configured threshold:

    python -m src.routing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibrate import CalibratedTfidfLRModel, calibration_cv
from src.config import load_config
from src.evaluate import REPO_ROOT, load_labelled_data
from src.features import full_text


def margin(proba: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 predicted-class probability, per row.

    "Which two departments is it torn between" (PLAN.md §6) is more
    actionable for a reviewer than the absolute confidence number alone.
    """
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def runner_up_category(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Second-highest-probability class label, per row."""
    order = np.argsort(proba, axis=1)
    return classes[order[:, -2]]


def needs_review(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """True where calibrated confidence falls below the auto-route threshold."""
    return confidence < threshold


def coverage_accuracy_curve(
    confidences: np.ndarray, correct: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """For each candidate threshold: what fraction of predictions would be
    auto-routed (coverage), and the accuracy on each side of the gate.

    This is what a threshold choice should be read off — not a number
    picked by eye against the 12 real test predictions, which would overfit
    the threshold to the one small unlabelled batch it needs to generalise
    beyond.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.85, 0.05), 2)

    rows = []
    for t in thresholds:
        auto_routed = confidences >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(auto_routed.mean()),
                "accuracy_auto_routed": float(correct[auto_routed].mean()) if auto_routed.any() else float("nan"),
                "accuracy_reviewed": float(correct[~auto_routed].mean()) if (~auto_routed).any() else float("nan"),
                "n_reviewed": int((~auto_routed).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    labelled_df = load_labelled_data(config)
    texts = full_text(labelled_df)
    labels = labelled_df["true_category"]
    seed = config["seed"]

    result = calibration_cv(lambda: CalibratedTfidfLRModel(config), texts, labels, seed=seed)
    curve = coverage_accuracy_curve(result.confidences, result.correct)

    print("=" * 10, "Coverage vs. accuracy: choosing the auto-route threshold (PLAN.md §6/§7)", "=" * 10)
    for _, row in curve.iterrows():
        print(
            f"  threshold={row['threshold']:.2f}  coverage={row['coverage']:.3f}  "
            f"acc(auto-routed)={row['accuracy_auto_routed']:.3f}  "
            f"acc(reviewed)={row['accuracy_reviewed']:.3f}  n_reviewed={int(row['n_reviewed'])}"
        )

    configured = config["routing"]["auto_route_threshold"]
    print(f"\n  config.yaml routing.auto_route_threshold = {configured}")

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / "coverage_accuracy.csv", index=False)
    print(f"Wrote {output_dir / 'coverage_accuracy.csv'}")


if __name__ == "__main__":
    main()
