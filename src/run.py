"""Single-command pipeline entrypoint: data on disk -> outputs/predictions.csv.

    python -m src.run
    python -m src.run --auto-route-threshold 0.55

Trains TF-IDF + Logistic Regression on all 44 labelled training emails and
predicts the 12 unlabelled test emails. Uses the sigmoid-calibrated model
(``CalibratedTfidfLRModel``), not the raw pipeline — see PLAN.md §6 and
NOTES.md for the measured before/after: calibration cuts pooled-CV Brier
0.544 -> 0.215 and ECE 0.613 -> 0.359, changes 0 of the 12 actual test
predictions (2.5% flip rate across the full nested-CV comparison, and that
change is within fold-to-fold noise on macro-F1), and is what makes a
routing threshold (§6, step 10) meaningful at all — raw confidences never
left a 0.26-0.49 band.

``top_features`` comes from a *separate*, uncalibrated fit on the same
training data — see ``src/explain.py``'s module docstring for why that
isn't a discrepancy with the shipped calibrated model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.calibrate import CalibratedTfidfLRModel
from src.config import load_config
from src.explain import explain_predictions
from src.features import full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.models.tfidf_lr import TfidfLRModel
from src.routing import margin, needs_review, runner_up_category

REPO_ROOT = Path(__file__).resolve().parent.parent

# Required by the brief (first three, exact names): email_id,
# predicted_category, confidence_score. Additional columns per PLAN.md §6's
# output contract.
OUTPUT_COLUMNS = [
    "email_id",
    "predicted_category",
    "confidence_score",
    "source_filename",
    "margin",
    "runner_up_category",
    "needs_review",
    "top_features",
]


def build_predictions(config: dict[str, Any]) -> pd.DataFrame:
    """Ingest, train on all labelled data, and predict the test set.

    Returns the predictions DataFrame with columns matching OUTPUT_COLUMNS;
    does not write to disk.
    """
    train_dir = REPO_ROOT / config["paths"]["train_dir"]
    test_dir = REPO_ROOT / config["paths"]["test_dir"]
    labels_csv = REPO_ROOT / config["paths"]["train_labels"]

    train_df = records_to_dataframe(ingest_directory(train_dir))
    test_df = records_to_dataframe(ingest_directory(test_dir))
    labels_df = load_train_labels(labels_csv)
    labelled_df = attach_labels(train_df, labels_df)
    train_texts = full_text(labelled_df)
    train_labels = labelled_df["true_category"]
    test_texts = full_text(test_df)

    model = CalibratedTfidfLRModel(config)
    model.fit(train_texts, train_labels)

    # Explanations come from a separate, uncalibrated fit — see
    # src/explain.py for why this isn't the shipped model's own internals.
    explain_model = TfidfLRModel(config)
    explain_model.fit(train_texts, train_labels)

    proba = model.predict_proba(test_texts)
    predicted_category, confidence_score = model.predict_with_confidence(test_texts)
    threshold = config["routing"]["auto_route_threshold"]

    predictions = pd.DataFrame(
        {
            "email_id": test_df["email_id"],
            "predicted_category": predicted_category,
            "confidence_score": confidence_score,
            "source_filename": test_df["source_filename"],
            "margin": margin(proba),
            "runner_up_category": runner_up_category(proba, model.classes_),
            "needs_review": needs_review(confidence_score, threshold),
            "top_features": explain_predictions(explain_model, test_texts).to_numpy(),
        }
    )
    return predictions[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the email classification pipeline end-to-end.")
    parser.add_argument(
        "--auto-route-threshold",
        type=float,
        default=None,
        help="Override config.yaml's routing.auto_route_threshold "
        "(calibrated-confidence cutoff for auto-route vs. needs_review).",
    )
    args = parser.parse_args()

    config = load_config()
    if args.auto_route_threshold is not None:
        config["routing"]["auto_route_threshold"] = args.auto_route_threshold

    predictions = build_predictions(config)

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(predictions["predicted_category"].value_counts().to_string())
    print(f"needs_review: {int(predictions['needs_review'].sum())} of {len(predictions)}")


if __name__ == "__main__":
    main()
