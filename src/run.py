"""Single-command entrypoint: data on disk -> outputs/predictions.csv.

    python -m src.run
    python -m src.run --auto-route-threshold 0.55

Trains on all 44 labelled emails and predicts the 12 unlabelled ones. Ships the
sigmoid-calibrated model; ``top_features`` comes from a separate uncalibrated
fit (see ``src/model.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import full_text
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import (
    CalibratedTfidfLRModel,
    TfidfLRModel,
    explain_predictions,
    predict_with_confidence,
)
from src.routing import margin, needs_review, runner_up_category

REPO_ROOT = Path(__file__).resolve().parent.parent

PROBABILITY_DECIMALS = 6

# First three are the brief's required columns, in the required order.
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
    """Ingest, train on all labelled data, predict the test set."""
    train_df = records_to_dataframe(ingest_directory(REPO_ROOT / config["paths"]["train_dir"]))
    test_df = records_to_dataframe(ingest_directory(REPO_ROOT / config["paths"]["test_dir"]))
    labels_df = load_train_labels(REPO_ROOT / config["paths"]["train_labels"])
    labelled_df = attach_labels(train_df, labels_df)

    train_texts, train_labels = full_text(labelled_df), labelled_df["true_category"]
    test_texts = full_text(test_df)

    model = CalibratedTfidfLRModel(config).fit(train_texts, train_labels)
    explain_model = TfidfLRModel(config).fit(train_texts, train_labels)

    proba = model.predict_proba(test_texts)
    predicted_category, confidence_score = predict_with_confidence(model, test_texts)

    # Round before deriving needs_review, so the flag is verifiable from the
    # confidence printed beside it. BLAS reassociation makes these differ in
    # the last couple of ULPs across platforms (~3e-16); at 6dp the CSV is
    # byte-identical anywhere, which a reproducibility claim needs and no
    # routing decision is close enough to notice.
    confidence_score = np.round(confidence_score, PROBABILITY_DECIMALS)

    predictions = pd.DataFrame(
        {
            "email_id": test_df["email_id"],
            "predicted_category": predicted_category,
            "confidence_score": confidence_score,
            "source_filename": test_df["source_filename"],
            "margin": np.round(margin(proba), PROBABILITY_DECIMALS),
            "runner_up_category": runner_up_category(proba, model.classes_),
            "needs_review": needs_review(
                confidence_score, config["routing"]["auto_route_threshold"]
            ),
            "top_features": explain_predictions(explain_model, test_texts),
        }
    )
    return predictions[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify RedRock emails end to end.")
    parser.add_argument(
        "--auto-route-threshold",
        type=float,
        help="Override config.yaml's routing.auto_route_threshold.",
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
