"""Single-command pipeline entrypoint: data on disk -> outputs/predictions.csv.

    python -m src.run

Trains TF-IDF + Logistic Regression on all 44 labelled training emails and
predicts the 12 unlabelled test emails. Calibration, routing/abstain
thresholds, and per-prediction explanations are later build steps (§6, §9
steps 9-10 in PLAN.md); this version produces the required deliverable —
predicted_category and confidence_score per email — end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe
from src.model import build_pipeline, build_text_input, predict_with_confidence

REPO_ROOT = Path(__file__).resolve().parent.parent

# Required by the brief; source_filename is added per PLAN.md's settled
# email_id decision (§3) so the internal id <-> filename mapping stays
# unambiguous for a reviewer cross-checking against the provided files.
OUTPUT_COLUMNS = ["email_id", "predicted_category", "confidence_score", "source_filename"]


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

    pipeline = build_pipeline(config)
    pipeline.fit(build_text_input(labelled_df), labelled_df["true_category"])

    predicted_category, confidence_score = predict_with_confidence(
        pipeline, build_text_input(test_df)
    )

    predictions = pd.DataFrame(
        {
            "email_id": test_df["email_id"],
            "predicted_category": predicted_category,
            "confidence_score": confidence_score,
            "source_filename": test_df["source_filename"],
        }
    )
    return predictions[OUTPUT_COLUMNS]


def main() -> None:
    config = load_config()
    predictions = build_predictions(config)

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(predictions["predicted_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
