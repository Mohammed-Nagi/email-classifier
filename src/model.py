"""TF-IDF + Logistic Regression baseline classifier.

Shipped as the baseline arm on engineering grounds (deterministic, no
model download, millisecond inference, directly auditable coefficients) —
see PLAN.md section 5. A swappable ``models/base.py`` fit/predict_proba
contract is deferred until there is a second arm (embeddings, step 7) to
swap against; building that abstraction for one arm now would be
speculative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the TF-IDF + Logistic Regression pipeline from config.yaml's `model` section."""
    tfidf_cfg = config["model"]["tfidf"]
    lr_cfg = config["model"]["logistic_regression"]
    seed = config["seed"]

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        stop_words=tfidf_cfg["stop_words"],
        min_df=tfidf_cfg["min_df"],
    )
    classifier = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        random_state=seed,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def build_text_input(df: pd.DataFrame) -> pd.Series:
    """Assemble the model's text input: subject + full body (title + paragraphs).

    This is the "subject + full body" variant from PLAN.md's ablation table
    (§4) — the unstripped input. Stripped variants for the ablation harness
    are built separately in features.py (step 6).
    """
    return df["subject"] + " " + df["body_text"]


def predict_with_confidence(pipeline: Pipeline, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Predict categories and confidence scores (the predicted class's probability)."""
    proba = pipeline.predict_proba(texts)
    classes = pipeline.classes_
    pred_idx = proba.argmax(axis=1)
    predicted = classes[pred_idx]
    confidence = proba[np.arange(len(proba)), pred_idx]
    return predicted, confidence
