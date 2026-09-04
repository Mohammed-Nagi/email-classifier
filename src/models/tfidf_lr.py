"""TF-IDF + Logistic Regression: the shipped baseline arm.

Shipped on engineering grounds (deterministic, no model download, millisecond
inference, directly auditable coefficients) — see PLAN.md §5. Implements the
:class:`~src.models.base.ClassifierModel` interface so it is swappable with
other arms behind the same CV, ablation, calibration, and routing code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.base import ClassifierModel


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


class TfidfLRModel(ClassifierModel):
    """TF-IDF + Logistic Regression, built fresh (unfitted) from config."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._pipeline = build_pipeline(config)

    def fit(self, texts: pd.Series, labels: pd.Series) -> "TfidfLRModel":
        self._pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        return self._pipeline.predict_proba(texts)

    @property
    def classes_(self) -> np.ndarray:
        return self._pipeline.classes_
