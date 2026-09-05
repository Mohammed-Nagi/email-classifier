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

    def top_contributing_tokens(self, texts: pd.Series, top_k: int = 5) -> list[list[tuple[str, float]]]:
        """Top-``top_k`` tokens driving each row's predicted-class score.

        The audit trail PLAN.md §6 asks for. Cheap because the model is
        linear over a sparse bag-of-tokens: a token's contribution to the
        predicted class's decision score is exactly
        ``tfidf_value * coef_[predicted_class]``. Only positive
        contributions are returned — a token with a negative or zero
        contribution isn't evidence *for* the predicted class, so it
        wouldn't belong in "why was this routed here."
        """
        vectorizer = self._pipeline.named_steps["tfidf"]
        classifier = self._pipeline.named_steps["clf"]
        feature_names = vectorizer.get_feature_names_out()

        tfidf_matrix = vectorizer.transform(texts)
        predicted_idx = self._pipeline.predict_proba(texts).argmax(axis=1)

        results = []
        for row in range(tfidf_matrix.shape[0]):
            contributions = tfidf_matrix[row].toarray().ravel() * classifier.coef_[predicted_idx[row]]
            top_indices = np.argsort(contributions)[::-1][:top_k]
            results.append(
                [(feature_names[i], float(contributions[i])) for i in top_indices if contributions[i] > 0]
            )
        return results
