"""The shipped model: TF-IDF + Logistic Regression, optionally calibrated.

Every arm-shaped consumer in this project (CV harness, ablation, calibration,
routing) takes a ``model_factory: () -> model`` callable and needs only
``fit``/``predict_proba``/``classes_`` from what it returns — a plain callable
seam, no base class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """TF-IDF + Logistic Regression, per config.yaml's `model` section."""
    tfidf_cfg = config["model"]["tfidf"]
    lr_cfg = config["model"]["logistic_regression"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=tfidf_cfg["max_features"],
                    ngram_range=tuple(tfidf_cfg["ngram_range"]),
                    stop_words=tfidf_cfg["stop_words"],
                    min_df=tfidf_cfg["min_df"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=lr_cfg["C"],
                    max_iter=lr_cfg["max_iter"],
                    class_weight=lr_cfg["class_weight"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


class TfidfLRModel:
    """Uncalibrated TF-IDF + LR. Shipped for explanations; the CV baseline."""

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

    def top_contributing_tokens(
        self, texts: pd.Series, top_k: int = 5
    ) -> list[list[tuple[str, float]]]:
        """Top-``top_k`` tokens driving each row's predicted-class score.

        A token's contribution is exactly ``tfidf_value * coef_[predicted]``,
        the model being linear over a sparse bag of tokens. Only positive
        contributions are returned — a token that argues against the predicted
        class doesn't belong in "why was this routed here." Ties are broken on
        the token string so the output is stable across platforms rather than
        depending on sort order among equal floats.
        """
        vectorizer = self._pipeline.named_steps["tfidf"]
        classifier = self._pipeline.named_steps["clf"]
        names = vectorizer.get_feature_names_out()

        matrix = vectorizer.transform(texts)
        predicted = self._pipeline.predict_proba(texts).argmax(axis=1)

        rows = []
        for row in range(matrix.shape[0]):
            contributions = matrix[row].toarray().ravel() * classifier.coef_[predicted[row]]
            ranked = sorted(
                range(len(contributions)), key=lambda i: (-contributions[i], names[i])
            )
            rows.append(
                [(names[i], float(contributions[i])) for i in ranked[:top_k] if contributions[i] > 0]
            )
        return rows


class CalibratedTfidfLRModel:
    """TF-IDF + LR with cross-validated sigmoid (Platt) calibration on top.

    ``cv=3``, not 5: this model is evaluated inside a 5-fold outer CV harness
    that already holds out ~20% of the 44 examples, leaving as few as 4-5 of
    the smallest class (n=6) in the training fold that ``CalibratedClassifierCV``
    splits again internally. ``cv=5`` needs 5 per class there and raises on some
    outer folds; 3 is the largest split that survives every fold.
    """

    def __init__(self, config: dict[str, Any], cv: int = 3) -> None:
        self._model = CalibratedClassifierCV(
            estimator=build_pipeline(config), method="sigmoid", cv=cv
        )

    def fit(self, texts: pd.Series, labels: pd.Series) -> "CalibratedTfidfLRModel":
        self._model.fit(texts, labels)
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        return self._model.predict_proba(texts)

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_


def predict_with_confidence(model: Any, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Predicted category and its probability, for any model in this module."""
    proba = model.predict_proba(texts)
    predicted_idx = proba.argmax(axis=1)
    return model.classes_[predicted_idx], proba[np.arange(len(proba)), predicted_idx]


def explain_predictions(model: TfidfLRModel, texts: pd.Series, top_k: int = 5) -> list[str]:
    """Per-row ``top_features`` strings, e.g. ``"loan (+0.42), officer (+0.31)"``.

    Uses an uncalibrated fit rather than the shipped calibrated model's
    internals: ``CalibratedClassifierCV``'s 3 sub-fits each train on a
    different ~29-example slice with its own TF-IDF vocabulary (measured: only
    135 of 1018 union words shared by all three), so averaging their
    coefficients would misrepresent the model rather than approximate it. Safe
    because calibration changes the argmax for 0 of the 12 test predictions.
    """
    return [
        ", ".join(f"{token} (+{weight:.2f})" for token, weight in row)
        for row in model.top_contributing_tokens(texts, top_k=top_k)
    ]
