"""Shared model interface: one fit/predict_proba contract for every arm.

This is the architectural choice PLAN.md (§8) calls out as central: the CV
harness (``evaluate.py``), the ablation harness, calibration, and routing are
all written once against :class:`ClassifierModel` — an arm (TF-IDF+LR,
embeddings+LR, zero-shot NLI) is a drop-in as long as it implements ``fit``
and ``predict_proba``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ClassifierModel(ABC):
    """A text classifier that produces class probabilities, not just labels.

    Probabilities (not bare predictions) are required throughout this
    project: the deliverable needs a confidence score, routing needs a
    margin, and calibration needs a probability to calibrate.
    """

    @abstractmethod
    def fit(self, texts: pd.Series, labels: pd.Series) -> "ClassifierModel":
        """Fit on training texts and their category labels. Returns self."""

    @abstractmethod
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Class probabilities, shape (n_samples, n_classes), columns ordered as ``classes_``."""

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Class labels in the column order used by ``predict_proba``."""

    def predict_with_confidence(self, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Predict categories and confidence scores (the predicted class's probability).

        Shared across every arm since it is derived entirely from
        ``predict_proba`` and ``classes_`` — no arm needs to reimplement it.
        """
        proba = self.predict_proba(texts)
        classes = self.classes_
        pred_idx = proba.argmax(axis=1)
        predicted = classes[pred_idx]
        confidence = proba[np.arange(len(proba)), pred_idx]
        return predicted, confidence
