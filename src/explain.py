"""Per-prediction explainability: top contributing tokens from LR
coefficients — the audit trail PLAN.md §6 requires for a regulated
deployment ("why was this routed here", not just where).

Uses a separate, uncalibrated ``TfidfLRModel`` fit on all 44 training
examples, not the shipped ``CalibratedTfidfLRModel`` ``run.py`` predicts
with. See NOTES.md / PLAN.md §6 for why: ``CalibratedClassifierCV``'s 3
internal sub-fits (``cv=3``) each train on a different ~29-example subset
with substantially different TF-IDF vocabularies — measured directly: only
135 of 1018 union vocabulary words are shared across all three. Averaging
their coefficients would not approximate one coherent model, it would
misrepresent one. A single fit on all 44 examples is both simpler and more
faithful to what actually decided the label: calibration only rescales the
confidence number, it doesn't change the argmax for 0 of the 12 real test
predictions (PLAN.md §6), so explaining against the uncalibrated fit
explains the same decision the calibrated model made — just not literally
the same fitted object.
"""

from __future__ import annotations

import pandas as pd

from src.models.tfidf_lr import TfidfLRModel


def format_top_features(contributions: list[tuple[str, float]]) -> str:
    """Render a row's ranked ``(token, contribution)`` list as a single
    display string for the ``top_features`` output column, e.g.
    ``"loan (+0.42), officer (+0.31)"``.
    """
    return ", ".join(f"{token} (+{weight:.2f})" for token, weight in contributions)


def explain_predictions(model: TfidfLRModel, texts: pd.Series, top_k: int = 5) -> pd.Series:
    """Top-``top_k`` contributing tokens per row, formatted for output."""
    contributions = model.top_contributing_tokens(texts, top_k=top_k)
    return pd.Series([format_top_features(row) for row in contributions], index=texts.index)
