"""Tests for src.routing: margin, runner-up, the threshold gate, and the
coverage/accuracy curve used to justify it."""

from __future__ import annotations

import numpy as np
import pytest

from src.routing import coverage_accuracy_curve, margin, needs_review, runner_up_category


def test_margin_is_top1_minus_top2() -> None:
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.34, 0.33, 0.33],
        ]
    )

    result = margin(proba)

    assert result == pytest.approx([0.5, 0.01])


def test_runner_up_category_is_second_highest_probability() -> None:
    classes = np.array(["A", "B", "C"])
    proba = np.array(
        [
            [0.7, 0.2, 0.1],  # A wins, B is runner-up
            [0.1, 0.2, 0.7],  # C wins, B is runner-up
        ]
    )

    result = runner_up_category(proba, classes)

    assert list(result) == ["B", "B"]


def test_needs_review_boundary_is_exclusive_at_threshold() -> None:
    confidence = np.array([0.44, 0.45, 0.46])

    result = needs_review(confidence, threshold=0.45)

    # Exactly at threshold counts as meeting it (auto-route), not below it.
    assert list(result) == [True, False, False]


def test_coverage_accuracy_curve_reports_full_coverage_at_zero_threshold() -> None:
    confidences = np.array([0.2, 0.5, 0.8, 0.9])
    correct = np.array([False, True, True, True])

    curve = coverage_accuracy_curve(confidences, correct, thresholds=np.array([0.0]))

    row = curve.iloc[0]
    assert row["coverage"] == pytest.approx(1.0)
    assert row["accuracy_auto_routed"] == pytest.approx(0.75)
    assert row["n_reviewed"] == 0
    assert np.isnan(row["accuracy_reviewed"])


def test_coverage_accuracy_curve_concentrates_errors_above_a_separating_threshold() -> None:
    # The one wrong prediction is also the lowest-confidence one.
    confidences = np.array([0.2, 0.5, 0.8, 0.9])
    correct = np.array([False, True, True, True])

    curve = coverage_accuracy_curve(confidences, correct, thresholds=np.array([0.3]))

    row = curve.iloc[0]
    assert row["coverage"] == pytest.approx(0.75)
    assert row["accuracy_auto_routed"] == pytest.approx(1.0)
    assert row["accuracy_reviewed"] == pytest.approx(0.0)
    assert row["n_reviewed"] == 1
