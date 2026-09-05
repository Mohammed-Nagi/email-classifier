"""Tests for src.routing: margin, runner-up, and the threshold gate."""

from __future__ import annotations

import numpy as np
import pytest

from src.routing import coverage_accuracy_curve, margin, needs_review, runner_up_category


def test_margin_is_top1_minus_top2() -> None:
    assert margin(np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])) == pytest.approx([0.5, 0.01])


def test_runner_up_is_the_second_highest_class() -> None:
    result = runner_up_category(
        np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]), np.array(["A", "B", "C"])
    )
    assert list(result) == ["B", "B"]


def test_confidence_exactly_at_threshold_auto_routes() -> None:
    """The boundary decides whether an email reaches a human; it shouldn't be
    ambiguous. At the threshold counts as meeting it."""
    assert list(needs_review(np.array([0.44, 0.45, 0.46]), threshold=0.45)) == [
        True,
        False,
        False,
    ]


def test_coverage_accuracy_curve_separates_the_two_sides_of_the_gate() -> None:
    confidences = np.array([0.2, 0.5, 0.8, 0.9])
    correct = np.array([False, True, True, True])  # the error is the least confident

    curve = coverage_accuracy_curve(confidences, correct, thresholds=np.array([0.0, 0.3]))

    everything, gated = curve.iloc[0], curve.iloc[1]
    assert everything["coverage"] == pytest.approx(1.0)
    assert everything["accuracy_auto_routed"] == pytest.approx(0.75)
    assert everything["n_reviewed"] == 0
    assert gated["coverage"] == pytest.approx(0.75)
    assert gated["accuracy_auto_routed"] == pytest.approx(1.0)
    assert gated["n_reviewed"] == 1
