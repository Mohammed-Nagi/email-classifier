"""Tests for src.features: what each ablation variant does and doesn't include."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    FEATURE_VARIANTS,
    core_only,
    full_text,
    generic_salutation,
    greeting_and_core,
    no_title,
    subject_only,
)
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def labelled_df() -> pd.DataFrame:
    config = load_config()
    train_df = records_to_dataframe(ingest_directory(ROOT / config["paths"]["train_dir"]))
    labels_df = load_train_labels(ROOT / config["paths"]["train_labels"])
    return attach_labels(train_df, labels_df)


def test_every_email_has_exactly_three_paragraphs(labelled_df: pd.DataFrame) -> None:
    """The (greeting, core, signature) invariant every stripped variant slices on."""
    assert (labelled_df["body_paragraphs"].apply(len) == 3).all()


def test_no_title_drops_inner_title_keeps_subject_and_paragraphs() -> None:
    # inner_title shares no substring with subject here; in the real corpus it
    # does ("Account Transfer" within "Account Transfer Request"), which would
    # make a naive "not in" assertion pass vacuously.
    df = pd.DataFrame(
        {
            "subject": ["Loan Application"],
            "inner_title": ["Financing Request"],
            "body_paragraphs": [("Dear Loan Officer,", "I need a loan.", "Jane Doe")],
        }
    )
    result = no_title(df).iloc[0]

    assert "Financing Request" not in result
    assert "Loan Application" in result
    assert all(p in result for p in df["body_paragraphs"].iloc[0])


def test_stripped_variants_exclude_exactly_what_they_claim(labelled_df: pd.DataFrame) -> None:
    """core_only and greeting_and_core are the ablation's load-bearing rungs —
    if either silently kept the subject or title, the measured drop is wrong."""
    row = labelled_df.iloc[0]
    core = core_only(labelled_df).iloc[0]
    greeting_core = greeting_and_core(labelled_df).iloc[0]

    assert core == row["body_paragraphs"][1]
    assert greeting_core == f"{row['body_paragraphs'][0]} {row['body_paragraphs'][1]}"
    for result in (core, greeting_core):
        assert row["subject"] not in result
        assert row["inner_title"] not in result
        assert row["body_paragraphs"][2] not in result


def test_subject_only_is_just_the_subject(labelled_df: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(
        subject_only(labelled_df), labelled_df["subject"], check_names=False
    )


def test_generic_salutation_replaces_greeting_and_keeps_the_title(
    labelled_df: pd.DataFrame,
) -> None:
    """It swaps paragraph 0 only — keeping the title is what makes it pair with
    greeting_and_core/core_only to isolate the greeting's contribution."""
    row = labelled_df.iloc[0]
    result = generic_salutation(labelled_df).iloc[0]

    assert "Dear Sir/Madam," in result
    assert row["body_paragraphs"][0] not in result
    assert row["inner_title"] in result
    assert row["subject"] in result
    assert row["body_paragraphs"][1] in result


def test_full_text_is_subject_plus_body(labelled_df: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(
        full_text(labelled_df),
        labelled_df["subject"] + " " + labelled_df["body_text"],
        check_names=False,
    )


def test_every_registered_variant_returns_nonempty_text_per_row(
    labelled_df: pd.DataFrame,
) -> None:
    for name, builder in FEATURE_VARIANTS.items():
        result = builder(labelled_df)
        assert len(result) == len(labelled_df), name
        assert (result.str.strip().str.len() > 0).all(), name
