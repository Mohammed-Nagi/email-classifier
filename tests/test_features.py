"""Tests for src.features: text-variant builders for the ablation harness."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    FEATURE_VARIANTS,
    core_only,
    distractor_text,
    full_text,
    generic_salutation,
    greeting_and_core,
    no_subject,
    no_title,
    subject_only,
    synonym_substitution,
    truncated_first_sentence,
    typo_noise,
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
    """core_only's slicing assumes (greeting, core, signature) — verify it holds."""
    assert (labelled_df["body_paragraphs"].apply(len) == 3).all()


def test_full_text_matches_old_build_text_input(labelled_df: pd.DataFrame) -> None:
    result = full_text(labelled_df)
    expected = labelled_df["subject"] + " " + labelled_df["body_text"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_no_title_drops_inner_title_keeps_subject_and_paragraphs() -> None:
    # inner_title deliberately shares no substring with subject here — in the
    # real corpus inner_title ("Account Transfer") is a substring of subject
    # ("Account Transfer Request"), which would make a naive "not in" check
    # pass vacuously.
    df = pd.DataFrame(
        {
            "subject": ["Loan Application"],
            "inner_title": ["Financing Request"],
            "body_text": ["Financing Request Dear Loan Officer, I need a loan. Jane Doe"],
            "body_paragraphs": [("Dear Loan Officer,", "I need a loan.", "Jane Doe")],
        }
    )
    result = no_title(df).iloc[0]

    assert "Financing Request" not in result
    assert "Loan Application" in result
    for paragraph in df["body_paragraphs"].iloc[0]:
        assert paragraph in result


def test_core_only_excludes_subject_title_greeting_and_signature(labelled_df: pd.DataFrame) -> None:
    row = labelled_df.iloc[0]
    result = core_only(labelled_df).iloc[0]

    assert result == row["body_paragraphs"][1]
    assert row["subject"] not in result
    assert row["inner_title"] not in result
    assert row["body_paragraphs"][0] not in result
    assert row["body_paragraphs"][2] not in result


def test_greeting_and_core_excludes_subject_title_and_signature(labelled_df: pd.DataFrame) -> None:
    row = labelled_df.iloc[0]
    result = greeting_and_core(labelled_df).iloc[0]

    assert result == f"{row['body_paragraphs'][0]} {row['body_paragraphs'][1]}"
    assert row["subject"] not in result
    assert row["inner_title"] not in result
    assert row["body_paragraphs"][2] not in result


def test_subject_only_is_just_the_subject(labelled_df: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(
        subject_only(labelled_df), labelled_df["subject"], check_names=False
    )


def test_no_subject_drops_subject_keeps_title_and_paragraphs(labelled_df: pd.DataFrame) -> None:
    row = labelled_df.iloc[0]
    result = no_subject(labelled_df).iloc[0]

    assert row["subject"] not in result
    assert row["inner_title"] in result


def test_generic_salutation_replaces_greeting_only(labelled_df: pd.DataFrame) -> None:
    row = labelled_df.iloc[0]
    result = generic_salutation(labelled_df).iloc[0]

    assert "Dear Sir/Madam," in result
    assert row["body_paragraphs"][0] not in result
    assert row["body_paragraphs"][1] in result
    assert row["body_paragraphs"][2] in result
    assert row["subject"] in result


def test_truncated_first_sentence_drops_title_and_signature() -> None:
    df = pd.DataFrame(
        {
            "subject": ["Loan Application"],
            "inner_title": ["Financing Request"],
            "body_paragraphs": [
                (
                    "Dear Loan Officer,",
                    "I need a loan of $5000. It is for a car. Thank you.",
                    "Jane Doe",
                )
            ],
        }
    )
    result = truncated_first_sentence(df).iloc[0]

    assert "Financing Request" not in result
    assert "Jane Doe" not in result
    assert "Dear Loan Officer," in result
    assert "Loan Application" in result
    assert "I need a loan of $5000." in result
    assert "It is for a car" not in result


def test_distractor_text_appends_boilerplate_without_removing_signal(labelled_df: pd.DataFrame) -> None:
    row = labelled_df.iloc[0]
    result = distractor_text(labelled_df).iloc[0]

    assert result.startswith(full_text(labelled_df).iloc[0])
    assert "RedRock Support" in result
    assert "confidential" in result
    assert row["subject"] in result


def test_synonym_substitution_replaces_known_department_words() -> None:
    df = pd.DataFrame(
        {
            "subject": ["Loan Application"],
            "body_text": ["I need a loan for my account. My insurance claim was denied."],
        }
    )
    result = synonym_substitution(df).iloc[0]

    assert "loan" not in result.lower()
    assert "financing" in result.lower()
    assert "account" not in result.lower() or "profile" in result.lower()
    assert "insurance" not in result.lower()
    assert "claim" not in result.lower()


def test_typo_noise_is_deterministic_and_preserves_word_count(labelled_df: pd.DataFrame) -> None:
    first = typo_noise(labelled_df, seed=7)
    second = typo_noise(labelled_df, seed=7)
    pd.testing.assert_series_equal(first, second)

    original_words = full_text(labelled_df).iloc[0].split(" ")
    noisy_words = first.iloc[0].split(" ")
    assert len(noisy_words) == len(original_words)


def test_typo_noise_different_seeds_diverge(labelled_df: pd.DataFrame) -> None:
    a = typo_noise(labelled_df, seed=1)
    b = typo_noise(labelled_df, seed=2)
    assert not a.equals(b)


def test_all_registered_variants_return_nonempty_strings_for_every_row(
    labelled_df: pd.DataFrame,
) -> None:
    for name, builder in FEATURE_VARIANTS.items():
        result = builder(labelled_df)
        assert len(result) == len(labelled_df), name
        assert (result.str.strip().str.len() > 0).all(), name
