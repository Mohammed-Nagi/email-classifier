"""Tests for src.ingest: nested-HTML unwrap, email_id decision, malformed input."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest import (
    attach_labels,
    ingest_directory,
    load_train_labels,
    parse_email_file,
    records_to_dataframe,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"

EXPECTED_CLASS_COUNTS = {
    "Account Management": 13,
    "Investment Advisory": 12,
    "Insurance Claims": 7,
    "Other": 6,
    "Loan Processing": 6,
}

VALID_EMAIL_HTML = """<!DOCTYPE html>
<html><head><title>Email 31</title></head>
<body>
<div class="meta">
  <div data-field="email_id">31</div>
  <div data-field="subject">Account Transfer Request</div>
  <div data-field="sender">cjones@gmail.com</div>
  <div data-field="date_received">2025-04-15</div>
</div>
<div class="email-body">
<!DOCTYPE html><html><head><title>Account Transfer</title></head><body>\
<p>Dear Account Services,</p><p>I would like to transfer my account.</p>\
<p>Anthony Scott</p></body></html>
</div>
</body></html>
"""


def test_parse_email_1_metadata_matches_known_fixture() -> None:
    record = parse_email_file(TRAIN_DIR / "email_1.html")

    assert record.email_id == 31
    assert record.source_filename == "email_1.html"
    assert record.subject == "Account Transfer Request"
    assert record.sender == "cjones@gmail.com"
    assert record.date_received == "2025-04-15"


def test_nested_body_unwrapped_correctly() -> None:
    record = parse_email_file(TRAIN_DIR / "email_1.html")

    assert record.inner_title == "Account Transfer"
    assert record.body_paragraphs == (
        "Dear Account Services,",
        "I would like to transfer my account to another financial institution. "
        "What is the process and any associated fees?",
        "Anthony Scott",
    )
    # The outer chrome ("Email 31") must not leak into the unwrapped body.
    assert "Email 31" not in record.body_text


def test_no_whitespace_glue_between_adjacent_elements() -> None:
    record = parse_email_file(TRAIN_DIR / "email_1.html")

    assert "Account Transfer Dear Account Services" in record.body_text
    assert "TransferDear" not in record.body_text


@pytest.mark.parametrize("directory", [TRAIN_DIR, TEST_DIR])
def test_email_id_differs_from_filename_for_every_file(directory: Path) -> None:
    records = ingest_directory(directory)
    assert records, f"expected files in {directory}"

    for record in records:
        filename_number = int("".join(ch for ch in record.source_filename if ch.isdigit()))
        assert record.email_id != filename_number, (
            f"{record.source_filename}: email_id unexpectedly equals the "
            "filename number — the internal id / filename mismatch this "
            "codebase relies on no longer holds"
        )


def test_ingest_directory_returns_expected_counts() -> None:
    assert len(ingest_directory(TRAIN_DIR)) == 44
    assert len(ingest_directory(TEST_DIR)) == 12


def test_parse_file_missing_required_field_raises(tmp_path: Path) -> None:
    html = VALID_EMAIL_HTML.replace('<div data-field="subject">Account Transfer Request</div>', "")
    bad_file = tmp_path / "bad.html"
    bad_file.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        parse_email_file(bad_file)


def test_parse_file_missing_email_body_div_raises(tmp_path: Path) -> None:
    html = VALID_EMAIL_HTML.replace('class="email-body"', 'class="not-the-body"')
    bad_file = tmp_path / "bad.html"
    bad_file.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="no .email-body div"):
        parse_email_file(bad_file)


def test_parse_file_missing_inner_title_raises(tmp_path: Path) -> None:
    html = VALID_EMAIL_HTML.replace("<title>Account Transfer</title>", "")
    bad_file = tmp_path / "bad.html"
    bad_file.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="no <title>"):
        parse_email_file(bad_file)


def test_load_train_labels_has_44_rows_matching_train_filenames() -> None:
    labels_df = load_train_labels(TRAIN_LABELS)
    train_filenames = {p.name for p in TRAIN_DIR.glob("*.html")}

    assert len(labels_df) == 44
    assert set(labels_df["filename"]) == train_filenames


def test_attach_labels_produces_expected_class_counts() -> None:
    records_df = records_to_dataframe(ingest_directory(TRAIN_DIR))
    labels_df = load_train_labels(TRAIN_LABELS)

    merged = attach_labels(records_df, labels_df)

    assert merged["true_category"].value_counts().to_dict() == EXPECTED_CLASS_COUNTS


def test_attach_labels_raises_on_unmatched_record() -> None:
    records_df = records_to_dataframe(ingest_directory(TRAIN_DIR))
    labels_df = load_train_labels(TRAIN_LABELS).iloc[1:]  # drop email_1.html's label

    with pytest.raises(ValueError, match="no matching label"):
        attach_labels(records_df, labels_df)
