"""Parse RedRock's HTML email files and the labels CSV into records.

Each file is an outer HTML document with metadata in ``data-field`` divs and a
body div whose content is itself a second complete HTML document. This module
unwraps that and keeps the body decomposed, so feature builders can select
parts of it without re-parsing HTML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

REQUIRED_META_FIELDS = ("email_id", "subject", "sender", "date_received")


@dataclass(frozen=True)
class EmailRecord:
    """A parsed email. ``inner_title`` and ``body_paragraphs`` stay separate
    (not pre-flattened) so ablations can drop artifacts without re-parsing."""

    email_id: int
    source_filename: str
    subject: str
    sender: str
    date_received: str
    inner_title: str
    body_paragraphs: tuple[str, ...]
    body_text: str


def _extract_meta_fields(soup: BeautifulSoup, source_filename: str) -> dict[str, str]:
    fields = {
        div["data-field"]: div.get_text(strip=True)
        for div in soup.find_all("div", attrs={"data-field": True})
    }
    missing = [name for name in REQUIRED_META_FIELDS if not fields.get(name)]
    if missing:
        raise ValueError(f"{source_filename}: missing required field(s) {missing}")
    return fields


def _unwrap_body(soup: BeautifulSoup, source_filename: str) -> tuple[str, tuple[str, ...]]:
    body_div = soup.find("div", class_="email-body")
    if body_div is None:
        raise ValueError(f"{source_filename}: no .email-body div found")

    inner_soup = BeautifulSoup(body_div.decode_contents(), "html.parser")

    inner_title_tag = inner_soup.find("title")
    if inner_title_tag is None or not inner_title_tag.get_text(strip=True):
        raise ValueError(f"{source_filename}: nested document has no <title>")
    inner_title = inner_title_tag.get_text(strip=True)

    # Extract per element, never one get_text() across tags: every file has
    # <title>X</title><p>Y</p> with no whitespace between, which a single call
    # would glue into "XY" — one garbage token per email after tokenization.
    paragraphs = tuple(
        p.get_text(separator=" ", strip=True) for p in inner_soup.find_all("p")
    )
    paragraphs = tuple(p for p in paragraphs if p)
    if not paragraphs:
        raise ValueError(f"{source_filename}: nested document has no <p> content")

    return inner_title, paragraphs


def parse_email_file(path: Path) -> EmailRecord:
    """Parse one RedRock email HTML file into an :class:`EmailRecord`.

    Raises ``ValueError`` if required metadata fields, the nested body div,
    the inner title, or the inner paragraphs are missing.
    """
    source_filename = path.name
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    fields = _extract_meta_fields(soup, source_filename)
    inner_title, paragraphs = _unwrap_body(soup, source_filename)

    try:
        email_id = int(fields["email_id"])
    except ValueError as exc:
        raise ValueError(
            f"{source_filename}: email_id {fields['email_id']!r} is not an integer"
        ) from exc

    body_text = " ".join((inner_title, *paragraphs))

    return EmailRecord(
        email_id=email_id,
        source_filename=source_filename,
        subject=fields["subject"],
        sender=fields["sender"],
        date_received=fields["date_received"],
        inner_title=inner_title,
        body_paragraphs=paragraphs,
        body_text=body_text,
    )


def _numeric_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def ingest_directory(dir_path: Path) -> list[EmailRecord]:
    """Parse every ``*.html`` file in ``dir_path``, in filename-numeric order."""
    files = sorted(Path(dir_path).glob("*.html"), key=_numeric_sort_key)
    return [parse_email_file(f) for f in files]


def records_to_dataframe(records: list[EmailRecord]) -> pd.DataFrame:
    """Convert parsed records into a DataFrame, one row per email."""
    return pd.DataFrame([asdict(r) for r in records])


def load_train_labels(labels_csv: Path) -> pd.DataFrame:
    """Load ``train_labels.csv`` (columns: ``filename``, ``true_category``)."""
    return pd.read_csv(labels_csv)


def attach_labels(records_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join parsed records with their labels on filename.

    Raises ``ValueError`` if any record fails to match a label, since a
    silent drop would quietly shrink the training set.
    """
    merged = records_df.merge(
        labels_df, left_on="source_filename", right_on="filename", how="left"
    )
    unmatched = merged[merged["true_category"].isna()]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} record(s) had no matching label: "
            f"{unmatched['source_filename'].tolist()}"
        )
    return merged.drop(columns=["filename"])
