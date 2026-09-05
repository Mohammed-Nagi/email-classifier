"""Text-assembly variants: one builder per ablation condition.

Each builder maps the ingested DataFrame to a ``pd.Series`` of assembled text,
so the ablation is the same CV harness over different builders rather than a
parallel codebase. Every builder relies on one corpus invariant, checked in
``tests/test_features.py``: ``body_paragraphs`` is always
``(greeting, core, signature)``.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

TextVariantBuilder = Callable[[pd.DataFrame], pd.Series]

GENERIC_SALUTATION = "Dear Sir/Madam,"


def full_text(df: pd.DataFrame) -> pd.Series:
    """subject + inner title + all paragraphs — the unstripped baseline."""
    return df["subject"] + " " + df["body_text"]


def no_title(df: pd.DataFrame) -> pd.Series:
    """subject + paragraphs, inner ``<title>`` (a paraphrase of subject) removed."""
    return df["subject"] + " " + df["body_paragraphs"].apply(lambda ps: " ".join(ps))


def greeting_and_core(df: pd.DataFrame) -> pd.Series:
    """Greeting + core paragraph. Isolates the greeting's own contribution
    against ``core_only``: dropping the department-naming greeting costs real
    macro-F1 once title and subject are already gone, while replacing it costs
    almost nothing when they aren't (``generic_salutation``)."""
    return df["body_paragraphs"].apply(lambda ps: " ".join(ps[:2]))


def core_only(df: pd.DataFrame) -> pd.Series:
    """Core paragraph(s) only — no subject, title, greeting, or signature.

    ``[1:-1]`` rather than ``[1]`` so extra body paragraphs degrade gracefully
    instead of being silently mis-sliced.
    """
    return df["body_paragraphs"].apply(
        lambda ps: " ".join(ps[1:-1]) if len(ps) > 2 else ps[1]
    )


def subject_only(df: pd.DataFrame) -> pd.Series:
    """Subject line alone."""
    return df["subject"]


def generic_salutation(df: pd.DataFrame) -> pd.Series:
    """Full text with the department-naming greeting ("Dear Loan Officer")
    swapped for a generic one — **keeps the inner title**; only paragraph 0
    changes. Real clients don't reliably name the department they're writing to.
    """

    def assemble(row: pd.Series) -> str:
        paragraphs = (GENERIC_SALUTATION, *row["body_paragraphs"][1:])
        return f"{row['subject']} {row['inner_title']} {' '.join(paragraphs)}"

    return df.apply(assemble, axis=1)


# Progressive artifact stripping (the ablation ladder), plus generic_salutation,
# which is a perturbation rather than a rung: it swaps the greeting instead of
# removing content, and pairs with greeting_and_core/core_only to show the
# greeting is redundant leakage until nothing else leaks the label.
ABLATION_VARIANTS: dict[str, TextVariantBuilder] = {
    "full_text": full_text,
    "no_title": no_title,
    "greeting_and_core": greeting_and_core,
    "core_only": core_only,
    "subject_only": subject_only,
}

FEATURE_VARIANTS: dict[str, TextVariantBuilder] = {
    **ABLATION_VARIANTS,
    "generic_salutation": generic_salutation,
}
