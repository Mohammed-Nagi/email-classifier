"""Text-assembly variants: one builder per ablation/perturbation condition.

Each builder takes the ingested (and, for training, labelled) DataFrame and
returns a ``pd.Series`` of assembled text — the same shape ``TfidfVectorizer``
expects. ``FEATURE_VARIANTS`` is the registry the ablation harness in
``evaluate.py`` iterates over, so adding a stress condition is: write a
builder, register it, done — no parallel CV code.

Every builder relies on one corpus invariant, checked in
``tests/test_features.py``: every email's ``body_paragraphs`` has exactly
three entries, in order ``(greeting, core, signature)``. Confirmed across all
56 provided files (44 train + 12 test) before writing ``core_only`` below —
see NOTES.md's caution about not trusting this without checking.

Two groups:

- **Ablation variants** (PLAN.md §4): progressively strip structural
  artifacts (inner ``<title>``, greeting/signature, subject) to see how much
  of the saturated CV score they were responsible for.
- **Perturbation variants**: PLAN.md's suggested stress conditions
  (corrupted subject, generic salutation, truncation, distractor text) plus
  two extra ones proposed this session — synonym substitution as a cheap,
  dependency-free paraphrase proxy, and character-level typo noise, which a
  word-level TF-IDF vectorizer has no defence against. Both simulate
  conditions a real inbox has that this generated corpus does not.
"""

from __future__ import annotations

import random
import re
from typing import Callable

import pandas as pd

TextVariantBuilder = Callable[[pd.DataFrame], pd.Series]

GENERIC_SALUTATION = "Dear Sir/Madam,"

# Deterministic, topic-agnostic boilerplate appended by distractor_text.
# Modelled on real reply-chain/disclaimer noise; content is generic on
# purpose so it adds volume without adding class signal.
_QUOTED_REPLY = (
    "On Mon, Jan 5, 2025 at 9:14 AM, RedRock Support <support@redrock.com> wrote: "
    "> Thank you for contacting RedRock. > We will respond within 2 business days."
)
_DISCLAIMER = (
    "This email and any files transmitted with it are confidential and intended "
    "solely for the use of the individual to whom they are addressed. If you have "
    "received this email in error please notify the sender."
)

# Deterministic paraphrase proxy: swap department-indicative words for a
# synonym a real client might use instead. Not a substitute for a real
# paraphrase model, but needs no weight download and directly tests whether
# the model learned the exact keyword or something more robust.
_SYNONYMS = {
    "loan": "financing",
    "account": "profile",
    "insurance": "coverage",
    "claim": "case",
    "investment": "portfolio",
    "advisory": "consultation",
    "advisor": "consultant",
    "transfer": "move",
    "policy": "plan",
}
_SYNONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _SYNONYMS) + r")\b", re.IGNORECASE
)

# Light character-level noise: adjacent-character transposition, applied to
# a fixed fraction of words long enough to survive it. Simulates fat-fingered
# typing / OCR noise that a word-level TF-IDF vectorizer's exact-match
# vocabulary has no tolerance for.
_TYPO_RATE = 0.15


def _core(paragraphs: tuple[str, ...]) -> str:
    return paragraphs[1]


def _join_middle(paragraphs: tuple[str, ...]) -> str:
    """Everything between the first and last paragraph, space-joined.

    Equivalent to ``paragraphs[1]`` for this corpus (always exactly 3
    paragraphs) but written to degrade gracefully rather than silently
    mis-slice if a future variant of the data has more body paragraphs.
    """
    middle = paragraphs[1:-1]
    return " ".join(middle) if middle else _core(paragraphs)


def full_text(df: pd.DataFrame) -> pd.Series:
    """subject + inner title + all paragraphs — the unstripped baseline input."""
    return df["subject"] + " " + df["body_text"]


def no_title(df: pd.DataFrame) -> pd.Series:
    """subject + paragraphs, inner ``<title>`` (a paraphrase of subject) removed."""
    return df["subject"] + " " + df["body_paragraphs"].apply(lambda ps: " ".join(ps))


def core_only(df: pd.DataFrame) -> pd.Series:
    """Core paragraph(s) only — no subject, title, greeting, or signature."""
    return df["body_paragraphs"].apply(_join_middle)


def greeting_and_core(df: pd.DataFrame) -> pd.Series:
    """Greeting + core paragraph — no subject, title, or signature.

    Sits between ``no_title`` and ``core_only`` in the ablation ladder to
    isolate the greeting's own contribution. Added after re-deriving PLAN.md
    §4's numbers turned up a real feature-interaction effect: dropping the
    department-naming greeting (§3) costs ~0.19 macro-F1 once title/subject
    are already gone (this variant vs. ``core_only``), but replacing it with
    a generic one costs almost nothing while title/subject are still present
    (``generic_salutation`` vs. ``full_text``). The greeting only matters
    once nothing else is left to leak the label.
    """
    return df["body_paragraphs"].apply(lambda ps: " ".join(ps[:2]))


def subject_only(df: pd.DataFrame) -> pd.Series:
    """Subject line alone."""
    return df["subject"]


def no_subject(df: pd.DataFrame) -> pd.Series:
    """Full body (title + paragraphs), subject dropped.

    Simulates a forwarded/replied email that lost its subject line, or a
    subject genericised to "Fwd:"/"Re:" by a mail client.
    """
    return df["body_text"]


def generic_salutation(df: pd.DataFrame) -> pd.Series:
    """Full text with the department-naming greeting replaced by a generic one.

    Tests whether the model learned the topic or learned that greetings like
    "Dear Loan Officer" name the department outright — real clients do not
    reliably write department-naming salutations.
    """

    def assemble(row: pd.Series) -> str:
        paragraphs = (GENERIC_SALUTATION, *row["body_paragraphs"][1:])
        return f"{row['subject']} {row['inner_title']} {' '.join(paragraphs)}"

    return df.apply(assemble, axis=1)


def truncated_first_sentence(df: pd.DataFrame) -> pd.Series:
    """subject + greeting + only the first sentence of the core paragraph.

    Simulates a mobile-preview or partial-ingestion truncation: title,
    signature, and everything past the first period of the core message are
    dropped.
    """

    def truncate(row: pd.Series) -> str:
        greeting, core, _signature = row["body_paragraphs"]
        first_sentence = core.split(".", 1)[0].strip()
        if first_sentence:
            first_sentence += "."
        return f"{row['subject']} {greeting} {first_sentence}".strip()

    return df.apply(truncate, axis=1)


def distractor_text(df: pd.DataFrame) -> pd.Series:
    """Full text with a quoted reply chain and a confidentiality disclaimer appended.

    Both are topic-agnostic boilerplate that real client emails routinely
    carry. Tests robustness to added volume that carries no class signal.
    """
    return full_text(df) + " " + _QUOTED_REPLY + " " + _DISCLAIMER


def synonym_substitution(df: pd.DataFrame) -> pd.Series:
    """Full text with department-indicative keywords swapped for a synonym.

    A cheap, dependency-free proxy for paraphrase: no weight download, but it
    directly probes whether the model keyed on the exact word ("loan") or
    would fail on a client who wrote "financing" instead.
    """

    def substitute(match: re.Match[str]) -> str:
        return _SYNONYMS[match.group(0).lower()]

    return full_text(df).apply(lambda text: _SYNONYM_PATTERN.sub(substitute, text))


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or rng.random() > _TYPO_RATE:
        return word
    i = rng.randrange(1, len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def typo_noise(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Full text with ~15% of eligible words character-transposed.

    Simulates fat-fingered typing / OCR noise. Deterministic given ``seed``:
    each call rebuilds its own ``random.Random`` so results don't depend on
    call order or global RNG state.
    """
    rng = random.Random(seed)

    def corrupt(text: str) -> str:
        return " ".join(_typo(word, rng) for word in text.split(" "))

    return full_text(df).apply(corrupt)


# Ablation variants (PLAN.md §4): progressive artifact stripping.
# greeting_and_core sits between no_title and core_only — see its docstring
# for why that intermediate rung was worth adding.
ABLATION_VARIANTS: dict[str, TextVariantBuilder] = {
    "full_text": full_text,
    "no_title": no_title,
    "greeting_and_core": greeting_and_core,
    "core_only": core_only,
    "subject_only": subject_only,
}

# Perturbation variants: stress conditions simulating real inbox noise.
PERTURBATION_VARIANTS: dict[str, TextVariantBuilder] = {
    "no_subject": no_subject,
    "generic_salutation": generic_salutation,
    "truncated_first_sentence": truncated_first_sentence,
    "distractor_text": distractor_text,
    "synonym_substitution": synonym_substitution,
    "typo_noise": typo_noise,
}

FEATURE_VARIANTS: dict[str, TextVariantBuilder] = {**ABLATION_VARIANTS, **PERTURBATION_VARIANTS}
