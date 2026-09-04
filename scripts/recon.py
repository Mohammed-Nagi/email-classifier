"""Throwaway data-reconnaissance probe.

Reproduces the §3 data recon table in PLAN.md from this codebase's own
ingest module, so no claim about the data rests on a number that was only
ever computed in a scratch notebook. Scope is deliberately limited to
*data* properties (counts, structure, length, sender noise, the taxonomy
gap). Model performance numbers (§2's CV / macro-F1 scores, the §4 ablation
table) are a separate concern, verified later by evaluate.py once it
exists (build step 5+) — printing them here would just be a second place
for them to go stale.

Not part of the shipped pipeline; not covered by tests; run directly:
    python scripts/recon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.ingest import attach_labels, ingest_directory, load_train_labels, records_to_dataframe

DEPARTMENT_WORDS = (
    "account",
    "advisory",
    "advisor",
    "loan",
    "insurance",
    "claim",
    "investment",
    "financial planner",
    "security",
)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    config = load_config()
    root = Path(__file__).resolve().parent.parent
    train_dir = root / config["paths"]["train_dir"]
    test_dir = root / config["paths"]["test_dir"]
    labels_csv = root / config["paths"]["train_labels"]

    train_records = ingest_directory(train_dir)
    test_records = ingest_directory(test_dir)
    train_df = records_to_dataframe(train_records)
    test_df = records_to_dataframe(test_records)
    labels_df = load_train_labels(labels_csv)
    labelled_df = attach_labels(train_df, labels_df)

    section("Corpus size")
    print(f"train: {len(train_df)} labelled emails")
    print(f"test:  {len(test_df)} unlabelled emails")

    section("Class balance (train)")
    counts = labelled_df["true_category"].value_counts()
    for category, count in counts.items():
        print(f"  {category:<22} {count}")

    section("email_id vs filename")
    all_df = records_to_dataframe(train_records + test_records)
    filename_numbers = all_df["source_filename"].str.extract(r"(\d+)")[0].astype(int)
    mismatches = int((all_df["email_id"] != filename_numbers).sum())
    print(f"mismatched: {mismatches} / {len(all_df)}")
    example = all_df.iloc[0]
    print(f"  example: {example['source_filename']} -> internal email_id {example['email_id']}")

    section("Body length (chars, body_text = inner_title + paragraphs)")
    lengths = all_df["body_text"].str.len()
    print(f"min={lengths.min()}  max={lengths.max()}  median={lengths.median()}")

    section("Sender field noise")
    all_df["sender_domain"] = all_df["sender"].str.split("@").str[-1]
    print(all_df["sender_domain"].value_counts().to_string())
    hr_from_security = labelled_df[
        labelled_df["sender"].eq("security@redrock.com")
        & labelled_df["subject"].str.contains("Job Application", case=False)
    ]
    it_from_gmail = labelled_df[
        labelled_df["sender"].str.endswith("@gmail.com")
        & labelled_df["subject"].str.contains("Maintenance", case=False)
    ]
    print(f"HR-topic mail from security@redrock.com: {len(hr_from_security)} match(es)")
    print(f"IT-topic mail from a gmail.com address:   {len(it_from_gmail)} match(es)")

    section("Salutations naming a department")
    first_paragraphs = all_df["body_paragraphs"].apply(lambda ps: ps[0].lower() if ps else "")
    named = first_paragraphs.apply(lambda s: any(word in s for word in DEPARTMENT_WORDS))
    print(f"{named.sum()} / {len(named)} opening lines mention a department-indicative word")
    print("examples:")
    for text in first_paragraphs[named].head(3):
        print(f"  - {text}")

    section("Taxonomy gap: test/email_4.html")
    fraud_email = next(r for r in test_records if r.source_filename == "email_4.html")
    print(f"subject: {fraud_email.subject}")
    print(f"body: {' '.join(fraud_email.body_paragraphs)}")
    print("-> no department in the 5-category taxonomy covers a fraud report")

    section("Extension beyond section 3: within-train nearest-neighbour similarity")
    print("(diagnostic only - checks the 'not near-duplicate leakage' claim; not a model score)")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(labelled_df["body_text"])
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    nearest_neighbour_sim = sim.max(axis=1)
    print(f"mean nearest-neighbour cosine similarity: {nearest_neighbour_sim.mean():.3f}")


if __name__ == "__main__":
    main()
