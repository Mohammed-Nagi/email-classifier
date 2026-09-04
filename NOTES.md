# Handoff notes

Context that's expensive to reconstruct from the commits/code alone. Written at the end
of the session that completed PLAN.md §9 steps 1–4 (scaffold, ingest.py + tests, recon.py,
TF-IDF+LR baseline → predictions.csv). Read this before continuing at step 5.

## Decisions and reasoning

- **Parser: stdlib `html.parser`, not `lxml`/`html5lib`.** The nested-doc unwrap
  (`.email-body` contains a second full `<!DOCTYPE html>` document) is the highest-risk
  part of ingestion because HTML5-spec parsers apply tree-construction rules (foster
  parenting) that can hoist a nested `<html>`/`<head>`/`<title>` out of its containing div.
  I tested `html.parser` and `lxml` directly against `data/train/email_1.html`: both kept
  the nested doc correctly nested inside `.email-body` for this data (no reparenting
  triggered). I did not test `html5lib` — it wasn't installed, and it implements the HTML5
  tree-construction algorithm most strictly, so it's the most likely of the three to
  reparent. Chose `html.parser` because it's stdlib (no dependency) and was empirically
  verified, not because `lxml` was shown to be wrong. If a future session switches parsers,
  `tests/test_ingest.py::test_nested_body_unwrapped_correctly` and
  `test_no_whitespace_glue_between_adjacent_elements` will catch a regression.

- **`EmailRecord` keeps the body decomposed, not flattened.** `inner_title` and
  `body_paragraphs` are separate fields; `body_text` is a convenience concatenation, not
  the canonical representation. This is deliberate: `features.py` (step 6, unbuilt) needs
  to build stripped variants (no title, no greeting/signature, subject-only) for the
  ablation harness, and that's cheap only if ingest.py hasn't already thrown the structure
  away.

- **Fail loudly, no fallback paths.** `parse_email_file` raises `ValueError` on any missing
  field, missing `.email-body`, missing inner `<title>`, or empty paragraph list. No
  malformed file exists in the 56 provided (verified in `scripts/recon.py` and asserted
  across all of them in tests), so there was no real case to design graceful degradation
  for. If real malformed input ever shows up, it should surface immediately, not produce a
  silently-wrong row.

- **No `models/base.py` abstraction yet.** PLAN.md's target architecture has a swappable
  fit/predict_proba interface from the start; I built `src/model.py` as one concrete
  pipeline instead. Building the interface for a single arm would be speculative — do it
  in step 5 when the embeddings arm exists to swap against.

- **`recon.py` deliberately excludes §2/§4 numbers.** It reproduces only §3's data-only
  facts (counts, structure, lengths, sender noise, taxonomy gap) plus one extension (the
  near-duplicate check, see below). CV accuracy, macro-F1, and the ablation table are
  model-evaluation results that belong in `evaluate.py` (step 5+) — computing them twice in
  two different scripts would just create a second place for them to drift out of sync.

- **Confidence score is raw, uncalibrated LR probability.** No calibration in step 4 by
  design (that's step 9). Don't read anything into the current values beyond "the model's
  own softmax output."

- **`sender` and `date_received` are captured but not fed to the model.** Only
  `subject + body_text` goes into TF-IDF. This was the correct default given the sender
  trap-feature finding (below), but it was never an explicit filtering decision inside
  `model.py` — it's just that nothing there references those columns. If a later session
  adds sender/date features, that needs the deliberate "measure, then exclude, with
  evidence" treatment PLAN.md calls for in §3, not a silent inclusion.

## Exact text assembly (ingest.py → model.py)

This is the contract the step-6 ablation harness needs to build on top of, unambiguously:

1. `_unwrap_body()` extracts `inner_title` via `title_tag.get_text(strip=True)` (one tag,
   one string).
2. Each `<p>` in the inner doc is extracted independently via
   `p.get_text(separator=" ", strip=True)`; empty results are filtered out. Order is
   document order — in this corpus that's consistently `[greeting, body sentence(s),
   signature name]`, typically 3 paragraphs.
3. `EmailRecord.body_text = " ".join((inner_title, *body_paragraphs))` — single space
   between every element, inner_title first, then paragraphs in order. No newlines.
4. `model.build_text_input()` = `df["subject"] + " " + df["body_text"]` — final string fed
   to `TfidfVectorizer` is `subject, inner_title, paragraph_0, ..., paragraph_n`, all
   single-space-joined.
5. Nothing dropped is silently dropped: the outer document chrome (outer `<title>`,
   `<h1>`, the header `<p>` duplicating subject) is never extracted in the first place —
   only content inside `.email-body` and the `data-field` divs is read.
6. No lowercasing/stopword-removal/punctuation-stripping happens in `ingest.py`. That's
   entirely `TfidfVectorizer`'s job downstream (`stop_words: english` in config.yaml).
   `ingest.py`'s contract stops at "clean structured text out of HTML."

For the "body only, inner `<title>` removed" ablation variant: drop `inner_title` from the
join in step 3/4, keep everything else. For "core paragraphs only": use `body_paragraphs`
directly, excluding index 0 (greeting) and index -1 (signature) — but verify that
first/last-index assumption against the corpus rather than trusting it; I didn't check
whether every email has exactly the greeting-then-signature shape.

## The get_text() separator bug

**Symptom:** `BeautifulSoup.get_text()` with the default `separator=""` concatenates text
from adjacent tags with zero whitespace when there's no whitespace text node between them
in the raw HTML. Every one of the 56 inner documents has `<title>X</title><p>Y</p>` with no
whitespace in between, e.g. `<title>Account Transfer</title><p>Dear Account Services,</p>`.
A naive single `body_div.get_text()` call produces `"...Account TransferDear Account
Services,..."` — one glued token where there should be two words.

**What it would have corrupted:** not a rare edge case — this pattern is in all 56 files,
so every email's body text would have had a glued title/first-paragraph boundary. After
TF-IDF tokenization that becomes a garbage compound token (e.g. `transferdear`) instead of
two real words, in every row. It would specifically have undermined the step-6 ablation:
the "inner title removed" variant depends on cleanly separating title from paragraph text,
which a glued string can't do correctly.

**Guard now in place:** `_unwrap_body()` never calls `get_text()` across multiple tags in
one call — `inner_title` and each paragraph are extracted as separate strings, then
explicitly joined with `" "`. Per-paragraph extraction also uses `separator=" "` internally
as a defensive measure against multi-tag paragraphs (none exist yet, but a future data
variant might have inline tags inside a `<p>`). Regression test:
`test_no_whitespace_glue_between_adjacent_elements` asserts the joined form contains
`"Account Transfer Dear Account Services"` (with space) and does not contain `"TransferDear"`.

## Data surprises / fragile spots

- **The corpus is a near-perfect template.** All 56 files share identical HTML/CSS
  boilerplate and structure. `ingest.py`'s fail-loudly design is correct for this data but
  would need real hardening (encoding variants, missing fields, multiple body divs, etc.)
  before pointing it at an actual inbox.
- **Sender field is a textbook trap feature, confirmed exactly as PLAN.md claimed:**
  `security@redrock.com` sends a "Job Application Response" (HR topic); `jamesg@gmail.com`
  sends a "System Maintenance Notice" (IT topic). Domain/address carries no reliable
  category signal here.
- **Inner `<title>` is a near-verbatim paraphrase of `subject`** (e.g. subject "Account
  Transfer Request" → inner title "Account Transfer"). This is a strong artifact/leakage
  signal sitting right at the start of the body text — explains why PLAN.md's ablation
  table shows a real drop once it's stripped.
- **Baseline confidence scores are uniformly low.** All 12 test predictions from the
  current (uncalibrated) model land in 0.26–0.49 — none above 0.5, even for emails that
  look unambiguous. This is expected for raw multinomial softmax over 5 classes with 44
  training rows and a diffuse TF-IDF feature space, but it means `confidence_score` as
  currently computed does **not** yet meaningfully separate confident from uncertain
  predictions. Calibration (step 9) is a real prerequisite for §6's routing/abstain policy
  to mean anything, not just a nice-to-have.
- **`test/email_4.html` (internal id 53), the fraud-report taxonomy gap, is confirmed** —
  but the current baseline predicts it Account Management at confidence **0.39**, not
  PLAN.md's cited 0.64. Different hyperparameters (and possibly different text assembly)
  than whatever produced that number. The qualitative finding (no category fits) still
  holds; the specific number in PLAN.md §6's confidence table is stale and needs
  regenerating from this codebase before it goes in the README.
- **Python 3.14.4 was the only interpreter available on this machine.** `requirements.txt`
  is pinned to versions that resolved cleanly against it (see commit history), but nobody
  has verified this on an older Python. Worth a fresh-clone test on whatever Python version
  is actually common before submission (PLAN.md step 13).

## Where PLAN.md is wrong, underspecified, or worth doing differently

- **The unwrap's hardest part — parser choice and the separator hazard — isn't mentioned
  at all**, despite PLAN.md flagging the unwrap generally as "easy to get subtly wrong."
  This was the single biggest gap between PLAN.md's stated risk level and the guidance it
  gave for avoiding it.
- **§3's body-length figures (127–204, median 170) don't state what was measured.** It
  turns out `body_text` (inner_title + paragraphs, joined) reproduces PLAN.md's numbers
  exactly — confirmed in `scripts/recon.py`. Worth stating explicitly for whoever reads
  PLAN.md next, since "body only" is ambiguous between several plausible definitions.
- **The 0.26 near-duplicate figure was off** (now corrected in PLAN.md itself to 0.245,
  with a note). This is a useful reminder that every number in PLAN.md came from the same
  throwaway probe and needs the same re-derivation treatment — I'd trust none of the
  remaining §2/§4/§6 numbers until re-measured against this actual codebase.
- **§6's confidence/margin table is almost certainly stale** for the same reason (see
  email_4 above) — flagging clearly so a future session doesn't copy it into the README
  without regenerating it from `evaluate.py`.
- **PLAN.md's repo sketch shows `models/base.py` from the start; I deliberately didn't
  build it in step 1–4** (see Decisions above). Step 5 should either build it when the
  embeddings arm lands, or make an explicit call that the abstraction isn't worth it for
  two arms — don't default into it without deciding.

## Known gaps — do not mistake for finished work

- **No CV/holdout evaluation exists for this exact pipeline.** `src/run.py` fits on all 44
  labelled rows with zero held-out validation. There is currently no measured accuracy or
  macro-F1 anywhere in this codebase for `src/model.py`'s actual configuration — the §2
  headline numbers in PLAN.md are from the old throwaway probe, not from this code, and
  have not been re-verified against it. This is the most important open item for step 5.
- `config.yaml`'s model hyperparameters (`max_features`, `ngram_range`, `C`,
  `class_weight`) are untuned, unvalidated defaults — documented as "sane defaults" in a
  comment, nothing more.
- `Other` is handled as a plain 5th label only (strategy 1 of 3 in PLAN.md §6) — abstain-as-
  Other and the hybrid gate are not implemented or compared (step 8).
- No routing/threshold/abstain logic, no `needs_review` column, no `--auto-route-threshold`.
- No explainability / token attribution (`explain.py` doesn't exist).
- No embeddings arm, no zero-shot NLI arm.
- No README.
- `lxml` is installed in `.venv` (from the parser investigation) but is **not** in
  `requirements.txt` and nothing shipped imports it — intentional, but don't be surprised
  it's in a local `pip freeze`.
- Git remote is set: `https://github.com/Mohammed-Nagi/email-classifier.git`, pushed to
  `master`, tracking `origin/master`.
