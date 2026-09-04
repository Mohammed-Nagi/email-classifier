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
  throwaway probe and needs the same re-derivation treatment — confirmed right again this
  session: §2/§4's headline numbers were also off, materially (see the step 5/6 section
  above). §6's confidence/margin table is the one remaining unverified figure; it needs
  regenerating from `evaluate.py`/calibration in step 9, same treatment.
- ~~PLAN.md's repo sketch shows `models/base.py` from the start; I deliberately didn't
  build it in step 1–4~~ — built this session (`src/models/base.py`, `ClassifierModel`).
  The trigger wasn't a second arm existing yet (it doesn't — embeddings is still step 7)
  but the CV harness needing a `model_factory: () -> ClassifierModel` contract to build a
  fresh unfitted model per fold; that contract is exactly the fit/predict_proba interface
  PLAN.md wanted, so it made sense to build now rather than duplicate it later.

## Step 5/6 session: PLAN.md §2/§4 numbers re-derived, materially different

The single biggest open item from the previous handoff — "no CV/holdout evaluation exists
for this exact pipeline" — is resolved. Running `python -m src.evaluate` (5-fold × 10
repeats, `TfidfLRModel` + `features.full_text`) does **not** reproduce PLAN.md's original
§2/§4 numbers: `full_text` scores 0.929 ± 0.108 (min 0.731), not 0.986 ± 0.038, and 0/50
independent single 5-fold splits score a perfect macro-F1.

Diagnosed, not just measured: the instability is almost entirely the `Other` class (F1
0.727 ± 0.424, min 0.000 across the same 50 folds) — the four real categories score
0.980 ± 0.031 excluding it, matching the original probe closely. PLAN.md §2 and §4 have
been rewritten in place with the corrected numbers and this diagnosis; do not use the old
figures (0.986/0.839/0.950) anywhere downstream (README included) — they do not reproduce
from this codebase and the session transcript has the full diagnostic if the "why" is
needed again.

One specific number was also traced to a probable definitional difference rather than a
bug: the original "core paragraphs only" figure (0.839) is close to this session's
`greeting_and_core` variant (0.863) — which keeps the greeting — and far from the literal
"no greeting" reading (`core_only`, 0.674). Most likely the original probe's "core
paragraphs only" kept the greeting despite its own written definition excluding it. Verified
this isn't an ingest/assembly bug on this session's side: `core_only`'s output was eyeballed
against the raw HTML for 3 emails across categories and matches exactly (see `src/features.py`
`_join_middle`/`core_only`), and `subject_only` reproduces the original probe's number almost
exactly (0.670 vs 0.667), which also rules out the CV protocol/estimator as the cause.

## Exact semantics of every features.py variant — read before touching step 7

The embeddings arm (step 7) has to run through **these exact same `FEATURE_VARIANTS`
builders** for the comparison to mean anything — if it builds its own text assembly, the
ablation/perturbation table becomes two different experiments wearing one table. Precise
composition of each, in terms of the `EmailRecord` fields it reads:

| Variant | Group | Exact text | Fields touched |
|---|---|---|---|
| `full_text` | ablation | `subject` + `inner_title` + `greeting` + `core` + `signature` | `subject`, `body_text` (= `inner_title`+paragraphs, pre-joined in ingest.py) |
| `no_title` | ablation | `subject` + `greeting` + `core` + `signature` | `subject`, `body_paragraphs` |
| `greeting_and_core` | ablation | `greeting` + `core` | `body_paragraphs[:2]` |
| `core_only` | ablation | `core` (= `body_paragraphs[1:-1]` joined; equals `body_paragraphs[1]` alone for this corpus — always exactly 3 paragraphs, checked) | `body_paragraphs` |
| `subject_only` | ablation | `subject` | `subject` |
| `no_subject` | perturbation | `inner_title` + `greeting` + `core` + `signature` (literally `body_text` as-is) | `body_text` |
| `generic_salutation` | perturbation | `subject` + `inner_title` + `"Dear Sir/Madam,"` + `core` + `signature` — **keeps title**, only paragraph 0 is swapped | `subject`, `inner_title`, `body_paragraphs[1:]` |
| `truncated_first_sentence` | perturbation | `subject` + `greeting` + first sentence of `core` up to and including the first `.` (title and signature dropped, rest of core dropped) | `subject`, `body_paragraphs[0]`, `body_paragraphs[1]` |
| `distractor_text` | perturbation | `full_text` + fixed quoted-reply string + fixed disclaimer string (both hardcoded in `features.py`, topic-agnostic) | `full_text()` + literals |
| `synonym_substitution` | perturbation | `full_text` with a regex word-boundary, case-insensitive swap over 9 hardcoded department words (`loan→financing, account→profile, insurance→coverage, claim→case, investment→portfolio, advisory→consultation, advisor→consultant, transfer→move, policy→plan`) | `full_text()` |
| `typo_noise` | perturbation | `full_text` with ~15% of words (length ≥ 4) getting one adjacent-character swap, via a local `random.Random(seed)` | `full_text()` |

Two things in that table that are easy to get wrong from memory alone:

- **`generic_salutation` keeps the inner `<title>`.** It is *not* built on `no_title` or
  `core_only` — it's `full_text` with only the greeting paragraph replaced. If a future
  session wants a "generic salutation AND no title" variant, that's a new builder, not a
  tweak to the existing one.
- **`typo_noise`'s seed defaults to a hardcoded `42`, independent of `config['seed']`.**
  `FEATURE_VARIANTS["typo_noise"]` is called as `builder(df)` everywhere (registry loop in
  `run_ablation`, in `evaluate.py`'s `main()`), so it always uses the hardcoded default. If
  `config.yaml`'s seed ever changes, `typo_noise`'s specific corruption pattern will not move
  with it — the CV splits will use the new seed, but the noise injected into the text won't.
  Not a bug today (nothing depends on the two seeds matching), but a latent trap if a future
  session assumes "seed everywhere" means all seeds are threaded from one place.

## How the greeting/title interaction was isolated — re-checkable, not just asserted

**Mechanism**: `RepeatedStratifiedKFold.split(X, y)` stratifies on `y` (labels) only — `X`
just has to match `y`'s length. Verified directly: built fold splits from `full_text` and
from `core_only` at the same seed and compared every `(train_idx, test_idx)` pair —
identical. This means **for a fixed seed, every `FEATURE_VARIANTS` builder gets the exact
same train/test fold membership**, since they're all built from the same `labelled_df`. Two
variants' macro-F1 scores at the same seed are therefore a paired comparison for free — a
difference between them is attributable to the text change alone, not to which examples
happened to be hard in that split. No paired-t-test machinery needed; the CV protocol
already provides the pairing.

That's what makes these two comparisons valid:

- `generic_salutation` (0.920 ± 0.109) vs. `full_text` (0.929 ± 0.108): swapping the greeting
  while title/subject are still present costs **0.009** — noise-level.
- `greeting_and_core` (0.863 ± 0.114) vs. `core_only` (0.674 ± 0.118): removing the greeting
  once title/subject are already gone costs **0.189** — real.

To re-verify this later (e.g. after the embeddings arm exists, to check the same interaction
holds for a different model family): run `repeated_stratified_cv` on any two
`FEATURE_VARIANTS` builders at the same `seed`, and the difference in `.macro_f1_scores`
arrays (not just the means) is the fold-by-fold effect of that specific text change —
`(scores_a - scores_b)` is a valid paired difference vector.

## CV harness gotchas for whoever calls evaluate.py next

- **No caching, no shared fits.** `main()` in `evaluate.py` calls `repeated_stratified_cv`
  and `per_class_f1_cv` separately for `full_text`, and `run_ablation` calls it again for
  `full_text` as part of `ABLATION_VARIANTS`. Same splits (same seed), but models are
  refit from scratch each time — nothing is cached across these calls. `python -m
  src.evaluate` takes a few seconds on this corpus; if a heavier arm (embeddings) makes that
  noticeably slow, consider caching fold assignments or fitted vectorizers, but don't assume
  that exists today.
- **`model_factory` closures over `config`, not over loop variables.** In `run_ablation`,
  `lambda: TfidfLRModel(config)` is safe (no late-binding bug) because `config` doesn't
  change across the loop — only `texts`, which isn't captured by the lambda, is used
  directly in the `repeated_stratified_cv` call. If a future session parametrizes the model
  itself per-variant (e.g. different `C` per condition), watch for the classic
  loop-variable-capture bug.
- **Variance is reported everywhere in `evaluate.py`'s own output** (`CVResult.summary()`
  always prints mean ± std, min, max), but **`run_ablation`'s returned DataFrame (and
  `outputs/ablation_results.csv`) only stores the four summary stats, not the raw per-fold
  arrays.** If `plots.py` (step 9) wants a fold-distribution plot (box/violin per variant),
  it needs to call `repeated_stratified_cv` directly for that variant, not read the CSV —
  the CSV has already thrown the per-fold detail away.
- **`std` is sample std (`ddof=1`)**, not population std — matters if PLAN.md's original
  numbers ever need re-checking against a `ddof=0` computation (unlikely to matter at n=50
  folds, but worth knowing which one this codebase uses).
- `_DummyModel` wraps `sklearn.dummy.DummyClassifier(strategy="most_frequent")` — it has no
  `config` dependency, so `count_perfect_single_fold_runs(_DummyModel, ...)` works directly
  (see `test_dummy_model_never_scores_a_perfect_single_fold_run`).

## PLAN.md §5–§7 concerns surfaced this session (not yet acted on)

- **§5's claim that the stripped-artifact ablation gives arms "real headroom to separate
  them" is untested for whether that headroom is clean signal or more `Other`-noise.** §2's
  finding — that `Other` (not general unsaturation) drives most of `full_text`'s variance —
  was only checked on `full_text`. Nobody has run `per_class_f1_cv` on `core_only` or
  `subject_only` to see whether `Other` is still the dominant source of instability in the
  stripped conditions. If it is, comparing TF-IDF+LR vs. embeddings on `core_only` will
  still be partly deciding the comparison by `Other`-noise, not by which arm handles sparse
  text better — worth checking before leaning on that comparison in the README. Quick check
  for step 7: `per_class_f1_cv(model_factory, features.core_only(df), labels, seed=...)`.
- **§7 asks for per-class precision/recall, a confusion matrix, and a coverage-vs-accuracy
  curve** — `evaluate.py` currently only computes per-class **F1** (`per_class_f1_cv`), not
  precision/recall separately, and has no confusion-matrix aggregation across folds yet.
  Both are step-11 (`evaluation_report.md`) work, not done here — flagging so step 11 doesn't
  assume `evaluate.py` already has everything §7 lists.
- **§6's confidence/margin table is still stale** (carried over from the previous handoff —
  unrelated to this session's changes, still needs regenerating once calibration exists).

## Known gaps — do not mistake for finished work

- ~~No CV/holdout evaluation exists for this exact pipeline~~ — resolved this session:
  `src/evaluate.py` has the repeated stratified CV harness, and `src/run.py` still fits on
  all 44 labelled rows for the actual `predictions.csv` output (correct — CV is for
  evaluation, not for shrinking the training set of the shipped model).
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
