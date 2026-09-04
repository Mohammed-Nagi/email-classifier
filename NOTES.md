# Handoff notes

Context that's expensive to reconstruct from the commits/code alone, organized by topic
rather than by session. PLAN.md §9 steps 1–6 are done (ingest, recon, TF-IDF+LR baseline,
CV harness, ablation harness). Read this before continuing at step 7.

For §2/§4's actual CV/ablation numbers, **PLAN.md is the source of truth** — this file cites
them rather than restating them, so the two documents don't drift apart.

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
  the canonical representation. This is deliberate: `features.py`'s ten variant builders
  need `body_paragraphs` decomposed to build stripped variants cheaply — that's cheap only
  because `ingest.py` never flattened the structure away. See the variant-semantics table
  below for exactly what each builder does with it.

- **Fail loudly, no fallback paths.** `parse_email_file` raises `ValueError` on any missing
  field, missing `.email-body`, missing inner `<title>`, or empty paragraph list. No
  malformed file exists in the 56 provided (verified in `scripts/recon.py` and asserted
  across all of them in tests), so there was no real case to design graceful degradation
  for. If real malformed input ever shows up, it should surface immediately, not produce a
  silently-wrong row.

- **`models/base.py` was deliberately deferred past ingest/recon/baseline, then built once
  the CV harness needed it.** PLAN.md's target architecture has a swappable
  fit/predict_proba interface from the start; the first three build steps shipped one
  concrete pipeline instead (an interface for a single arm would have been speculative).
  Built when `evaluate.py`'s CV harness needed a `model_factory: () -> ClassifierModel`
  contract to build a fresh unfitted model per fold — not because a second arm existed yet
  (it doesn't; embeddings is still step 7), but because that contract *is* the
  fit/predict_proba interface PLAN.md wanted. The old `src/model.py` no longer exists —
  it's `src/models/tfidf_lr.py` (`TfidfLRModel`) behind `src/models/base.py`.

- **`recon.py` deliberately excludes §2/§4 numbers.** It reproduces only §3's data-only
  facts (counts, structure, lengths, sender noise, taxonomy gap) plus one extension (the
  near-duplicate check, see below). CV accuracy, macro-F1, and the ablation table are
  model-evaluation results that belong in `evaluate.py` — computing them twice in two
  different scripts would just create a second place for them to drift out of sync.

- **Confidence score is calibrated (sigmoid/Platt), as of step 9.** `run.py` ships
  `CalibratedTfidfLRModel`, not the raw pipeline — see "Calibration (step 9)" below for the
  measured before/after and why the raw model turned out to be underconfident, not
  overconfident as PLAN.md §6 originally assumed.

- **`sender` and `date_received` are captured but not fed to the model.** Only
  `features.full_text()` (subject + body_text) goes into TF-IDF. This was the correct
  default given the sender trap-feature finding (below), but it was never an explicit
  filtering decision — it's just that no `features.py` builder references those columns.
  If a later session adds sender/date features, that needs the deliberate "measure, then
  exclude, with evidence" treatment PLAN.md calls for in §3, not a silent inclusion.

## Investigated and deferred: the embeddings arm (step 7)

**Decision: built the diagnostic PLAN.md §5 called for, then deferred the arm itself on
time-budget grounds — not because the diagnostic came back negative.**

The check: run `per_class_f1_cv` on `core_only` and `subject_only` (never done before this
session — §2's diagnosis had only ever been run on `full_text`). Full numbers, all five
ablation rungs, in `outputs/ablation_per_class.csv` (`run_ablation_per_class` +
`macro_f1_excluding_class`, `src/evaluate.py`):

| Variant | Other F1 (std, min) | Loan Processing F1 (std, min) | macro-F1 excl. Other |
|---|---|---|---|
| full_text | 0.727 (±0.424, min 0.000) | 0.956 (±0.112, min 0.667) | 0.980 |
| no_title | 0.467 (±0.481, min 0.000) | 0.960 (±0.109, min 0.667) | 0.967 |
| greeting_and_core | 0.440 (±0.477, min 0.000) | 1.000 (±0.000, min 1.000) | 0.969 |
| core_only | 0.047 (±0.191, min 0.000) | 0.672 (±0.421, min 0.000) | 0.831 |
| subject_only | 0.000 (±0.000, min 0.000) | 0.665 (±0.402, min 0.000) | 0.837 |

**Result: the headroom premise holds, but not for the reason originally assumed.** `Other`
doesn't just stay noisy under stripping, it collapses toward zero — worse than PLAN.md §5
anticipated. And a second n=6 class, Loan Processing, newly destabilises at the two most-
stripped rungs (min F1 0.000 at both), something §2's `full_text`-only diagnosis had no way
to see. So a **flat 5-way macro-F1** comparison between arms on `core_only`/`subject_only`
would have been an even worse coin-flip than on `full_text` — contaminated by two unstable
n=6 classes instead of one. But the `Other`-excluded macro-F1 (last column above) still
degrades genuinely and monotonically — 0.980 → 0.831 — independent of either small class.
That's real, decontaminated headroom on the four saturated categories. See PLAN.md §2 for
the two-mechanism read this produced (small-n fragility vs. semantic-incoherence fragility —
`Other` has both, Loan Processing only the first) and §4 for the ablation-table fix this
forced (report `Other`-excluded macro-F1 alongside flat, not flat alone).

**Given that the premise held, the arm was still cut — deliberately, not as a fallback:**
1. Confirmatory, not load-bearing: PLAN.md §5 already argues model choice is an engineering
   decision once accuracy saturates; the arm would have strengthened, not changed, that.
2. The brief's actual required confidence score is not yet meaningful (see below — 0.26–0.49,
   no separation) and calibration → routing → explainability is the chain that fixes that.
   That chain doesn't fit the remaining time budget alongside a new model arm.
3. `sentence-transformers` pulls in `torch` — a real, one-sided dependency cost (~0.5–1.5GB
   install, ~90MB model download on first run) that would have needed a second requirements
   file and a guarded import to keep out of the required single-command path, adding exactly
   the kind of conditional complexity the rest of this project avoids.

If a future session revisits this: `src/models/embed_lr.py` doesn't exist yet, but the plan
was to implement `ClassifierModel` with a module-level cached `SentenceTransformer` singleton
(load weights once, not per fold — nothing about the encoder is *fit* to this data, so
sharing it across folds isn't a leakage risk the way sharing a fitted `TfidfVectorizer`
would be) and an `LogisticRegression` head, run through the same `FEATURE_VARIANTS`
builders. Scope it to `full_text` (parity check) + `core_only` (the rung with the largest
decontaminated headroom) — `subject_only` tells nearly the same story as `core_only`
(0.837 vs 0.831) and the six perturbation variants test TF-IDF's specific token-exact-match
weakness, not the scaffolding-dependence question an embeddings arm would answer.

## Calibration (step 9): the underconfidence finding, and why run.py switched

**The raw model isn't overconfident — it's badly underconfident, the opposite of what
PLAN.md §6 originally assumed before this was measured.** Pooled out-of-fold across the
repeated CV harness (5-fold × 10 repeats, `calibration_cv` in `src/calibrate.py`): raw mean
confidence is **0.35** against **96%** actual pooled accuracy. Every raw confidence for the
12 real test emails sits in a **0.26–0.49** band regardless of how easy the email actually
was — this is *why* a routing threshold (step 10) against raw confidence would have been
close to meaningless: there's no usable spread to threshold against. `Brier 0.544, ECE
0.613` raw.

Sigmoid (Platt) calibration (`CalibratedTfidfLRModel`) substantially fixes this: `Brier
0.215, ECE 0.359`, mean confidence 0.62 against 97% pooled accuracy. On the actual 12 test
emails, calibrated confidence spreads 0.43–0.79 (was 0.26–0.49) and **0 of the 12
predicted categories change** — calibration rescaled the numbers without touching the
decisions on the emails that matter. The paired check across the full 50-fold CV shows the
same thing at the aggregate level: 11/440 pooled
predictions (2.5%) flip label, and the macro-F1 delta (calibrated − raw) is +0.022 ± 0.063
— inside fold-to-fold noise, i.e. no reliable accuracy change either way. ECE improves a
lot but doesn't reach zero (0.359 remains) — expected and stated up front in PLAN.md §6:
calibrating on 44 points with a 6-example class is thin, and this is the measured
confirmation of that caveat, not a new problem.

**Decision: `run.py` now ships `CalibratedTfidfLRModel`, not the raw pipeline**, made this
session rather than deferred to step 10 as originally planned — because step 10's routing
threshold is only as meaningful as the probability scale it operates on, and raw confidence
had no usable spread to set a threshold against. The evidence above (0 flips on the real
test set, macro-F1 change within noise, large Brier/ECE improvement) was the basis; see
PLAN.md §6 for the full before/after table.

**Why `cv=3`, not 5, inside `CalibratedClassifierCV`** (also documented as a docstring in
`src/calibrate.py`, repeated here per instruction — this is the kind of parameter that looks
arbitrary a session later without the reasoning attached): `CalibratedTfidfLRModel` is
evaluated inside the same 5-fold *outer* CV harness the raw model is. That outer split
already removes ~20% of the 44 examples into a held-out test fold, leaving as few as 4-5
examples of the smallest classes (Other/Loan Processing, n=6 each) in the *training* fold
that `CalibratedClassifierCV` then splits again internally for its own calibration-vs-fit
separation. `cv=5` there needs 5 examples of every class in that inner split and fails
outright on some outer folds ("n_splits=5 cannot be greater than the number of members in
each class"). `cv=3` is the largest inner split that survives every outer fold this model is
actually evaluated with — chosen empirically against the real fold structure, not a default.

**Why Brier/ECE are computed on *pooled out-of-fold* predictions, not per-fold or in-sample**
(also worth keeping, same reason): computing them on training-fit probabilities would be
badly optimistic (the model has already seen those labels). Computing them per-fold would
put ~9 held-out examples into ECE's 10 bins — too few for a bin to mean anything. Pooling
every repeat's held-out predictions across the full repeated CV gives 44 examples × 10
repeats = 440 points, which is what `calibration_cv` returns before deriving Brier/ECE from
it. This mirrors `evaluate.py`'s existing repeated-CV pattern but keeps raw predictions
instead of throwing them away the way `repeated_stratified_cv` does (see "CV harness
gotchas" below).

**The paired comparison (`paired_calibration_comparison`) is structurally paired, not just
same-seed paired.** Unlike the embeddings-arm headroom check above (which relied on
"same seed ⇒ same fold membership" across two separate `per_class_f1_cv` calls), this
function fits both the raw and calibrated model inside the *same* loop iteration, from the
*same* `splitter.split()` call — the pairing doesn't depend on trusting that two separate CV
invocations happened to get identical folds, it's guaranteed by construction. Worth using
this pattern again for any future "did X change predictions, not just a score" question,
rather than reaching for two independent CV calls plus the same-seed argument.

## Exact text assembly (ingest.py → features.py)

The contract every `features.py` variant is built on top of, unambiguously:

1. `_unwrap_body()` extracts `inner_title` via `title_tag.get_text(strip=True)` (one tag,
   one string).
2. Each `<p>` in the inner doc is extracted independently via
   `p.get_text(separator=" ", strip=True)`; empty results are filtered out. Order is
   document order. **Confirmed, not assumed:** every one of the 56 provided files (44
   train + 12 test) has exactly 3 paragraphs, in order `(greeting, core, signature)` — this
   was checked before writing `core_only`/`greeting_and_core`, not trusted. See the
   variant-semantics table below for the confirmed per-variant detail.
3. `EmailRecord.body_text = " ".join((inner_title, *body_paragraphs))` — single space
   between every element, inner_title first, then paragraphs in order. No newlines.
4. `features.full_text()` = `df["subject"] + " " + df["body_text"]` — final string fed to
   `TfidfVectorizer` is `subject, inner_title, paragraph_0, ..., paragraph_n`, all
   single-space-joined.
5. Nothing dropped is silently dropped: the outer document chrome (outer `<title>`,
   `<h1>`, the header `<p>` duplicating subject) is never extracted in the first place —
   only content inside `.email-body` and the `data-field` divs is read.
6. No lowercasing/stopword-removal/punctuation-stripping happens in `ingest.py`. That's
   entirely `TfidfVectorizer`'s job downstream (`stop_words: english` in config.yaml).
   `ingest.py`'s contract stops at "clean structured text out of HTML."

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
two real words, in every row. It would specifically have undermined the ablation harness:
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
  signal sitting right at the start of the body text — see PLAN.md §4 for the measured drop
  once it's stripped.
- **Baseline confidence scores were uniformly low — RESOLVED, step 9.** All 12 raw-model
  test predictions landed in 0.26–0.49, none above 0.5, even for unambiguous emails. Flagged
  here as expected-but-a-problem for routing; confirmed exactly right once measured
  (mean confidence 0.35 vs. 96% actual accuracy — underconfidence, not the overconfidence
  PLAN.md §6 assumed) and fixed by calibration. See "Calibration (step 9)" above for the
  numbers; this entry is kept as the prediction that turned out correct, not a live gap.
- **`test/email_4.html` (internal id 53), the fraud-report taxonomy gap, is confirmed** —
  no category fits a fraud report, exactly as PLAN.md §3 argues. Its current measured
  confidence is tracked in PLAN.md §6 (marked stale there pending recalibration) — not
  restated here to avoid the two files drifting.
- **Python 3.14.4 was the only interpreter available on this machine.** `requirements.txt`
  is pinned to versions that resolved cleanly against it (see commit history), but nobody
  has verified this on an older Python. Worth a fresh-clone test on whatever Python version
  is actually common before submission (PLAN.md step 13).

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

That's what makes these two comparisons valid (see PLAN.md §4 for the exact figures):

- `generic_salutation` vs. `full_text`: swapping the greeting while title/subject are still
  present costs next to nothing.
- `greeting_and_core` vs. `core_only`: removing the greeting once title/subject are already
  gone costs real macro-F1.

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
- **`std` is sample std (`ddof=1`)**, not population std — matters if PLAN.md's numbers
  ever need re-checking against a `ddof=0` computation (unlikely to matter at n=50 folds,
  but worth knowing which one this codebase uses).
- `_DummyModel` wraps `sklearn.dummy.DummyClassifier(strategy="most_frequent")` — it has no
  `config` dependency, so `count_perfect_single_fold_runs(_DummyModel, ...)` works directly
  (see `test_dummy_model_never_scores_a_perfect_single_fold_run`).
- **`run_ablation_per_class` + `macro_f1_excluding_class`** (added this session) refit the
  ablation ladder a second time via `per_class_f1_cv` — same folds (same seed), but this is
  a second set of fits, not a reuse of `run_ablation`'s. If the ablation ladder ever grows
  past 5 variants and `python -m src.evaluate`'s runtime becomes annoying, this is the first
  place to look for caching, per the "no caching, no shared fits" gotcha above.
  `outputs/ablation_per_class.csv` is long-format (one row per variant × category) — pivot
  or `groupby` it rather than reading a wide table.

## Where PLAN.md is wrong, underspecified, or worth doing differently

- **The unwrap's hardest part — parser choice and the separator hazard — isn't mentioned
  at all**, despite PLAN.md flagging the unwrap generally as "easy to get subtly wrong."
  This was the single biggest gap between PLAN.md's stated risk level and the guidance it
  gave for avoiding it.
- **§3's body-length figures (127–204, median 170) don't state what was measured.** It
  turns out `body_text` (inner_title + paragraphs, joined) reproduces PLAN.md's numbers
  exactly — confirmed in `scripts/recon.py`. Worth stating explicitly for whoever reads
  PLAN.md next, since "body only" is ambiguous between several plausible definitions.
- **Every number in PLAN.md traces back to the same throwaway probe, and needed the same
  re-derivation treatment every time it was checked.** The 0.26 near-duplicate figure was
  off (corrected in PLAN.md §3 to 0.245). §2's headline CV score and §4's ablation table
  were also materially wrong when re-derived in code — see PLAN.md §2/§4 for the corrected
  figures and full diagnosis (not restated here, to keep one home for those numbers). §6's
  confidence/margin table is the one remaining unverified block — PLAN.md §6 now carries
  its own STALE marker in place, with the confirmed `email_4` discrepancy.
- **§5's "real headroom to separate arms" claim on stripped conditions has been checked and
  confirmed** — see "Investigated and deferred: the embeddings arm" above. The headroom is
  real once `Other` is excluded, but flat macro-F1 on `core_only`/`subject_only` is worse
  contaminated than §5 originally worried (a second n=6 class, Loan Processing, collapses
  too) — this reshaped §2's diagnosis and §4's ablation table, not just §5.

## Known gaps — do not mistake for finished work

- `config.yaml`'s model hyperparameters (`max_features`, `ngram_range`, `C`,
  `class_weight`) are untuned, unvalidated defaults — documented as "sane defaults" in a
  comment, nothing more.
- **`evaluate.py` doesn't yet cover everything PLAN.md §7 asks for.** It computes per-class
  **F1** (`per_class_f1_cv`) but not precision/recall separately, and has no
  confusion-matrix aggregation across folds. Both are step-11 (`evaluation_report.md`)
  work — flagging so step 11 doesn't assume `evaluate.py` already has everything §7 lists.
- `Other` is handled as a plain 5th label only (strategy 1 of 3 in PLAN.md §6) — abstain-as-
  Other and the hybrid gate are not implemented or compared (step 8).
- No routing/threshold/abstain logic, no `needs_review` column, no `--auto-route-threshold`.
- No explainability / token attribution (`explain.py` doesn't exist).
- No embeddings arm (investigated and deliberately deferred, see above — not an oversight),
  no zero-shot NLI arm (always optional, never started).
- No README.
- `lxml` is installed in `.venv` (from the parser investigation) but is **not** in
  `requirements.txt` and nothing shipped imports it — intentional, but don't be surprised
  it's in a local `pip freeze`.
- Git remote is set: `https://github.com/Mohammed-Nagi/email-classifier.git`, pushed to
  `master`, tracking `origin/master`.
