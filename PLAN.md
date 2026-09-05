# Email Classification Challenge — Implementation Plan

Working spec for the Isazi take-home. Read this first in the Claude Code session.

---

## 1. The task (from the brief)

RedRock, a financial services company, needs incoming client emails routed to the right
department. Misrouting causes delays and regulatory penalties.

Required deliverables:

1. **Ingestion** — read the email folder + `train_labels.csv`; extract and preprocess body
   and metadata (subject, etc.).
2. **Modelling** — classify into 5 categories: `Account Management`, `Investment Advisory`,
   `Loan Processing`, `Insurance Claims`, `Other`. Must output **a predicted category AND a
   confidence score** per email.
3. **Reporting** — CSV with columns `email_id, predicted_category, confidence_score`.
4. **Written analysis** — (a) how you would measure success, and (b) what additional
   features/data sources would raise accuracy for regulatory compliance.
5. **Single-command end-to-end run.** Open-source tools only.

Stated evaluation criteria: technical quality, architecture, code quality, problem-solving,
maintainability, overall engineering approach.

---

## 2. The finding that shapes everything

*(§2 and §4 were rewritten against `src/evaluate.py`'s actual repeated-CV harness. The
original figures here came from a throwaway probe and did not reproduce — see NOTES.md for
the full diagnostic (variant semantics, isolation methodology, verification steps). Every
number in this section is re-derivable by running `python -m src.evaluate`; NOTES.md cites
these numbers rather than restating them, to avoid the two files drifting apart.)*

**Four of the five categories are saturated; the fifth is not, and averaging them together
produces a headline number that describes neither.** Measured with 5-fold × 10-repeat
stratified CV on the actual `TfidfLRModel` + `features.full_text` pipeline (`src/evaluate.py`):

| | macro-F1 |
|---|---|
| Insurance Claims | 1.000 ± 0.000 (min 1.000) |
| Investment Advisory | 0.988 ± 0.048 (min 0.800) |
| Account Management | 0.976 ± 0.056 (min 0.800) |
| Loan Processing | 0.956 ± 0.112 (min 0.667) |
| **Other** | **0.727 ± 0.424 (min 0.000)** |
| **All 5, flat softmax (headline)** | **0.929 ± 0.108 (min 0.731, max 1.000)** |
| All 4 real categories, `Other` excluded | 0.980 ± 0.031 (min 0.914) |

`Other` is a residual class (HR replies, IT notices, marketing — no topic coherence, see
below) with only 6 training examples. Folded into one 5-way softmax, its instability drags
the headline macro-F1 down and its variance up; on its own it accounts for essentially all
of the spread — some folds get **zero** F1 on `Other` entirely. Exclude it and the four
substantive categories are genuinely near-saturated at 0.980 ± 0.031, closely matching what
the original probe reported as the headline number. The four real categories *are*
keyword-separable — mean nearest-neighbour cosine similarity within train is only 0.245
(re-derived in code; see §3), so this is not near-duplicate leakage, just short,
template-generated text where the category words appear near-verbatim ("applying for a
personal loan of $25,000", "Dear Loan Officer").

**A sharper, verified version of this claim: the instability has two separate causes, not
one, and `Other` is worst because it has both.** §4's per-class ablation breakdown
(`outputs/ablation_per_class.csv`) shows Loan Processing — also n=6, but topically coherent
— stays perfectly stable through moderate stripping (F1 0.956 on `full_text`, a clean
**1.000 ± 0.000** on `greeting_and_core`) and only collapses to min-F1-zero at the two most
stripped rungs, `core_only` (0.672 ± 0.421, min 0.000) and `subject_only` (0.665 ± 0.402,
min 0.000). `Other` destabilises immediately, already unstable at the mildest stripping
(`no_title`: 0.467 ± 0.481, min 0.000) and collapsing further as stripping increases (0.047
at `core_only`, exactly **0.000 ± 0.000** at `subject_only`). So: **small n (6) predicts
fragility once enough information is stripped that lexical cues run out** — both n=6
classes eventually hit min F1 = 0 — but **semantic incoherence (`Other` specifically)
predicts fragility to any information loss at all**, because unlike Loan Processing it has
no topic to fall back on even when title, greeting, and subject are all still present. This
also means flat macro-F1 is not a safe metric for the ablation ladder in §4 once stripping
is heavy — see there for the decontaminated figure.

**A single CV run would not have shown this.** Across 50 independent single (non-repeated)
stratified 5-fold splits, **0 of 50** scored a perfect macro-F1 — every one of them landed a
few `Other` examples awkwardly and dipped. A report of "5-fold CV: macro-F1 1.000" from this
exact pipeline is not a lucky outlier away from being reproducible; it does not reproduce at
all in 50 tries. That is the strongest evidence in this submission for the methodology
argument below: a single CV run, run once and reported, is not a safe way to evaluate a
5-way classifier at n=44 with a 6-example residual class in it — repeats are not a nicety,
they are what makes the instability visible at all.

### What follows from this

**Accuracy on the flat 5-way task is not a stable number, and the instability has a name:
`Other`.** There is no headroom for a heavier model to compete for on the four saturated
categories — a zero-shot cross-encoder cannot beat 0.98, and any comparison between models
on `full_text` will be decided by `Other`-driven noise, not signal. An earlier draft of this
plan proposed a three-arm model bake-off as the technical centrepiece; the measurements
above make that centrepiece hollow.

**This is also the argument for §6's `Other` strategy, not a separate observation.** The
exact class destabilising the headline CV score is the one class where the label
`Other` semantically means "none of the four departments" rather than a topic to be learned
from 6 examples. Abstain-as-`Other` (§6, strategy 2) is not just "semantically honest" — it
is the direct fix for the specific instability measured here.

**The submission's value must come from everything that happens after the headline number is
understood, not from the headline number itself:**

1. Recognising that a single-run "perfect" CV score would not be a real result even if one
   had been produced — it doesn't reproduce (0/50 seeds, above) — and diagnosing *why* the
   flat 5-way score is unstable instead of stopping at "it's high."
2. Building a **harder evaluation** than the one provided, since the provided one — a
   single CV run, reported once — is **unstable, not merely uninformative**: it can land
   anywhere from 0.73 to 1.00 depending on which fold `Other`'s six examples fall into.
3. Justifying model choice on **engineering grounds** (determinism, auditability, latency,
   zero-download) once the four real categories have stopped discriminating between arms.
4. A **calibrated confidence and abstain policy** that demonstrably fires on the genuinely
   ambiguous cases — and, per the finding above, directly addresses `Other`'s instability
   rather than being "semantically honest" alone.
5. **Per-prediction explainability** — an audit trail, which is what a regulated deployment
   actually requires.

This reframing *is* the submission. Lead the README with it.

---

## 3. Data reconnaissance

Derived from the provided files with a throwaway probe. **Re-derive every number in code —
no README claim should rest on a figure neither you nor the tests re-confirmed.**

| Fact | Value | Implication |
|---|---|---|
| Train / test | 44 labelled / 12 unlabelled | Test set far too small to evaluate on. Say so. |
| Class balance | AM 13, IA 12, IC 7, Other 6, LP 6 | Mild imbalance; stratify all CV. |
| Body length | 127–204 chars (median 170) | Very short. |
| Structure | `data-field` divs: `email_id`, `subject`, `sender`, `date_received`; plus `.email-body` | Parseable with BeautifulSoup. |
| **Nested HTML body** | `.email-body` contains a *second* full `<!DOCTYPE html>` document | Must unwrap twice. Easy to get subtly wrong. |
| **`email_id` ≠ filename** | True for **all 56 files** (`train/email_1.html` → internal id 31) | Structural, not a data error. Affects the output column. |
| Sender field | Noisy: HR-topic mail from `security@redrock.com`; IT notice from a `gmail.com` address | Trap feature. Measure, then exclude, with evidence. |
| **Inner `<title>`** | Each body opens with a topic restatement ("Account Transfer", "Pet Insurance") | Generation artifact. Inflates scores — see ablation below. |
| **Salutations name the department** | "Dear Loan Officer", "Dear Pet Insurance Claims", "Dear Account Services" | The label is partly *written into* the greeting. Real clients do not reliably do this. |
| Near-duplicate check | Mean nearest-neighbour cosine similarity within train: **0.245** (re-derived in `scripts/recon.py`; corrected from an earlier throwaway-probe figure of 0.26, likely a differently-fitted `TfidfVectorizer` — see `NOTES.md`) | Rules out near-duplicate leakage as the explanation for the saturated CV score in §2; the task is keyword-separable, not duplicated. |

### Decision: which `email_id` goes in the output CSV — SETTLED

Emit the **internal `email_id`** as the `email_id` column, plus **`source_filename`** as an
additional column. The internal id is the document's own unique identifier and is what a
downstream system would key on; the filename is a packaging artifact. Emitting both makes
the mapping unambiguous. Assert the mismatch in `test_ingest.py`.

### `Other` is a residual class, not a topic

It covers HR replies, IT notices, and marketing in train. No semantic centroid. It is
structurally hostile to zero-shot NLI (there is no coherent "this email is about Other"
hypothesis) and thin for supervised learning at 6 examples. Handle it deliberately —
see §6.

### Taxonomy gap — a genuine business finding

Test `email_4.html` ("Account Freeze Request") reads: *"Dear Security Department, I suspect
fraudulent activity on my account. Please freeze my account immediately."* **None of the
five categories fit a fraud report.** The baseline assigns it to Account Management, not
`Other` (current confidence figure: §6, marked stale pending regeneration). In a regulated
setting a suspected-fraud notification is plausibly the
highest-stakes item in the whole inbox, and the taxonomy has nowhere to put it.

Raise this explicitly in the written analysis. Noticing that the label schema is
incomplete — rather than silently forcing a prediction — is exactly the judgement the
compliance framing is testing.

---

## 4. Ablation / stress testing — the technical centrepiece

Since the provided evaluation cannot separate a good model from a lucky split (§2), **build
a harder one.** Progressively strip the structural artifacts and measure degradation.
Re-derived from `src/features.py` + `src/evaluate.py` (`python -m src.evaluate`), 5-fold ×
10 repeats, on `TfidfLRModel`:

| Input | flat macro-F1 (all 5 classes) | macro-F1, `Other` excluded (4 real categories) |
|---|---|---|
| subject + title + full body | 0.929 ± 0.108 (min 0.731, max 1.000) | 0.980 |
| subject + body, inner `<title>` removed | 0.867 ± 0.118 | 0.967 |
| greeting + core sentence only (no subject, title, or signature) | 0.863 ± 0.114 | 0.969 |
| core sentence only (no subject, title, greeting, or signature) | 0.674 ± 0.118 | **0.831** |
| subject only | 0.670 ± 0.096 | 0.837 |

**Report both columns — the flat column is contaminated once stripping is heavy.** §2's
refined diagnosis explains why: at the two most-stripped rungs, `Other` *and* Loan
Processing (both n=6) collapse toward zero, dragging the 5-class average down regardless of
what happens to the four saturated categories. The `Other`-excluded column isolates that:
the real, decontaminated effect on the four saturated categories is **0.980 → 0.831, a 0.15
drop** — smaller than the flat column's 0.26, but still substantial and not an artifact of
the residual class. Both numbers are true; the flat one just isn't the one that isolates
"does stripping structure hurt the four real categories," which is the question this ladder
exists to answer. Full per-class breakdown, all five rungs: `outputs/ablation_per_class.csv`
(generated by `python -m src.evaluate`, backed by `run_ablation_per_class` /
`macro_f1_excluding_class` in `src/evaluate.py`).

**A genuine feature-interaction finding fell out of building that intermediate rung.** The
greeting is the giveaway PLAN.md §3 already flags ("Dear Loan Officer" names the
department outright), but how much it matters depends on what else is present:

- With title/subject still in the text, **swapping the greeting for a generic one costs
  almost nothing** — `generic_salutation` below scores 0.920 ± 0.109 against `full_text`'s
  0.929 ± 0.108, a 0.009 difference.
- With title/subject already stripped, **removing the greeting outright costs 0.19** —
  `greeting_and_core` (0.863) vs. `core_only` (0.674).

The department-naming greeting is redundant leakage while richer artifacts are present, and
becomes load-bearing leakage once they're gone. A single ablation number would have missed
this; the intermediate rung is why it didn't.

**The perturbations below tell a different, and more useful, story than "the model is
fragile."** It is not fragile to noise — it is dependent on scaffolding:

| Perturbation | macro-F1 | Δ vs. full_text |
|---|---|---|
| `no_subject` (drop subject, keep title+body) | 0.948 ± 0.092 | +0.019 |
| `distractor_text` (append quoted-reply + disclaimer boilerplate) | 0.934 ± 0.104 | +0.005 |
| `synonym_substitution` (swap department keywords for synonyms) | 0.927 ± 0.107 | −0.002 |
| `generic_salutation` (replace greeting with "Dear Sir/Madam,") | 0.920 ± 0.109 | −0.009 |
| `truncated_first_sentence` (subject + greeting + first sentence only) | 0.907 ± 0.113 | −0.022 |
| `typo_noise` (~15% of words character-transposed) | 0.902 ± 0.118 | −0.027 |

Every perturbation that *adds* noise, corrupts individual words, or swaps individual
phrases costs under 0.03 macro-F1 — within fold-to-fold variance, i.e. indistinguishable
from no perturbation at all. Only *removing structural content* (title, greeting/signature,
subject) causes real degradation (§4 ablation table, 0.26 drop). **The model is robust to
messy input and dependent on structural scaffolding** — a different, more specific claim
than "fragile to real-world conditions," and the one the data actually supports.

This ablation table, the greeting-interaction finding, and the 0/50-seeds reproducibility
result (§2) are the most defensible artefacts in the submission: together they show the
provided evaluation could not distinguish a good solution from a lucky one, and this one
can, with a named mechanism for why.

---

## 5. Modelling

**Model selection cannot be decided by accuracy here.** Say this plainly, then decide on
engineering grounds instead.

### Ship: TF-IDF + Logistic Regression

Justify on the criteria that still discriminate once accuracy does not:

- **Deterministic and reproducible** — a reviewer re-running gets identical output.
- **No model download, no network, no API key** — "we can run your solution" is a stated
  requirement.
- **Millisecond inference** — relevant at "thousands of emails per day".
- **Directly auditable** — per-class coefficients give token-level attribution for every
  prediction. In a regulated setting, being able to answer "why was this routed here?" is a
  requirement, not a nicety.
- **Native, usable probabilities** for the required confidence score.

### Contrasting arm (sentence embeddings + LR) — investigated, deferred

The plan was to run **sentence embeddings + LR** (`all-MiniLM-L6-v2`) through the identical
CV and ablation harness, not to win but to show the saturation ceiling is a property of the
*data*, not of one model family. The headroom premise this depended on was checked first, as
planned: §2's refined diagnosis confirms real, `Other`-independent headroom exists on
stripped conditions (0.980 → 0.831 macro-F1 excluding `Other`, §4) — so the arm would have
had a real question to answer, and this was not the reason it was cut.

**Deferred anyway, on time-budget grounds, once the check came back positive:**

- The arm is confirmatory, not load-bearing. This section already argues model choice is
  decided on engineering grounds (determinism, no download, auditability) once accuracy
  saturates — the embeddings arm would strengthen that argument without changing it.
- The brief's actual required deliverable — a **meaningful confidence score** — is not yet
  satisfied: NOTES.md records the current uncalibrated confidences sitting at 0.26–0.49 with
  no separation between confident and uncertain predictions. Calibration → routing →
  explainability is the chain that satisfies the brief and carries the regulated-deployment
  framing; that's where the remaining time went instead.
- The dependency cost is real and one-sided: `sentence-transformers` pulls in `torch`
  (~0.5–1.5GB install, plus a ~90MB model download on first run). Isolating that behind a
  second requirements file and a guarded import — so a reviewer who just wants
  `predictions.csv` doesn't pay for it — is exactly the kind of conditional complexity this
  project otherwise avoids.

This is a scope decision, recorded here and in NOTES.md with the diagnostic that justifies
it, not a gap. See NOTES.md for the full per-class numbers behind the headroom check.

Optionally add **zero-shot NLI** as a third arm *only if time permits after §6–§7 are done*.
If included, use `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, not `facebook/bart-large-mnli`
— the latter is the canonical reference baseline rather than the strongest option, and
cross-encoders trained on more diverse NLI corpora outperform it by several macro-F1 points.
Note that `-c` variants are trained only on commercially-licensed data, which is a relevant
distinction for a financial client.

### Excluded, with reasons for the README

- **Fine-tuning a transformer** — 44 examples cannot support it; the result would be
  unstable and unreportable.
- **A pretrained classifier with a foreign label space** — forcing a mapping onto RedRock's
  five categories adds an opaque, undebuggable translation layer.
- **LLM API classifier** — non-deterministic and needs a key, conflicting with the
  reproducibility requirement. Explaining why it was excluded is worth more than including it.

---

## 6. Confidence, `Other`, and routing

**Regenerated, step 10 — no longer stale.** Full 12-row table in `outputs/predictions.csv`
(`python -m src.run`); the four cases originally singled out here, with real calibrated
confidence and margin:

| Email | Subject | Conf | Margin | `needs_review` (t=0.45) | Situation |
|---|---|---|---|---|---|
| `email_6` | New Product Announcement | 0.44 | 0.16 | **True** | Marketing — Other vs Account Management |
| `email_1` | Financial Education Workshop | 0.43 | 0.16 | **True** | Event marketing vs advisory |
| `email_9` | Market Research Request | 0.57 | 0.43 | False | Borderline, but above threshold |
| `email_4` | Account Freeze Request | 0.68 | 0.48 | **False** | Fraud report — taxonomy gap (§3) |

**The design validates on the two genuinely ambiguous marketing emails — `email_6` and
`email_1` are exactly the two the gate flags for review, out of 12.** That's the mechanism
working as intended: real, empirical evidence the abstain gate fires on the right cases,
not an assertion that it should.

**`email_4` is the mechanism's honest limitation, not a hidden failure — say so in the
README, don't bury it.** The fraud report is *not* flagged for review, and its
top-contributing token is unambiguous about why: `account (+0.61)` — "freeze my account" is
lexically almost indistinguishable from a routine Account Management request, so the model
is confidently wrong, not uncertain. A confidence gate only catches what the model is
unsure about; it cannot catch a case where the model is sure and wrong for a structural
reason (the taxonomy has no category for this email at all — §3). This is exactly why §3
raises the missing fraud/security category as a business finding independent of the
modelling: no threshold tuning fixes a label space that's missing a label.

### Confidence score

1. Take the predicted-class probability.
2. **Calibrate it.** This section originally assumed raw LR probabilities on 44 samples
   would be *overconfident* — measured, that assumption was backwards: the raw model is
   **badly underconfident**. Pooled out-of-fold across the repeated CV harness (5-fold ×
   10 repeats, `python -m src.calibrate`), mean confidence is 0.35 against 96% actual
   accuracy — every raw confidence sits in a narrow 0.26–0.49 band regardless of how
   easy the email actually was (NOTES.md), which is exactly why a routing threshold
   against raw confidence would have been close to meaningless. Cross-validated sigmoid
   (Platt) calibration (`CalibratedTfidfLRModel`, `src/calibrate.py`) fixes this
   substantially:

   | | Brier (0–2, lower better) | ECE (0–1, lower better) | mean confidence | pooled accuracy |
   |---|---|---|---|---|
   | Raw | 0.544 | 0.613 | 0.35 | 0.96 |
   | Calibrated | 0.215 | 0.359 | 0.62 | 0.97 |

   **Paired check (same fold, same test examples, both models, per iteration — not a
   comparison of two means):** macro-F1 delta (calibrated − raw) is +0.022 ± 0.063 across
   50 folds — inside fold-to-fold noise, so calibration does not reliably help or hurt
   classification accuracy either way. Only 11 of 440 pooled predictions (2.5%) flip
   label at all, and **0 of the 12 actual test emails** flip category. Calibration is
   doing what it should: rescaling confidence to match reality without materially
   changing what gets predicted. **`run.py` now ships the calibrated model** on this
   evidence (see NOTES.md for the full reasoning).

   **Caveat honestly, as originally planned:** ECE improves a lot (0.613 → 0.359) but
   doesn't reach zero — calibrating on 44 points with 6 in the smallest class is
   statistically thin, exactly as anticipated, and this is the measured confirmation of
   that caveat rather than a hedge added after the fact. With production volume this
   would be re-fit properly.
3. Report **margin** (top-1 − top-2) as an extra column — for routing, "which two
   departments is it torn between" is more actionable than absolute confidence.
   **Done, step 10** (`src/routing.py`).

**On treating calibrated confidence as a ranking, not a probability for cost arithmetic.**
ECE 0.359 after calibration is a real, substantial improvement over raw (0.613) — but it
is not zero, and mean confidence (0.62) still trails pooled accuracy (0.97) by a wide
margin. Calibrated confidence is a genuinely useful *relative ordering* — the coverage/
accuracy curve below shows it separates hard cases from easy ones cleanly — but it is not
precise enough to support a claim like "this prediction is 68% likely to be correct,
multiply by the misrouting cost." The routing policy below is designed around that
distinction deliberately: the threshold is read off an empirical coverage/accuracy curve,
not derived from a formal expected-cost calculation.

**Connecting this to §4's ablation finding, for the README.** It's tempting to conclude
that calibrating on this corpus risks *baking in overconfidence* for messier real-world
input, since accuracy craters under stripping (§4: 0.98 → 0.83 four-class, `core_only`).
**Checked directly, not just reasoned about — and the result is the opposite of what that
argument predicts.** Fitting and calibrating on the training fold's `full_text`, then
evaluating on the *held-out* fold's `core_only` text (a genuine, non-leaking proxy for
"harder input arrives at inference time"): the accuracy–confidence gap **shrinks**, for
both models — raw 0.613→0.468, calibrated 0.359→0.241. Confidence tracks the accuracy drop
reasonably well here rather than lagging behind it; nothing inverts into overconfidence in
this test. The theoretical risk (a calibration curve fit on clean data could miscalibrate
on production input) remains real in principle and worth stating in the README as a
limitation — synthetic, single-topic training data is a poor proxy for real inbox
noise (§7) — but this specific, measurable check does not confirm it materialising, and
that null result belongs in the README next to the theoretical concern, not instead of it.

### `Other` strategy — implement and compare at least two

**Deferred (step 8), same time-budget reasoning as the embeddings arm** — see §5, §9.
Shipped: strategy 1 only.

1. **Normal fifth label** (supervised baseline). **Shipped.**
2. **Abstain-as-Other** — predict over the four real categories; if none clears threshold,
   assign `Other`. Semantically honest: "none of these departments" is what `Other` means.
   Not built.
3. **Hybrid** — four-way topic model plus a binary "is this routable business
   correspondence?" gate. Not built.

Strategy 2 would be the one to beat and aligns with the routing framing, if time permits a
revisit after step 12.

### Routing policy — done, step 10

Explicit **abstain / human-review band**: `confidence >= threshold` → auto-route;
`< threshold` → flag `needs_review` (`src/routing.py`). **The threshold is chosen from the
empirical coverage/accuracy curve, not a formal cost-weighted calculation** — calibrated
confidence is a much better ranking than raw but still measurably imperfect (ECE 0.359,
above), so treating it as precise enough for expected-cost arithmetic would overstate what
it actually is. `config.yaml`'s `routing.auto_route_threshold: 0.45` was picked because,
on pooled out-of-fold calibrated CV predictions (`python -m src.routing`): it auto-routes
89.1% of predictions at **100.0%** accuracy among them, while the reviewed 10.9% has only
75.0% accuracy — the gate concentrates the model's actual errors into the review queue
rather than passing them through, which is the property that matters, independent of
whether 0.45 is precisely the "right" number. Exposed as `--auto-route-threshold`; no
magic numbers in code.

### Explainability (regulated-context requirement) — done, step 10

For every prediction, `top_features` (`src/explain.py`) emits the **top contributing
tokens** from LR coefficients — cheap with a linear model, and it turns the output into an
audit trail. **Computed from a separate, uncalibrated fit on all 44 training examples, not
the shipped `CalibratedTfidfLRModel`'s own internals** — `CalibratedClassifierCV`'s 3
internal sub-fits (`cv=3`) hold substantially different TF-IDF vocabularies (measured: only
135 of 1018 union words shared across all three), so averaging their coefficients would
misrepresent the model rather than approximate it. A single fit on all 44 examples is both
simpler and faithful to the actual decision: calibration doesn't change the argmax for 0 of
the 12 real test predictions (above), so this explains the same decision the shipped model
made. State this explicitly in the README — it's a real design choice with a measured
justification, not an inconsistency to gloss over.

**Worked example for the README:** `email_4` (Account Freeze Request, the fraud-report
taxonomy gap from §3) is predicted Account Management with top features `account (+0.61),
request (+0.05), request account (+0.05), immediately (+0.02), transactions (+0.01)` — "my
account... immediately" reads, lexically, almost identically to a routine account-service
request. This is exactly why the confidence gate above doesn't flag it: the model isn't
uncertain, it's confidently applying the wrong (because missing) category. The explanation
makes that failure legible to a reviewer instead of hiding it behind a single number.

### Output contract — done, step 10

`email_id, predicted_category, confidence_score` kept exactly as specified, first and
correctly named. Additional columns, in order: `source_filename`, `margin`,
`runner_up_category`, `needs_review`, `top_features` (`src/run.py`, `OUTPUT_COLUMNS`).

---

## 7. Evaluation and written analysis

The 12 test emails are unlabelled and far too few to evaluate on — treating them as a test
set would be the methodological error the brief is likely watching for.

Evaluate on the 44 labelled emails with **stratified 5-fold CV, repeated**, reporting:
per-class precision/recall/F1; macro-F1 as headline; **fold-to-fold variance** (at n=44 the
spread matters as much as the mean); confusion matrix; the **ablation degradation table**
(§4); calibration quality; and a **coverage vs. accuracy curve** — accuracy on auto-routed
mail as a function of threshold. Note that with 44 points the curve has ~2% granularity;
present it as indicative.

### "How would you measure success"

- Offline: macro-F1 and per-class recall, cross-validated, with the fold-to-fold interval at
  n=44 acknowledged (never a single run) — broken out by class so `Other`'s instability
  doesn't hide inside the average (§2), and reported alongside the **stripped-artifact**
  condition (§4), not the unstripped headline alone.
- Deployment: **per-class precision at the operating threshold**, because misrouting cost is
  asymmetric — a misrouted fraud report is not a misrouted marketing email.
- Operational: auto-route coverage, human-review queue volume, and **downstream reassignment
  rate** (how often a department bounces an email back) — free ground truth accruing over time.
- Drift: category mix and confidence distribution tracked monthly; the sample spans
  Jan–Aug 2025, so drift and seasonality are concrete, not hypothetical.

### "What additional features/data would help"

- **A category for fraud/security** — the taxonomy gap found in `email_4` (§3). Lead with this.
- Thread history and prior emails from the same client — context beyond one 170-char message.
- CRM record: does the client hold a loan, a policy, an investment account? A strong prior.
- Attachment presence and type (claim form vs. statement vs. ID document).
- Verified sender identity — the current sender field is demonstrably unreliable; cite the
  measured evidence.
- Reassignment logs as continuously collected labels for retraining.
- **Realistic labelled data**, quantified: the current corpus is clean, single-topic and
  artifact-rich. Real mail is multi-topic, forwarded, truncated and mistyped — the ablation
  in §4 is a proxy for that gap, and closing it needs real samples, not more synthetic ones.

---

## 8. Repository structure

```
email-classifier/
├── README.md                   # THE submission document — finding first, then setup/run,
│                                 #   ablation, calibration/routing, both written answers,
│                                 #   investigated-and-deferred, limitations, next steps
├── PLAN.md                     # this file (optional to ship)
├── NOTES.md                    # session handoff notes (not shipped)
├── requirements.txt            # pinned
├── config.yaml                 # threshold, model names, paths — no magic numbers in code
├── data/                       # provided folder, unmodified
├── scripts/
│   └── recon.py                # throwaway §3 data-recon probe; not part of the shipped pipeline
├── src/
│   ├── ingest.py                # HTML parsing -> records (nested body, artifact-aware)
│   ├── features.py              # text assembly variants (full / stripped / core) for ablation
│   ├── models/
│   │   ├── base.py              # shared fit/predict_proba interface — arms are swappable
│   │   └── tfidf_lr.py          # shipped baseline arm (embeddings arm investigated, deferred — §5)
│   ├── evaluate.py              # CV, ablation harness, per-class/reproducibility diagnostics
│   ├── calibrate.py             # sigmoid calibration, Brier/ECE, paired raw-vs-calibrated check
│   ├── routing.py               # margin, runner-up, threshold gate, coverage/accuracy curve
│   ├── explain.py               # per-prediction token attribution (separate uncalibrated fit — §6)
│   └── run.py                   # single-command entrypoint — ships CalibratedTfidfLRModel,
│                                 #   full output contract (margin, needs_review, top_features, ...)
├── tests/
│   ├── test_ingest.py           # nested unwrap, id mismatch, missing fields
│   ├── test_features.py         # variant-builder semantics
│   ├── test_models.py           # ClassifierModel interface + TfidfLRModel
│   ├── test_evaluate.py         # CV harness, per-class breakdown, reproducibility check
│   ├── test_calibrate.py        # calibrated model, Brier/ECE correctness, paired comparison
│   ├── test_routing.py          # margin, runner-up, threshold boundary, coverage/accuracy curve
│   ├── test_explain.py          # top-token attribution, formatting
│   └── test_run.py              # end-to-end output contract
└── outputs/
    ├── predictions.csv          # THE required deliverable — calibrated confidence, full contract
    ├── ablation_results.csv     # §4 degradation table, generated by `python -m src.evaluate`
    ├── ablation_per_class.csv   # per-class F1 across the ablation ladder (§2/§4 decontamination)
    ├── calibration_reliability.csv  # reliability-curve bins, raw vs. calibrated (§6)
    └── coverage_accuracy.csv    # coverage/accuracy per threshold candidate (§6 routing policy)
```

`evaluation_report.md`, `figures/`, and `plots.py` were all cut (step 11) — see §9.

`models/base.py` is the key architectural choice: one `fit`/`predict_proba` contract means
the CV harness, ablation, calibration, routing and reporting are written once and the arms
are genuinely swappable. Say so in the README — the routing policy is what a compliance team
changes without touching the model; the model is what a data scientist swaps without
touching parsing or routing.

**`features.py` producing multiple text variants is what makes §4 cheap** — the ablation is
then just the same harness over different feature builders, not a parallel codebase.

**No notebook.** There is no exploratory analysis worth one (class balance is five numbers),
and a notebook is a maintainability liability in a submission judged on maintainability:
hidden state, stale outputs, no tests.

**No `evaluation_report.md`, no `plots.py` rendering, no `figures/` — cut deliberately, step
11.** A PLAN-internal invention, not a submission requirement, and a second document to keep
in sync with the README once one exists. The underlying data (reliability curve,
coverage/accuracy curve, per-class F1) already ships as CSV in `outputs/`; the README cites
the command that regenerates each number instead of embedding a static figure.

---

## 9. Build order

1. ~~Scaffold, pinned `requirements.txt`, `config.yaml`.~~ **(done)**
2. ~~**`ingest.py` first, with tests.** Nested-HTML unwrap and the `email_id` decision must be
   right before anything else — everything downstream inherits these bugs.~~ **(done)**
3. ~~Reproduce the §3 recon numbers in a throwaway script.~~ **(done —** `scripts/recon.py`;
   do not carry a claim into the README that you have not re-measured.
4. ~~**TF-IDF + LR end-to-end to `predictions.csv`.**~~ **(done)**
5. ~~`models/base.py` + CV harness in `evaluate.py`.~~ **(done)**
6. ~~**`features.py` variants + ablation harness** (§4) — this is the centrepiece.~~
   **(done —** ten variants, five-rung ablation ladder, six perturbations; see NOTES.md.
7. ~~Embeddings arm through the same harness.~~ **Investigated, deferred** — headroom
   premise checked and confirmed real (§5), arm judged confirmatory rather than load-bearing
   given the time budget; see §5 and NOTES.md for the full reasoning.
8. `Other` strategies; compare. **Deferred with step 7**, same time-budget reasoning —
   revisit only if time remains after step 9–12.
9. ~~Calibration.~~ **(done —** raw model measured badly *underconfident* (not overconfident,
   correcting §6's original assumption); sigmoid calibration cuts pooled-CV Brier
   0.544→0.215 and ECE 0.613→0.359 without changing any of the 12 actual test predictions
   (2.5% flip rate on the full paired nested-CV check, macro-F1 delta within fold noise).
   `run.py` now ships `CalibratedTfidfLRModel` on this evidence — see §6, NOTES.md.
   `plots.py`'s actual figure-rendering deferred to step 11: the reliability-curve data is
   already written to `outputs/calibration_reliability.csv`, so step 11 renders from data
   already produced rather than choosing between "build plots.py now" and "duplicate the
   computation later."
10. ~~`routing.py` + `explain.py`; wire `needs_review` and `top_features` into output.~~
    **(done —** threshold=0.45 chosen from the coverage/accuracy curve (100.0% accuracy on
    the 89.1% auto-routed, 75.0% on the 10.9% reviewed); the two genuinely ambiguous test
    emails from §6's old stale table (`email_6`, `email_1`) are exactly the two flagged
    `needs_review`; `email_4` (fraud-report taxonomy gap) is confidently *not* flagged —
    a named limitation, not a hidden one. `explain.py` uses a separate uncalibrated fit,
    reasoned and measured in §6/NOTES.md. Full output contract now in `predictions.csv`.
11. ~~`evaluate.py` generates `evaluation_report.md`.~~ **Cut, deliberately** — a
    PLAN-internal invention, not a submission requirement, and a second document to keep in
    sync with the README once the README exists. Every number the README states cites the
    command that regenerates it instead (`python -m src.evaluate` / `calibrate` /
    `routing`), which satisfies re-derivability without the second file. `plots.py`'s figure
    rendering is cut with it — no reviewer asked for embedded images, and the underlying data
    (reliability curve, coverage/accuracy curve, per-class F1) already ships as CSV.
12. ~~**README.**~~ **(done —** `README.md`, root of the repo. Leads with the finding (§2),
    not setup; both written answers included; "Investigated and deferred" section with
    evidence, not apology, for the embeddings arm, `Other` strategies, sender field,
    fine-tuning, foreign-label-space classifiers, and LLM APIs.
13. ~~**Fresh-clone test**: `pip install -r requirements.txt && python -m src.run` from
    nothing.~~ **(done, ahead of the README write, deliberately —** real `git clone` into a
    scratch directory, fresh Python 3.14.4 venv, cold `pip install`, `python -m src.run`:
    output byte-identical to the committed `predictions.csv`, 62/62 tests passing in that
    same clean environment. Stated plainly in the README as evidence, not hedged.

**Time budget:** steps 1–4 in the first two days. The remaining days go to §4, §6 and the
README — the parts that actually differentiate.

---

## 10. Guardrails

- **Determinism.** Seeds everywhere; identical `predictions.csv` on re-run.
- **Never break the required CSV contract.**
- **Never report a single CV run's number alone.** Any headline macro-F1 must appear beside
  the per-class breakdown (`Other` vs. the rest), the fold-to-fold variance, and the
  stripped-artifact figure. A bare "macro-F1 1.000" in this submission would read as
  naivety, not achievement — it doesn't even reproduce (§2: 0/50 seeds).
- **Do not declare a winner between arms separated by less than fold-to-fold variance.**
  Say they are indistinguishable at this sample size and explain which you shipped and why.
- **Flag the synthetic-data caveat once, clearly, early** — then move on. State it; do not
  belabour it.
- Docstrings and type hints throughout; maintainability is an explicit criterion.
- Small, meaningful commits.

---

## 11. What "good" looks like on submission day

- One command, clean environment, produces `predictions.csv`.
- A README whose opening section reframes the problem: the four real categories are
  saturated, the fifth (`Other`) is what actually destabilises the headline CV score, a
  single CV run can't be trusted to show that (0/50 seeds reproduce a perfect one) — here is
  what I did instead. A reviewer who reads only that section already knows the submission is
  a cut above.
- An ablation table showing 0.93 → 0.67 when structural artifacts are stripped, and the
  0/50-seeds reproducibility result showing why a single CV run at n=44 can't be trusted.
- A confidence table showing the abstain mechanism firing on the genuinely ambiguous emails.
- The fraud-report taxonomy gap raised as a business finding.
- Per-prediction explanations that make the routing auditable.
- Limitations stated plainly: `email_id` ambiguity, unreliable sender field, residual
  `Other`, n=44, and synthetic data.
