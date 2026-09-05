# Email Classification for RedRock

## The finding

Four of the five categories are saturated. The fifth, `Other`, is not — and averaging them
into one macro-F1 produces a headline number that describes neither.

| | macro-F1 (5-fold × 10-repeat CV) |
|---|---|
| Insurance Claims | 1.000 ± 0.000 |
| Investment Advisory | 0.988 ± 0.048 |
| Account Management | 0.976 ± 0.056 |
| Loan Processing | 0.956 ± 0.112 |
| **Other** (n=6, residual class) | **0.727 ± 0.424 (min 0.000)** |
| **All 5, flat (the headline number)** | **0.929 ± 0.108 (min 0.731, max 1.000)** |
| All 4 real categories, `Other` excluded | 0.980 ± 0.031 |

`Other` is a residual bucket (HR replies, IT notices, marketing) with 6 training examples
and no topic coherence — some CV folds score it 0 outright. Folded into one 5-way average,
its instability is what makes the headline number swing, not any weakness in the four real
categories, which are genuinely near-saturated at 0.980.

That saturation isn't leakage, which is the first thing worth ruling out: mean
nearest-neighbour cosine similarity within the training set is **0.245**, so the corpus
doesn't near-duplicate itself. The task is genuinely keyword-separable — short, templated
text where the category words appear near-verbatim.

**A single CV run would not have shown this.** Across 50 independent single (non-repeated)
5-fold splits on this exact pipeline, **0 of 50** scored a perfect macro-F1 — a "macro-F1 =
1.000" report from this data isn't a lucky outlier, it doesn't reproduce at all in 50 tries.
At n=44 with a 6-example residual class, one CV run reported once is not a safe way to
evaluate this task; repeats are what make the instability visible.

Once that's established, accuracy stops being the thing to optimize — there's no headroom
left for a heavier model to compete for on the four saturated categories, and any
model-vs-model comparison on the raw task is decided by `Other`-driven noise, not signal.
What follows in this README is the harder evaluation this project builds instead: an
ablation that measures what the model actually depends on, a calibrated and honestly-scoped
confidence score, a routing policy with real evidence it fires on the right cases, and an
audit trail for every prediction — the things a regulated deployment actually needs once
"the accuracy number is high" stops being a meaningful claim.

## Running it

```
pip install -r requirements.txt
python -m src.run
```

Produces `outputs/predictions.csv`: `email_id, predicted_category, confidence_score`
(required columns, in that order), plus `source_filename`, `margin`, `runner_up_category`,
`needs_review`, `top_features`. No network access is needed for this command — the shipped
model is TF-IDF + Logistic Regression, chosen partly *because* it has no download (see
"Investigated and deferred" below).

**Verified with a genuine fresh-clone test**, not asserted: cloned the repo into a scratch
directory, built a fresh Python 3.14.4 virtualenv, ran `pip install -r requirements.txt`
cold, then `python -m src.run`. It produced `outputs/predictions.csv` byte-identical to the
one committed here, and the full test suite passed in that same clean environment.

Every number in this README is reproducible from a second command, not just asserted:

| Command | Reproduces |
|---|---|
| `python -m src.evaluate` | CV headline, per-class breakdown, 0/50 reproducibility check, near-duplicate check, ablation table |
| `python -m src.calibrate` | Raw vs. calibrated Brier/ECE, paired flip-rate check, domain-shift check |
| `python -m src.routing` | Coverage/accuracy curve behind the 0.45 threshold |
| `python -m pytest` | 42 tests |

`config.yaml` holds every threshold and hyperparameter (no magic numbers in code);
`--auto-route-threshold` overrides the routing cutoff at the command line. The one seam
worth naming is a callable, not a class hierarchy: every consumer (CV, ablation,
calibration, routing) takes a `model_factory: () -> model` and needs only
`fit`/`predict_proba`/`classes_`, so swapping the model touches none of them.

## The ablation: what the model actually depends on

The provided evaluation can't distinguish a good model from a lucky split. This project
builds a harder one instead: progressively strip structural artifacts (the inner `<title>`,
the department-naming greeting — "Dear Loan Officer" — subject) and measure the drop.

| Input | flat macro-F1 (5 classes) | macro-F1, `Other` excluded (4 real categories) |
|---|---|---|
| full text (subject + title + body) | 0.929 ± 0.108 | 0.980 |
| title removed | 0.867 ± 0.118 | 0.967 |
| greeting + core only | 0.863 ± 0.114 | 0.969 |
| core sentence only | 0.674 ± 0.118 | **0.831** |
| subject only | 0.670 ± 0.096 | 0.837 |

**The flat column is contaminated and shouldn't be the headline here either.** At the two
most-stripped rungs, `Other` *and* Loan Processing — both 6-example classes — collapse to a
minimum F1 of 0.000, dragging the 5-class average down regardless of what's happening to
the categories with real signal. They don't fail for the same reason, though: `Other`
destabilises under even mild stripping (title removal alone), because it has no topic to
fall back on; Loan Processing stays perfectly stable until stripping is extreme, because a
small-but-coherent class survives on lexical cues until those cues are gone. Small sample
size predicts fragility to enough information loss; semantic incoherence predicts fragility
to any information loss — `Other` has both. The `Other`-excluded column isolates the real
effect: **0.980 → 0.831, a genuine 0.15 drop** on the four categories that actually matter,
smaller than the flat column's 0.26 but not an artifact of the residual class.

**A feature-interaction finding fell out of building the intermediate rungs.** The
department-naming greeting matters differently depending on what else survives: swapping it
for "Dear Sir/Madam," while title and subject are still present costs almost nothing
(0.920 vs. 0.929, a 0.009 difference) — but removing it once title and subject are already
gone costs 0.19 (0.674 vs. 0.863). The greeting is redundant leakage when richer artifacts
are present and becomes load-bearing leakage once they're gone.

## Confidence, calibration, and routing

The original plan assumed raw LR probabilities would be *overconfident*. Measured, that was
backwards: the raw model is **badly underconfident** — mean confidence 0.35 against 96%
actual pooled accuracy, with every raw confidence pinned in a narrow 0.26–0.49 band
regardless of how easy the email actually was. A routing threshold against that number
would have been close to meaningless.

Cross-validated sigmoid (Platt) calibration fixes this substantially:

| | Brier (↓ better) | ECE (↓ better) | mean confidence | pooled accuracy |
|---|---|---|---|---|
| Raw | 0.544 | 0.613 | 0.35 | 0.96 |
| Calibrated | 0.215 | 0.359 | 0.62 | 0.97 |

A fold-paired check (raw and calibrated models fit inside the same CV iteration, not just
the same seed) confirms this is a rescaling, not a behaviour change: macro-F1 delta is
+0.022 ± 0.063 across 50 folds — inside fold-to-fold noise — and only 2.5% of pooled
predictions flip label, **0 of the 12 real test predictions among them**. `run.py` ships
the calibrated model on that evidence.

ECE improves a lot but doesn't reach zero. **Calibrated confidence is treated as a ranking,
not a probability precise enough for cost arithmetic** — the auto-route threshold (0.45) is
read off the empirical coverage/accuracy curve, not derived from an expected-cost formula:
at 0.45, 89.1% of predictions auto-route at 100.0% accuracy, while the 10.9% held back for
review are only 75.0% accurate. The gate concentrates the model's actual errors into the
review queue rather than passing them through.

**On the real 12 test emails, the gate fires on exactly the two genuinely ambiguous ones —
and, just as tellingly, doesn't fire on the one that's actually wrong.** "New Product
Announcement" and "Financial Education Workshop" (marketing copy that reads like Other or
Investment Advisory depending on the sentence) are the two flagged `needs_review`. "Account
Freeze Request" — a fraud report, for which none of the five categories are correct — is
*not* flagged: confidence 0.68, well clear of the threshold. That email isn't uncertain.
It's confidently wrong against a taxonomy that has no correct answer in it for a fraud
report; the top contributing token is `account (+0.61)`, because "freeze my account" reads,
lexically, like a routine account-service request. A confidence gate catches what a model
doesn't know it doesn't know. It cannot catch a case where the model is sure and wrong
because the label space is missing a category — that's a taxonomy problem, not a modelling
one, and no threshold tuning fixes it.

Every prediction's `top_features` column gives the audit trail a compliance reviewer would
need — computed from a separate model fit on all 44 examples rather than the shipped
calibrated model's internals, because `CalibratedClassifierCV`'s three internal sub-fits
hold materially different TF-IDF vocabularies; a single full fit is both simpler and, given
the 0-flip result above, faithful to the same decision the shipped model made.

## Written answers

### How would you measure success?

- **Offline:** macro-F1 and per-class recall, cross-validated with the fold-to-fold interval
  reported (never a single run), broken out by class so `Other`'s instability doesn't hide
  inside the average, alongside the stripped-artifact ablation condition — not the
  unstripped headline alone.
- **Deployment:** per-class precision at the operating threshold — misrouting cost is
  asymmetric, a misrouted fraud report is not a misrouted marketing email.
- **Operational:** auto-route coverage, human-review queue volume, and downstream
  reassignment rate (how often a department bounces an email back) — free ground truth that
  accrues over time without further labelling.
- **Drift:** category mix and confidence distribution tracked monthly. The sample spans
  January–August 2025, so seasonality is concrete here, not hypothetical.

### What additional features or data would help?

**Lead with the taxonomy gap, because it's the highest-stakes finding in the data, not a
modelling one.** The test set contains a fraud report ("I suspect fraudulent activity on my
account. Please freeze my account immediately.") that fits none of the five categories. In
a regulated setting that's plausibly the single highest-stakes email in the inbox, and the
label schema has nowhere to put it — **a category for fraud/security should exist before
any amount of additional modelling effort would help.**

Beyond that: thread history and prior emails from the same client (context beyond one
170-character message); a CRM lookup (does this client hold a loan, a policy, an investment
account — a strong prior the model currently has no access to); attachment presence and
type (claim form vs. statement vs. ID document); verified sender identity (the current
sender field is a measured trap feature — `security@redrock.com` sends HR mail,
a `gmail.com` address sends an IT notice — and was excluded from the model on that
evidence, not by default); reassignment logs as continuously-collected labels for
retraining; and realistic, non-synthetic labelled data. This corpus is clean,
single-topic, and artifact-rich — real mail is multi-topic, forwarded, truncated, and
mistyped — and the ablation above is a proxy for that gap, not a substitute for closing it.

## Investigated and deferred

**Sentence embeddings + LR, as a second modelling arm.** Checked the premise first: does
`Other`'s instability, measured on `full_text` in the finding above, also explain the
apparent headroom on stripped-text conditions? Per-class CV on `core_only` confirms `Other`
collapses further under stripping (F1 0.047, then 0.000 at `subject_only`) and a second
n=6 class, Loan Processing, newly destabilises there too — but the `Other`-excluded
macro-F1 still degrades genuinely (0.980 → 0.831), so real headroom exists. Deferred anyway:
the arm would be confirmatory, not load-bearing — model choice here is already decided on
engineering grounds once accuracy saturates — while the brief's actual required deliverable,
a meaningful confidence score, wasn't yet satisfied when this decision was made. Calibration
and routing were the better use of the remaining time, and `sentence-transformers` would
have added a ~0.5–1.5GB conditional dependency for a confirmatory result.

**Abstain-as-`Other` and a hybrid routing gate**, as alternatives to the plain 5th-label
strategy shipped here. Both remain unbuilt, deferred with the embeddings arm on the same
time-budget grounds. Abstain-as-`Other` is the one to beat if revisited — it's semantically
closer to what the label means and would likely reduce `Other`'s specific instability.

**The sender field.** Measured, not assumed, to be unreliable: an HR-topic email arrives
from `security@redrock.com`, an IT notice from a personal Gmail address. Excluded from the
model on that evidence.

**Fine-tuning a transformer.** 44 examples cannot support it; the result would be unstable
and unreportable, not merely unnecessary.

**A pretrained classifier with a foreign label space**, remapped onto RedRock's five
categories. Rejected — an opaque, undebuggable translation layer between the model's real
output and the one that matters.

**An LLM API classifier.** Non-deterministic and requires a key, which conflicts directly
with the reproducibility requirement this submission is built around. Excluding it is worth
more than including it.

**Zero-shot NLI as a third arm.** Optional from the start of this project, never started —
lower priority than the other two arms.

## Limitations

- **n=44 training / 12 unlabelled test** — the test set is far too small to evaluate on;
  every number here is cross-validated on the 44, never measured against the 12.
- **Synthetic, single-topic, artifact-rich corpus.** Real inbox mail is messier in ways this
  data isn't; the ablation table is the closest proxy available, not a substitute.
- **Calibration is indicative, not authoritative.** ECE 0.359 after calibration, fit on 44
  points with a 6-example class — stated plainly rather than glossed over. The obvious
  follow-up worry — that calibrating on clean synthetic data bakes in overconfidence for
  messier real input — was checked rather than assumed: training on full text and scoring
  the held-out fold on stripped text (`core_only`), the confidence–accuracy gap *shrinks*
  (ECE 0.359 → 0.241) rather than inverting. That's reassuring, not conclusive; `core_only`
  is still synthetic text, and real mail could behave differently.
- **The confidence gate has a structural blind spot.** It can't catch a confidently wrong
  prediction caused by a missing category — see the fraud report above.
- **`email_id` ≠ filename** for all 56 provided files. Both are emitted in the output CSV so
  the mapping stays unambiguous, but this is worth knowing before cross-checking by hand.
- **The sender field is present in the data but excluded from the model**, on the trap-
  feature evidence above — not silently dropped.

## What I'd do next

With more time: build the abstain-as-`Other` strategy and measure whether it actually
reduces `Other`'s instability, rather than just being semantically cleaner. Add a category
for fraud/security to the taxonomy and re-run the whole pipeline against it — the biggest
single improvement available isn't a modelling change. And get real (or realistically
noisy) labelled mail, even a few hundred examples, to check whether the scaffolding-
dependence found here is a property of this synthetic corpus specifically or of short
templated business email in general.
