# Email Classification for RedRock

Routes client emails to one of five departments, with a calibrated confidence score and an
escalation flag for cases a human should check.

## Quick start

```
pip install -r requirements.txt
python -m src.run
```

Writes `outputs/predictions.csv`: `email_id, predicted_category, confidence_score` (required
columns, in that order), plus `source_filename`, `margin`, `runner_up_category`,
`needs_review`, `top_features`. No network access needed.

| Command | Reproduces |
|---|---|
| `python -m src.evaluate` | CV headline, per-class breakdown, 0/50 seed check, leakage check, ablation |
| `python -m src.calibrate` | Raw vs. calibrated Brier/ECE, paired flip-rate, domain-shift check |
| `python -m src.routing` | Coverage/accuracy curve behind the 0.45 threshold |
| `python -m pytest` | 44 tests |

Verified from a fresh clone: new virtualenv, cold `pip install`, then `python -m src.run`
reproduces the committed `predictions.csv` exactly and the suite passes. Output floats are
rounded to 6dp on write, so this holds across platforms rather than only on mine.

`config.yaml` holds every threshold and hyperparameter. `--auto-route-threshold` overrides
the routing cutoff. The one seam worth naming is a callable, not a class hierarchy: each
consumer takes a `model_factory: () -> model` needing only `fit`/`predict_proba`/`classes_`.

## The finding

Four of five categories are saturated. `Other` is not, and averaging them into one macro-F1
gives a number that describes neither.

| Category | macro-F1 (5-fold × 10-repeat CV) |
|---|---|
| Insurance Claims | 1.000 ± 0.000 |
| Investment Advisory | 0.988 ± 0.048 |
| Account Management | 0.976 ± 0.056 |
| Loan Processing | 0.956 ± 0.112 |
| **`Other`** (n=6, residual class) | **0.727 ± 0.424 (min 0.000)** |
| **All 5, flat — the headline number** | **0.929 ± 0.108** |
| All 4 real categories, `Other` excluded | 0.980 ± 0.031 |

`Other` is a residual bucket (HR replies, IT notices, marketing) with 6 examples and no topic
coherence; some folds score it 0. Its instability drives the headline number's swing, not any
weakness in the four real categories.

Not leakage: mean nearest-neighbour cosine similarity within train is **0.245**. The task is
genuinely keyword-separable — short templated text where category words appear near-verbatim.

**A single CV run would not have shown this.** Across 50 independent single 5-fold splits,
**0 of 50** scored a perfect macro-F1. At n=44 with a 6-example residual class, one run
reported once is not a safe estimate.

Consequence: accuracy is not the thing to optimise. There is no headroom on the four
saturated categories, and any model-vs-model comparison is decided by `Other`-driven noise.
The rest of this README is the harder evaluation built instead.

## Ablation: what the model depends on

Progressively strip structural artifacts (inner `<title>`, department-naming greeting —
"Dear Loan Officer" — subject) and measure the drop.

| Input | flat macro-F1 | macro-F1, `Other` excluded |
|---|---|---|
| full text | 0.929 ± 0.108 | 0.980 |
| title removed | 0.867 ± 0.118 | 0.967 |
| greeting + core only | 0.863 ± 0.114 | 0.969 |
| core sentence only | 0.674 ± 0.118 | **0.831** |
| subject only | 0.670 ± 0.096 | 0.837 |

The flat column is contaminated. At the two most-stripped rungs, `Other` *and* Loan
Processing — both n=6 — hit min F1 0.000, dragging the average regardless of the categories
with real signal. They fail differently: `Other` destabilises under mild stripping (title
removal alone) because it has no topic to fall back on; Loan Processing stays stable until
stripping is extreme, because a small-but-coherent class survives on lexical cues. Small
sample size predicts fragility to *enough* information loss; semantic incoherence predicts
fragility to *any*. `Other` has both.

The `Other`-excluded column isolates the real effect: **0.980 → 0.831**, a genuine 0.15 drop.

**Feature interaction.** Replacing the department-naming greeting with "Dear Sir/Madam"
costs almost nothing while title and subject remain (0.920 vs. 0.929) but costs 0.19 once
they are gone (0.674 vs. 0.863). The greeting is redundant leakage until richer artifacts
disappear, then becomes load-bearing.

## Confidence, calibration, routing

The raw model is **underconfident**, not overconfident as assumed: mean confidence 0.35
against 96% pooled accuracy, every value pinned in a 0.26–0.49 band. A threshold against
that number would be meaningless.

| | Brier ↓ | ECE ↓ | mean confidence | pooled accuracy |
|---|---|---|---|---|
| Raw | 0.544 | 0.613 | 0.35 | 0.96 |
| Calibrated (CV sigmoid) | 0.215 | 0.359 | 0.62 | 0.97 |

A fold-paired check — raw and calibrated fit inside the same CV iteration, not merely the
same seed — confirms rescaling rather than behaviour change: macro-F1 delta +0.022 ± 0.063
(inside fold noise), 2.5% of pooled predictions flip label, **0 of the 12 test predictions**.
`run.py` ships the calibrated model on that evidence.

ECE improves but does not reach zero, so **calibrated confidence is used as a ranking, not a
probability precise enough for cost arithmetic.** The 0.45 threshold is the lowest point on
the empirical coverage/accuracy curve reaching 100% accuracy on auto-routed mail: 89.1%
auto-routes at 100.0% accuracy, the 10.9% held for review is 75.0% accurate. The gate
concentrates errors into the review queue rather than passing them through.

**On the 12 test emails, the gate fires on the two ambiguous cases and not on the one that
is wrong.** "New Product Announcement" and "Financial Education Workshop" are flagged
`needs_review`. "Account Freeze Request" — a fraud report, for which no category is correct
— is *not* flagged: confidence 0.68, top token `account (+0.61)`, because "freeze my account"
reads lexically like routine account service. A confidence gate catches what the model knows
it doesn't know. It cannot catch a model that is sure and wrong because the label space is
missing a category. That is a taxonomy problem; no threshold fixes it.

A manual read of all 12 agrees — as a sanity check, not a measured accuracy, since 12
unlabelled examples cannot support one. By my own reading, 10 of 12 are unambiguously right
and the two I would contest are exactly the two flagged. `email_1` ("Financial Education
Workshop", an events notice from `hr@redrock.com`) is most likely `Other` given how train
uses that bucket; the model picks Investment Advisory at 0.43 with `Other` as runner-up and
`needs_review=True`. `email_9` is borderline but defensible. The gate caught both contestable
cases without being tuned on them.

`top_features` gives the per-prediction audit trail, computed from a separate uncalibrated
fit on all 44 examples: `CalibratedClassifierCV`'s three internal sub-fits hold materially
different TF-IDF vocabularies (135 of 1018 union terms shared), so averaging their
coefficients would misrepresent the model. Given the 0-flip result above, the single fit
explains the same decision the shipped model made.

## Measuring success

- **Offline:** macro-F1 and per-class recall, cross-validated with the fold interval reported
  — never a single run — broken out by class so `Other` cannot hide inside the average, and
  reported on the stripped-artifact condition, not the unstripped headline alone.
- **Deployment:** per-class precision at the operating threshold. Misrouting cost is
  asymmetric; a misrouted fraud report is not a misrouted marketing email.
- **Operational:** auto-route coverage, review queue volume, and downstream reassignment rate
  (how often a department bounces an email back) — free ground truth that accrues without
  further labelling.
- **Drift:** category mix and confidence distribution monthly. The sample spans January–August
  2025, so seasonality is concrete.

## Additional features and data

**The taxonomy gap comes first.** The test set contains a fraud report ("I suspect fraudulent
activity on my account. Please freeze my account immediately.") that fits none of the five
categories — plausibly the highest-stakes email in a regulated inbox, with nowhere to go. A
fraud/security category should exist before any further modelling effort.

Then: thread history and prior mail from the same client; a CRM lookup (does this client hold
a loan, a policy, an investment account — a strong prior currently unavailable to the model);
attachment presence and type; verified sender identity (the sender field is a measured trap —
`security@redrock.com` sends HR mail, a Gmail address sends an IT notice — and was excluded
on that evidence); reassignment logs as continuously-collected labels; and realistic
non-synthetic mail. This corpus is clean, single-topic and artifact-rich; the ablation above
is a proxy for that gap, not a substitute for closing it.

## Investigated and deferred

**Sentence embeddings + LR.** Premise checked first: per-class CV on stripped text shows
`Other` collapsing further (F1 0.047, then 0.000) and Loan Processing newly destabilising —
but `Other`-excluded macro-F1 still degrades genuinely, so real headroom exists. Deferred
anyway: the arm is confirmatory, not load-bearing, since model choice is already settled on
engineering grounds once accuracy saturates — and a ~0.5–1.5GB dependency for a confirmatory
result was worse value than calibration and routing, which the brief actually requires.

**Abstain-as-`Other` and a hybrid gate**, as alternatives to the plain fifth label. Unbuilt,
same time-budget grounds. Abstain-as-`Other` is the one to beat if revisited — semantically
closer to what the label means, and likely to reduce `Other`'s instability specifically.

**Fine-tuning a transformer.** 44 examples cannot support it; the result would be unstable
and unreportable, not merely unnecessary.

**A pretrained classifier with a foreign label space**, remapped onto these five categories —
an opaque, undebuggable translation layer.

**An LLM API classifier.** Non-deterministic and key-dependent, conflicting with the
reproducibility this submission is built on.

## Limitations

- **n=44 train / 12 unlabelled test.** Every number here is cross-validated on the 44, never
  measured against the 12.
- **Synthetic, single-topic, artifact-rich corpus.** The ablation is the closest available
  proxy for real mail, not a substitute.
- **Calibration is indicative, not authoritative** — ECE 0.359, fit on 44 points with a
  6-example class. The follow-up worry, that calibrating on clean data bakes in overconfidence
  for messier input, was checked rather than assumed: training on full text and scoring the
  held-out fold on stripped text, the confidence–accuracy gap *shrinks* (ECE 0.359 → 0.241)
  rather than inverting. Reassuring, not conclusive — `core_only` is still synthetic.
- **The confidence gate has a structural blind spot**: it cannot catch a confidently wrong
  prediction caused by a missing category.
- **`email_id` ≠ filename** for all 56 files. Both are emitted so the mapping is unambiguous.

## Next steps

Build abstain-as-`Other` and measure whether it actually reduces `Other`'s instability rather
than merely being semantically cleaner. Add a fraud/security category and re-run the pipeline
against it — the largest available improvement is not a modelling change. Get real or
realistically noisy labelled mail to test whether the scaffolding dependence found here is a
property of this corpus or of short templated business email generally.