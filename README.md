# Email Classification for RedRock

Routes client emails to one of five departments, with a calibrated confidence score and an
escalation flag for cases a human should check.

## Quick start

```
pip install -r requirements.txt
python -m src.run
```

Writes `outputs/predictions.csv`: `email_id, predicted_category, confidence_score` (required
columns, in that order), then `source_filename`, `margin`, `runner_up_category`, `needs_review`,
`top_features`. No network access is required.

| Command | Reproduces |
|---|---|
| `python -m src.evaluate` | CV headline, per-class breakdown, 0/50 seed check, leakage check, ablation, confusion matrix, abstain sweep and ladder |
| `python -m src.calibrate` | Raw against calibrated Brier/ECE, paired flip-rate, domain-shift check |
| `python -m src.routing` | Coverage/accuracy curve behind the 0.45 threshold |
| `python -m pytest` | 48 tests |

Verified from a fresh clone: new virtualenv, cold `pip install`, then `python -m src.run`
reproduces the committed `predictions.csv` exactly and the suite passes. Output floats are
rounded to 6dp on write, so this holds across platforms and not only on the machine it was built
on.

`config.yaml` holds every threshold and hyperparameter, and `--auto-route-threshold` overrides
the routing cutoff. The one seam worth naming is a callable rather than a class hierarchy: each
consumer takes a `model_factory: () -> model` and needs only `fit`, `predict_proba` and
`classes_` from what it returns.

## The finding

**Four of five categories are saturated and the fifth is not.** Averaging them into one macro-F1
gives a number that describes neither.

| Category | macro-F1 (5-fold × 10-repeat CV) |
|---|---|
| Insurance Claims | 1.000 ± 0.000 |
| Investment Advisory | 0.988 ± 0.048 |
| Account Management | 0.976 ± 0.056 |
| Loan Processing | 0.956 ± 0.112 |
| **`Other`** (n=6, residual class) | **0.727 ± 0.424 (min 0.000)** |
| **All 5, flat, the headline number** | **0.929 ± 0.108** |
| All 4 real categories, `Other` excluded | 0.980 ± 0.031 |

`Other` is a residual bucket of HR replies, IT notices and marketing, with 6 examples and no
topic coherence; some folds score it 0. Its instability drives the headline number's swing, not
any weakness in the four real categories.

**This is not near-duplicate leakage.** Mean nearest-neighbour cosine similarity within train is
0.245. The task is keyword-separable by construction: the text is short and templated, and
category words appear near-verbatim.

**A single CV run would not have shown any of this.** Across 50 independent single 5-fold splits,
0 of 50 scored a perfect macro-F1. At n=44 with a 6-example residual class, one run reported once
is not a safe estimate.

Accuracy is therefore not the thing to optimise. There is no headroom on the four saturated
categories, and any model-against-model comparison is decided by `Other`-driven noise. What
follows is the harder evaluation built in its place.

## Ablation: what the model depends on

Structural artifacts are stripped progressively: the inner `<title>`, the department-naming
greeting ("Dear Loan Officer"), and the subject.

| Input | flat macro-F1 | macro-F1, `Other` excluded |
|---|---|---|
| full text | 0.929 ± 0.108 | 0.980 |
| title removed | 0.867 ± 0.118 | 0.967 |
| greeting and core only | 0.863 ± 0.114 | 0.969 |
| core sentence only | 0.674 ± 0.118 | **0.831** |
| subject only | 0.670 ± 0.096 | 0.837 |

**The flat column is contaminated.** At the two most-stripped rungs `Other` and Loan Processing,
both n=6, hit min F1 0.000 and drag the average down regardless of what the categories with real
signal do. They fail differently. `Other` destabilises under mild stripping, title removal alone,
because it has no topic to fall back on. Loan Processing stays stable until stripping is extreme,
because a small but coherent class survives on lexical cues. Small sample size predicts fragility
to *enough* information loss; semantic incoherence predicts fragility to *any*. `Other` has both.

The `Other`-excluded column isolates the real effect: 0.980 to 0.831, a drop of 0.15.

**The greeting is redundant leakage until nothing else leaks the label.** Replacing it with "Dear
Sir/Madam" costs almost nothing while the title and subject remain, 0.920 against 0.929, and
costs 0.19 once they are gone, 0.674 against 0.863.

## Confusion matrix: where wrong predictions land

Pooled out-of-fold predictions over the same 5-fold × 10-repeat CV, 440 predictions, rows true
and columns predicted:

|                     | Account Mgmt | Insurance | Investment | Loan | Other |
|---|---|---|---|---|---|
| **Account Mgmt**    | 130 | 0  | 0   | 0  | 0  |
| **Insurance**       | 0   | 70 | 0   | 0  | 0  |
| **Investment**      | 0   | 0  | 120 | 0  | 0  |
| **Loan**            | 0   | 0  | 0   | 60 | 0  |
| **Other**           | 8   | 0  | 3   | 7  | 42 |

**Every confusion is `Other` landing in a real department**, 18 of 60 or 30%, split 8/7/3/0 over
Account Management, Loan Processing, Investment Advisory and Insurance Claims. The four real
categories never leak into each other or into `Other`. This is not a model that confuses
departments, it is one with no reliable handle on off-topic mail, which is a taxonomy or
negative-class-signal problem rather than a decision-boundary one. The cost is one-directional:
non-client mail clutters a queue instead of a client's loan or claim reaching the wrong desk.

This corroborates the `Other` diagnosis independent of fold variance. The confusion matrix is a
single pooled count and is not sensitive to how folds happen to split, and it lands on the same
class as the sole source of error.

## Confidence, calibration, routing

**The raw model is underconfident, not overconfident as assumed:** mean confidence 0.35 against
96% pooled accuracy, with every value pinned inside a 0.26 to 0.49 band. A threshold read against
that number would be meaningless.

| | Brier ↓ | ECE ↓ | mean confidence | pooled accuracy |
|---|---|---|---|---|
| Raw | 0.544 | 0.613 | 0.35 | 0.96 |
| Calibrated (CV sigmoid) | 0.215 | 0.359 | 0.62 | 0.97 |

A fold-paired check, with raw and calibrated fit inside the same CV iteration rather than merely
under the same seed, confirms this is a rescaling and not a change in behaviour: macro-F1 delta
+0.022 ± 0.063, inside fold noise, 2.5% of pooled predictions flip label, and 0 of the 12 test
predictions among them. `run.py` ships the calibrated model on that evidence.

**ECE improves but does not reach zero, so calibrated confidence is used as a ranking rather than
a probability precise enough for cost arithmetic.** The 0.45 threshold is the lowest point on the
empirical coverage/accuracy curve reaching 100% accuracy on auto-routed mail: 89.1% auto-routes
at 100.0% accuracy, and the 10.9% held for review is 75.0% accurate. The gate concentrates errors
into the review queue instead of passing them through.

**On the 12 test emails the gate fires on the two ambiguous cases and not on the one that is
wrong.** "New Product Announcement" and "Financial Education Workshop" are flagged
`needs_review`. "Account Freeze Request", a fraud report for which no category is correct, is not
flagged: confidence 0.68, top token `account (+0.61)`, because "freeze my account" reads lexically
like routine account service. A confidence gate catches what the model knows it does not know. It
cannot catch a model that is sure and wrong because the label space is missing a category, which
is a taxonomy problem and no threshold fixes it.

A manual read of all 12 agrees, as a sanity check rather than a measured accuracy, since 12
unlabelled examples cannot support one. By my own reading 10 of 12 are unambiguously right, and
the two I would contest are exactly the two flagged. `email_1`, "Financial Education Workshop",
an events notice from `hr@redrock.com`, is most likely `Other` given how train uses that bucket;
the model picks Investment Advisory at 0.43 with `Other` as runner-up and `needs_review=True`.
`email_9` is borderline but defensible. The gate caught both contestable cases without being
tuned on them.

`top_features` gives the per-prediction audit trail, computed from a separate uncalibrated fit on
all 44 examples. `CalibratedClassifierCV`'s three internal sub-fits hold materially different
TF-IDF vocabularies, only 135 of 1018 union terms shared by all three, so averaging their
coefficients would misrepresent the model rather than approximate it. Given the 0-flip result
above, the single fit explains the same decision the shipped model made.

## Abstain-as-`Other`: built, measured, rejected

The alternative to a plain fifth label is to train on the four substantive categories only and
abstain to `Other` when no class clears a confidence threshold.

**Stated before building it:** this conflates "genuinely off-topic" with "merely not confident".
Shipped, every low-confidence email becomes both `Other` and `needs_review` by construction, so
the residual class and the review queue become the same set. The threshold therefore stays its own
config value, `abstain.threshold`, tuned on its own sweep rather than reused from
`routing.auto_route_threshold`.

Same folds and seed as the flat model, threshold chosen to maximise `Other` F1 at 0.35:

| Category | Flat 5-way (current) | Abstain-as-`Other` |
|---|---|---|
| Insurance Claims | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Investment Advisory | 0.988 ± 0.048 | 0.925 ± 0.126 |
| Account Management | 0.976 ± 0.056 | 0.935 ± 0.095 |
| Loan Processing | 0.956 ± 0.112 | **0.727 ± 0.424 (min 0.000)** |
| `Other` | **0.727 ± 0.424** | **0.602 ± 0.380 (min 0.000)** |
| Flat macro-F1 | **0.929 ± 0.108** | **0.838 ± 0.139** |
| macro-F1, `Other` excluded | 0.980 | 0.897 |

**It loses on every axis.** `Other`'s own F1 gets worse rather than better, 0.727 to 0.602. Flat
macro-F1 drops nine points. The `Other`-excluded figure drops too, 0.980 to 0.897, because Loan
Processing, the next-smallest real class at n=6, inherits `Other`'s exact failure signature,
0.727 ± 0.424 with min 0.000. That is the ablation's small-n and semantic-incoherence split
resurfacing a second, independent way: a coherent but small class destabilising under threshold
pressure instead of stripping pressure, which is what the small-n mechanism alone predicts. The
threshold is sharp rather than robust as well, with macro-F1 falling from 0.838 at 0.35 to 0.629
at 0.40.

**Recommendation: do not ship it.** The flat model and the existing confidence gate already flag
low-confidence predictions for review, without relabelling them and without merging two failure
modes into one bucket. Full ladder and sweep results come from `python -m src.evaluate`.

## Measuring success

- **Offline.** Macro-F1 and per-class recall, cross-validated with the fold interval reported and
  never a single run, broken out by class so `Other` cannot hide inside the average, and reported
  on the stripped-artifact condition rather than the unstripped headline alone.
- **Deployment.** Per-class precision at the operating threshold. Misrouting cost is asymmetric;
  a misrouted fraud report is not a misrouted marketing email.
- **Operational.** Auto-route coverage, review queue volume, and downstream reassignment rate,
  meaning how often a department bounces an email back. This is free ground truth that accrues
  without further labelling.
- **Drift.** Category mix and confidence distribution monthly. The sample spans January to August
  2025, so seasonality is concrete rather than hypothetical.

## Additional features and data

**The taxonomy gap comes first.** The test set contains a fraud report, "I suspect fraudulent
activity on my account. Please freeze my account immediately.", that fits none of the five
categories. It is plausibly the highest-stakes email in a regulated inbox and has nowhere to go.
A fraud or security category should exist before any further modelling effort.

Then: thread history and prior mail from the same client; a CRM lookup for whether the client
holds a loan, a policy or an investment account, which is a strong prior currently unavailable to
the model; attachment presence and type; verified sender identity, parsed in `ingest.py` but used
by no feature variant here, so not measured and not claimed either way; reassignment logs as
continuously collected labels; and realistic non-synthetic mail. This corpus is clean,
single-topic and artifact-rich, and the ablation above is a proxy for that gap rather than a
substitute for closing it.

## Investigated and deferred

**Sentence embeddings with LR.** The premise was checked first. Per-class CV on stripped text
shows `Other` collapsing further, F1 0.047 then 0.000, and Loan Processing newly destabilising,
but `Other`-excluded macro-F1 still degrades genuinely, so real headroom exists. Deferred anyway:
the arm is confirmatory rather than load-bearing, since model choice is already settled on
engineering grounds once accuracy saturates, and a 0.5 to 1.5GB dependency for a confirmatory
result was worse value than calibration and routing, which the brief requires.

**A hybrid gate** combining abstain-as-`Other` with the existing confidence gate. Not built:
abstain-as-`Other` was measured and rejected above, and since it does not reduce `Other`'s
instability there is no improved abstain signal for a hybrid to add.

**Fine-tuning a transformer.** 44 examples cannot support it. The result would be unstable and
unreportable, not merely unnecessary.

**A pretrained classifier with a foreign label space**, remapped onto these five categories. An
opaque and undebuggable translation layer.

**An LLM API classifier.** Non-deterministic and key-dependent, which conflicts with the
reproducibility this submission is built on.

## Limitations

- **n=44 train and 12 unlabelled test.** Every number here is cross-validated on the 44 and never
  measured against the 12.
- **The corpus is synthetic, single-topic and artifact-rich.** The ablation is the closest
  available proxy for real mail, not a substitute for it.
- **Calibration is indicative rather than authoritative**, at ECE 0.359 and fit on 44 points with
  a 6-example class. The follow-up worry, that calibrating on clean data bakes in overconfidence
  for messier input, was checked rather than assumed: training on full text and scoring the
  held-out fold on stripped text, the confidence to accuracy gap shrinks, ECE 0.359 to 0.241,
  rather than inverting. Reassuring but not conclusive, since `core_only` is still synthetic.
- **The confidence gate has a structural blind spot.** It cannot catch a confidently wrong
  prediction caused by a missing category.
- **The 0.45 threshold is selected and evaluated on the same 440 pooled out-of-fold predictions.**
  The coverage/accuracy curve it is read off is the same data used to report 89.1% auto-routing at
  100.0% accuracy. At n=44 behind those 440 points, that operating-point figure is optimistic.
  Nested CV, selecting the threshold inside an outer fold and evaluating on data the selection
  never saw, is the proper fix and is not built here.
- **Why 100% and not 0.40's 93.6% coverage at 99.8% accuracy.** The four points of coverage given
  up are a deliberate trade rather than the cleanest-looking number on the curve. A misrouted
  client request, a loan or a claim reaching the wrong desk, is materially worse in a regulated
  setting than one more email waiting in a human queue, so the threshold drives auto-routed error
  to zero instead of minimising review volume.
- **`email_id` does not match the filename** for any of the 56 files. Both are emitted so the
  mapping is unambiguous.

## Next steps

Add a fraud or security category and re-run the pipeline against it. The largest available
improvement is not a modelling change, and abstain-as-`Other` is not a substitute for it, since an
off-topic label still has nowhere correct to go. Obtain real or realistically noisy labelled mail
to test whether the scaffolding dependence found here is a property of this corpus or of short
templated business email generally. If budget opens up for modelling, nested CV for the
auto-route threshold would replace the optimistic single-dataset estimate flagged above.
