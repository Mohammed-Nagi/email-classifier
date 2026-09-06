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
`top_features`.

| Command | Reproduces |
|---|---|
| `python -m src.evaluate` | CV headline, per-class breakdown, 0/50 seed check, leakage check, ablation, confusion matrix, abstain sweep and ladder |
| `python -m src.calibrate` | Raw against calibrated Brier/ECE, paired flip-rate, domain-shift check |
| `python -m src.routing` | Coverage/accuracy curve behind the 0.45 threshold |
| `python -m pytest` | 48 tests |

Verified from a fresh clone: new virtualenv, cold `pip install`, then `python -m src.run`
reproduces the committed `predictions.csv` exactly. Output floats are rounded to 6dp on write.

`config.yaml` holds every threshold and hyperparameter, and `--auto-route-threshold` overrides
the routing cutoff.

## Results

**Four of five categories are saturated.**

| Category | macro-F1 (5-fold × 10-repeat CV) |
|---|---|
| **All 5 categories** | **0.929 ± 0.108** |
| All 4 real categories, `Other` excluded | 0.980 ± 0.031 |
| Insurance Claims | 1.000 ± 0.000 |
| Investment Advisory | 0.988 ± 0.048 |
| Account Management | 0.976 ± 0.056 |
| Loan Processing | 0.956 ± 0.112 |
| **`Other`** (n=6, residual class) | **0.727 ± 0.424 (min 0.000)** |

`Other` is a residual bucket of HR replies, IT notices and marketing, with 6 examples and no
topic coherence; some folds score it 0. Its instability drives the headline result's swing, not
any weakness in the four real categories.

Mean nearest-neighbour cosine similarity within train is 0.245. The task is keyword-separable
by construction: the text is short and templated, and category words appear near-verbatim.

Across 50 independent single 5-fold splits, 0 of 50 scored a perfect macro-F1. At n=44 with a
6-example residual class, one run reported once is not a safe estimate.

Accuracy is therefore not the thing to optimise. There is no headroom on the four saturated
categories, and any model-against-model comparison is decided by `Other`-driven noise.

## Ablation: what the model depends on

Structural artefacts are stripped progressively: the inner `<title>`, the department-naming
greeting ("Dear Loan Officer"), and the subject.

| Input | macro-F1 | macro-F1, `Other` excluded |
|---|---|---|
| full text | 0.929 ± 0.108 | 0.980 |
| title removed | 0.867 ± 0.118 | 0.967 |
| greeting and core only | 0.863 ± 0.114 | 0.969 |
| core sentence only | 0.674 ± 0.118 | **0.831** |
| subject only | 0.670 ± 0.096 | 0.837 |

`Other` and Loan Processing, both n=6, hit min F1 of 0.000 and drag the average down regardless
of what the categories with real signal do. They fail differently. `Other` destabilises under mild
stripping, title removal alone, because it has no topic to fall back on. Loan Processing stays
stable until stripping is extreme, because a small but coherent class survives on lexical cues.

The `Other`-excluded column isolates the real effect: 0.980 to 0.831, a drop of 0.15.

**The greeting is redundant leakage.** Replacing it with "Dear Sir/Madam" costs almost nothing, while
the title and subject remain (0.920 against 0.929) and cost 0.19 once they are gone (0.674 against 0.863).

## Confusion matrix

Pooled out-of-fold predictions over the same 5-fold × 10-repeat CV, 440 predictions, rows true
and columns predicted:

|                     | Account Mgmt | Insurance | Investment | Loan | Other |
|---|---|---|---|---|---|
| **Account Mgmt**    | 130 | 0  | 0   | 0  | 0  |
| **Insurance**       | 0   | 70 | 0   | 0  | 0  |
| **Investment**      | 0   | 0  | 120 | 0  | 0  |
| **Loan**            | 0   | 0  | 0   | 60 | 0  |
| **Other**           | 8   | 0  | 3   | 7  | 42 |

**Every model mistake is `Other` landing in a real department**, 18 of 60 or 30%, split 8/7/3/0
over Account Management, Loan Processing, Investment Advisory and Insurance Claims. The four
real categories never leak into each other or into `Other`. This is not a model that confuses
departments; it is one with no reliable handle on off-topic mail, which is a taxonomy or
negative-class-signal problem rather than a decision-boundary one. The cost is manageable:
non-client mail clutters a queue instead of a client's loan or claim reaching the wrong desk.

This affirms the `Other` diagnosis independent of fold variance. The confusion matrix is a
single pooled count and is not sensitive to how folds happen to split, and it lands on the same
class as the sole source of error.

## Confidence and Calibration

**The raw model is underconfident.** Mean confidence 0.35 against 96% pooled accuracy, with every
value inside a 0.26 to 0.49 band.

| | Brier ↓ | ECE ↓ | mean confidence | pooled accuracy |
|---|---|---|---|---|
| Raw | 0.544 | 0.613 | 0.35 | 0.96 |
| Calibrated (CV sigmoid) | 0.215 | 0.359 | 0.62 | 0.97 |

A fold-paired check confirms this is a rescaling and not a change in behaviour: macro-F1 delta
+0.022 ± 0.063, within fold noise, 2.5% of pooled predictions flip label, and 0 of the 12 test
predictions among them. `run.py` ships the calibrated model on that evidence.

**ECE improves but does not reach zero, so calibrated confidence is used as a ranking.** The 0.45
threshold is the lowest point on the coverage/accuracy curve reaching 100% accuracy on auto-routed
mail: 89.1% auto-routes at 100.0% accuracy, and the 10.9% held for review is 75.0% accurate.
The gate concentrates errors into the review queue instead of passing them through.

**On the 12 test emails, the gate fires on the two ambiguous cases.** "New Product Announcement"
and "Financial Education Workshop" are flagged as `needs_review`.

By my own reading, 10 of 12 are unambiguously right, and the two I would contest are exactly the
two flagged. `email_1`, "Financial Education Workshop", an events notice from `hr@redrock.com`,
is most likely `Other`; the model picks Investment Advisory at 0.43 with `Other` as runner-up and
`needs_review=True`. `email_9` is a similar case. The gate caught both contestable cases without
being tuned on them.

`top_features` comes from a separate uncalibrated fit on all 44 examples, not from the shipped
calibrated model: `CalibratedClassifierCV`'s three internal sub-fits use different vocabularies,
so their coefficients can't be averaged into one explanation. Since calibration changes 0 of the
12 test predictions, this fit explains the same decisions the shipped model makes.

## Abstain-as-`Other`: built, measured, rejected

The alternative to a plain fifth label is to train on the four substantive categories only and
abstain to `Other` when no class clears a confidence threshold. Built and measured on the same
folds as the flat model: it loses on every axis. `Other`'s own F1 gets worse (0.727 → 0.602),
flat macro-F1 drops nine points (0.929 → 0.838), and Loan Processing, the next-smallest real
class, inherits `Other`'s instability (0.727 ± 0.424, min 0.000), confirming that the fragility
tracks small sample size rather than the `Other` label itself. The threshold is also sharp
rather than robust, with macro-F1 falling from 0.838 at 0.35 to 0.629 at 0.40. The flat model
and the existing confidence gate already flag low-confidence predictions for review.
Full ladder and sweep results: `python -m src.evaluate`.

## Measuring success

Success is measured by how well the emails actually get routed, not by overall accuracy.
In practice that means tracking per-department precision on the emails the system routes
automatically, since a wrong routing costs more in some departments than others, and
watching the size of the human-review queue to make sure it stays small enough to be useful.
Reassignments, a department bouncing an email back, could become a source of ground truth
for catching drift over time, and the category mix and confidence distribution are worth a
monthly check given the training data only spans January to August 2025.

## Additional features and data

Thread history and prior mail from the same client would give the model context beyond a single
message. A CRM lookup for whether the client already holds a loan, a policy or an investment
account would supply a strong prior the model has no access to today. Attachment presence and
type, and verified sender identity, parsed in `ingest.py` but unused by any feature variant here,
are both plausible signals not yet tested. Reassignment logs would provide continuously collected
labels for retraining without further manual annotation, and realistic, non-synthetic mail would
be the most valuable addition overall.

## Investigated and deferred

**A hybrid gate** combining abstain-as-`Other` with the existing confidence gate. Not built:
abstain-as-`Other` was measured and rejected above, and since it does not reduce `Other`'s
instability, there is no improved abstain signal for a hybrid to add.

**Fine-tuning a transformer.** 44 examples cannot support it. The result would be unstable and
unreportable.

**A pretrained classifier with a foreign label space**, remapped onto these five categories. An
opaque and undebuggable translation layer.

**An LLM API classifier.** Non-deterministic and key-dependent, which conflicts with the
reproducibility this submission is built on.

## Limitations

- **n=44 train and 12 unlabelled test.** Every number here is cross-validated on the 44 and never
  measured against the 12.
- **The corpus is synthetic.** The ablation is the closest available proxy for real mail, not a
  substitute for it.
- **Calibration is indicative rather than authoritative**, at ECE 0.359 and fit on 44 points with
  a 6-example class.
- **The 0.45 threshold is selected and evaluated on the same 440 pooled out-of-fold predictions.**
  The coverage/accuracy curve used is the same data used to report 89.1% auto-routing at
  100.0% accuracy. At n=44 behind those 440 points, that operating-point figure is optimistic.
  Nested CV, selecting the threshold inside an outer fold and evaluating on data the selection
  never saw, is the recommended approach.
- **`email_id` does not match the filename** for any of the 56 files. Both are emitted to the output
  csv, so the mapping is unambiguous.
