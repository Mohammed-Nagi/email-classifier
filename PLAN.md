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

**A TF-IDF + Logistic Regression baseline achieves a perfect cross-validated score on the
44 labelled emails.** Measured with 5-fold stratified CV: accuracy 1.000, macro-F1 1.000.
Under repeated stratified CV (5-fold × 10 repeats): macro-F1 **0.986 ± 0.038**. A
most-frequent-class dummy scores 0.091.

This is not near-duplicate leakage — mean nearest-neighbour cosine similarity within train is
only 0.245 (re-derived in code; see the near-duplicate check row in §3). The task is simply
keyword-separable: the category words appear near-verbatim in the text ("applying for a
personal loan of $25,000", "Dear Loan Officer").

### What follows from this

**Accuracy is saturated, so accuracy cannot be the deliverable.** There is no headroom for a
heavier model to compete for — a zero-shot cross-encoder cannot beat 1.000, and any
comparison between models will be decided by noise, not signal. An earlier draft of this
plan proposed a three-arm model bake-off as the technical centrepiece; the measurements
above make that centrepiece hollow.

**The submission's value must come from everything that happens after accuracy saturates:**

1. Recognising the perfect score as a **red flag rather than a result**, and saying so.
2. Building a **harder evaluation** than the one provided, since the provided one is
   uninformative.
3. Justifying model choice on **engineering grounds** (determinism, auditability, latency,
   zero-download) once accuracy has stopped discriminating.
4. A **calibrated confidence and abstain policy** that demonstrably fires on the genuinely
   ambiguous cases.
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
five categories fit a fraud report.** The baseline assigns Account Management at 0.64 with
`Other` as runner-up. In a regulated setting a suspected-fraud notification is plausibly the
highest-stakes item in the whole inbox, and the taxonomy has nowhere to put it.

Raise this explicitly in the written analysis. Noticing that the label schema is
incomplete — rather than silently forcing a prediction — is exactly the judgement the
compliance framing is testing.

---

## 4. Ablation / stress testing — the technical centrepiece

Since the provided evaluation is saturated, **build a harder one.** Progressively strip the
structural artifacts and measure degradation. Probe results to reproduce and extend
(5-fold × 10 repeats, macro-F1):

| Input | macro-F1 |
|---|---|
| subject + full body | 0.986 ± 0.038 |
| body only, inner `<title>` removed | 0.950 ± 0.080 |
| core paragraphs only (no subject, title, greeting or signature) | **0.839 ± 0.129** |
| subject only | 0.667 ± 0.087 |

The headline: **strip the artifacts and macro-F1 falls from 0.99 to 0.84, with variance
more than tripling.** That gap is the honest measure of the task, and 0.84 ± 0.13 is a very
different claim from "100% accurate".

Extend with perturbations that mimic real inbox conditions:

- Remove or corrupt the subject line (forwarded/replied mail often loses it).
- Replace department-naming salutations with generic ones ("Dear Sir/Madam") — tests whether
  the model learned the topic or learned the greeting.
- Paraphrase / synonym substitution on key terms.
- Truncate to the first sentence (mobile previews, partial ingestion).
- Add signature blocks, disclaimers, quoted reply chains as distractor text.

Report a degradation table. This is the most defensible artefact in the submission: it
shows you understood that the provided evaluation could not distinguish a good solution
from a lucky one, and built one that could.

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

### Include: one contrasting arm, to prove the ceiling is real

Run **sentence embeddings + LR** (`all-MiniLM-L6-v2`) through the identical CV and ablation
harness. Its purpose is not to win — it is to demonstrate that the ceiling is a property of
the *data*, not of one model family, and to show the comparison was actually run. Report it
on the stripped-artifact ablation too, where there is real headroom to separate them.

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

The observed confidence behaviour on the 12 test emails already validates the design — the
lowest-confidence predictions are exactly the genuinely ambiguous ones:

| Email | Subject | Conf | Margin | Situation |
|---|---|---|---|---|
| `email_6` | New Product Announcement | 0.34 | 0.08 | Marketing — Other vs Investment Advisory |
| `email_9` | Market Research Request | 0.37 | 0.07 | Genuinely borderline |
| `email_1` | Financial Education Workshop | 0.46 | 0.23 | Event marketing vs advisory |
| `email_4` | Account Freeze Request | 0.64 | 0.49 | Fraud report — taxonomy gap (§3) |

Report this table. Empirical evidence that the abstain mechanism fires on the right cases is
far stronger than asserting that it should.

### Confidence score

1. Take the predicted-class probability.
2. **Calibrate it** — raw LR probabilities on 44 samples are overconfident. Use
   cross-validated Platt/sigmoid calibration and show reliability curve + Brier/ECE before
   and after. **Caveat honestly:** calibrating on 44 points with 6 in the smallest class is
   itself statistically thin. State that the calibration is indicative, not authoritative,
   and that with production volume it would be re-fit properly. Saying this is worth more
   than pretending otherwise.
3. Report **margin** (top-1 − top-2) as an extra column — for routing, "which two
   departments is it torn between" is more actionable than absolute confidence.

### `Other` strategy — implement and compare at least two

1. **Normal fifth label** (supervised baseline).
2. **Abstain-as-Other** — predict over the four real categories; if none clears threshold,
   assign `Other`. Semantically honest: "none of these departments" is what `Other` means.
3. **Hybrid** — four-way topic model plus a binary "is this routable business
   correspondence?" gate.

Strategy 2 is the one to beat and aligns with the routing framing.

### Routing policy

Explicit **abstain / human-review band**: above threshold → auto-route; below → flag
`needs_review` for a human queue. Choose the threshold by **cost-weighted reasoning** — a
misrouted fraud report or insurance claim costs far more than an extra email in a human
queue. The threshold lives in `config.yaml`, exposed as `--auto-route-threshold`. No magic
numbers in code.

### Explainability (regulated-context requirement)

For every prediction, emit the **top contributing tokens** from the LR coefficients. Cheap
with a linear model, and it turns the output into an audit trail: a compliance reviewer can
see *why* an email was routed, not just where. Include a worked example in the README.

### Output contract

Keep `email_id, predicted_category, confidence_score` exactly as specified, first and
correctly named. Additional columns after: `source_filename`, `margin`,
`runner_up_category`, `needs_review`, `top_features`. Never break the required contract.

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

- Offline: macro-F1 and per-class recall, cross-validated, with the interval at n=44
  acknowledged — and reported on the **stripped-artifact** condition, not the inflated one.
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
├── README.md                  # setup, run, design decisions, trade-offs, written analysis
├── PLAN.md                    # this file (optional to ship)
├── requirements.txt           # pinned
├── config.yaml                # threshold, model names, paths — no magic numbers in code
├── data/                      # provided folder, unmodified
├── src/
│   ├── ingest.py              # HTML parsing -> records (nested body, artifact-aware)
│   ├── features.py            # text assembly variants (full / stripped / core) for ablation
│   ├── models/
│   │   ├── base.py            # shared fit/predict_proba interface — arms are swappable
│   │   ├── tfidf_lr.py
│   │   └── embed_lr.py
│   ├── calibrate.py
│   ├── routing.py             # threshold, abstain, Other strategy
│   ├── explain.py             # per-prediction token attribution
│   ├── evaluate.py            # CV, ablation harness, metrics, comparison
│   ├── plots.py               # calibration, coverage-accuracy, confusion matrix
│   └── run.py                 # single-command entrypoint
├── tests/
│   ├── test_ingest.py         # nested unwrap, id mismatch, missing fields
│   └── test_routing.py        # threshold boundaries
└── outputs/
    ├── predictions.csv        # THE required deliverable
    ├── evaluation_report.md   # generated: metrics, ablation table, embedded figures
    └── figures/
```

`models/base.py` is the key architectural choice: one `fit`/`predict_proba` contract means
the CV harness, ablation, calibration, routing and reporting are written once and the arms
are genuinely swappable. Say so in the README — the routing policy is what a compliance team
changes without touching the model; the model is what a data scientist swaps without
touching parsing or routing.

**`features.py` producing multiple text variants is what makes §4 cheap** — the ablation is
then just the same harness over different feature builders, not a parallel codebase.

**No notebook.** There is no exploratory analysis worth one (class balance is five numbers),
and a notebook is a maintainability liability in a submission judged on maintainability:
hidden state, stale outputs, no tests. Every figure is generated by `plots.py` during the
normal run and embedded in `evaluation_report.md`.

---

## 9. Build order

1. Scaffold, pinned `requirements.txt`, `config.yaml`.
2. **`ingest.py` first, with tests.** Nested-HTML unwrap and the `email_id` decision must be
   right before anything else — everything downstream inherits these bugs.
3. Reproduce the §3 recon numbers in a throwaway script. Do not carry a claim into the
   README that you have not re-measured.
4. **TF-IDF + LR end-to-end to `predictions.csv`.** A valid submittable artefact must exist
   by day 2.
5. `models/base.py` + CV harness in `evaluate.py`.
6. **`features.py` variants + ablation harness** (§4) — this is the centrepiece; give it
   real time.
7. Embeddings arm through the same harness.
8. `Other` strategies; compare.
9. Calibration + `plots.py`.
10. `routing.py` + `explain.py`; wire `needs_review` and `top_features` into output.
11. `evaluate.py` generates `evaluation_report.md`.
12. **README** — the §2 framing up front, then setup, run, design decisions, both written
    answers, and an **"Investigated and rejected"** section (sender field, arms that lost,
    `Other` strategies not chosen, why fine-tuning and LLM APIs were excluded).
13. **Fresh-clone test**: `pip install -r requirements.txt && python -m src.run` from nothing.

**Time budget:** steps 1–4 in the first two days. The remaining days go to §4, §6 and the
README — the parts that actually differentiate.

---

## 10. Guardrails

- **Determinism.** Seeds everywhere; identical `predictions.csv` on re-run.
- **Never break the required CSV contract.**
- **Never report the inflated number alone.** Any headline accuracy must appear beside the
  stripped-artifact figure and the fold variance. A bare "100% accuracy" in this submission
  would read as naivety, not achievement.
- **Do not declare a winner between arms separated by less than fold-to-fold variance.**
  Say they are indistinguishable at this sample size and explain which you shipped and why.
- **Flag the synthetic-data caveat once, clearly, early** — then move on. State it; do not
  belabour it.
- Docstrings and type hints throughout; maintainability is an explicit criterion.
- Small, meaningful commits.

---

## 11. What "good" looks like on submission day

- One command, clean environment, produces `predictions.csv`.
- A README whose opening section reframes the problem: accuracy is saturated, here is what
  I did instead — and a reviewer who reads only that section already knows the submission is
  a cut above.
- An ablation table showing 0.99 → 0.84 when artifacts are stripped.
- A confidence table showing the abstain mechanism firing on the genuinely ambiguous emails.
- The fraud-report taxonomy gap raised as a business finding.
- Per-prediction explanations that make the routing auditable.
- Limitations stated plainly: `email_id` ambiguity, unreliable sender field, residual
  `Other`, n=44, and synthetic data.
