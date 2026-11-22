# DriftLens: Evaluating Machine-Learning Reliability under Temporal Data Shift

**Research Report.** Syed Hanzala Ali (24K0025), Syed Zahid Hussain (24K0015), Saadullah (24K0004)

> Every number below comes from
> `results/bank_logistic/`, `results/bank_hgb/` and `results/bank_logistic_leakageDemo/`.

## Abstract
We compare three maintenance strategies on the UCI Bank Marketing data (41,188 rows, ordered by date
2008–2010): a frozen model, a model whose probabilities are re-fitted with a sigmoid calibrator on
the latest labelled window, and rolling retraining on all earlier rows. We also evaluate
confidence-based abstention. All strategies are scored on four sequential, leakage-audited evaluation
blocks with Logistic Regression (LR) and HistGradientBoosting (HGB).

Rolling retraining gave the best mean PR-AUC, Brier score, log loss and ECE for both model families.
Its paired-bootstrap PR-AUC gains were significant in blocks 1–2 (LR +0.12 and +0.20; HGB +0.04 and
+0.06), but it was worse than the frozen model in block 0.

Sigmoid recalibration improved Brier score in most blocks. However, on LR it fitted **negative slopes**
on three of four windows and so reversed the model's ranking. This happened in windows with strong prior and
covariate shift.

Simple drift indicators were saturated: KS = 1.0 for the macro-economic features in every block. They
correlated only weakly with frozen-model PR-AUC (r = 0.05 for LR and 0.22 for HGB, n = 4 blocks).

## 1. Introduction and research questions
See proposal §1–3. RQ1: how do metrics evolve for a frozen model? RQ2: how do recalibration and
rolling retraining compare with the frozen model? RQ3: what does abstention buy, and at what coverage?
RQ4: do drift indicators track performance?

## 2. Related work
See `docs/literatureMatrix.md`. In short: concept-drift adaptation [Gama 2014; Lu 2018], post-hoc
calibration [Platt 1999; Guo 2017], selective classification [Geifman & El-Yaniv 2017], shift detection
[Rabanser 2019], and the original Bank Marketing study [Moro 2014]. Our contribution is a joint,
reproducible comparison, not a new method.

## 3. Method
**Data.** UCI `bank-additional-full.csv`, retrieved 2026-11-21. SHA-256 `74adfc57…afb4d8` (see
`data/raw/SOURCE.json`). 41,188 rows, 20 inputs, overall positive rate 0.113. `duration` is excluded
because it is known only after the call ends. Row order is used as a proxy for time; there are no
per-row timestamps.

**Split** (saved in `splitManifest.csv`):

| Segment | Rows |
|---|---|
| Train | 0 – 20,594 |
| Initial calibration | 20,594 – 24,712 |
| Evaluation blocks 0–3 | 24,712 – 41,188, about 4,119 rows each |

**Models.** Both are scikit-learn pipelines with median/constant imputation, scaling and one-hot
encoding, fitted on training rows only, seed 42. The two classifiers are LR (`max_iter=1000`) and HGB
(`max_iter=100`).

**Strategies.**
- *Frozen:* fit once on the train segment.
- *Recalibrated:* same base model, plus a sigmoid (Platt) calibrator. Block 0 uses a calibrator
  fitted on the initial calibration window; block k uses one fitted on block k−1.
- *Rolling:* refit on all rows before block k.

An automated audit confirms that no strategy uses labels from its current or future block
(`leakageAudit.csv`, all `True`). A unit test also flips those labels and checks that predictions do
not change.

**Metrics.**
- PR-AUC, Brier score, log loss, ECE (10 equal-width bins) and precision at top 10%.
- Risk-coverage: error at a 0.5 threshold among the most-confident fraction of cases. Confidence is
  max(p, 1−p), and no threshold is tuned on evaluation labels.
- Drift per feature against the train segment: KS and Wasserstein for numeric features,
  Jensen–Shannon for categorical ones.
- Paired bootstrap 95% intervals (200 resamples) for each strategy minus frozen.
- Synthetic stress tests on the frozen model: numeric shift, missing values and unseen categories.

## 4. Results

### 4.1 RQ1: the frozen model across windows

| Block | Positive rate | LR PR-AUC | LR ECE | HGB PR-AUC | HGB ECE |
|---|---|---|---|---|---|
| 0 | 0.101 | 0.178 | 0.130 | 0.113 | 0.025 |
| 1 | 0.120 | 0.117 | 0.401 | 0.128 | 0.042 |
| 2 | 0.157 | 0.149 | 0.347 | 0.161 | 0.081 |
| 3 | 0.459 | 0.549 | 0.170 | 0.499 | 0.384 |

The last block has a very different base rate (0.46 against 0.10–0.16). PR-AUC rises there, but
PR-AUC is bounded below by prevalence [Saito 2015], so the rise mostly reflects prior shift, not
better ranking. In blocks 0–2 the frozen models' PR-AUC is only about 1.0–1.8 times the prevalence, and frozen LR is
at or below chance in blocks 1–2 (0.117 vs 0.120 and 0.149 vs 0.157).

Calibration of the frozen LR model degrades sharply after block 0 (ECE 0.13 → 0.40). HGB stays fairly
well calibrated until the prior shift in block 3 (ECE 0.38).

### 4.2 RQ2: maintenance strategies

Mean over the four blocks (`summarizeByStrategy`):

| Model | Strategy | PR-AUC | Brier | Log loss | ECE | Prec@10% | Update s | New labels |
|---|---|---|---|---|---|---|---|---|
| LR | frozen | 0.248 | 0.232 | 0.647 | 0.262 | 0.250 | 0 | 0 |
| LR | recalibrated | 0.182 | 0.174 | 0.614 | 0.131 | 0.126 | 0.04 | 4,119 |
| LR | rolling | **0.301** | **0.146** | **0.582** | **0.056** | **0.308** | 0.70 | 4,119 |
| HGB | frozen | 0.225 | 0.183 | 0.606 | 0.133 | 0.239 | 0 | 0 |
| HGB | recalibrated | 0.220 | 0.168 | 0.528 | 0.105 | 0.225 | 0.05 | 4,119 |
| HGB | rolling | **0.253** | **0.155** | **0.498** | **0.079** | **0.286** | 1.54 | 4,119 |

Paired bootstrap, strategy minus frozen, 95% CI (`bootstrap.csv`):

- **Rolling, PR-AUC.** Significantly better in blocks 1 and 2 for both models (LR +0.116 [0.093,
  0.143] and +0.199 [0.170, 0.231]; HGB +0.037 and +0.058), and for HGB in block 3 (+0.028
  [0.002, 0.053]). Significantly *worse* in block 0 (LR −0.106, HGB −0.012).
- **Rolling, Brier.** Better in blocks 1–2 (LR) and block 3 (HGB, −0.114).
- **Recalibrated, Brier.** Better in LR blocks 1–2 (−0.160, −0.132) and HGB blocks 1–3. Worse in LR
  block 3 (+0.078) and HGB block 0 (+0.003).
- **Recalibrated, PR-AUC.** A monotone increasing map cannot change PR-AUC, yet LR PR-AUC fell in
  blocks 0 (−0.107) and 3 (−0.173). The cause is that the fitted Platt slopes were negative for
  blocks 0, 2 and 3 (−1.73, −0.04 and −0.10; only block 1 had a positive slope, +0.68). In block 2
  the reversal happened to *raise* PR-AUC slightly (+0.014, CI includes 0), because the frozen LR
  ranking there was already below chance. On those
  windows the base model's scores were *inversely* related to the labels, so the calibrator reversed
  the ranking. This is a real failure mode of unconstrained recalibration under concept and prior
  shift, not a code defect (verified by inspecting `calibrator.model.coef_`).

**Answer to RQ2.** Rolling retraining was the most reliable strategy overall, at about 15–30× the
update cost of recalibration but still under 2 s per update on a laptop. Recalibration is cheap and
often improves probability quality, but it is fragile when the latest window differs from the next
one. Neither strategy helped in block 0.

### 4.3 RQ3: selective prediction
Mean error among accepted cases over blocks (`riskCoverage.csv`):

| Coverage | LR frozen | LR recal | LR rolling | HGB frozen | HGB recal | HGB rolling |
|---|---|---|---|---|---|---|
| 1.0 | 0.414 | 0.210 | 0.191 | 0.210 | 0.210 | 0.203 |
| 0.8 | 0.397 | 0.224 | 0.169 | 0.204 | 0.207 | 0.194 |
| 0.5 | 0.417 | 0.242 | 0.170 | 0.197 | 0.200 | 0.179 |

Abstention helped only when the model was reasonably well specified. HGB rolling went from an error
of 0.203 to 0.179 by abstaining on half the cases. For the miscalibrated frozen LR model, abstaining
did **not** reduce error (0.414 → 0.417), because its most confident predictions were confidently
wrong under shift. For recalibrated LR, error *increased* as coverage fell, which is consistent with
the reversed ranking described in §4.2.

Under strong prior shift, a 0.5-threshold error rate is dominated by block 3. Confidence-based
abstention is therefore only as good as the calibration of the confidence score.

### 4.4 RQ4: drift indicators versus performance
The macro-economic features (`euribor3m`, `nr.employed`, `emp.var.rate`, `cons.price.idx`) have
KS = 1.0 against the train segment in every block, because their values in later periods do not
overlap the training range. As a result, mean KS is high and nearly flat (0.52–0.62). Its correlation
with frozen PR-AUC is r = 0.05 (LR) and r = 0.22 (HGB), with only 4 blocks.

Block 1 has the highest mean KS (0.62) but not the largest performance change. Block 3 has the
largest change, driven by prior shift, which feature-drift scores do not measure.

**Answer to RQ4.** Feature drift was present everywhere and did not single out the windows where
performance moved. Drift scores should prompt a review; they are not evidence of failure.

### 4.5 Stress tests (synthetic)
- **Frozen LR.** PR-AUC drops from 0.378 to 0.280 at 30% missing values, and to 0.362 with 20% unseen
  categories. A uniform numeric shift leaves PR-AUC unchanged, because a constant shift only moves a
  linear model's logit and keeps the ranking, while Brier changes from 0.232 to 0.165.
- **Frozen HGB.** Largely insensitive: all PR-AUC values are within 0.214–0.226.

These are artificial perturbations and are not estimates of real-world drift.

### 4.6 Leakage demonstration
Keeping `duration` raises frozen LR mean PR-AUC from **0.248 to 0.452** and precision at top 10% from
0.250 to 0.536 (`results/bank_logistic_leakageDemo`). This size of gain is what a careless pipeline
would report, and it shows why `duration` must be excluded. These numbers are not valid results.

## 5. Discussion
The order-based simulation contains covariate shift (macro indicators), prior shift (block 3) and
apparent concept shift (the reversed score–label relationship in the calibration window).

Retraining on all past data was the most robust choice. Recalibrating on only the most recent window
assumes that window resembles the next one, and that assumption failed in three of four windows. Practical safeguards
that follow from this:

- constrain the calibration slope to be ≥ 0, or fall back to intercept-only (prior) correction;
- monitor the label rate directly;
- read PR-AUC relative to block prevalence.

These are suggestions for future work and were **not** applied, so that the pre-declared protocol
stays intact.

## 6. Limitations
- Row order is a proxy for time; no calendar dates are inferred.
- Only four evaluation blocks, so correlations and trends are indicative.
- One dataset and one seed. The optional UCI Adult replication was not done.
- ECE depends on the binning; the 0.5 threshold used for the risk-coverage error is a fixed choice.
- Update time is wall-clock time on one machine.
- The data come from an older Portuguese marketing campaign. This is an educational prototype, not
  a banking decision system, and no results should be used for real customers.

## 7. Ethics and reproducibility
Public dataset (CC BY 4.0), with no new personal data. Each run saves its config, split manifest,
seed, software versions (Python 3.15.0, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1) and data
checksum. The clean-run steps are in `docs/reproducibility.md`. Negative findings (block 0, recalibration reversals, weak drift correlation) are
reported in full.

## References
See `docs/literatureMatrix.md` (proposal references [1]–[6] plus [7]–[12]).
