# EDA: Parkinson's Disease Motor Score (UPDRS)

_Generated from `data/eda.py` on 2025-10-07._

## Dataset at a glance

- **Tables:** 2 (train: features + target, test: features only)
- **Shape:** train 44 590 × 14 (13 features + `target`), test 11 013 × 13
- **Target:** `target` — continuous UPDRS motor score (regression)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | Dtype | Null % | n\_unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0.0% | 44 590 | Row identifier — unique per row, drop before modelling |
| `patient_id` | String | 0.0% | 5 576 | 5 576 distinct patients across 44 590 rows — many rows per patient |
| `cohort` | String | 0.0% | 2 | Binary study cohort label |
| `sexM` | Int64 | 0.0% | 2 | Binary: 0 / 1 |
| `gene` | String | **32.4%** | 4 | High missingness; 4 gene-mutation categories |
| `age_at_diagnosis` | Float64 | **5.2%** | 563 | Some missing; highly correlated with `age` (Pearson 0.94) |
| `age` | Float64 | 0.0% | 1 082 | No missing |
| `ledd` | Float64 | **36.6%** | 1 320 | Levodopa equivalent daily dose; heavy missingness |
| `time_since_intake_on` | Float64 | **46.4%** | 64 | Nearly half missing — measured only when medication status is "on" |
| `time_since_intake_off` | Float64 | **78.8%** | 178 | Very heavy missingness — measured only when medication status is "off" |
| `rater_id` | String | 0.0% | 50 | 50 distinct raters; potential random effect / batch variable |
| `on` | Float64 | **29.6%** | 83 | UPDRS score in "on" medication state — strong predictor |
| `off` | Float64 | **42.4%** | 101 | UPDRS score in "off" medication state — strongest predictor |
| `target` | Float64 | 0.0% | 867 | The outcome to predict; no missing values |

**Highlights:**
- `time_since_intake_off` (79%) and `time_since_intake_on` (46%) are the two most-missing columns. Their missingness is likely structural (only recorded at the matching medication state), not random.
- `gene` has 32% missingness — needs imputation or an indicator.
- `age_at_diagnosis` and `age` are near-collinear (Pearson 0.94); one may be redundant.
- `Index` is a row index with unique ratio 1.0 — must be excluded from modelling features.

## Target

Continuous UPDRS motor score ranging from **0.0 to 109.5**.

- **Mean:** 37.47 | **Median:** 37.3 | **IQR:** 25.6 – 49.3 | **SD:** 16.5
- **Distribution:** roughly bell-shaped, slightly right-skewed. The histogram shows a near-normal bulk between 20 and 65, with a long tail toward 100+.

  | Decile bin | Count |
  |---|---|
  | 0 – 11 | 2 477 |
  | 11 – 22 | 5 641 |
  | 22 – 33 | 9 764 |
  | 33 – 44 | 10 703 |
  | 44 – 55 | 8 844 |
  | 55 – 66 | 5 203 |
  | 66 – 77 | 1 757 |
  | 77 – 88 | 180 |
  | 88 – 99 | 13 |
  | 99 – 110 | 8 |

- No missing values in the target.

## Structure

- **No datetime columns** detected by skrub (no `date`-typed columns).
- **Group structure:** `patient_id` has 5 576 unique values across 44 590 rows (unique ratio 0.125) — each patient contributes ~8 rows on average. This is the dominant structural signal.
- `rater_id` has 50 distinct raters, a secondary grouping.
- No temporal ordering column found, though the study likely has visit order within a patient (not present as a feature).

## Associations

**Strongest feature → target links (Pearson):**

| Feature | Pearson r | Cramér V | Notes |
|---|---|---|---|
| `off` | **0.871** | 0.406 | UPDRS "off" state — very strong linear signal |
| `on` | **0.669** | 0.368 | UPDRS "on" state — strong |
| `age` | 0.310 | 0.114 | Moderate |
| `ledd` | 0.298 | 0.236 | Moderate |
| `age_at_diagnosis` | 0.133 | 0.051 | Weak |
| `time_since_intake_off` | 0.008 | 0.092 | Near-zero Pearson but moderate Cramér V — possible non-linear link |
| `time_since_intake_on` | 0.000 | 0.177 | Practically no linear signal |
| `cohort` | n/a | 0.090 | Weak |
| `gene` | n/a | 0.052 | Weak |
| `patient_id` | n/a | 0.028 | Low Cramér V — expected, as it is an identifier |

**Notable feature ↔ feature links:**
- `age_at_diagnosis` ↔ `age`: Pearson 0.94, Cramér V 0.56 — near-collinear.
- `on` ↔ `off`: Pearson 0.87, Cramér V 0.33 — correlated (expected medically).
- `ledd` ↔ `on`: Pearson 0.20.

**⚠ Leakage flag:** `off` and `on` are UPDRS motor scores measured *at the same assessment visit* as `target`. Depending on the exact definition of `target`, these could be close to direct leaks or partial leaks (e.g. if `target` is computed from a combination that includes `on` or `off`). **Verify with the problem definition before using them as input features.**

## Modelling implications

- **Group leakage risk:** `patient_id` repeats across rows (~8 rows/patient). A plain `KFold` would mix the same patient across train and validation folds, inflating the CV score. → Use `GroupKFold` (or `GroupShuffleSplit`) with `groups=patient_id` to ensure patient-level hold-out.
- **Regression task, near-normal target:** `KFold` suffices (no stratification needed for continuous targets). The right-skew tail (a few scores > 80) may warrant a log/sqrt target transform if residuals are heteroscedastic — test in experiment 01.
- **High missingness in `time_since_intake_off` (79%) and `ledd` (37%):** These columns need imputation. skrub's `tabular_pipeline` applies `SimpleImputer` by default; verify it covers these.
- **`off` / `on` leakage check:** If `target` is indeed derived from `on`/`off`, drop them. If they are legitimate predictors (a different measurement), they will dominate the model — consider running a baseline with and without them.
- **`age_at_diagnosis` ↔ `age` collinearity:** Dropping one is safe; keep `age` (no missing).
- **`rater_id` as a random effect:** 50 raters could introduce systematic bias. A model-agnostic approach is to include `rater_id` as a categorical feature; a more rigorous one is to residualise it. Start simple (include as feature) and inspect rater residuals.
- **`Index` must be excluded** from the feature matrix.
- **Metric:** RMSE or MAE are natural for this continuous target. Check the competition's official metric (the `sample_submission.csv` implies a Kaggle-style setup).

## Open questions

1. **Is `target` derived from `on` and/or `off`?** If so, these are leaks and must be excluded (or handled as the "previous assessment" score if the task is longitudinal).
2. **What does `cohort` encode?** The two cohorts may have different data-collection protocols, which would matter for the splitter.
3. **Visit ordering within a patient:** Is there a visit sequence column not included here? A `TimeSeriesSplit`-style approach within patients would be more realistic than `GroupKFold` if the task is longitudinal prediction.
4. **Is `rater_id` present in the test set?** (It is — same 50 raters.) Including it as a feature is fine.
5. **Competition metric:** What is the official evaluation metric? The sample submission format suggests a standard regression leaderboard; confirm before choosing between RMSE and MAE.
