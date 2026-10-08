# 02_day2_models

## Question / hypothesis

Does adding patient-grouped cross-validation, missingness-aware boosting, and
categorical features improve over the Day 1 dummy baseline (RMSE 16.48)?

## Motivation

- **Sourcing strategy:** Day 2 lab guide (docs/DAY2.md)
- **Why this matters:** The Day 1 dummy used a random 80/20 row holdout that
  can leak the same patient across train and validation. Grouped CV (GroupKFold
  by patient_id) gives a realistic estimate of out-of-patient performance,
  matching how Kaggle evaluates submissions.

## Data source

`data/kaggle_official/` — official Kaggle competition ZIP, extracted
2026-10-07. Columns: `Index`, `patient_id`, `cohort`, `sexM`, `gene`,
`age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
`time_since_intake_off`, `on`, `off`. No `rater_id` column (differs from
local fork files).

## Method

- **CV:** `GroupKFold(n_splits=5, groups=patient_id)` — splits precomputed
  once and shared by all numeric models. Patient disjointness verified
  (0 patients in both train and fold-5 holdout).
- **Excluded from X:** `target` (outcome), `Index` (row id), `patient_id`
  (grouping key only).
- **y alignment:** `y_train` indexed by `Index`, reindexed to `X_train` row
  order before any split.
- **Per-fold RMSE (published script):** extracted with
  `cv_report.metrics.get("rmse", data_source="test", aggregate=None)`.
  Exactly five finite values are required before their mean is calculated.
  The default `cv_report.metrics.rmse()` returns aggregated mean and std;
  those two summary values must not be averaged together.
- **Held-out EstimatorReport:** `cv_report.reports_[4]` (fold 5) — used
  directly without any refit. No call to `create_estimator_report`.

### Models compared

| Name | Description |
|---|---|
| `dummy_mean` | `DummyRegressor(strategy="mean")` — floor |
| `ridge` | `SimpleImputer(median)` → `Ridge(alpha=1.0)` |
| `hgbr_num` | `HistGradientBoostingRegressor(random_state=0)` on 8 numeric cols (NaN-native) |
| `hgbr_mixed` | `TableVectorizer()` → `HistGradientBoostingRegressor(random_state=0)` on 8 numeric + `cohort` + `gene` as `category` |

`hgbr_mixed` used a separate `cv_splits_mixed` (same row order, same groups).

## Results

### 5-fold GroupKFold RMSE

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean** | Std |
|---|---|---|---|---|---|---|---|
| `dummy_mean` | 16.5852 | 16.8800 | 15.9478 | 16.4698 | 16.6132 | **16.4992** | 0.3067 |
| `ridge` | 10.4414 | 10.7578 | 10.3068 | 10.2501 | 10.4267 | **10.4365** | 0.1760 |
| `hgbr_num` | 7.4859 | 7.5796 | 7.2635 | 7.3098 | 7.5327 | **7.4343** | 0.1250 |
| `hgbr_mixed` | 7.4849 | 7.5502 | 7.2587 | 7.3109 | 7.5303 | **7.4270** | 0.1192 |

**Best model: `hgbr_mixed`** — mean grouped-CV RMSE **7.43** vs dummy floor
16.50, a 55 % reduction. Adding `cohort` and `gene` (categorical) gave a
marginal additional improvement over numeric-only HGBR (7.43 vs 7.43; ~0.007
RMSE difference, less than one standard deviation — not material at this scale).

### Hub reports (printed URLs, not constructed)

| Report | Key | Printed URL |
|---|---|---|
| Best CV (5-fold, `hgbr_mixed`) | `02_cv_hgbr_mixed_othmane_2` | <https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50335> |
| Held-out fold-5 EstimatorReport | `02_fold5_hgbr_mixed_othmane_2` | <https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50341> |

Note: an earlier run (`_othmane` without suffix) was also uploaded during
development. The `_2` run supplied the CSV actually submitted to Kaggle.
The final script was verified in a subsequent `_3` run using the metric
accessor documented above: [CV 50437](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50437),
[EstimatorReport 50443](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50443).
The `_2` and `_3` CSVs are byte-for-byte identical, and both have the same
five-fold RMSEs. The Kaggle description still correctly points to report 50341.

### Submission CSV

- **File:** `submissions/02_fold5_hgbr_mixed_othmane_2.csv`
- **Rows:** 11 013 (matches `sample_submission.csv`)
- **Columns:** `Index`, `target`
- **Prediction range:** [4.00, 85.81] — finite, no NaNs
- **Prediction mean:** 36.78 (vs training target mean 37.47)
- **Index order:** aligned to `sample_submission.csv`

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** mean grouped-CV RMSE **7.43** (`hgbr_mixed`, 5-fold
  patient-grouped, official Kaggle data)
- **Comparison against the grouped-CV dummy:** 16.50 → 7.43 (−55 %).
  The Day 1 random-row holdout uses a different protocol.
- **Implication:** HGBR improves substantially over Ridge in this comparison.
  No feature-removal or imputation ablation was run, so this experiment
  does not isolate the cause of that gain. Adding `cohort`/`gene` produces
  a very small mean improvement (< 0.01 RMSE), insufficient to establish
  a robust advantage. Use these existing scores before trying one targeted
  change in Bob; see [the team handoff](OTHMANE_HANDOFF.md).

## Kaggle submission — 7 October 2026

Submitted by Othmane HASSANI to team Julien Gouel. Kaggle showed **Success** and public RMSE **7.142**, versus Julien’s dummy submission **16.421** (56.5% lower error).

File: `02_fold5_hgbr_mixed_othmane_2.csv`. Submission Description includes the exact EstimatorReport URL: <https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50341>.

[Submission page](https://www.kaggle.com/competitions/bobathon-esilv/submissions). Official event eligibility remains unconfirmed because the Overview and Rules list conflicting submission windows. The user approved submitting today with this limitation.
