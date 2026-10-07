# 01_dummy

## Question / hypothesis

Does always predicting the mean UPDRS score give a sensible lower-bound
baseline, and what RMSE does that establish for future experiments to beat?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Run a dummy regressor from sklearn to act as my baseline model."
- **Why this matters:** A `DummyRegressor(strategy="mean")` ignores all
  features and just predicts the training-set mean for every visit. Its
  error tells us the absolute worst we should tolerate from any real
  model — if a real model can't beat this, something is wrong with it.

## Method

- **Files touched:** `src/parkinsons/pipeline.py`, `src/parkinsons/evaluate.py`
- **Feature columns (fixed):** `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
  `patient_id` and `Index` are never in X; `patient_id` is used only as
  the grouping key for the splitter.
- **Change versus baseline:** This IS the first baseline. The pipeline
  wraps a single `DummyRegressor(strategy="mean")` inside a skrub DataOps
  graph so that `split_kwargs={"groups": patient_id}` at `mark_as_X`
  flows naturally into `GroupShuffleSplit` via `skore.evaluate`.
- **Cross-validation:** `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)`
  with `groups=patient_id`, so no patient leaks across the train/validation
  boundary. Groups are wired at `mark_as_X(split_kwargs={"groups": ...})`
  and flow into `skore.evaluate` → `CrossValidationReport` automatically.
- **Report pushed to skore hub:** project `bobathon-esilv`, workspace
  `julien.gouel`, run name `01_dummy`.
- **Submission:** after hub push, fit the dummy on all X_train, write
  `submissions/01_dummy.csv` with columns `Index,target` validated against
  `data/sample_submission.csv`.
- **Out of scope for this experiment:** feature engineering, imputation,
  any real estimator. Purpose is only to establish the floor.

## Risks / things that could invalidate the result

- The mean is computed on the training split only; RMSE on the held-out
  split is the number that matters.
- If the train/test split happens to be unrepresentative (e.g., one cohort
  ends up mostly in one partition), the baseline RMSE may be misleading —
  a multi-fold GroupKFold would be more stable, but for a dummy baseline
  one split is fine.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 16.48 (0.2 holdout, default skore split)
- **Implication for next iteration:** Any real model must beat RMSE 16.48.
  The dominant predictors `off` (r=0.87) and `on` (r=0.67) suggest a
  linear or gradient-boosting regressor on the 8 feature columns should
  do substantially better.
