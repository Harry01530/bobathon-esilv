# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's Disease UPDRS Motor Score — regression
- **Goal:** Predict the UPDRS motor score (`target`) for each patient visit from clinical features; minimise prediction error on the held-out test set.
- **Last experiment:** 01_dummy — done
- **Last result:** RMSE 16.48 (0.2 holdout)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-10-07
  - env manager: pip+venv - recorded: 2025-10-07
  - agent feature: installed - recorded: 2025-10-07
  - optional features: none - recorded: 2025-10-07
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: hub - recorded: 2025-10-07
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: n/a - recorded: 2025-10-07
  - student prior: beginner - recorded: 2025-10-07
  - CV splitter family: GroupKFold (patient_id) - recorded: 2025-10-07

## Data understanding (EDA)

- **Status:** done - 2025-10-07
- **Summary:** Train set 44 590 rows × 13 features; `target` is a continuous UPDRS motor score (range 0–110, roughly bell-shaped). `patient_id` repeats ~8×/patient → GroupKFold required. `off` (r=0.87) and `on` (r=0.67) are the dominant predictors — verify they are not leaks from `target`'s definition. Three columns have heavy structural missingness (`time_since_intake_off` 79%, `ledd` 37%, `time_since_intake_on` 46%).
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) baseline — floor for all future models | done | RMSE 16.48 (0.2 holdout) | [01_dummy.md](01_dummy.md) |

## Backlog

| # | Item | Source |
|---|---|---|
