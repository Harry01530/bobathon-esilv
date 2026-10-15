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
- **Last experiment:** 13_hybrid_gru — done
- **Last result:** mean grouped-CV RMSE 2.5598 ± 0.0117 (HybridModel HGB+BiGRU, GroupKFold 5, 2026-10-14)
- **Verified Kaggle public result:** 2.642 — 2026-10-14, team Julien Gouel (top 16).
- **Team handoff:** [Othmane's solution, data provenance and next steps](OTHMANE_HANDOFF.md).
- **Final report:** [journal/FINAL_REPORT.md](FINAL_REPORT.md)

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
- **Data version note (2026-10-08):** the inherited EDA above describes the fork CSVs. Othmane's modelling results use the official Kaggle download in `data/kaggle_official/`, which omits `rater_id` and differs in `on`/`off`. See [the handoff](OTHMANE_HANDOFF.md) before reusing the earlier EDA conclusions.

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) baseline — floor for all future models | done | RMSE 16.48 (0.2 holdout) | [01_dummy.md](01_dummy.md) |
| `02_day2_models` | Four-model comparison with 5-fold patient-grouped CV; best: hgbr_mixed | done | mean CV RMSE 7.43 (hgbr_mixed, GroupKFold 5, official data) · Kaggle 7.142 — [CV 50335](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50335) · [Est. 50341](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50341) | [02_day2_models.md](02_day2_models.md) |
| `04_hgb_rich_patient_feats` | Rich per-patient temporal features + tuned HGB | done | mean CV RMSE 3.4831 ± 0.0167 (hgb_tuned_rich, GroupKFold 5) | [04_hgb_rich_patient_feats.md](04_hgb_rich_patient_feats.md) |
| `05_two_stage` | Two-stage HGB: stage-1 OOF predictions as extra features for stage 2 | done | mean CV RMSE 3.2618 ± 0.0255 (TwoStageHGB, GroupKFold 5) — [CV 60514](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60514) · [Est. 60520](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60520) | [05_two_stage.md](05_two_stage.md) |
| `06_two_stage_stack` | Stack HGB+Ridge+ET sur OOF features, log1p target transform | done | mean CV RMSE 3.2583 (two_stage_stack_log1p, GroupKFold 5) — [CV 60813](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60813) · [Est. 60819](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60819) | [06_two_stage_stack.md](06_two_stage_stack.md) |
| `08_two_stage_envelope` | TwoStageHGB + features enveloppe bloc 3 (expectile par patient) | done | mean CV RMSE 2.7129 ± 0.0270 (TwoStageHGB, GroupKFold 5) — [CV 60938](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60938) · [Est. 60944](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60944) | [08_two_stage_envelope.md](08_two_stage_envelope.md) |
| `09_two_stage_mle` | TwoStageMLE : bloc 5 EMG, calibration per-fold, prior_scale=0.5 | done | mean CV RMSE 2.6905 ± 0.0449 (TwoStageMLE, GroupKFold 5) — [CV 61428](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/61428) · [Est. 61434](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/61434) | [09_two_stage_mle.md](09_two_stage_mle.md) |
| `10_mle_tuning` | Grid search TwoStageMLE ; meilleur : prior_scale=0.3 | done | mean CV RMSE 2.6772 ± 0.0338 (+0.0133 vs 09) — [CV 61684](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/61684) · [Est. 61690](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/61690) | — |
| `11_target_vcorr` | Stage 1 sur map_Vcorr clippé [0,132], stage 2 sur résidu y−ŷ₁ | done | mean CV RMSE 2.8127 ± 0.0445 (régresse vs 10 : −0.1355) | — |
| `12_voting_ensemble` | VotingRegressor(model_a, model_b) avec prior_scale=0.3 | done | mean CV RMSE 2.6784 ± 0.0297 · Kaggle 2.708 — [CV 83280](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/83280) · [Est. 82849](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/82849) | — |
| `13_hybrid_gru` | HybridModel : HGB ensemble (smooth=None) + BiGRU 2 couches hidden=64, puis smooth_per_patient quad_iso | done | mean CV RMSE **2.5598** ± 0.0117 · Kaggle **2.642** (meilleur équipe) — [CV 85779](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/85779) · [Est. 85398](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/85398) | — |

## Backlog

| # | Item | Source |
|---|---|---|
