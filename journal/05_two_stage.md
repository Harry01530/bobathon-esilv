# 05_two_stage

**Question:** Does a two-stage HGB (stage-1 OOF predictions as extra features for stage 2) beat the single-stage model from 04?
**Method:** `TwoStageHGB` in `src/parkinsons/two_stage.py`. Stage 1 = HGB(300, lr=0.05); inner GroupKFold(5) on `patient_id` for OOF predictions. Stage 2 = HGB(300, lr=0.05) on base features + prediction features (shifts, poly trends, rolling means of stage-1 preds). Eval on identical GroupKFold(5) as 03/04. Expected ~3.18 mean RMSE; gate at 3.4.
**State:** approved — 2025-10-07
