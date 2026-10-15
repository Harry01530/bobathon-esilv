# 06_two_stage_stack

**Question:** Does stage 2 of `TwoStageHGB` (05) improve with (1) a target transform, (2) a stack of HGB + Ridge + ExtraTrees and (3) clipping to the MDS-UPDRS III range?

**Method:** `TwoStageStack` in `src/parkinsons/two_stage_stack.py`. Stage 1 is the same as in 05. Stage 2 = `TransformedTargetRegressor(StackingRegressor(HGB, Ridge, ExtraTrees), log1p)`, with meta-model `LinearRegression(positive=True)`. The stacking's internal CV reuses the patient-grouped inner splits (GroupKFold on `patient_id`), never row KFold. Predictions are clipped to [0, 132] inside `predict()`. Evaluation on the same GroupKFold(5) as 02-05, on the official data.

**Target:** min 0, max 109.5, skew 0.05 (nearly symmetric), so a log transform has little to correct.

**Pre-measurement (same folds, same stage 1, every stage-2 variant):**

| Stage 2 | Fold RMSE | Mean | Δ vs 05 | Folds won |
|---|---|---:|---:|---:|
| HGB alone (= 05) | 3.285 3.223 3.280 3.277 3.248 | 3.2627 | — | — |
| Stack HGB+Ridge+ET, no transform | 3.287 3.218 3.313 3.271 3.259 | 3.2696 | +0.0069 | 2/5 |
| Stack HGB+ET, no transform | 3.267 3.211 3.300 3.273 3.251 | 3.2603 | −0.0024 | 3/5 |
| **Stack HGB+Ridge+ET, log1p** | 3.270 3.204 3.302 3.276 3.239 | **3.2583** | **−0.0045** | 4/5 |

- Clipping changes RMSE by at most 0.0002: only 0-25 predictions per fold fall outside [0, 132].
- With log1p, the meta-model gives Ridge a weight of about 0.005, so the stack is in practice HGB (~0.42) + ExtraTrees (~0.57).
- `"standard"` (StandardScaler on the target) is available in the code but was not measured on the 5 folds.
- The best gain (0.0045) is far below the fold-to-fold spread (~0.025) and the gap to the top of the leaderboard (~1.0).

**Conclusion:** the three techniques are implemented and validated but bring no meaningful gain. The remaining error is not in the stage-2 model. It is in how each patient's latent "true OFF" trajectory is estimated from noisy ON/OFF visits.

**State:** script ready — run `experiments/06_two_stage_stack.py` in Bob to get the official Hub reports.
