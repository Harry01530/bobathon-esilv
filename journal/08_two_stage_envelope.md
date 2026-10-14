# 08_two_stage_envelope

**Question:** Does a linear mixed-effects (LME) model of the corrected ON/OFF measurements beat 07? What actually limits 07?

**LME (block 4 of `features.py`, `lme_trajectory`):** fixed effects = population curve (cubic in years since diagnosis + diagnosis age, sex, cohort, gene). Random effects per patient = intercept, slope, curvature and ON offset, with a full 4×4 covariance. One residual variance per measurement type. Fitted by EM (maximum likelihood) in numpy. `statsmodels.MixedLM` cannot have one residual variance per type, and is too slow for 4 random effects on ~60k measurements refitted per fold.

- Alone, it estimates the latent better than 07 (proxy RMSE 5.28 vs 5.65).
- But it does not help the HGB:

| Model | Mean RMSE |
|---|---:|
| Single-stage HGB, 07 features | 3.320 |
| Single-stage HGB, LME instead of 07 latent | 3.337 |
| Two-stage, 08 features | 2.7129 |
| Two-stage, 08 features + LME | 2.7176 |

So the LME stays in the code, off by default (`lme_params=False`).

**What limits 07 (official train; target used for analysis only):**
- Within a visit, the ON and OFF deviations from the target correlate at **0.977**. There is one multiplicative "rater" bias per visit, shared by ON and OFF. 07 and the LME wrongly counted them as two independent measurements.
- That bias is independent across visits (lag-1 autocorrelation 0.01) and unrelated to cohort, gene, sex, age, LEDD or timing. Without `rater_id` it cannot be removed, only averaged.
- It is very asymmetric. OFF/target has its mode at 1.02 and its q75 at 1.00, but a q25 of 0.66. Measurements are "right or too low", so the latent lies near the upper envelope of the visit values. The running maximum of the visit values alone scores proxy 5.66, as good as the whole 07 latent.
- The ratio ON/OFF within a visit is almost bias-free, so it gives the drug effect and each patient's drug response precisely.

**Method (block 3, `visit_envelope`, all target-free):**
1. Drug curve `m(t) = median(ON/OFF)` per bin of `time_since_intake_on`, over visits with both scores.
2. Patient drug factor `rho = median((ON/OFF)/m(t_on))`, shrunk to 1 with 2 pseudo-visits.
3. One value per visit: OFF, or `ON / (m·rho)`, or the mean of both.
4. Per-patient penalised quadratic fitted by expectile regression (asymmetric least squares, IRLS) at tau = 0.5 / 0.8 / 0.9, then isotonic projection.
5. Running maximum of the visit values, and each visit's ratio to the upper envelope.

**Result (TwoStageHGB, same GroupKFold(5), features rebuilt per fold):**

| Model | Fold RMSE | Mean |
|---|---|---:|
| 05 | 3.285 3.223 3.280 3.277 3.248 | 3.2627 |
| 07 | 3.176 3.095 3.127 3.126 3.188 | 3.1424 |
| **08** | 2.733 2.709 2.671 2.703 2.748 | **2.7129** |

Gain vs 07: −0.43, better on all 5 folds; fold std 0.027. Block-3 settings were not tuned (λ reused from 07, taus fixed a priori).

**State:** script ready — run `experiments/08_two_stage_envelope.py` in Bob.
