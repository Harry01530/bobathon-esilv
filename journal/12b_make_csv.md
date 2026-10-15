# 12b_make_csv / 12c_hub_report

**Question:** Can we recover the missing `submissions/12_final_ensemble.csv` and push an EstimatorReport to the hub without re-running the full 12 CV?
**Method:** `12b_make_csv.py` refits `model_a` and `model_b` (TwoStageMLE, prior_scale=0.3, CAT_DTYPES) on all training rows; writes `submissions/12_model_a_only.csv` and `submissions/12_final_ensemble.csv` (unweighted mean, clipped [0,132]). `src/parkinsons/ensemble.py` provides `AvgEnsemble(BaseEstimator)`. `12c_hub_report.py` pushes an `EstimatorReport` (default 0.2 splitter) as `12_final_ensemble`.
**State:** approved — 2025-10-07
