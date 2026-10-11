# 04_hgb_rich_patient_feats

**Question:** Do rich per-patient temporal features (neighbour shifts, cumulative stats, polynomial trends) lower GroupKFold RMSE below experiment 03?
**Method:** Same `GroupKFold(5)` eval folds as 03. `build_features(raw)` sorts by `(patient_id, age)`, computes all features, returns in original row order. `HistGradientBoostingRegressor` with NaN-native handling + `RandomizedSearchCV(n_iter=20)` on shuffled tuning folds (seed 42). Best of default vs tuned pushed to hub as `04_cv_hgb_rich` / `04_fold5_hgb_rich`.
**State:** approved — 2025-10-07
