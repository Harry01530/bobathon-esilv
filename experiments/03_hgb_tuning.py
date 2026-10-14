# %% [markdown]
# # 03_hgb_tuning — hyperparameter search on HistGradientBoosting
#
# Builds on experiments/02_day2_models.py (Othmane, best = hgbr_mixed,
# grouped-CV RMSE 7.427018).
#
# Data source: data/kaggle_official/ (same as 02).
#
# Protocol (patient_id is ONLY used to build groups, never as a feature):
#   1. Tuning folds: 5 patient-grouped folds built from a SHUFFLED patient order
#      (seed 42). They differ from the evaluation folds, which limits the
#      optimism of picking hyperparameters on the same folds we report on.
#   2. RandomizedSearchCV over HGB hyperparameters on those tuning folds.
#   3. Evaluation folds: GroupKFold(n_splits=5) on patient_id, identical to 02,
#      so the tuned model is compared to 7.427018 on the same patients/folds.
#      The default HGB is re-evaluated on the same X and folds as a control.
#   4. Two feature sets are tried:
#        - "mixed"         : 8 numeric cols + cohort + gene (category dtype)
#        - "mixed_patient" : "mixed" + per-patient aggregates built from X only
#                            (never from target); test patients have several
#                            visits too, so these are computable on X_test.
#   5. Best (feature set, params) by mean grouped-CV RMSE is uploaded to the Hub
#      (CrossValidationReport + held-out fold-5 EstimatorReport, as in 02), then
#      refit on all training rows to write the submission CSV.

# %%
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from parkinson.hub import load_skore_credentials
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skore import login

# ── Settings ──────────────────────────────────────────────────────────────────
N_ITER = 30  # random-search candidates per feature set
TUNE_SEED = 42  # seed for the shuffled tuning folds
FEATURE_SETS = ["mixed", "mixed_patient"]
OTHMANE_BEST_RMSE = 7.427018

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CAT_COLS = ["cohort", "gene"]
FORBIDDEN = {"Index", "patient_id", "target"}

# ── Load and align by Index (same as 02) ──────────────────────────────────────
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
y = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]
    .values
)
groups = X_train_raw["patient_id"].values

# Same category list on train and test, so codes mean the same thing.
CAT_DTYPES = {
    c: pd.CategoricalDtype(
        sorted(pd.concat([X_train_raw[c], X_test_raw[c]]).dropna().unique())
    )
    for c in CAT_COLS
}


# ── Features ──────────────────────────────────────────────────────────────────
def patient_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-visit and per-patient features built from X columns only."""
    out = pd.DataFrame(index=raw.index)
    g = raw.groupby("patient_id")
    out["years_dx"] = raw["age"] - raw["age_at_diagnosis"]
    for c in ["off", "on", "ledd"]:
        out[f"{c}_na"] = raw[c].isna().astype(int)
    out["off_minus_on"] = raw["off"] - raw["on"]
    out["n_vis"] = g["age"].transform("size")
    out["age_first"] = g["age"].transform("min")
    out["since_first"] = raw["age"] - out["age_first"]
    out["vis_rank"] = g["age"].rank(method="first")
    for c in ["off", "on", "ledd"]:
        out[f"{c}_pmean"] = g[c].transform("mean")
        out[f"{c}_pmax"] = g[c].transform("max")
    out["off_dev"] = raw["off"] - out["off_pmean"]
    return out


def build_X(raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    X = raw[NUMERIC_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(CAT_DTYPES[c])
    if feature_set == "mixed_patient":
        X = pd.concat([X, patient_features(raw)], axis=1)
    elif feature_set != "mixed":
        raise ValueError(f"Unknown feature set: {feature_set}")
    leaked = FORBIDDEN & set(X.columns)
    assert not leaked, f"Forbidden columns in features: {leaked}"
    return X


# ── Folds ─────────────────────────────────────────────────────────────────────
def shuffled_group_folds(groups: np.ndarray, n_splits: int, seed: int):
    """Patient-grouped folds from a shuffled patient order (any sklearn version)."""
    patients = np.unique(groups)
    np.random.default_rng(seed).shuffle(patients)
    fold_of = {p: i % n_splits for i, p in enumerate(patients)}
    fold = np.array([fold_of[p] for p in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def check_disjoint(splits, label):
    for k, (tr, te) in enumerate(splits):
        assert not set(groups[tr]) & set(groups[te]), (
            f"{label}: patient leak in fold {k}"
        )


tune_splits = shuffled_group_folds(groups, 5, TUNE_SEED)
# Evaluation folds: identical to 02 (GroupKFold does not depend on X content).
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
check_disjoint(tune_splits, "tuning")
check_disjoint(eval_splits, "evaluation")
print(f"Train: {len(X_train_raw)} rows, {len(np.unique(groups))} patients")
print("Tuning and evaluation folds are patient-disjoint [ok]")

# ── Search space ──────────────────────────────────────────────────────────────
base_hgb = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    early_stopping=False,
    random_state=0,
)
param_distributions = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_iter": randint(150, 801),
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 50, 100, 200],
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": [0.5, 0.8, 1.0],
}


# ── Helpers (same logic as 02) ────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), f"{label}: {folds}"
    arr = np.array(folds)
    print(
        f"  {label:28s} folds={[round(v, 3) for v in folds]} "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )
    return folds, float(arr.mean()), float(arr.std())


# ── Search + evaluation per feature set ───────────────────────────────────────
results = {}
for fs in FEATURE_SETS:
    X = build_X(X_train_raw, fs)
    print(f"\n=== Feature set '{fs}' ({X.shape[1]} columns) ===")
    print(f"Random search: {N_ITER} candidates x 5 tuning folds ...")
    search = RandomizedSearchCV(
        base_hgb,
        param_distributions,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=tune_splits,
        refit=False,
        n_jobs=-1,
        random_state=0,
    )
    search.fit(X, y)
    best_params = search.best_params_
    print(f"  best tuning-fold RMSE = {-search.best_score_:.4f}")
    print(f"  best params = {best_params}")

    tuned = clone(base_hgb).set_params(**best_params)
    default = clone(base_hgb)

    print("Evaluation on the 02 folds (GroupKFold on patient_id):")
    comp = skore.evaluate(
        {f"hgb_default_{fs}": default, f"hgb_tuned_{fs}": tuned},
        X,
        y,
        splitter=eval_splits,
    )
    for name, cv_rep in comp.reports_.items():
        folds, mean, std = cv_fold_rmses(cv_rep, name)
        results[name] = {
            "feature_set": fs,
            "cv_report": cv_rep,
            "model": default if name.startswith("hgb_default") else tuned,
            "params": None if name.startswith("hgb_default") else best_params,
            "folds": folds,
            "mean": mean,
            "std": std,
        }

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Summary (reference: Othmane hgbr_mixed = {OTHMANE_BEST_RMSE}) ===")
for name, r in sorted(results.items(), key=lambda kv: kv[1]["mean"]):
    delta = r["mean"] - OTHMANE_BEST_RMSE
    print(f"  {name:28s} mean={r['mean']:.4f}  std={r['std']:.4f}  delta={delta:+.4f}")

best_name = min(results, key=lambda k: results[k]["mean"])
best = results[best_name]
gain = OTHMANE_BEST_RMSE - best["mean"]
print(f"\nBest: {best_name}  mean={best['mean']:.4f}  gain vs 02 = {gain:+.4f}")
if gain <= best["std"]:
    print("WARNING: gain is within the fold-to-fold spread; it may not be real.")

# ── Hub upload (same pattern as 02) ───────────────────────────────────────────
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
existing_keys = set(project.summarize().frame()["key"].tolist())


def free_key(base: str) -> str:
    if base not in existing_keys:
        return base
    i = 2
    while f"{base}_{i}" in existing_keys:
        i += 1
    return f"{base}_{i}"


def put_and_capture_url(key: str, report) -> str:
    skore.console.begin_capture()
    project.put(key, report)
    captured = skore.console.end_capture()
    existing_keys.add(key)
    urls = re.findall(r"https://[^\s\]>]+", captured)
    return urls[-1].rstrip("/") if urls else "url-not-captured"


cv_key = free_key(f"03_cv_{best_name}")
cv_url = put_and_capture_url(cv_key, best["cv_report"])
est_key = free_key(f"03_fold5_{best_name}")
est_url = put_and_capture_url(est_key, best["cv_report"].reports_[4])

# ── Refit on all training rows and write the submission ───────────────────────
fs = best["feature_set"]
final_model = clone(best["model"]).fit(build_X(X_train_raw, fs), y)
X_test = build_X(X_test_raw, fs)
preds = final_model.predict(X_test)

assert len(preds) == len(sample_sub), f"{len(preds)} vs {len(sample_sub)} rows"
assert np.all(np.isfinite(preds)), "Non-finite predictions"
submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
assert list(submission.columns) == ["Index", "target"]
csv_path = SUBMISSIONS_DIR / f"{est_key}.csv"
submission.to_csv(csv_path, index=False)

metadata = {
    "data_source": str(DATA_DIR),
    "tuning_folds": f"5 patient-grouped folds, shuffled patients, seed {TUNE_SEED}",
    "evaluation_folds": "GroupKFold(n_splits=5, groups=patient_id), same as 02",
    "n_iter": N_ITER,
    "reference_rmse_02": OTHMANE_BEST_RMSE,
    "all_models": {
        k: {
            "feature_set": v["feature_set"],
            "params": v["params"],
            "fold_rmse": v["folds"],
            "mean": v["mean"],
            "std": v["std"],
        }
        for k, v in results.items()
    },
    "best_model": best_name,
    "cv_report_key": cv_key,
    "cv_report_url": cv_url,
    "estimator_report_key": est_key,
    "estimator_report_url": est_url,
    "csv_path": str(csv_path),
}
meta_path = SUBMISSIONS_DIR / f"{est_key}_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2, default=str))

print(f"\n{'=' * 60}")
print(f"CV report       : {cv_key}  {cv_url}")
print(f"EstimatorReport : {est_key}  {est_url}   <- put this URL on Kaggle")
print(f"Submission CSV  : {csv_path}")
print(f"Metadata JSON   : {meta_path}")
print(f"{'=' * 60}")
print("Done - stop before Kaggle upload.")
