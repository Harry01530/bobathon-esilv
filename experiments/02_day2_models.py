# %% [markdown]
# # 02_day2_models — four-model comparison, 5-fold patient-grouped CV
#
# Data source: data/kaggle_official/ (official Kaggle ZIP, no rater_id).
#
# Four estimators evaluated on **identical** GroupKFold(5) patient-grouped
# splits precomputed from patient_id:
#   - dummy_mean   : DummyRegressor(strategy="mean")
#   - ridge        : SimpleImputer(median) → Ridge(alpha=1)
#   - hgbr_num     : HistGradientBoostingRegressor (numeric cols, NaN native)
#   - hgbr_mixed   : TableVectorizer → HistGradientBoostingRegressor (+ cohort/gene)
#
# The three numeric models are compared via one ComparisonReport (same X).
# hgbr_mixed runs separately (different X with category dtypes).
#
# Per-model per-fold RMSEs are extracted via:
#   cv_report.metrics.get("rmse", data_source="test", aggregate=None)
# which returns a (1 × n_splits) DataFrame — one column per fold.
# .rmse() / .metrics.get("rmse") with default aggregate=("mean","std") returns
# only mean and std; averaging those two numbers would be wrong.
#
# The best individual CrossValidationReport is uploaded to Hub.
# The held-out fold EstimatorReport is reports_[4] of that CV (already fitted,
# no re-fit, no data leakage from fold reuse).
#
# project.put prints "Consult your report at <URL>" to the rich console.
# We intercept that with skore.console.begin_capture() / end_capture() to
# capture the exact numeric URL — never constructing it from a UUID.

# %%
# ── Imports ───────────────────────────────────────────────────────────────────
import json  # write the metadata file
import re  # regex used to extract the report URL from console output
from pathlib import Path  # file path handling

import numpy as np  # numerical computing
import pandas as pd  # CSV loading / data manipulation
import skore  # experiment tracking and ML reports
from skore import login  # Skore Hub authentication
from sklearn.base import clone  # copy an unfitted estimator with the same hyperparameters
from sklearn.dummy import DummyRegressor  # baseline model (predicts a constant)
from sklearn.ensemble import HistGradientBoostingRegressor  # gradient boosting, handles NaN natively
from sklearn.impute import SimpleImputer  # fills missing values
from sklearn.linear_model import Ridge  # linear regression with L2 regularisation
from sklearn.model_selection import GroupKFold  # CV that keeps groups (patients) together
from sklearn.pipeline import make_pipeline  # chain preprocessing + model
from skrub import TableVectorizer  # automatic encoding of mixed-type tables

from parkinsons.hub import load_skore_credentials  # loads Skore Hub credentials

# ── Paths ─────────────────────────────────────────────────────────────────────
# Project root = parent of the folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "kaggle_official"   # official competition source
# Output folder for submissions (created if it does not exist)
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ── Load and align by Index ───────────────────────────────────────────────────
# Training features
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
# Target, reordered to match the row order of X_train_raw
y_series = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]  # guarantee row alignment
)
# Plain numpy array of targets
y = y_series.values

# ── Feature sets (excludes target, Index, patient_id) ────────────────────────
# Purely numeric columns
NUMERIC_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
# Numeric columns plus two categorical ones
MIXED_COLS = NUMERIC_COLS + ["cohort", "gene"]

# Numeric-only feature matrix
X_num = X_train_raw[NUMERIC_COLS]

# Mixed feature matrix: cohort and gene are cast to pandas "category" dtype
X_mixed = X_train_raw[MIXED_COLS].copy()
X_mixed["cohort"] = X_mixed["cohort"].astype("category")
X_mixed["gene"]   = X_mixed["gene"].astype("category")

# ── Precompute identical GroupKFold(5) splits ─────────────────────────────────
# Splits are precomputed once so all models use the exact same fold boundaries.
# Patient id is the group: a patient is never in both train and test of a fold
groups = X_train_raw["patient_id"].values
gkf = GroupKFold(n_splits=5)
cv_splits_num   = list(gkf.split(X_num, y, groups=groups))
cv_splits_mixed = list(gkf.split(X_mixed, y, groups=groups))

# Sanity: patient disjointness on last fold
tr_idx, te_idx = cv_splits_num[-1]
tr_patients = set(groups[tr_idx])
te_patients = set(groups[te_idx])
# No patient may appear in both the train and the test side
assert len(tr_patients & te_patients) == 0, "Patient leak in fold!"
print(f"Train: {X_num.shape}  |  {len(np.unique(groups))} patients  |  5-fold GroupKFold")
print(f"Fold-5 holdout: {len(te_idx)} rows, {len(te_patients)} patients (disjoint ✓)")

# ── Define estimators ─────────────────────────────────────────────────────────
# Baseline: always predicts the training mean
dummy   = DummyRegressor(strategy="mean")
# Linear model: median imputation, then Ridge
ridge   = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
# Gradient boosting on numeric columns (NaN handled natively)
hgbr_num   = HistGradientBoostingRegressor(random_state=0)
# Gradient boosting on mixed columns: TableVectorizer encodes categoricals first
hgbr_mixed = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)

# ── Step 1: ComparisonReport for the three numeric models ─────────────────────
# evaluate({dict}, X, y, splitter=list_of_tuples) →
#   ComparisonReport whose .reports_ = {name: CrossValidationReport}
print("\nRunning GroupKFold(5) for dummy / ridge / hgbr_num …")
comp_report = skore.evaluate(
    {"dummy_mean": dummy, "ridge": ridge, "hgbr_num": hgbr_num},
    X_num,
    y,
    splitter=cv_splits_num,
)

# ── Step 2: Separate CrossValidationReport for hgbr_mixed ────────────────────
# Runs separately because its feature matrix (X_mixed) differs from X_num
print("Running GroupKFold(5) for hgbr_mixed …")
cv_mixed = skore.evaluate(hgbr_mixed, X_mixed, y, splitter=cv_splits_mixed)

# ── Extract per-model per-fold RMSE ──────────────────────────────────────────
# comp_report.reports_ is a dict[str, CrossValidationReport]

def cv_fold_rmses(cv_report: skore.CrossValidationReport, label: str):
    """Return per-fold RMSEs from a CrossValidationReport.

    Uses cv_report.metrics.get("rmse", data_source="test", aggregate=None),
    which returns a (1 × n_splits) DataFrame — the only correct way to obtain
    individual fold values.  .rmse() uses the default aggregate=("mean","std")
    and returns only those two summary statistics; averaging them is wrong.
    """
    # One RMSE per fold, computed on the test side of each split
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    fold_vals = df.values.flatten().tolist()
    # Sanity: exactly 5 finite values required before selecting best
    assert len(fold_vals) == 5, (
        f"{label}: expected 5 fold RMSEs, got {len(fold_vals)}: {fold_vals}"
    )
    assert all(np.isfinite(v) for v in fold_vals), (
        f"{label}: non-finite fold RMSE: {fold_vals}"
    )
    arr = np.array(fold_vals)
    # Print fold values, mean and standard deviation
    print(f"  {label:16s}: folds={[round(v,4) for v in arr]}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    return fold_vals, float(arr.mean())


print("\n=== Grouped CV RMSE (5-fold, patient-grouped) ===")
# Collect everything needed later (report, folds, mean, data, model, splits) per model
results = {}
for name, cv_rep in comp_report.reports_.items():
    folds, mean = cv_fold_rmses(cv_rep, name)
    results[name] = {"cv_report": cv_rep, "folds": folds, "mean": mean,
                     "X": X_num, "model": {"dummy_mean": dummy, "ridge": ridge,
                                           "hgbr_num": hgbr_num}[name],
                     "cv_splits": cv_splits_num}

# Same bookkeeping for the mixed model (uses X_mixed and its own splits)
mixed_folds, mixed_mean = cv_fold_rmses(cv_mixed, "hgbr_mixed")
results["hgbr_mixed"] = {"cv_report": cv_mixed, "folds": mixed_folds,
                         "mean": mixed_mean, "X": X_mixed,
                         "model": hgbr_mixed, "cv_splits": cv_splits_mixed}

# Best model = lowest mean CV RMSE
best_name = min(results, key=lambda k: results[k]["mean"])
print(f"\nBest model: {best_name}  mean CV RMSE = {results[best_name]['mean']:.4f}")

# ── Hub upload ────────────────────────────────────────────────────────────────
# Load credentials, log in and open the project on the Hub
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# Keys already present in the project, to avoid overwriting existing reports
existing_keys = set(project.summarize().frame()["key"].tolist())

def free_key(base: str, existing: set) -> str:
    # Returns `base` if free, otherwise `base_2`, `base_3`, … until an unused key is found
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"

def put_and_capture_url(key: str, report) -> str:
    """Call project.put and return the URL printed to the rich console.

    project.put prints: Consult your report at [link=URL]URL[/link]
    We capture the rich console output and extract the bare URL with a
    pattern that stops at whitespace or rich markup closing brackets.
    """
    # Start capturing the console output
    skore.console.begin_capture()
    project.put(key, report)
    # Stop capturing and get what was printed
    captured = skore.console.end_capture()
    # Match https:// followed by any non-whitespace, non-bracket chars
    urls = re.findall(r'https://[^\s\]>]+', captured)
    if not urls:
        return "url-not-captured"
    url = urls[-1].rstrip("/")           # strip any trailing slash
    assert " " not in url, f"Whitespace in captured URL: {url!r}"
    return url


# Upload only the best individual CrossValidationReport (not the ComparisonReport)
best_cv_key = free_key(f"02_cv_{best_name}_othmane", existing_keys)
print(f"\nUploading best CV report ({best_name}) as '{best_cv_key}' …")
best_cv_url = put_and_capture_url(best_cv_key, results[best_name]["cv_report"])
existing_keys.add(best_cv_key)  # remember the key to avoid a later collision
print(f"  Printed URL: {best_cv_url}")

# Upload the already-fitted held-out fold EstimatorReport (fold index 4 = last fold)
# CrossValidationReport.reports_[4] was fitted on folds 0-3, evaluated on fold 4.
# We do NOT call create_estimator_report — that would refit and conflate train rows.
fold_est_report = results[best_name]["cv_report"].reports_[4]
est_key = free_key(f"02_fold5_{best_name}_othmane", existing_keys)
print(f"Uploading held-out fold-5 EstimatorReport as '{est_key}' …")
est_url = put_and_capture_url(est_key, fold_est_report)
existing_keys.add(est_key)
print(f"  Printed URL: {est_url}")

# Recap of the two uploads
print(f"\n{'='*60}")
print(f"Best CV report key : {best_cv_key}")
print(f"Best CV report URL : {best_cv_url}")
print(f"EstimatorReport key: {est_key}")
print(f"EstimatorReport URL: {est_url}")
print(f"{'='*60}")

# ── Fit on all training data and write submission CSV ─────────────────────────
print(f"\nFitting {best_name} on all {len(X_num)} training rows …")
# Use the feature matrix that matches the best model (numeric or mixed)
X_best = results[best_name]["X"]
# Fresh unfitted copy of the best model, then fit on 100% of the training data
final_model = clone(results[best_name]["model"])
final_model.fit(X_best, y)

# Load test features
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
# Build test features with the same columns/dtypes the best model was trained on
if best_name == "hgbr_mixed":
    X_test_feat = X_test_raw[MIXED_COLS].copy()
    X_test_feat["cohort"] = X_test_feat["cohort"].astype("category")
    X_test_feat["gene"]   = X_test_feat["gene"].astype("category")
else:
    X_test_feat = X_test_raw[NUMERIC_COLS]

# Predict on the test set
preds = final_model.predict(X_test_feat)

# Validate
# Compare with the sample submission: same number of rows, no NaN/infinite values
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(preds) == len(sample_sub), f"Row count mismatch: {len(preds)} vs {len(sample_sub)}"
assert np.all(np.isfinite(preds)), "Non-finite predictions detected"

# Align to sample_submission Index order
submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
# Format check: exactly the two columns Kaggle expects
assert list(submission.columns) == ["Index", "target"]

# Save the CSV, named after the EstimatorReport key
csv_path = SUBMISSIONS_DIR / f"{est_key}.csv"
submission.to_csv(csv_path, index=False)
print(f"Submission CSV: {csv_path}  ({len(submission)} rows, columns={list(submission.columns)})")
print(f"  pred range: [{preds.min():.2f}, {preds.max():.2f}]  mean={preds.mean():.4f}")

# ── Metadata JSON ─────────────────────────────────────────────────────────────
# Traceability record: data, best model, CV scores, Hub keys/URLs and output path
metadata = {
    "data_source":           str(DATA_DIR),
    "model_name":            best_name,
    "cv_splits":             "GroupKFold(n_splits=5, groups=patient_id)",
    "fold_rmse":             results[best_name]["folds"],
    "mean_cv_rmse":          results[best_name]["mean"],
    "all_models_mean_rmse":  {k: v["mean"] for k, v in results.items()},
    "best_cv_key":           best_cv_key,
    "best_cv_url":           best_cv_url,
    "estimator_report_key":  est_key,
    "estimator_report_url":  est_url,
    "csv_path":              str(csv_path),
}
meta_path = SUBMISSIONS_DIR / f"{est_key}_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"Metadata JSON:  {meta_path}")
print("\nDone — stop before Kaggle upload.")
