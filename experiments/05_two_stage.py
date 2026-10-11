# %% [markdown]
# # 05_two_stage
#
# Two-stage HGB: stage-1 OOF predictions become extra features for stage 2.
# Same GroupKFold(5) eval folds as 03/04. Gate: mean RMSE must be <= 3.4.
#
# Feature engineering lives in src/parkinsons/features.py.
# TwoStageHGB lives in src/parkinsons/two_stage.py.
# This script does NOT import or execute experiment 04 in any way.

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from parkinsons.features import build_features, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage import TwoStageHGB

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_DIR        = PROJECT_ROOT / "data" / "Kaggle_officiel"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv")
y = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]
    .values
)
groups = X_train_raw["patient_id"].values

# ── Shared category dtype (train + test categories pooled) ───────────────────
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

# ── Eval folds: identical GroupKFold(5) on patient_id as in 02/03/04 ─────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
print(f"Train: {len(X_train_raw)} rows, {len(np.unique(groups))} patients")
print("Eval folds are patient-disjoint [ok]")

# ── Feature builder closure (fixes cat_dtypes) ────────────────────────────────
def _build(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw, CAT_DTYPES)


# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nBuilding features …")
X_full = _build(X_train_raw)
assert len(X_full) == len(X_train_raw), "Row count mismatch"
assert X_full.index.equals(X_train_raw.index), "Index mismatch"
print(f"Feature matrix: {X_full.shape[1]} columns")


# ── CV helper ─────────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df    = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), \
        f"{label}: bad folds {folds}"
    arr = np.array(folds)
    print(f"  {label:28s} folds={[round(v, 3) for v in folds]} "
          f"mean={arr.mean():.4f} std={arr.std():.4f}")
    return folds, float(arr.mean()), float(arr.std())


# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating TwoStageHGB on eval_splits …")

estimator = TwoStageHGB(
    build_features_fn=_build,
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
)

cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)
folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, "two_stage_hgb")

if mean_rmse > 3.4:
    raise RuntimeError(
        f"STOP: mean RMSE {mean_rmse:.4f} > 3.4 — probable bug in TwoStageHGB. "
        "Check build_features alignment and OOF prediction logic."
    )
print(f"Gate passed: mean RMSE {mean_rmse:.4f} <= 3.4 [ok]")

# ── Hub upload ────────────────────────────────────────────────────────────────
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


cv_key  = free_key("05_cv_two_stage")
try:
    cv_url = put_and_capture_url(cv_key, cv_report)
except Exception as e:
    print(f"STOP: project.put({cv_key!r}) failed: {type(e).__name__}: {e}")
    raise SystemExit(1)

est_key = free_key("05_fold5_two_stage")
try:
    est_url = put_and_capture_url(est_key, cv_report.reports_[4])
except Exception as e:
    print(f"STOP: project.put({est_key!r}) failed: {type(e).__name__}: {e}")
    raise SystemExit(1)

# ── Refit on all training rows and write submission ───────────────────────────
print(f"\nRefitting TwoStageHGB on all {len(X_train_raw)} training rows …")
final_model = clone(estimator).fit(X_train_raw, y)
preds = final_model.predict(X_test_raw)

assert len(preds) == len(sample_sub), f"{len(preds)} vs {len(sample_sub)}"
assert np.all(np.isfinite(preds)), "Non-finite predictions"

submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == 11013

csv_path = SUBMISSIONS_DIR / f"{est_key}.csv"
submission.to_csv(csv_path, index=False)

# ── Final output ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print(f"TwoStageHGB : folds={[round(v, 3) for v in folds]}  "
      f"mean={mean_rmse:.4f}  std={std_rmse:.4f}")
print(f"CV  report URL : {cv_url}  [{cv_key}]")
print(f"Est report URL : {est_url}  [{est_key}]")
print(f"Submission CSV : {csv_path}  ({len(submission)} rows)")
print(f"{'=' * 62}")
print("Done — stop before Kaggle upload.")
