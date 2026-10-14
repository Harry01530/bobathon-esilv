# %% [markdown]
# # 06_two_stage_stack
#
# Builds on 05_two_stage (TwoStageHGB). Stage 1 is unchanged; stage 2 becomes:
#   1. Target transform : stage 2 wrapped in a TransformedTargetRegressor
#                         (TARGET_TRANSFORM below: "none", "log1p", "standard").
#   2. Stacking         : StackingRegressor(HGB + Ridge + ExtraTrees), meta-model
#                         LinearRegression(positive=True). Its internal CV is
#                         patient-grouped (GroupKFold on patient_id), never row KFold.
#   3. Clipping         : predictions clipped to the MDS-UPDRS III range [0, 132]
#                         inside predict() (so CV scores include it) and again
#                         on the CSV.
#
# Same GroupKFold(5) eval folds on patient_id as 02-05. Gate: mean RMSE <= 3.4.
# Feature engineering: src/parkinsons/features.py (unchanged).
# Model: src/parkinsons/two_stage_stack.py (new).

# %%
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skore import login

from parkinsons.features import build_features, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_stack import TwoStageStack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────
TARGET_TRANSFORM = "log1p"  # "none" | "log1p" | "standard" (best measured: log1p)
USE_RIDGE = True
USE_EXTRA_TREES = True
ET_N_ESTIMATORS = 200
CLIP_RANGE = (0.0, 132.0)  # MDS-UPDRS Part III bounds
REF_05_RMSE = 3.26  # mean grouped-CV RMSE of 05_two_stage
GATE_RMSE = 3.4

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
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
log.info("Target: min=%.2f  max=%.2f  skew=%.3f", y.min(), y.max(), pd.Series(y).skew())

# ── Shared category dtype (train + test categories pooled) ───────────────────
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

# ── Eval folds: identical GroupKFold(5) on patient_id as in 02-05 ────────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups)))
log.info("Eval folds are patient-disjoint [ok]")


# ── Feature builder closure (fixes cat_dtypes) ────────────────────────────────
def _build(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw, CAT_DTYPES)


# ── Sanity check ──────────────────────────────────────────────────────────────
log.info("Building features …")
X_full = _build(X_train_raw)
assert len(X_full) == len(X_train_raw), "Row count mismatch"
assert X_full.index.equals(X_train_raw.index), "Index mismatch"
log.info("Feature matrix: %d columns", X_full.shape[1])


# ── CV helper ─────────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), (
        f"{label}: bad folds {folds}"
    )
    arr = np.array(folds)
    log.info(
        "  %-28s folds=%s  mean=%.4f  std=%.4f",
        label,
        [round(v, 3) for v in folds],
        arr.mean(),
        arr.std(),
    )
    return folds, float(arr.mean()), float(arr.std())


# ── Evaluation ────────────────────────────────────────────────────────────────
estimator = TwoStageStack(
    build_features_fn=_build,
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
    target_transform=TARGET_TRANSFORM,
    use_ridge=USE_RIDGE,
    use_extra_trees=USE_EXTRA_TREES,
    et_n_estimators=ET_N_ESTIMATORS,
    clip_range=CLIP_RANGE,
)
label = f"two_stage_stack_{TARGET_TRANSFORM}"
log.info("Evaluating %s on eval_splits (this takes a while) …", label)

cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)
folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, label)

if mean_rmse > GATE_RMSE:
    raise RuntimeError(
        f"STOP: mean RMSE {mean_rmse:.4f} > {GATE_RMSE} — probable bug. "
        "Check build_features alignment and the stacking CV splits."
    )
log.info("Gate passed: mean RMSE %.4f <= %s [ok]", mean_rmse, GATE_RMSE)
gain = REF_05_RMSE - mean_rmse
log.info("Gain vs 05_two_stage (%.2f): %+.4f", REF_05_RMSE, gain)
if gain <= std_rmse:
    log.warning("Gain is within the fold-to-fold spread; it may not be real.")

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


cv_key = free_key(f"06_cv_two_stage_stack_{TARGET_TRANSFORM}")
try:
    cv_url = put_and_capture_url(cv_key, cv_report)
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", cv_key, type(e).__name__, e)
    raise SystemExit(1)

est_key = free_key(f"06_fold5_two_stage_stack_{TARGET_TRANSFORM}")
try:
    est_url = put_and_capture_url(est_key, cv_report.reports_[4])
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", est_key, type(e).__name__, e)
    raise SystemExit(1)

# ── Refit on all training rows and write submission ───────────────────────────
log.info("Refitting %s on all %d training rows …", label, len(X_train_raw))
final_model = clone(estimator).fit(X_train_raw, y)
log.info("Stacking weights (meta-model): %s", final_model.stack_weights())

raw_preds = final_model.predict_unclipped(X_test_raw)
n_out = int(((raw_preds < CLIP_RANGE[0]) | (raw_preds > CLIP_RANGE[1])).sum())
preds = np.clip(raw_preds, final_model.clip_lo_, final_model.clip_hi_)
log.info("Clipping: %d test predictions outside %s were clipped", n_out, CLIP_RANGE)

assert len(preds) == len(sample_sub), f"{len(preds)} vs {len(sample_sub)}"
assert np.all(np.isfinite(preds)), "Non-finite predictions"
assert preds.min() >= CLIP_RANGE[0] and preds.max() <= CLIP_RANGE[1], "Clip failed"

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
log.info(
    "\n%s\n%s : folds=%s  mean=%.4f  std=%.4f  gain_vs_05=%+.4f"
    "\nCV  report URL : %s  [%s]"
    "\nEst report URL : %s  [%s]   <- put this URL on Kaggle"
    "\nSubmission CSV : %s  (%d rows)"
    "\n%s",
    "=" * 62,
    label,
    [round(v, 3) for v in folds],
    mean_rmse,
    std_rmse,
    gain,
    cv_url,
    cv_key,
    est_url,
    est_key,
    csv_path,
    len(submission),
    "=" * 62,
)
log.info("Done — stop before Kaggle upload.")
