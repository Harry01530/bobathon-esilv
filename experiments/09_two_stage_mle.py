# %% [markdown]
# # 09_two_stage_mle
#
# TwoStageMLE = the 05-08 two-stage HGB, plus:
#   * block 5 of features.py: per-patient maximum likelihood of the latent
#     "true OFF" curve under an exponentially-modified Gaussian model of the
#     asymmetric visit bias (one noise model per visit type), with an
#     empirical-Bayes prior on (level, slope, curvature);
#   * a final per-patient projection of the predictions (quadratic in age,
#     then non-decreasing).
# The bias model is calibrated INSIDE each training fold (TwoStageMLE.fit), so
# the cross-validation never uses the held-out patients' targets.
# Same GroupKFold(5) eval folds as 03-08.
# Gate: mean RMSE must be <= GATE_RMSE (08 scored 2.7129).
#
# Features: src/parkinsons/features.py.  Model: src/parkinsons/two_stage_mle.py.

# %%
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skore import login

from parkinsons.features import (
    BIAS_PARAMS,
    build_features,
    calibrate_bias_model,
    make_cat_dtypes,
)
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_mle import TwoStageMLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GATE_RMSE = 2.75  # pre-measured 09 (honest, calibration inside folds): 2.6877
REF_08_RMSE = 2.7129

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

# ── Shared category dtype (train + test categories pooled) ───────────────────
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

# ── Eval folds: identical GroupKFold(5) on patient_id as in 02-08 ────────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups)))
log.info("Eval folds are patient-disjoint [ok]")

# ── Sanity check: block 5 is present and the calibration works ───────────────
log.info("Calibrating the bias model on all training rows (sanity check) …")
calib_all = calibrate_bias_model(X_train_raw, y)
X_full = build_features(X_train_raw, CAT_DTYPES, bias_calibration=calib_all)
assert len(X_full) == len(X_train_raw) and X_full.index.equals(X_train_raw.index)
blk5 = [c for c in X_full.columns if c.startswith(("mle_", "map_"))]
assert blk5, "Block 5 missing: is features.py v4 installed?"
log.info("Feature matrix: %d columns, block 5: %s", X_full.shape[1], blk5)
names = ["off_untreated", "off_treated", "both", "on_t_known", "on_t_missing"]
for k, n in enumerate(names):
    p = calib_all["noise"][k]
    s40, m40 = np.exp(p[1] + p[2] * np.log(40)), np.exp(p[3] + p[4] * np.log(40))
    log.info(
        "  visit type %-14s scale c=%.3f  sigma@40=%.2f  mu@40=%.2f",
        n,
        np.exp(p[0]),
        s40,
        m40,
    )


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


# ── Evaluation (calibration refitted inside every training fold) ─────────────
log.info("Evaluating TwoStageMLE on eval_splits …")
estimator = TwoStageMLE(cat_dtypes=CAT_DTYPES, inner_n_splits=5, smooth="quad_iso")
cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)
folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, "two_stage_mle")

if mean_rmse > GATE_RMSE:
    raise RuntimeError(
        f"STOP: mean RMSE {mean_rmse:.4f} > {GATE_RMSE} — probable bug in TwoStageMLE "
        "or in block 5 of features.py."
    )
log.info(
    "Gate passed: mean RMSE %.4f <= %.2f [ok]  (gain vs 08: %+.4f)",
    mean_rmse,
    GATE_RMSE,
    REF_08_RMSE - mean_rmse,
)

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


cv_key = free_key("09_cv_two_stage_mle")
try:
    cv_url = put_and_capture_url(cv_key, cv_report)
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", cv_key, type(e).__name__, e)
    raise SystemExit(1)

est_key = free_key("09_fold5_two_stage_mle")
try:
    est_url = put_and_capture_url(est_key, cv_report.reports_[4])
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", est_key, type(e).__name__, e)
    raise SystemExit(1)

# ── Refit on all training rows (calibration on all of train) and submit ──────
log.info("Refitting TwoStageMLE on all %d training rows …", len(X_train_raw))
final_model = clone(estimator).fit(X_train_raw, y)
preds = final_model.predict(X_test_raw)  # already projected and clipped
preds = np.clip(preds, 0.0, 132.0)

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

meta = {
    "fold_rmse": folds,
    "mean_rmse": mean_rmse,
    "std_rmse": std_rmse,
    "bias_params": BIAS_PARAMS,
    "calibration_full_train": {
        k: np.asarray(v).round(6).tolist() for k, v in final_model.calibration_.items()
    },
    "cv_report": {"key": cv_key, "url": cv_url},
    "estimator_report": {"key": est_key, "url": est_url},
}
(SUBMISSIONS_DIR / f"{est_key}_metadata.json").write_text(json.dumps(meta, indent=2))

# ── Final output ──────────────────────────────────────────────────────────────
log.info(
    "\n%s\nTwoStageMLE : folds=%s  mean=%.4f  std=%.4f"
    "\nCV  report URL : %s  [%s]"
    "\nEst report URL : %s  [%s]   <- put this URL on Kaggle"
    "\nSubmission CSV : %s  (%d rows)"
    "\n%s",
    "=" * 62,
    [round(v, 3) for v in folds],
    mean_rmse,
    std_rmse,
    cv_url,
    cv_key,
    est_url,
    est_key,
    csv_path,
    len(submission),
    "=" * 62,
)
log.info("Done — stop before Kaggle upload.")
