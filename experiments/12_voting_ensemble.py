# %% [markdown]
# # 12_voting_ensemble
#
# Unified Skore Hub report: wrap model_a (shallow300/lr.05/leaf31) and
# model_b (deep600/lr.03/leaf63), both with prior_scale=0.3, inside a
# sklearn VotingRegressor.  Evaluate the VotingRegressor directly via
# skore.evaluate so the jury sees ONE report instead of two.
#
# Commit provenance: src/parkinsons/two_stage_mle.py from
#   https://github.com/Harry01530/bobathon-esilv @ a608a6f86a0f7886f46982bf19535fcfb8b99cd6
# MRO fix applied: RegressorMixin before BaseEstimator (sklearn 1.9 is_regressor).

# %%
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GroupKFold
from skore import login

from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_mle import TwoStageMLE

_LOG_FILE = Path(__file__).resolve().parent / "12_voting_ensemble_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

BEST_BIAS_PARAMS = {**BIAS_PARAMS, "prior_scale": 0.3}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "kaggle_official"
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
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups)))

# ── GroupKFold(5) — identical splits as exp. 09/10/12 ─────────────────────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak fold {k}"

# ── Build VotingRegressor — shallow + deep members ────────────────────────────
model_a = TwoStageMLE(
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
    smooth="quad_iso",
    bias_params=BEST_BIAS_PARAMS,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
)
model_b = TwoStageMLE(
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
    smooth="quad_iso",
    bias_params=BEST_BIAS_PARAMS,
    max_iter=600,
    learning_rate=0.03,
    max_leaf_nodes=63,
    min_samples_leaf=20,
)

# VotingRegressor with equal weights (default) → simple mean
voting = VotingRegressor(estimators=[("shallow", clone(model_a)), ("deep", clone(model_b))])

# Verify clone round-trips and is_regressor
from sklearn.base import clone as _clone, is_regressor
assert is_regressor(voting), "VotingRegressor is not recognised as regressor"
test_clone = _clone(voting)
assert test_clone.estimators[0][1].max_iter == 300
assert test_clone.estimators[1][1].max_iter == 600
log.info("VotingRegressor clone/is_regressor checks passed")

# ── Evaluate VotingRegressor directly via skore.evaluate ─────────────────────
log.info("Running skore.evaluate(VotingRegressor, GroupKFold5) …")
cv = skore.evaluate(voting, X_train_raw, y, splitter=eval_splits)

# Extract per-fold RMSE
df_rmse = cv.metrics.rmse(data_source="test", aggregate=None)
folds = df_rmse.values.flatten().tolist()
arr = np.array(folds)
log.info(
    "VotingRegressor  folds=%s  mean=%.4f  std=%.4f",
    [round(v, 3) for v in folds],
    arr.mean(),
    arr.std(),
)

# ── Hub upload (hard 180s budget, retry=False) ────────────────────────────────
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# Patch HUBClient to disable retries
try:
    import skore._plugins.hub.client.client as _hub_client_mod
    _OrigHUBClient = _hub_client_mod.HUBClient
    class _NoRetryHUBClient(_OrigHUBClient):
        def __init__(self, *args, **kwargs):
            kwargs["retry"] = False
            super().__init__(*args, **kwargs)
    _hub_client_mod.HUBClient = _NoRetryHUBClient
    log.info("HUBClient patched: retry=False")
except Exception as _e:
    log.warning("HUBClient patch skipped: %s", _e)

# Collect existing keys to avoid collisions
try:
    _summary = project.summarize()
    existing_keys: set[str] = set(_summary.frame()["key"].tolist())
except Exception:
    existing_keys = set()

def free_key(base: str) -> str:
    if base not in existing_keys:
        return base
    i = 2
    while f"{base}_{i}" in existing_keys:
        i += 1
    return f"{base}_{i}"

_HUB_DEADLINE = time.monotonic() + 180

def _put_timed(key: str, obj) -> str:
    """project.put with URL capture; raises TimeoutError if budget exhausted."""
    if time.monotonic() > _HUB_DEADLINE:
        raise TimeoutError("180s Hub budget exhausted")
    project.put(key, obj)
    existing_keys.add(key)
    log.info("  put '%s' ok", key)
    return key

est_key = free_key("12_voting_ensemble")
cv_key  = free_key("12_cv_voting_ensemble")

est_url = "not-uploaded"
cv_url  = "not-uploaded"

try:
    # EstimatorReport first (fold 4 / last fold)
    est_report = cv.reports_[4]
    _put_timed(est_key, est_report)
    est_url = f"https://app.skore.ai/{cfg['workspace']}/project/bobathon-esilv"
    log.info("EstimatorReport key=%s", est_key)

    _put_timed(cv_key, cv)
    cv_url = est_url  # same project URL
    log.info("CV report key=%s", cv_key)

except TimeoutError as _te:
    log.warning("Hub upload aborted: %s — artefacts saved locally", _te)
except Exception as _e:
    log.error("Hub upload failed: %s: %s — artefacts saved locally", type(_e).__name__, _e)

# ── Refit VotingRegressor on all training data ────────────────────────────────
log.info("Refitting VotingRegressor on all %d rows …", len(X_train_raw))
final_voting = clone(voting).fit(X_train_raw, y)

preds = np.clip(final_voting.predict(X_test_raw), 0.0, 132.0)

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

csv_path = SUBMISSIONS_DIR / f"{cv_key}.csv"
submission.to_csv(csv_path, index=False)
log.info("CSV saved → %s  (%d rows)", csv_path, len(submission))

# ── Save metadata ──────────────────────────────────────────────────────────────
import json
meta = {
    "experiment": "12_voting_ensemble",
    "provenance": "https://github.com/Harry01530/bobathon-esilv @ a608a6f86a0f7886f46982bf19535fcfb8b99cd6",
    "mro_fix": "RegressorMixin before BaseEstimator (sklearn 1.9)",
    "cv_folds_rmse": [round(v, 4) for v in folds],
    "cv_mean_rmse": round(float(arr.mean()), 4),
    "cv_std_rmse":  round(float(arr.std()),  4),
    "hub_est_key": est_key,
    "hub_cv_key":  cv_key,
    "hub_url":     est_url,
    "csv_path":    str(csv_path),
}
meta_path = SUBMISSIONS_DIR / "12_voting_ensemble_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
log.info("Metadata → %s", meta_path)

# ── Summary ───────────────────────────────────────────────────────────────────
log.info(
    "\n%s"
    "\nVotingRegressor  folds=%s  mean=%.4f  std=%.4f"
    "\nEst  key: %-35s  url: %s"
    "\nCV   key: %-35s  url: %s"
    "\nCSV path: %s  (%d rows)"
    "\n%s",
    "=" * 66,
    [round(v, 3) for v in folds], arr.mean(), arr.std(),
    est_key, est_url,
    cv_key,  cv_url,
    csv_path, len(submission),
    "=" * 66,
)
log.info("Done — stop before Kaggle upload.")

if __name__ == "__main__":
    pass
