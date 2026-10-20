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
# ── Imports: standard library ─────────────────────────────────────────────────
import logging  # logging (console + file)
import re  # (imported but not used in this script)
import time  # used to track the time budget of the Hub upload
from pathlib import Path  # file path handling

# ── Imports: data science ─────────────────────────────────────────────────────
import numpy as np  # numerical computing
import pandas as pd  # CSV loading / data manipulation
import skore  # experiment tracking and ML reports (Skore Hub)
from sklearn.base import clone  # copies an unfitted estimator with the same hyperparameters
from sklearn.ensemble import VotingRegressor  # ensemble of regressors (averages predictions)
from sklearn.model_selection import GroupKFold  # cross-validation with groups (here: patients)
from skore import login  # Skore Hub authentication

# ── Imports: project modules ──────────────────────────────────────────────────
from parkinsons.features import BIAS_PARAMS, make_cat_dtypes  # default bias parameters + categorical dtypes helper
from parkinsons.hub import load_skore_credentials  # loads Skore Hub credentials
from parkinsons.two_stage_mle import TwoStageMLE  # two-stage model (MLE-based estimation)

# ── Logging configuration ─────────────────────────────────────────────────────
# Log file placed next to the script
_LOG_FILE = Path(__file__).resolve().parent / "12_voting_ensemble_run.log"
logging.basicConfig(
    level=logging.INFO,  # minimum level displayed: INFO
    format="%(asctime)s  %(levelname)-8s  %(message)s",  # timestamp + level + message
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # print to console
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),  # write to file (overwritten at each run)
    ],
)
log = logging.getLogger(__name__)

# Best bias parameters: the defaults, with prior_scale set to 0.3
BEST_BIAS_PARAMS = {**BIAS_PARAMS, "prior_scale": 0.3}

# ── Paths ─────────────────────────────────────────────────────────────────────
# Project root = parent of the folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Folder with the official Kaggle data
DATA_DIR = PROJECT_ROOT / "data" / "kaggle_official"
# Output folder for submissions (created if it does not exist)
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
# Train and test features, and the sample submission file
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
# Target: index by "Index", then reorder to match X_train so that X and y are aligned
y = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]
    .values
)
# Patient identifier: used as the group to avoid leakage between train and test
groups = X_train_raw["patient_id"].values
# Consistent categorical dtypes across train and test
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups)))

# ── GroupKFold(5) — identical splits as exp. 09/10/12 ─────────────────────────
# 5-fold split where a given patient never appears in both train and test
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
# Safety check: no patient shared between train and test in any fold
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak fold {k}"

# ── Build VotingRegressor — shallow + deep members ────────────────────────────
# Model A: "shallow" (300 iterations, learning rate 0.05, max 31 leaves)
model_a = TwoStageMLE(
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,  # number of inner folds (first stage)
    smooth="quad_iso",  # smoothing method
    bias_params=BEST_BIAS_PARAMS,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
)
# Model B: "deep" (600 iterations, lower learning rate 0.03, max 63 leaves)
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
# The models are cloned so the ensemble holds its own independent copies
voting = VotingRegressor(estimators=[("shallow", clone(model_a)), ("deep", clone(model_b))])

# Verify clone round-trips and is_regressor
# Checks that sklearn recognises the ensemble as a regressor and that cloning preserves hyperparameters
from sklearn.base import clone as _clone, is_regressor
assert is_regressor(voting), "VotingRegressor is not recognised as regressor"
test_clone = _clone(voting)
assert test_clone.estimators[0][1].max_iter == 300
assert test_clone.estimators[1][1].max_iter == 600
log.info("VotingRegressor clone/is_regressor checks passed")

# ── Evaluate VotingRegressor directly via skore.evaluate ─────────────────────
log.info("Running skore.evaluate(VotingRegressor, GroupKFold5) …")
# Cross-validate the ensemble using the splits defined above; produces a skore report
cv = skore.evaluate(voting, X_train_raw, y, splitter=eval_splits)

# Extract per-fold RMSE
# RMSE on each fold's test data (aggregate=None → one value per fold)
df_rmse = cv.metrics.rmse(data_source="test", aggregate=None)
folds = df_rmse.values.flatten().tolist()
arr = np.array(folds)
# Log per-fold RMSE, their mean and standard deviation
log.info(
    "VotingRegressor  folds=%s  mean=%.4f  std=%.4f",
    [round(v, 3) for v in folds],
    arr.mean(),
    arr.std(),
)

# ── Hub upload (hard 180s budget, retry=False) ────────────────────────────────
# Load credentials and log in to the Hub
cfg = load_skore_credentials()
login(mode="hub")
# Open (or create) the project on the Hub
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# Patch HUBClient to disable retries
# Replace the Hub client class with a subclass that forces retry=False,
# so automatic retries cannot blow past the time budget
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
    # If the patch fails (e.g. skore version change), carry on without it
    log.warning("HUBClient patch skipped: %s", _e)

# Collect existing keys to avoid collisions
# Fetch the keys already in the project so existing reports are not overwritten
try:
    _summary = project.summarize()
    existing_keys: set[str] = set(_summary.frame()["key"].tolist())
except Exception:
    existing_keys = set()

def free_key(base: str) -> str:
    # Returns `base` if free, otherwise `base_2`, `base_3`, … until an unused key is found
    if base not in existing_keys:
        return base
    i = 2
    while f"{base}_{i}" in existing_keys:
        i += 1
    return f"{base}_{i}"

# Deadline: 180 seconds from now for the whole upload
_HUB_DEADLINE = time.monotonic() + 180

def _put_timed(key: str, obj) -> str:
    """project.put with URL capture; raises TimeoutError if budget exhausted."""
    # Refuse to upload if the time budget is exceeded
    if time.monotonic() > _HUB_DEADLINE:
        raise TimeoutError("180s Hub budget exhausted")
    project.put(key, obj)
    existing_keys.add(key)  # remember the key to avoid a later collision
    log.info("  put '%s' ok", key)
    return key

# Unique keys for the estimator report and the cross-validation report
est_key = free_key("12_voting_ensemble")
cv_key  = free_key("12_cv_voting_ensemble")

# Default values if the upload fails or is interrupted
est_url = "not-uploaded"
cv_url  = "not-uploaded"

try:
    # EstimatorReport first (fold 4 / last fold)
    # Report of the last fold (index 4) is uploaded first
    est_report = cv.reports_[4]
    _put_timed(est_key, est_report)
    est_url = f"https://app.skore.ai/{cfg['workspace']}/project/bobathon-esilv"
    log.info("EstimatorReport key=%s", est_key)

    # Then the full cross-validation report
    _put_timed(cv_key, cv)
    cv_url = est_url  # same project URL
    log.info("CV report key=%s", cv_key)

except TimeoutError as _te:
    # 180 s budget exceeded: keep going, results remain available locally
    log.warning("Hub upload aborted: %s — artefacts saved locally", _te)
except Exception as _e:
    # Any other upload error does not stop the rest of the script
    log.error("Hub upload failed: %s: %s — artefacts saved locally", type(_e).__name__, _e)

# ── Refit VotingRegressor on all training data ────────────────────────────────
log.info("Refitting VotingRegressor on all %d rows …", len(X_train_raw))
# Retrain on 100% of the training data (on a fresh, unfitted clone)
final_voting = clone(voting).fit(X_train_raw, y)

# Test predictions, clipped between 0 and 132 (valid range of the target)
preds = np.clip(final_voting.predict(X_test_raw), 0.0, 132.0)

# Sanity checks: correct number of rows and no NaN/infinite values
assert len(preds) == len(sample_sub), f"{len(preds)} vs {len(sample_sub)}"
assert np.all(np.isfinite(preds)), "Non-finite predictions"

# Build the submission file, reordered to follow sample_submission
submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
# Check the format expected by Kaggle: 2 columns and 11013 rows
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == 11013

# Save the CSV, named after the CV report key
csv_path = SUBMISSIONS_DIR / f"{cv_key}.csv"
submission.to_csv(csv_path, index=False)
log.info("CSV saved → %s  (%d rows)", csv_path, len(submission))

# ── Save metadata ──────────────────────────────────────────────────────────────
import json
# Metadata dictionary for experiment traceability
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
# Final recap framed by lines of "="
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

# Empty entry point: all the work runs at module level
if __name__ == "__main__":
    pass
