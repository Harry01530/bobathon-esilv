# %% [markdown]
# # 12_final_ensemble
#
# Variance-reduction ensemble: average predictions from two TwoStageMLE
# configurations that differ only in their HGB depth:
#
#   model_a  (shallow) : max_iter=300, lr=0.05, max_leaf_nodes=31  — best of 10
#   model_b  (deep)    : max_iter=600, lr=0.03, max_leaf_nodes=63  — A_deep_hgb
#
# Both use prior_scale=0.3 (best bias prior from exp. 10 / D_prior_03).
# The ensemble prediction is the unweighted mean of the two.
# Same GroupKFold(5) eval folds as 09/10.  Gate: mean RMSE must be < 2.6772 (10).
#
# Submission saved to submissions/12_final_ensemble.csv — exp. 10 CSV is untouched.

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

from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_mle import TwoStageMLE

_LOG_FILE = Path(__file__).resolve().parent / "12_run.log"
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

GATE_RMSE = 2.6772  # best so far: exp. 10 D_prior_03
BEST_BIAS_PARAMS = {**BIAS_PARAMS, "prior_scale": 0.3}

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
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

# ── Eval folds: identical GroupKFold(5) on patient_id as 09/10 ───────────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
log.info(
    "Train: %d rows, %d patients — folds are patient-disjoint [ok]",
    len(X_train_raw),
    len(np.unique(groups)),
)


# ── CV RMSE helper ────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str) -> tuple[list[float], float, float]:
    """Extract per-fold RMSE from a skore CV report and log a summary line."""
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), (
        f"{label}: bad folds {folds}"
    )
    arr = np.array(folds)
    log.info(
        "  %-30s folds=%s  mean=%.4f  std=%.4f",
        label,
        [round(v, 3) for v in folds],
        arr.mean(),
        arr.std(),
    )
    return folds, float(arr.mean()), float(arr.std())


# ── Ensemble CV via skore.evaluate on each member ────────────────────────────
log.info("Evaluating ensemble members on eval_splits …")

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

cv_a = skore.evaluate(model_a, X_train_raw, y, splitter=eval_splits)
folds_a, mean_a, std_a = cv_fold_rmses(cv_a, "model_a (shallow, prior=0.3)")

cv_b = skore.evaluate(model_b, X_train_raw, y, splitter=eval_splits)
folds_b, mean_b, std_b = cv_fold_rmses(cv_b, "model_b (deep,    prior=0.3)")

# ── Compute ensemble CV RMSE from OOF fold predictions ───────────────────────
# skore stores per-estimator reports in cv_report.reports_; each has .y_pred
# We must reconstruct test indices fold by fold to align the averaged OOF.
oof_ensemble = np.empty(len(y))
for fold_idx, (_, te_idx) in enumerate(eval_splits):
    pred_a = cv_a.reports_[fold_idx].get_predictions(data_source='test')
    pred_b = cv_b.reports_[fold_idx].get_predictions(data_source='test')
    oof_ensemble[te_idx] = 0.5 * (pred_a + pred_b)

ensemble_rmse_per_fold = [
    float(np.sqrt(np.mean((oof_ensemble[te] - y[te]) ** 2))) for _, te in eval_splits
]
mean_ensemble = float(np.mean(ensemble_rmse_per_fold))
std_ensemble = float(np.std(ensemble_rmse_per_fold))
log.info(
    "  %-30s folds=%s  mean=%.4f  std=%.4f",
    "ensemble (avg a+b)",
    [round(v, 3) for v in ensemble_rmse_per_fold],
    mean_ensemble,
    std_ensemble,
)

if mean_ensemble >= GATE_RMSE:
    log.warning(
        "Ensemble mean RMSE %.4f does NOT beat gate %.4f"
        " — no improvement over exp. 10.",
        mean_ensemble,
        GATE_RMSE,
    )
else:
    log.info(
        "Gate passed: ensemble RMSE %.4f < %.4f  (gain: %+.4f)",
        mean_ensemble,
        GATE_RMSE,
        GATE_RMSE - mean_ensemble,
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


cv_a_key = free_key("12_cv_model_a")
cv_b_key = free_key("12_cv_model_b")
est_a_key = free_key("12_fold5_model_a")
est_b_key = free_key("12_fold5_model_b")

try:
    cv_a_url = put_and_capture_url(cv_a_key, cv_a)
    cv_b_url = put_and_capture_url(cv_b_key, cv_b)
    est_a_url = put_and_capture_url(est_a_key, cv_a.reports_[4])
    est_b_url = put_and_capture_url(est_b_key, cv_b.reports_[4])
except Exception as e:
    log.critical("project.put failed: %s: %s", type(e).__name__, e)
    raise SystemExit(1)

# ── Refit both models on all training rows ────────────────────────────────────
log.info("Refitting both models on all %d training rows …", len(X_train_raw))
final_a = clone(model_a).fit(X_train_raw, y)
final_b = clone(model_b).fit(X_train_raw, y)

preds_a = final_a.predict(X_test_raw)
preds_b = final_b.predict(X_test_raw)
preds_ensemble = np.clip(0.5 * (preds_a + preds_b), 0.0, 132.0)

assert len(preds_ensemble) == len(sample_sub), (
    f"{len(preds_ensemble)} vs {len(sample_sub)}"
)
assert np.all(np.isfinite(preds_ensemble)), "Non-finite predictions"

submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds_ensemble})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == 11013

csv_path = SUBMISSIONS_DIR / "12_final_ensemble.csv"
submission.to_csv(csv_path, index=False)

# ── Final output ──────────────────────────────────────────────────────────────
log.info(
    "\n%s"
    "\nmodel_a (shallow) : folds=%s  mean=%.4f  std=%.4f"
    "\nmodel_b (deep)    : folds=%s  mean=%.4f  std=%.4f"
    "\nensemble (avg)    : folds=%s  mean=%.4f  std=%.4f"
    "\nGate (exp.10)     : %.4f   gain: %+.4f"
    "\nCV  model_a URL : %s  [%s]"
    "\nCV  model_b URL : %s  [%s]"
    "\nEst model_a URL : %s  [%s]"
    "\nEst model_b URL : %s  [%s]"
    "\nSubmission CSV  : %s  (%d rows)"
    "\n%s",
    "=" * 66,
    [round(v, 3) for v in folds_a],
    mean_a,
    std_a,
    [round(v, 3) for v in folds_b],
    mean_b,
    std_b,
    [round(v, 3) for v in ensemble_rmse_per_fold],
    mean_ensemble,
    std_ensemble,
    GATE_RMSE,
    GATE_RMSE - mean_ensemble,
    cv_a_url,
    cv_a_key,
    cv_b_url,
    cv_b_key,
    est_a_url,
    est_a_key,
    est_b_url,
    est_b_key,
    csv_path,
    len(submission),
    "=" * 66,
)
log.info("Done — stop before Kaggle upload.")
