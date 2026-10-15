# %% [markdown]
# # 13_hybrid_gru
#
# Hybrid model: unweighted average of two TwoStageMLE (HGB) models blended
# with a 2-layer bidirectional GRU that reads each patient's visit sequence.
#
#   HGB branch  : average of TwoStageMLE(smooth=None, iter=300/lr=0.05/leaves=31)
#                 and TwoStageMLE(smooth=None, iter=800/lr=0.03/leaves=63)
#   GRU branch  : BiGRU(hidden=64, layers=2, dropout=0.1) on sequences of
#                 [age, off, on, t_on, t_off, hgb_pred] sorted by age per patient
#   Final pred  : smooth_per_patient(0.5*HGB + 0.5*GRU, "quad_iso")
#
# Leakage contract: calibrate_bias_model and z-score statistics are fitted
# exclusively on the training rows of every fold.
# Evaluation: GroupKFold(5) on patient_id, same folds as 09-12.
# Gate: mean RMSE must be < 2.6772 (exp. 10 best).
#
# Hub keys: 13_cv_hybrid_gru  (CV report)
#           13_fold5_hybrid_gru  (EstimatorReport fold 5 + submission CSV)

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

from parkinsons.features import make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.hybrid_gru import HybridGRUModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GATE_RMSE = 2.6772  # best so far: exp. 10 D_prior_03

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

assert "rater_id" not in X_train_raw.columns, "rater_id must not be in X_train"
log.info(
    "Train: %d rows, %d patients — Test: %d rows",
    len(X_train_raw),
    len(np.unique(groups)),
    len(X_test_raw),
)

# ── Shared category dtype (train + test categories pooled) ───────────────────
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

# ── Eval folds: GroupKFold(5) on patient_id, identical to 09-12 ──────────────
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
log.info("Eval folds are patient-disjoint [ok]")

# ── Estimator ─────────────────────────────────────────────────────────────────
estimator = HybridGRUModel(cat_dtypes=CAT_DTYPES)

# ── Evaluation via skore ───────────────────────────────────────────────────────
log.info("Evaluating HybridGRUModel on eval_splits …")
cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)


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


folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, "hybrid_gru")

if mean_rmse > GATE_RMSE:
    log.warning(
        "Mean RMSE %.4f does NOT beat gate %.4f — ensemble still best.",
        mean_rmse,
        GATE_RMSE,
    )
else:
    log.info(
        "Gate passed: mean RMSE %.4f < %.4f  (gain: %+.4f)",
        mean_rmse,
        GATE_RMSE,
        GATE_RMSE - mean_rmse,
    )

# ── Hub upload ────────────────────────────────────────────────────────────────
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
existing_keys = set(project.summarize().frame()["key"].tolist())


def _check_key_conflict(base: str) -> str:
    """Return the key unchanged if free.  Raise if it is already taken."""
    if base in existing_keys:
        raise RuntimeError(
            f"Hub key '{base}' already exists. "
            "Either delete it on the Hub, or rename this experiment. "
            "Do NOT silently pick a new key — the submission description must "
            "reference the canonical key."
        )
    return base


def put_and_capture_url(key: str, report) -> str:
    skore.console.begin_capture()
    project.put(key, report)
    captured = skore.console.end_capture()
    existing_keys.add(key)
    urls = re.findall(r"https://[^\s\]>]+", captured)
    return urls[-1].rstrip("/") if urls else "url-not-captured"


cv_key = _check_key_conflict("13_cv_hybrid_gru")
try:
    cv_url = put_and_capture_url(cv_key, cv_report)
except RuntimeError:
    raise
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", cv_key, type(e).__name__, e)
    raise SystemExit(1)

est_key = _check_key_conflict("13_fold5_hybrid_gru")
try:
    est_url = put_and_capture_url(est_key, cv_report.reports_[4])
except RuntimeError:
    raise
except Exception as e:
    log.critical("project.put(%r) failed: %s: %s", est_key, type(e).__name__, e)
    raise SystemExit(1)

# ── Refit on all training rows and write submission CSV ───────────────────────
log.info("Refitting HybridGRUModel on all %d training rows …", len(X_train_raw))
final_model = clone(estimator).fit(X_train_raw, y)
preds = final_model.predict(X_test_raw)
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

csv_path = SUBMISSIONS_DIR / "13_fold5_hybrid_gru.csv"
submission.to_csv(csv_path, index=False)

# ── Metadata ─────────────────────────────────────────────────────────────────
meta = {
    "fold_rmse": folds,
    "mean_rmse": mean_rmse,
    "std_rmse": std_rmse,
    "cv_report": {"key": cv_key, "url": cv_url},
    "estimator_report": {"key": est_key, "url": est_url},
}
(SUBMISSIONS_DIR / "13_fold5_hybrid_gru_metadata.json").write_text(
    json.dumps(meta, indent=2)
)

# ── Final output ──────────────────────────────────────────────────────────────
log.info(
    "\n%s\nHybridGRU : folds=%s  mean=%.4f  std=%.4f"
    "\nGate (exp.10) : %.4f   gain: %+.4f"
    "\nCV  report URL : %s  [%s]"
    "\nEst report URL : %s  [%s]   <- put this URL on Kaggle"
    "\nSubmission CSV : %s  (%d rows)"
    "\n%s",
    "=" * 66,
    [round(v, 3) for v in folds],
    mean_rmse,
    std_rmse,
    GATE_RMSE,
    GATE_RMSE - mean_rmse,
    cv_url,
    cv_key,
    est_url,
    est_key,
    csv_path,
    len(submission),
    "=" * 66,
)
log.info("Done — stop before Kaggle upload.")
