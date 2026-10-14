# %% [markdown]
# # 11_target_vcorr
#
# Axe 2 : stage 1 entraîné sur map_Vcorr = (V + D̂) / c (valeur de visite
# corrigée du biais de sous-évaluation EMG) au lieu de y.  Stage 2 apprend
# le résidu y − ŷ₁.  A la prédiction : ŷ = ŷ₁ + ŷ₂.
#
# Hypothèse : map_Vcorr est un estimateur moins bruité de T que y pour les
# visites ON (biais asymétrique fort). Stage 1 converge plus vite vers T ;
# stage 2 n'a plus qu'un résidu de faible amplitude à corriger.
#
# Baseline : exp. 10 — TwoStageMLE(prior_scale=0.3) — mean CV RMSE 2.6772.
# Gate     : mean RMSE <= 2.6772.
#
# Même GroupKFold(5) eval folds que 09/10.

# %%
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_mle import TwoStageMLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GATE_RMSE = 2.6772  # exp. 10 baseline
REF_10_RMSE = 2.6772

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

eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))
for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups)))
log.info("Eval folds are patient-disjoint [ok]")


# ── CV helper ─────────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), (
        f"{label}: bad folds {folds}"
    )
    arr = np.array(folds)
    log.info(
        "  %-35s folds=%s  mean=%.4f  std=%.4f",
        label,
        [round(v, 3) for v in folds],
        arr.mean(),
        arr.std(),
    )
    return folds, float(arr.mean()), float(arr.std())


# ── Deux configs à comparer sur les mêmes eval_splits ─────────────────────────
# Config A : baseline (exp. 10) — use_vcorr=False, prior_scale=0.3
# Config B : axe 2 — use_vcorr=True,  prior_scale=0.3
bias_p = {**BIAS_PARAMS, "prior_scale": 0.3}

estimator = TwoStageMLE(
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
    smooth="quad_iso",
    bias_params=bias_p,
    use_vcorr=True,
)
label = "11_use_vcorr"

log.info("Evaluating %s on eval_splits …", label)
cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)
folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, label)
gain = REF_10_RMSE - mean_rmse
log.info("Gain vs 10 (%.4f): %+.4f", REF_10_RMSE, gain)

best_name = label
best = {
    "folds": folds,
    "mean": mean_rmse,
    "std": std_rmse,
    "estimator": estimator,
    "cv_report": cv_report,
}

if mean_rmse > GATE_RMSE + 1e-6:
    log.warning("RMSE %.4f does not beat gate %.4f", mean_rmse, GATE_RMSE)

# ── Hub upload ─────────────────────────────────────────────────────────────────
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


cv_key = free_key(f"11_cv_{best_name}")
est_key = free_key(f"11_fold5_{best_name}")
try:
    cv_url = put_and_capture_url(cv_key, best["cv_report"])
    est_url = put_and_capture_url(est_key, best["cv_report"].reports_[4])
except Exception as e:
    log.critical("project.put failed: %s: %s", type(e).__name__, e)
    raise SystemExit(1)

# ── Refit best config on all training rows ────────────────────────────────────
log.info("Refitting %s on all %d training rows …", best_name, len(X_train_raw))
final_model = clone(best["estimator"]).fit(X_train_raw, y)
preds = np.clip(final_model.predict(X_test_raw), 0.0, 132.0)

assert len(preds) == len(sample_sub)
assert np.all(np.isfinite(preds))
submission = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
csv_path = SUBMISSIONS_DIR / f"{est_key}.csv"
submission.to_csv(csv_path, index=False)

# ── Final output ───────────────────────────────────────────────────────────────
folds, mean_rmse, std_rmse = best["folds"], best["mean"], best["std"]
log.info(
    "\n%s\nBest config : %s"
    "\nRMSE        : folds=%s  mean=%.4f  std=%.4f"
    "\nGain vs 10  : %+.4f"
    "\nCV  report  : %s  [%s]"
    "\nEst report  : %s  [%s]   <- put this URL on Kaggle"
    "\nSubmission  : %s  (%d rows)"
    "\n%s",
    "=" * 66,
    best_name,
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
    "=" * 66,
)
log.info("Done — stop before Kaggle upload.")
