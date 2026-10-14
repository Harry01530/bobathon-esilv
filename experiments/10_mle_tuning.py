# %% [markdown]
# # 10_mle_tuning
#
# Grid search over TwoStageMLE hyperparameters on top of the 09 feature set.
# Candidates:
#   A. Deeper / slower HGB: max_iter=600, lr=0.03, max_leaf_nodes=63
#   B. Larger leaves: min_samples_leaf=50
#   C. Both A+B combined
#   D. Bias prior_scale=0.3 (tighter empirical-Bayes prior)
#   E. Bias prior_scale=0.7 (looser)
#   F. lam_slope=0.5 in BIAS_PARAMS (less regularisation on slope)
#   G. Best of above + inner_n_splits=10 (more OOF folds)
#
# All using same GroupKFold(5) eval folds as 09.
# Gate: mean RMSE <= 2.6905 (09 baseline).

# %%
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

GATE_RMSE = 2.6905  # 09 baseline

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
print(f"Train: {len(X_train_raw)} rows, {len(np.unique(groups))} patients")
print("Eval folds are patient-disjoint [ok]")


# ── CV helper ─────────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), (
        f"{label}: bad folds {folds}"
    )
    arr = np.array(folds)
    print(
        f"  {label:40s} folds={[round(v, 3) for v in folds]} "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )
    return folds, float(arr.mean()), float(arr.std())


# ── Candidate configurations ──────────────────────────────────────────────────
candidates = {
    "09_baseline": dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        bias_params=None,
    ),
    "A_deep_hgb": dict(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        bias_params=None,
    ),
    "B_large_leaf": dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        bias_params=None,
    ),
    "C_deep_large": dict(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        bias_params=None,
    ),
    "D_prior_03": dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        bias_params={**BIAS_PARAMS, "prior_scale": 0.3},
    ),
    "E_prior_07": dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        bias_params={**BIAS_PARAMS, "prior_scale": 0.7},
    ),
    "F_lam_slope_05": dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        bias_params={**BIAS_PARAMS, "lam_slope": 0.5},
    ),
}

results = {}
print("\nEvaluating candidates on eval_splits …\n")
for name, cfg in candidates.items():
    bp = cfg.pop("bias_params")
    estimator = TwoStageMLE(
        cat_dtypes=CAT_DTYPES,
        inner_n_splits=5,
        smooth="quad_iso",
        bias_params=bp,
        **cfg,
    )
    cv_report = skore.evaluate(estimator, X_train_raw, y, splitter=eval_splits)
    folds, mean_rmse, std_rmse = cv_fold_rmses(cv_report, name)
    results[name] = {
        "folds": folds,
        "mean": mean_rmse,
        "std": std_rmse,
        "estimator": estimator,
        "cv_report": cv_report,
        "bp": bp,
        "cfg": cfg,
    }

# ── Pick the best ──────────────────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["mean"])
best = results[best_name]
print(f"\nBest: {best_name}  mean={best['mean']:.4f}  (gate={GATE_RMSE})")

if best["mean"] >= GATE_RMSE:
    print(
        f"WARNING: best mean RMSE {best['mean']:.4f}"
        f" does not beat 09 baseline {GATE_RMSE}"
    )

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


cv_key = free_key(f"10_cv_{best_name}")
est_key = free_key(f"10_fold5_{best_name}")
cv_url = put_and_capture_url(cv_key, best["cv_report"])
est_url = put_and_capture_url(est_key, best["cv_report"].reports_[4])

# ── Refit best on all training rows ───────────────────────────────────────────
print(f"\nRefitting {best_name} on all {len(X_train_raw)} training rows …")
bp = best["bp"]
final_cfg = best["cfg"]
final_model = TwoStageMLE(
    cat_dtypes=CAT_DTYPES,
    inner_n_splits=5,
    smooth="quad_iso",
    bias_params=bp,
    **final_cfg,
).fit(X_train_raw, y)
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
print(f"\n{'=' * 66}")
print(f"Best config : {best_name}")
print(
    f"TwoStageMLE : folds={[round(v, 3) for v in folds]}  "
    f"mean={mean_rmse:.4f}  std={std_rmse:.4f}"
)
print(f"Gain vs 09 ({GATE_RMSE}): {GATE_RMSE - mean_rmse:+.4f}")
print(f"CV  report URL : {cv_url}  [{cv_key}]")
print(f"Est report URL : {est_url}  [{est_key}]   <- put this URL on Kaggle")
print(f"Submission CSV : {csv_path}  ({len(submission)} rows)")
print(f"{'=' * 66}")
print("Done — stop before Kaggle upload.")
