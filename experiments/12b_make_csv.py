# %% [markdown]
# # 12b_make_csv
#
# Recover submissions/12_final_ensemble.csv without re-running the full CV.
# Refits model_a and model_b on all training rows, averages their test
# predictions, and writes two CSVs:
#   - submissions/12_model_a_only.csv
#   - submissions/12_final_ensemble.csv  (unweighted mean, clipped [0, 132])

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd

from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons.two_stage_mle import TwoStageMLE

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"
if not DATA_DIR.exists():
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

assert "rater_id" not in X_train_raw.columns, "rater_id must not be in X_train"
print(f"Train: {len(X_train_raw)} rows — Test: {len(X_test_raw)} rows", flush=True)

# ── Model definitions (identical to exp. 12) ──────────────────────────────────
BEST_BIAS_PARAMS = {**BIAS_PARAMS, "prior_scale": 0.3}
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)

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

# ── Refit model_a ─────────────────────────────────────────────────────────────
print("Fitting model_a (shallow) …", flush=True)
t0 = time.perf_counter()
model_a.fit(X_train_raw, y)
print(f"  model_a fit done in {time.perf_counter() - t0:.1f}s", flush=True)

preds_a = model_a.predict(X_test_raw)

sub_a = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": np.clip(preds_a, 0, 132)})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
assert list(sub_a.columns) == ["Index", "target"]
assert len(sub_a) == 11013
assert np.all(np.isfinite(sub_a["target"].values))
sub_a.to_csv(SUBMISSIONS_DIR / "12_model_a_only.csv", index=False)
print(f"  12_model_a_only.csv written ({len(sub_a)} rows)", flush=True)

# ── Refit model_b ─────────────────────────────────────────────────────────────
print("Fitting model_b (deep) …", flush=True)
t0 = time.perf_counter()
model_b.fit(X_train_raw, y)
print(f"  model_b fit done in {time.perf_counter() - t0:.1f}s", flush=True)

preds_b = model_b.predict(X_test_raw)

# ── Ensemble CSV ──────────────────────────────────────────────────────────────
preds_ensemble = np.clip(0.5 * (preds_a + preds_b), 0.0, 132.0)

corr = float(np.corrcoef(preds_a, preds_b)[0, 1])
print(f"\nCorr(model_a, model_b) on test : {corr:.4f}", flush=True)
print(
    f"Ensemble test stats: "
    f"mean={preds_ensemble.mean():.3f}  std={preds_ensemble.std():.3f}  "
    f"min={preds_ensemble.min():.3f}  max={preds_ensemble.max():.3f}",
    flush=True,
)

sub_ensemble = (
    pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds_ensemble})
    .set_index("Index")
    .loc[sample_sub["Index"]]
    .reset_index()
)
assert list(sub_ensemble.columns) == ["Index", "target"]
assert len(sub_ensemble) == 11013
assert np.all(np.isfinite(sub_ensemble["target"].values))

csv_path = SUBMISSIONS_DIR / "12_final_ensemble.csv"
sub_ensemble.to_csv(csv_path, index=False)
print(f"\n12_final_ensemble.csv written to {csv_path}  ({len(sub_ensemble)} rows)", flush=True)
