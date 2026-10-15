# %% [markdown]
# # 12c_hub_report
#
# Push an EstimatorReport for the final AvgEnsemble to Skore Hub.
# Uses a float train_size splitter so skore can serialize the report.
# Runs a single train/test split (not full CV) — just enough to get a report
# onto the hub with diagnostic plots.
#
# Hub key: 12_final_ensemble  (suffixed _2, _3 … if already taken)

# %%
import re
from pathlib import Path

import skore
from skore import login

from parkinsons.ensemble import AvgEnsemble
from parkinsons.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons.hub import load_skore_credentials
from parkinsons.two_stage_mle import TwoStageMLE

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "data" / "kaggle_official"

# ── Load data ─────────────────────────────────────────────────────────────────
import pandas as pd

X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
y = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]
    .values
)

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

# ── Evaluate with skore (float splitter so report is serialisable) ────────────
print("Running skore.evaluate on AvgEnsemble (0.8 train split) …", flush=True)
report = skore.evaluate(
    AvgEnsemble(model_a, model_b),
    X_train_raw,
    y,
    splitter=0.8,
)
print("  evaluate done.", flush=True)

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


key = free_key("12_final_ensemble")

print(f"Pushing report under key '{key}' …", flush=True)
skore.console.begin_capture()
project.put(key, report)
captured = skore.console.end_capture()

# ── Extract URL from captured console output ──────────────────────────────────
url_match = re.search(
    r"https://skore\.probabl\.ai/\S+?/(?:estimators|cross-validations)/\d+",
    captured,
)
if url_match:
    url = url_match.group(0)
else:
    urls = re.findall(r"https://[^\s\]>]+", captured)
    url = urls[-1].rstrip("/") if urls else captured.strip()

print(f"\nKey : {key}", flush=True)
print(f"URL : {url}", flush=True)
