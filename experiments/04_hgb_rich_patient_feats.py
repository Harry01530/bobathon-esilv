# %% [markdown]
# # 04_hgb_rich_patient_feats
#
# Rich per-patient temporal features added to the 03 mixed feature set.
# Eval folds are identical GroupKFold(5) on patient_id as in 02/03.
# Sanity gate: default model must score below 3.7 RMSE, else STOP.

# %%
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skore import login

from parkinsons.hub import load_skore_credentials

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CAT_COLS = ["cohort", "gene"]
FORBIDDEN = {"Index", "patient_id", "target", "rater_id"}
TUNE_SEED = 42
N_ITER = 20

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

# Shared category dtype: codes are consistent across train and test.
CAT_DTYPES = {
    c: pd.CategoricalDtype(
        sorted(pd.concat([X_train_raw[c], X_test_raw[c]]).dropna().unique())
    )
    for c in CAT_COLS
}


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build all features from X columns only.

    Sorts by (patient_id, age) with a stable sort, computes features on
    that ordered view, then returns columns in the ORIGINAL row order of
    raw so alignment with y is preserved.
    """
    # patient_id is needed for groupby; check output cols later

    # ── Sort for ordered within-patient operations ─────────────────────────
    order = raw.sort_values(["patient_id", "age"], kind="stable").index
    s = raw.loc[order].copy()  # sorted view (index = original row labels)
    g = s.groupby("patient_id", sort=False)

    feat = pd.DataFrame(index=s.index)

    # 1. Base numeric + categoricals
    for c in NUMERIC_COLS:
        feat[c] = s[c]
    for c in CAT_COLS:
        feat[c] = s[c].astype(CAT_DTYPES[c])

    # 2. Derived simple
    feat["years_dx"] = s["age"] - s["age_at_diagnosis"]
    feat["off_na"] = s["off"].isna().astype(np.int8)
    feat["on_na"] = s["on"].isna().astype(np.int8)
    feat["ledd_na"] = s["ledd"].isna().astype(np.int8)

    # 3. Visit order
    feat["vis_rank"] = g["age"].rank(method="first") - 1  # 0-based
    feat["since_first"] = s["age"] - g["age"].transform("min")
    feat["n_vis"] = g["age"].transform("size")

    # 4. Neighbour values (within patient group)
    for c in ["off", "on"]:
        feat[f"{c}_prev"] = g[c].shift(1)
        feat[f"{c}_next"] = g[c].shift(-1)
    feat["dt_prev"] = s["age"] - g["age"].shift(1)
    feat["dt_next"] = g["age"].shift(-1) - s["age"]

    # 5. Cumulative (expanding, within patient; must shift to avoid same-row leakage)
    for c in ["off", "on", "ledd"]:
        expanding = g[c].expanding().mean().shift(1)
        expanding.index = expanding.index.droplevel(0)
        feat[f"{c}_cmean"] = expanding
        expanding_max = g[c].expanding().max().shift(1)
        expanding_max.index = expanding_max.index.droplevel(0)
        feat[f"{c}_cmax"] = expanding_max
    feat["off_dev_c"] = s["off"] - feat["off_cmean"]

    # 6. Whole-patient aggregates
    for c in ["off", "on", "ledd"]:
        feat[f"{c}_pmean"] = g[c].transform("mean")
        feat[f"{c}_pmax"] = g[c].transform("max")
    feat["off_dev"] = s["off"] - feat["off_pmean"]

    # 7. Per-patient polynomial trend (off, on)
    for c in ["off", "on"]:
        fit1 = np.full(len(s), np.nan)
        fit2 = np.full(len(s), np.nan)
        res1 = np.full(len(s), np.nan)
        res2 = np.full(len(s), np.nan)

        pos_array = np.arange(len(s))  # maps to s's positional index
        for pid, idx in g.groups.items():
            pos = pos_array[s.index.get_indexer(idx)]
            ages = s.loc[idx, "age"].values
            vals = s.loc[idx, c].values
            mask = ~np.isnan(vals)
            if mask.sum() >= 2 and ages[mask].max() - ages[mask].min() > 0:
                age_c = ages[mask] - ages[mask].mean()
                poly1 = np.polyfit(age_c, vals[mask], 1)
                f1 = np.polyval(poly1, ages - ages[mask].mean())
                fit1[pos] = f1
                res1[pos] = vals - f1
            if mask.sum() >= 3 and ages[mask].max() - ages[mask].min() > 0:
                age_c = ages[mask] - ages[mask].mean()
                poly2 = np.polyfit(age_c, vals[mask], 2)
                f2 = np.polyval(poly2, ages - ages[mask].mean())
                fit2[pos] = f2
                res2[pos] = vals - f2

        feat[f"{c}_fit1"] = fit1
        feat[f"{c}_fit2"] = fit2
        feat[f"{c}_res1"] = res1
        feat[f"{c}_res2"] = res2

    feat["gap1"] = feat["on_fit1"] - feat["off_fit1"]

    # ── Return in ORIGINAL row order ──────────────────────────────────────
    return feat.loc[raw.index]


# ── Folds (identical to 03) ───────────────────────────────────────────────────
def shuffled_group_folds(grps: np.ndarray, n_splits: int, seed: int):
    patients = np.unique(grps)
    np.random.default_rng(seed).shuffle(patients)
    fold_of = {p: i % n_splits for i, p in enumerate(patients)}
    fold = np.array([fold_of[p] for p in grps])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


tune_splits = shuffled_group_folds(groups, 5, TUNE_SEED)
eval_splits = list(GroupKFold(n_splits=5).split(X_train_raw, y, groups=groups))

for k, (tr, te) in enumerate(eval_splits):
    assert not (set(groups[tr]) & set(groups[te])), f"Patient leak in eval fold {k}"
print(f"Train: {len(X_train_raw)} rows, {len(np.unique(groups))} patients")
print("Eval folds are patient-disjoint [ok]")

# ── Build feature matrix ──────────────────────────────────────────────────────
print("\nBuilding features …")
X = build_features(X_train_raw)
assert len(X) == len(X_train_raw), "Row count mismatch after build_features"
assert X.index.equals(X_train_raw.index), "Index mismatch after build_features"
leaked = FORBIDDEN & set(X.columns)
assert not leaked, f"Forbidden columns in X: {leaked}"
print(f"Feature matrix: {X.shape[1]} columns")


# ── CV helpers ────────────────────────────────────────────────────────────────
def cv_fold_rmses(cv_report, label: str):
    df = cv_report.metrics.get("rmse", data_source="test", aggregate=None)
    folds = df.values.flatten().tolist()
    assert len(folds) == 5 and all(np.isfinite(v) for v in folds), (
        f"{label}: bad folds {folds}"
    )
    arr = np.array(folds)
    print(
        f"  {label:28s} folds={[round(v, 3) for v in folds]} "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )
    return folds, float(arr.mean()), float(arr.std())


# ── a) Sanity gate ────────────────────────────────────────────────────────────
print("\n[a] Sanity: default HGB on eval_splits …")
default_hgb = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    random_state=0,
    categorical_features="from_dtype",
)
sanity_report = skore.evaluate(default_hgb, X, y, splitter=eval_splits)
sanity_folds, sanity_mean, sanity_std = cv_fold_rmses(
    sanity_report, "hgb_default_rich_sanity"
)

if sanity_mean > 3.7:
    raise RuntimeError(
        f"STOP: sanity RMSE {sanity_mean:.4f} > 3.7 — "
        "probable row misalignment or wrong feature. Fix build_features."
    )
print(f"Sanity gate passed: mean RMSE {sanity_mean:.4f} <= 3.7 [ok]")

# ── b) RandomizedSearchCV on tuning folds ─────────────────────────────────────
print(f"\n[b] RandomizedSearchCV: {N_ITER} candidates x 5 tuning folds …")
base_hgb = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    early_stopping=False,
    random_state=0,
)
param_distributions = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_iter": randint(150, 801),
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 50, 100, 200],
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": [0.5, 0.8, 1.0],
}
search = RandomizedSearchCV(
    base_hgb,
    param_distributions,
    n_iter=N_ITER,
    scoring="neg_root_mean_squared_error",
    cv=tune_splits,
    refit=False,
    n_jobs=-1,
    random_state=0,
)
search.fit(X, y)
best_params = search.best_params_
print(f"  best tuning-fold RMSE = {-search.best_score_:.4f}")
print(f"  best params = {best_params}")

# ── c) Evaluation: default vs tuned ───────────────────────────────────────────
print("\n[c] Evaluating default vs tuned on eval_splits …")
default_eval = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    random_state=0,
    categorical_features="from_dtype",
)
tuned_eval = clone(base_hgb).set_params(**best_params)

comp = skore.evaluate(
    {"hgb_default_rich": default_eval, "hgb_tuned_rich": tuned_eval},
    X,
    y,
    splitter=eval_splits,
)

print("\n=== Evaluation results ===")
eval_results = {}
for name, cv_rep in comp.reports_.items():
    folds, mean, std = cv_fold_rmses(cv_rep, name)
    eval_results[name] = {
        "cv_report": cv_rep,
        "model": default_eval if "default" in name else tuned_eval,
        "folds": folds,
        "mean": mean,
        "std": std,
    }

best_name = min(eval_results, key=lambda k: eval_results[k]["mean"])
best = eval_results[best_name]
other_name = [n for n in eval_results if n != best_name][0]
other = eval_results[other_name]
gain = other["mean"] - best["mean"]
print(f"\nBest: {best_name}  mean={best['mean']:.4f}  gain vs other: {gain:+.4f}")
if gain <= best["std"]:
    print("WARNING: gain is within fold-to-fold spread.")

# ── d) Hub upload ─────────────────────────────────────────────────────────────
print("\n[d] Uploading to hub …")
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


cv_key = free_key("04_cv_hgb_rich")
cv_url = put_and_capture_url(cv_key, best["cv_report"])
est_key = free_key("04_fold5_hgb_rich")
est_url = put_and_capture_url(est_key, best["cv_report"].reports_[4])

# ── e) Refit on all training rows and write submission ────────────────────────
print(f"\n[e] Refitting {best_name} on all {len(X)} training rows …")
final_model = clone(best["model"]).fit(X, y)

X_test_feat = build_features(X_test_raw)
preds = final_model.predict(X_test_feat)

assert len(preds) == len(sample_sub), f"{len(preds)} vs {len(sample_sub)} rows"
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

# ── Final output ──────────────────────────────────────────────────────────────
d = eval_results["hgb_default_rich"]
t = eval_results["hgb_tuned_rich"]
print(f"\n{'=' * 62}")
print(
    f"hgb_default_rich : folds={[round(v, 3) for v in d['folds']]}  "
    f"mean={d['mean']:.4f}  std={d['std']:.4f}"
)
print(
    f"hgb_tuned_rich   : folds={[round(v, 3) for v in t['folds']]}  "
    f"mean={t['mean']:.4f}  std={t['std']:.4f}"
)
print(f"Best params      : {best_params}")
print(f"Best model       : {best_name}  (gain vs other: {gain:+.4f})")
print(f"CV  report URL   : {cv_url}  [{cv_key}]")
print(f"Est report URL   : {est_url}  [{est_key}]")
print(f"Submission CSV   : {csv_path}  ({len(submission)} rows)")
print(f"{'=' * 62}")
print("Done — stop before Kaggle upload.")

# ── Metadata JSON ─────────────────────────────────────────────────────────────
metadata = {
    "data_source": str(DATA_DIR),
    "feature_count": X.shape[1],
    "evaluation_folds": "GroupKFold(n_splits=5, groups=patient_id), same as 02/03",
    "tuning_folds": f"5 patient-grouped shuffled folds, seed {TUNE_SEED}",
    "n_iter": N_ITER,
    "sanity_mean_rmse": sanity_mean,
    "hgb_default_rich": {"folds": d["folds"], "mean": d["mean"], "std": d["std"]},
    "hgb_tuned_rich": {"folds": t["folds"], "mean": t["mean"], "std": t["std"]},
    "best_params": {
        k: float(v) if hasattr(v, "item") else v for k, v in best_params.items()
    },
    "best_model": best_name,
    "cv_report_key": cv_key,
    "cv_report_url": cv_url,
    "estimator_report_key": est_key,
    "estimator_report_url": est_url,
    "csv_path": str(csv_path),
}
meta_path = SUBMISSIONS_DIR / f"{est_key}_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2, default=str))
