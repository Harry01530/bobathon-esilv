# %% [markdown]
# # 01_dummy — DummyRegressor baseline
#
# Evaluates ``DummyRegressor(strategy="mean")`` with skore's default
# 80/20 holdout split, pushes the report to the hub project
# ``bobathon-esilv``, writes a Kaggle submission CSV, and prints the
# RMSE and the report URL.

# %%
import math

import pandas as pd
import skore
from skore import login
from sklearn.dummy import DummyRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import FEATURE_COLS, TARGET_COL, load_test_dataset
from parkinsons.hub import load_skore_credentials

# %% [markdown]
# ## Open the skore hub project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Load training data and evaluate

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").set_index("Index")[TARGET_COL]
y_train = y_train.loc[X_train["Index"]]

X = X_train[FEATURE_COLS]
y = y_train.values

report = skore.evaluate(DummyRegressor(strategy="mean"), X, y)

project.put("01_dummy", report)

# %% [markdown]
# ## Print RMSE and report URL

# %%
summary_frame = project.summarize().frame()
row = summary_frame.query("key == '01_dummy'").iloc[-1]
report_id = row.name

rmse_val = None
for col in summary_frame.columns:
    if "rmse" in str(col).lower():
        rmse_val = summary_frame.loc[report_id, col]
        break

if rmse_val is None or (isinstance(rmse_val, float) and math.isnan(rmse_val)):
    try:
        stored = project.get(report_id)
        metrics_df = stored.metrics.report_metrics()
        for col in metrics_df.columns:
            if "rmse" in str(col).lower():
                rmse_val = metrics_df[col].iloc[0]
                break
    except Exception:
        rmse_val = "unavailable"

print(f"RMSE (0.2 holdout): {rmse_val}")
print(f"Report URL: https://app.skore.probabl.ai/{cfg['workspace']}/projects/bobathon-esilv")

# %% [markdown]
# ## Fit on all training data and write the submission CSV

# %%
dummy_final = DummyRegressor(strategy="mean")
dummy_final.fit(X, y)

X_test = load_test_dataset(DATA_DIR)
preds = dummy_final.predict(X_test[FEATURE_COLS])

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert list(submission.columns) == list(sample.columns), (
    f"Column mismatch: {list(submission.columns)} vs {list(sample.columns)}"
)
assert len(submission) == len(sample), (
    f"Row count mismatch: {len(submission)} vs {len(sample)}"
)

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
submission.to_csv(SUBMISSIONS_DIR / "01_dummy.csv", index=False)

print(f"Submission written: submissions/01_dummy.csv ({len(submission)} rows)")
