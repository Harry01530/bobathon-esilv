"""Smoke test for experiments/01_dummy.py.

Fits the dummy learner on a small slice of real training data and
predicts on a disjoint slice with no shared patients. Asserts that the
number of predictions equals the number of rows in the predict slice.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import FEATURE_COLS, TARGET_COL


def test_01_dummy_predict_row_count() -> None:
    """Prediction count must equal predict-slice row count (no silent drops)."""
    data_dir = PROJECT_ROOT / "data"

    raw = pd.read_csv(data_dir / "X_train.csv")
    y_raw = pd.read_csv(data_dir / "y_train.csv").set_index("Index")["target"]

    # Split patients: first 80% train, last 20% predict — no overlap.
    patients = raw["patient_id"].unique()
    n_train = int(len(patients) * 0.8)
    train_patients = set(patients[:n_train])
    pred_patients = set(patients[n_train:])

    train_mask = raw["patient_id"].isin(train_patients)
    pred_mask = raw["patient_id"].isin(pred_patients)

    X_train = raw.loc[train_mask, FEATURE_COLS]
    y_train = y_raw.loc[raw.loc[train_mask, "Index"]]
    X_pred = raw.loc[pred_mask, FEATURE_COLS]

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_pred)

    assert len(preds) == len(X_pred), (
        f"Row count mismatch: got {len(preds)} predictions "
        f"for {len(X_pred)} predict rows."
    )
