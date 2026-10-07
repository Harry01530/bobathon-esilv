"""Data loading for the Parkinson's UPDRS prediction project.

Loads raw CSV files from the ``data/`` directory and exposes the
fixed feature columns used across all experiments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Feature columns used in every experiment.
# ``patient_id`` is kept separate — it is the grouping key for the CV
# splitter and must never appear in X.
# ``Index`` is a row identifier and must never appear in X either.
FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

TARGET_COL = "target"


def load_train_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Load X_train and y_train merged on ``Index``.

    Parameters
    ----------
    data_dir :
        Directory containing ``X_train.csv`` and ``y_train.csv``.

    Returns
    -------
    pd.DataFrame
        Merged frame with all feature columns plus ``patient_id`` (for
        grouping) and ``target``.
    """
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "X_train.csv")
    y = pd.read_csv(data_dir / "y_train.csv")
    return X.merge(y, on="Index")


def load_test_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Load X_test (features only, no target).

    Parameters
    ----------
    data_dir :
        Directory containing ``X_test.csv``.

    Returns
    -------
    pd.DataFrame
        Test frame with all feature columns plus ``Index`` (for the
        submission file) and ``patient_id``.
    """
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "X_test.csv")
