"""Cross-validation strategy for experiment 01_dummy.

``skore.evaluate`` in 0.26.0 does not accept a ``groups=`` parameter
directly. To guarantee no patient leaks across splits, we pre-compute
the ``(train, test)`` index arrays with ``GroupShuffleSplit.split``
and pass the resulting generator to ``skore.evaluate(splitter=...)``.
The ``splitter`` type hint for ``skore.evaluate`` accepts an iterable
of ``(train, test)`` tuples (confirmed from the source).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_train_dataset


def make_group_split(data_dir: str | None = None):
    """Return a list of one ``(train_idx, test_idx)`` tuple for 01_dummy.

    Groups by ``patient_id`` so no patient appears in both train and
    validation. Uses 20 % of patients as the validation set.

    Parameters
    ----------
    data_dir :
        Path to the data directory. Defaults to ``PROJECT_ROOT / "data"``.

    Returns
    -------
    list of (ndarray, ndarray)
        One element — the single GroupShuffleSplit fold.
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data")

    df = load_train_dataset(data_dir)
    groups = df["patient_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    return list(gss.split(df, df["target"], groups=groups))
