"""Pipeline declaration for experiment 01_dummy.

Wraps a ``DummyRegressor(strategy="mean")`` in a skrub DataOps graph.
The dummy ignores all feature values; the DataOps wrapper lets the
experiment script use the standard ``skore.evaluate`` path.
"""

from __future__ import annotations

from pathlib import Path

import skrub
from sklearn.dummy import DummyRegressor

from parkinsons.data import FEATURE_COLS, TARGET_COL, load_train_dataset


def build_learner(data_dir_preview: str | Path | None = None) -> skrub.SkrubLearner:
    """Return an unfit skrub SkrubLearner wrapping DummyRegressor(strategy="mean").

    Parameters
    ----------
    data_dir_preview :
        Optional path used by ``.skb.preview()`` during interactive
        exploration. Leave ``None`` for production fit / cross-validate.

    Returns
    -------
    skrub.SkrubLearner
        Unfit learner ready for ``skore.evaluate`` or ``.fit()``.
    """
    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )

    # Layer 1: load raw training data.
    data = data_dir.skb.apply_func(load_train_dataset)

    # Layer 2: mark X and y.
    # patient_id is not a feature; it is used in evaluate.py to build
    # the GroupShuffleSplit indices before passing them to skore.evaluate.
    X = data[FEATURE_COLS].skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    # Layer 3: estimator (DummyRegressor ignores X entirely).
    predictions = X.skb.apply(DummyRegressor(strategy="mean"), y=y)
    return predictions.skb.make_learner()
