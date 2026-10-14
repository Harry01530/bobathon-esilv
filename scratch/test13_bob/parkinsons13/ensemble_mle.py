"""Standalone HybridModel for experiment 13.

Two TwoStageMLE members (defaults + 800/lr.03/63/leaf50), each with its own
calibration fitted on its own training rows.  GRU uses features from member A's
calibration (train-only).  Blend: 0.5 * mean(HGB.predict_unsmoothed) + 0.5 * GRU,
then smooth_per_patient('quad_iso').

No dependency on parkinsons.ensemble_mle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .two_stage_mle import TwoStageMLE, smooth_per_patient
from .seq_gru import SeqGRU


class HybridModel(RegressorMixin, BaseEstimator):
    """Two TwoStageMLE members + SeqGRU, blended 50/50 before quad_iso projection.

    Parameters
    ----------
    cat_dtypes : dict or None
        Shared CategoricalDtype mapping from make_cat_dtypes.
    bias_params : dict or None
        Passed to both TwoStageMLE instances.
    gru_epochs : int
        Training epochs for the GRU (60 for real run; 2 for benchmark).
    gru_hidden : int
        Hidden size of the GRU (64).
    gru_random_state : int
        Seed (0 = team default).
    """

    def __init__(
        self,
        cat_dtypes=None,
        bias_params=None,
        gru_epochs: int = 60,
        gru_hidden: int = 64,
        gru_random_state: int = 0,
    ):
        self.cat_dtypes = cat_dtypes
        self.bias_params = bias_params
        self.gru_epochs = gru_epochs
        self.gru_hidden = gru_hidden
        self.gru_random_state = gru_random_state

    def fit(self, X: pd.DataFrame, y) -> "HybridModel":
        y = np.asarray(y, dtype=float)

        # Member A: defaults (max_iter=300, lr=0.05, leaf=31, min_leaf=20)
        self.member_a_ = TwoStageMLE(
            cat_dtypes=self.cat_dtypes,
            inner_n_splits=5,
            smooth=None,           # smoothing applied jointly at predict time
            bias_params=self.bias_params,
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        ).fit(X, y)

        # Member B: deep (max_iter=800, lr=0.03, leaf=63, min_leaf=50)
        self.member_b_ = TwoStageMLE(
            cat_dtypes=self.cat_dtypes,
            inner_n_splits=5,
            smooth=None,
            bias_params=self.bias_params,
            max_iter=800,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=50,
        ).fit(X, y)

        # GRU: trained with features built from member A's calibration (train-only)
        from .features import build_features
        Xf_a = build_features(
            X,
            self.cat_dtypes,
            bias_calibration=self.member_a_.calibration_,
            bias_params=self.bias_params,
        )
        self.gru_ = SeqGRU(
            hidden=self.gru_hidden,
            epochs=self.gru_epochs,
            batch_size=128,
            lr=3e-3,
            weight_decay=1e-4,
            n_seeds=1,
            random_state=self.gru_random_state,
        ).fit(X, Xf_a, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["member_a_", "member_b_", "gru_"])

        pred_a = self.member_a_.predict_unsmoothed(X)
        pred_b = self.member_b_.predict_unsmoothed(X)
        hgb_mean = 0.5 * (pred_a + pred_b)

        from .features import build_features
        Xf_a = build_features(
            X,
            self.cat_dtypes,
            bias_calibration=self.member_a_.calibration_,
            bias_params=self.bias_params,
        )
        gru_pred = self.gru_.predict(X, Xf_a)

        blended = 0.5 * hgb_mean + 0.5 * gru_pred
        return smooth_per_patient(X, blended, "quad_iso")
