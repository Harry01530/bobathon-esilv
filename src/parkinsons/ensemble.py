"""Unweighted average ensemble of two TwoStageMLE models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class AvgEnsemble(BaseEstimator, RegressorMixin):
    """Unweighted average of two regressors, clipped to [0, 132].

    Parameters
    ----------
    model_a, model_b : sklearn-compatible regressors
        The two base estimators. They are cloned on ``fit`` so the originals
        are not modified.
    """

    def __init__(self, model_a=None, model_b=None):
        self.model_a = model_a
        self.model_b = model_b

    def fit(self, X, y):
        self.a_ = clone(self.model_a).fit(X, y)
        self.b_ = clone(self.model_b).fit(X, y)
        return self

    def predict(self, X):
        pred = 0.5 * (self.a_.predict(X) + self.b_.predict(X))
        return np.clip(pred, 0, 132)
