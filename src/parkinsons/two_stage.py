"""Two-stage HistGradientBoosting regressor for Parkinson's UPDRS prediction.

Stage 1 fits a HGB on the full feature set (built from X columns only).
Out-of-fold predictions from stage 1 are used to build prediction features
(temporal derivatives, polynomial trends, rolling averages of the prediction
column), which are appended to the base features for stage 2.

``patient_id`` is required in X for within-patient grouping; it is dropped
before any model sees the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

# ── Import build_features and CAT_DTYPES from the experiment that defined them ─
# We re-import at call time to avoid circular imports; the actual objects are
# injected via TwoStageHGB(build_features_fn=..., cat_dtypes=...) kwargs so the
# class stays self-contained when called from skore.evaluate.
# Default: look up the 04 experiment module at runtime.


def _poly_trend_col(vals: np.ndarray, ages: np.ndarray, degree: int) -> np.ndarray:
    """Fit poly of `degree` per non-NaN visits; return fitted values at all ages.

    Falls back to the raw values when there are too few non-NaN points or no
    age spread (avoids ill-conditioned polyfit).
    """
    result = vals.copy().astype(float)
    mask = ~np.isnan(vals)
    age_spread = ages[mask].max() - ages[mask].min() if mask.sum() >= 2 else 0.0
    min_pts = degree + 1
    if mask.sum() >= min_pts and age_spread > 0:
        age_c = ages[mask] - ages[mask].mean()
        poly = np.polyfit(age_c, vals[mask], degree)
        result = np.polyval(poly, ages - ages[mask].mean())
    return result


def _build_pred_features(pred: np.ndarray, raw: pd.DataFrame) -> pd.DataFrame:
    """Build prediction features sorted by (patient_id, age), returned in raw order."""
    order = raw.sort_values(["patient_id", "age"], kind="stable").index

    # order is sorted index labels; pred is in raw row order.
    pred_sorted = pred[raw.index.get_indexer(order)]
    ages_sorted = raw.loc[order, "age"].values
    pids_sorted = raw.loc[order, "patient_id"].values

    n = len(pred_sorted)
    feat_s = pd.DataFrame(index=order)
    feat_s["pred"] = pred_sorted

    # Group in sorted space
    tmp = pd.DataFrame(
        {"pid": pids_sorted, "pred": pred_sorted, "age": ages_sorted}, index=order
    )
    g = tmp.groupby("pid", sort=False)

    feat_s["pred_s1"] = g["pred"].shift(1)
    feat_s["pred_s2"] = g["pred"].shift(2)
    feat_s["pred_m1"] = g["pred"].shift(-1)
    feat_s["pred_m2"] = g["pred"].shift(-2)
    feat_s["pred_pmean"] = g["pred"].transform("mean")

    feat_s["pred_cmean"] = g["pred"].transform(lambda s: s.expanding().mean())

    # Rolling (centered, min_periods=1)
    for w, name in [(3, "pred_roll3"), (5, "pred_roll5")]:
        rolled = tmp.groupby("pid")["pred"].apply(
            lambda x: x.rolling(w, center=True, min_periods=1).mean()
        )
        rolled.index = rolled.index.droplevel(0)
        feat_s[name] = rolled

    # Per-patient polynomial trend of pred vs age
    fit1_arr = np.full(n, np.nan)
    fit2_arr = np.full(n, np.nan)
    pos_array = np.arange(n)

    for pid, grp_idx in g.groups.items():
        pos = pos_array[order.get_indexer(grp_idx)]
        ages_p = ages_sorted[pos]
        vals_p = pred_sorted[pos]
        fit1_arr[pos] = _poly_trend_col(vals_p, ages_p, 1)
        fit2_arr[pos] = _poly_trend_col(vals_p, ages_p, 2)

    feat_s["pred_fit1"] = fit1_arr
    feat_s["pred_fit2"] = fit2_arr
    feat_s["pred_res1"] = pred_sorted - fit1_arr
    feat_s["pred_res2"] = pred_sorted - fit2_arr

    # Return in raw (original) order
    return feat_s.loc[raw.index]


class TwoStageHGB(BaseEstimator, RegressorMixin):
    """Two-stage HistGradientBoosting regressor.

    Parameters
    ----------
    build_features_fn :
        Callable ``(raw_df) -> feature_df``.  Must accept the raw X DataFrame
        that includes ``patient_id`` (used for sorting / grouping only) and
        return a feature DataFrame aligned to ``raw_df.index`` with no
        forbidden columns.
    cat_dtypes :
        ``dict[str, CategoricalDtype]`` used inside ``build_features_fn``.
        Passed for scikit-learn cloning; the function itself must use it.
    inner_n_splits : int
        Number of inner GroupKFold folds for OOF stage-1 predictions.
    """

    def __init__(
        self,
        build_features_fn=None,
        cat_dtypes=None,
        inner_n_splits: int = 5,
    ):
        self.build_features_fn = build_features_fn
        self.cat_dtypes = cat_dtypes
        self.inner_n_splits = inner_n_splits

    # ------------------------------------------------------------------
    @staticmethod
    def _hgb():
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=0,
            categorical_features="from_dtype",
        )

    def fit(self, X: pd.DataFrame, y):
        bfn = self.build_features_fn
        if bfn is None:
            raise ValueError("build_features_fn must be provided.")

        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values

        # ── Stage-1 features ──────────────────────────────────────────
        Xf = bfn(X)

        # ── OOF stage-1 predictions ───────────────────────────────────
        oof_pred = np.full(len(y), np.nan)
        inner_cv = GroupKFold(n_splits=self.inner_n_splits)
        for tr, te in inner_cv.split(Xf, y, groups=groups):
            m = self._hgb().fit(Xf.iloc[tr], y[tr])
            oof_pred[te] = m.predict(Xf.iloc[te])

        # ── Stage 1 fit on all training rows ─────────────────────────
        self.stage1_ = self._hgb().fit(Xf, y)

        # ── Stage-2 feature matrix ────────────────────────────────────
        pred_feats = _build_pred_features(oof_pred, X)
        Xf2 = pd.concat(
            [Xf.reset_index(drop=True), pred_feats.reset_index(drop=True)], axis=1
        )

        # ── Stage 2 fit ───────────────────────────────────────────────
        self.stage2_ = self._hgb().fit(Xf2, y)
        self.n_features_in_ = Xf.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["stage1_", "stage2_"])
        bfn = self.build_features_fn
        Xf = bfn(X)
        pred1 = self.stage1_.predict(Xf)
        pred_feats = _build_pred_features(pred1, X)
        Xf2 = pd.concat(
            [Xf.reset_index(drop=True), pred_feats.reset_index(drop=True)], axis=1
        )
        return self.stage2_.predict(Xf2)
