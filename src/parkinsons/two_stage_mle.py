"""Two-stage HGB with the bias-likelihood features (block 5) and a final
per-patient projection of the predictions.

Differences from ``TwoStageHGB`` (``parkinsons.two_stage``):

1. ``fit`` first calls ``calibrate_bias_model(X, y)`` on its own training
   rows, then builds the features with that calibration. In a cross-
   validation, the noise model and the empirical-Bayes prior therefore never
   see the held-out patients' targets.
2. ``predict`` projects each patient's predictions onto a quadratic in age,
   then onto non-decreasing values (isotonic regression). The true OFF never
   decreases within a patient and is a quadratic in age to within 0.35 point,
   so this removes visit-to-visit noise from the predictions.

``patient_id`` and ``age`` must be in X (grouping and ordering only).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import isotonic_regression
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted

from parkinsons.features import build_features, calibrate_bias_model
from parkinsons.two_stage import _build_pred_features

_SmoothMode = Literal["quad_iso", "iso"] | None


def smooth_per_patient(
    raw: pd.DataFrame,
    pred: np.ndarray | list[float],
    mode: _SmoothMode = "quad_iso",
    clip: tuple[float, float] | None = (0.0, 132.0),
) -> np.ndarray:
    """Project per-patient predictions onto a smooth, non-decreasing curve.

    True UPDRS motor scores never decrease within a patient and lie close to a
    quadratic in age (residual RMSE ≈ 0.35 point). Applying this constraint in
    post-processing removes visit-to-visit noise from the raw model output
    without altering the average level of predictions.

    Three projection modes are supported:

    ``None``
        No projection; predictions are returned as-is (only clipping applies).

    ``"iso"``
        Isotonic regression (PAVA) per patient on the age-sorted predictions.
        Enforces non-decreasing order at the cost of creating flat plateaus.

    ``"quad_iso"`` *(default)*
        First fits a quadratic (degree-2 polynomial) in age to the patient's
        predictions by ordinary least squares, then applies isotonic regression
        to the fitted values. The quadratic step smooths out large visit-to-
        visit swings before the monotone projection, which avoids the long flat
        segments that ``"iso"`` alone can produce.  Requires ≥ 3 visits with
        non-zero age spread; falls back to ``"iso"`` otherwise.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw input DataFrame; must contain ``"patient_id"`` (used for grouping)
        and ``"age"`` (used for ordering and the quadratic fit).
    pred : array-like of shape (n_samples,)
        Raw model predictions in the same row order as ``raw``.
    mode : {"quad_iso", "iso", None}, default ``"quad_iso"``
        Projection mode; see above.
    clip : tuple(float, float) or None, default ``(0.0, 132.0)``
        If not ``None``, predictions are clipped to ``[clip[0], clip[1]]``
        **after** the projection step.  The default bounds correspond to the
        clinical range of the MDS-UPDRS Part III scale.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Projected (and optionally clipped) predictions, in the same row order
        as ``raw``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``"quad_iso"``, ``"iso"``, or ``None``.
    """
    pred = np.asarray(pred, dtype=float)
    out = pred.copy()
    if mode not in (None, "iso", "quad_iso"):
        raise ValueError(f"Unknown smoothing mode: {mode!r}")
    if mode is not None:
        order = np.lexsort((raw["age"].to_numpy(), pd.factorize(raw["patient_id"])[0]))
        pid = raw["patient_id"].to_numpy()[order]
        age = raw["age"].to_numpy(dtype=float)[order]
        p = pred[order]
        sm = p.copy()
        starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(p)]
        for a, b in zip(starts[:-1], starts[1:]):
            seg = p[a:b]
            if mode == "quad_iso" and b - a >= 3 and np.ptp(age[a:b]) > 0:
                xx = age[a:b] - age[a:b].mean()
                A = np.c_[np.ones_like(xx), xx, xx**2]
                seg = A @ np.linalg.lstsq(A, seg, rcond=None)[0]
            sm[a:b] = isotonic_regression(seg, increasing=True)
        out[order] = sm
    if clip is not None:
        out = np.clip(out, *clip)
    return out


class TwoStageMLE(BaseEstimator, RegressorMixin):
    """Two-stage HGB with calibrated EMG bias-likelihood features (block 5).

    Architecture
    ------------
    **Stage 1** — a ``HistGradientBoostingRegressor`` trained on the full
    feature matrix produced by ``build_features(..., bias_calibration=...)``,
    which includes the 112-column set (blocks 1-3 + block 5: MLE/MAP curves
    under the exponentially-modified Gaussian bias model).

    **Stage 2** — a second HGB whose feature matrix is the stage-1 features
    augmented with out-of-fold (OOF) prediction features built by
    ``_build_pred_features`` (temporal shifts, rolling means, polynomial
    trends of the stage-1 predictions). The two-stage design lets stage 2
    correct systematic per-patient errors that stage 1 could not capture.

    **Prediction projection** — after stage 2, each patient's predictions are
    optionally projected onto a smooth, non-decreasing curve via
    ``smooth_per_patient``, exploiting the physiological constraint that true
    UPDRS scores never decrease within a patient.

    The EMG calibration (``calibrate_bias_model``) is always refitted on the
    training rows of the current fold so that held-out targets are never
    leaked, making the model safe to use inside a ``GroupKFold`` CV loop
    (e.g. via ``skore.evaluate``).

    Parameters
    ----------
    cat_dtypes : dict[str, pd.CategoricalDtype] or None, default ``None``
        Shared ``{col: CategoricalDtype}`` mapping from ``make_cat_dtypes``.
        Required for the HGB ``categorical_features="from_dtype"`` option.
    inner_n_splits : int, default 5
        Number of folds in the inner ``GroupKFold`` used to generate the
        stage-1 OOF predictions that feed into stage 2.
    bias_params : dict or None, default ``None``
        Keyword overrides for ``features.BIAS_PARAMS`` (e.g.
        ``{"prior_scale": 0.3}``).  ``None`` uses the module-level defaults.
    smooth : {"quad_iso", "iso", None}, default ``"quad_iso"``
        Per-patient projection mode applied at the end of ``predict``; see
        ``smooth_per_patient`` for details.
    two_stage : bool, default ``True``
        If ``False``, skip stage 2 entirely and return the stage-1 predictions
        directly (with smoothing).  Useful for ablation studies.
    max_iter : int, default 300
        Maximum number of boosting iterations for both HGB stages.
    learning_rate : float, default 0.05
        Shrinkage factor for each boosting step.
    max_leaf_nodes : int, default 31
        Maximum number of leaves per tree.
    min_samples_leaf : int, default 20
        Minimum number of samples per leaf; acts as a regulariser.

    Attributes
    ----------
    calibration_ : dict
        Output of ``calibrate_bias_model`` fitted on the training data.
        Keys: ``"noise"`` (5 × 5 EMG parameters), ``"B"`` (prior-mean
        regression coefficients), ``"Sigma"`` (prior covariance).
    stage1_ : HistGradientBoostingRegressor
        Stage-1 model fitted on the full training set.
    stage2_ : HistGradientBoostingRegressor or None
        Stage-2 model fitted on the augmented feature matrix, or ``None``
        when ``two_stage=False``.
    n_features_in_ : int
        Number of features in the stage-1 feature matrix (set for sklearn
        compatibility).

    Notes
    -----
    * ``patient_id`` and ``age`` must be columns of ``X``; they are used for
      grouping and ordering only and are never passed to any sklearn estimator.
    * Measured HGB settings: 800 trees / lr 0.03 / 63 leaves / 50 per leaf
      gave RMSE 2.6899 vs 2.6877 for the defaults in the two-stage setup
      (no gain, 2.5× slower).  A tighter empirical-Bayes prior
      (``bias_params={"prior_scale": 0.3}``) gave the best result: 2.6772.
    """

    def __init__(
        self,
        cat_dtypes: dict[str, pd.CategoricalDtype] | None = None,
        inner_n_splits: int = 5,
        bias_params: dict | None = None,
        smooth: _SmoothMode = "quad_iso",
        two_stage: bool = True,
        use_vcorr: bool = False,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
    ) -> None:
        self.cat_dtypes = cat_dtypes
        self.inner_n_splits = inner_n_splits
        self.bias_params = bias_params
        self.smooth = smooth
        self.two_stage = two_stage
        self.use_vcorr = use_vcorr
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def _hgb(self) -> HistGradientBoostingRegressor:
        """Return a fresh ``HistGradientBoostingRegressor`` with instance settings."""
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
            categorical_features="from_dtype",
        )

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature matrix from raw ``X`` using the fitted calibration."""
        return build_features(
            X,
            self.cat_dtypes,
            bias_calibration=self.calibration_,
            bias_params=self.bias_params,
        )

    @staticmethod
    def _stage2_matrix(
        Xf: pd.DataFrame,
        pred: np.ndarray,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate stage-1 features with OOF prediction features for stage 2."""
        return pd.concat(
            [
                Xf.reset_index(drop=True),
                _build_pred_features(pred, X).reset_index(drop=True),
            ],
            axis=1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TwoStageMLE":
        """Fit both stages on training data.

        The EMG bias model is calibrated on ``(X, y)`` first, so the
        bias-likelihood features (block 5) are computed with a calibration that
        has only seen the current training fold. This is the key property that
        prevents target leakage in cross-validation.

        When ``use_vcorr=True``, stage 1 is trained on ``map_Vcorr`` — the
        bias-corrected visit value ``(V + D̂) / c`` — instead of ``y``.  This
        pre-centres the target on the physical estimate of the latent T, leaving
        stage 2 to learn only the residual ``y − stage1_pred``.  At predict
        time the two outputs are **added**: ``ŷ = stage1_pred + stage2_pred``.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features.  Must contain ``"patient_id"`` and ``"age"``
            plus all columns expected by ``build_features``.
        y : array-like of shape (n_samples,)
            Target UPDRS motor scores (continuous, ≥ 0).

        Returns
        -------
        self : TwoStageMLE
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].to_numpy()
        self.calibration_ = calibrate_bias_model(X, y)  # training rows only
        Xf = self._features(X)
        self.n_features_in_ = Xf.shape[1]

        # ── Stage-1 target: map_Vcorr (bias-corrected visit value) or y ──────
        if self.use_vcorr and "map_Vcorr" in Xf.columns:
            # map_Vcorr = (V + D̂) / c  — best single-visit estimate of T.
            # It can contain extreme outliers from ill-conditioned MAP patients;
            # we clip to the clinical range [0, 132] and fall back to y for any
            # remaining non-finite values so stage 1 always sees a clean target.
            vcorr = np.clip(Xf["map_Vcorr"].to_numpy(), 0.0, 132.0)
            y1 = np.where(np.isfinite(vcorr), vcorr, y)
        else:
            y1 = y

        self.stage1_ = self._hgb().fit(Xf, y1)

        if not self.two_stage:
            self.stage2_ = None
            return self

        # ── Stage-2 target: residual y − OOF stage-1 prediction on y ─────────
        # OOF predictions are always on the original scale (y), not y1, so
        # the residual is well-defined and the stage-2 model corrects the
        # systematic gap between the physical estimate and the true label.
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=self.inner_n_splits).split(
            Xf, y, groups=groups
        ):
            if self.use_vcorr and "map_Vcorr" in Xf.columns:
                vcorr_tr = np.clip(
                    Xf.iloc[tr]["map_Vcorr"].to_numpy(), 0.0, 132.0
                )
                y1_tr = np.where(np.isfinite(vcorr_tr), vcorr_tr, y[tr])
            else:
                y1_tr = y[tr]
            oof[te] = self._hgb().fit(Xf.iloc[tr], y1_tr).predict(Xf.iloc[te])

        # stage2 learns the residual between y and OOF stage-1 prediction
        y2 = y - oof
        self.stage2_ = self._hgb().fit(self._stage2_matrix(Xf, oof, X), y2)
        return self

    def predict_unsmoothed(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw stage-2 predictions without the per-patient projection.

        Useful for diagnostics: comparing ``predict_unsmoothed`` with
        ``predict`` quantifies the effect of the monotone-quadratic projection.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features (same schema as ``fit``).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions before smoothing or clipping.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        check_is_fitted(self, ["stage1_"])
        Xf = self._features(X)
        pred1 = self.stage1_.predict(Xf)
        if self.stage2_ is None:
            return pred1
        # stage2 always predicts the residual (y − OOF stage-1); add it back.
        pred2 = self.stage2_.predict(self._stage2_matrix(Xf, pred1, X))
        return pred1 + pred2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict UPDRS motor scores with per-patient smoothing.

        Calls ``predict_unsmoothed`` then passes the result through
        ``smooth_per_patient`` using ``self.smooth`` as the projection mode.
        The default ``"quad_iso"`` mode fits a quadratic per patient then
        enforces monotone non-decreasing order via isotonic regression.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features (same schema as ``fit``).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Projected and clipped predictions.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        return smooth_per_patient(X, self.predict_unsmoothed(X), self.smooth)
