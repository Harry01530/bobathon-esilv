"""Two-stage model with a stacked, target-transformed, clipped stage 2.

Same stage 1 as ``TwoStageHGB`` (``parkinsons.two_stage``): a HGB on the
features from ``build_features``; its out-of-fold predictions become extra
"prediction features" for stage 2.

Stage 2 changes:

1. **Stacking** — ``StackingRegressor`` over three different model families:
   HistGradientBoosting, Ridge (imputation + scaling + one-hot) and
   ExtraTrees. The meta-model is a ``LinearRegression(positive=True)``, i.e. a
   non-negative weighted blend of the three base predictions.
   The stacking's internal CV uses **patient-grouped** splits. The default
   (row ``KFold``) would put visits of the same patient on both sides, make
   the tree models look better than they are, and over-weight them.
2. **Target transform** — stage 2 is optionally wrapped in a
   ``TransformedTargetRegressor`` (``"log1p"``, ``"standard"`` or ``"none"``).
3. **Clipping** — predictions are clipped to the clinical range of the
   MDS-UPDRS Part III (0-132) inside ``predict``, so the cross-validation
   scores already include the clipping.

``patient_id`` must be in X: it is only used for grouping and is dropped
by ``build_features`` before any model sees the data.
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted

from parkinsons.two_stage import _build_pred_features

# ExtraTrees handles NaN natively from scikit-learn 1.6 onwards.
_ET_NATIVE_NAN = parse_version(sklearn.__version__) >= parse_version("1.6")

_NUM = make_column_selector(dtype_exclude="category")
_CAT = make_column_selector(dtype_include="category")

_TargetTransform = Literal["none", "log1p", "standard"]


def make_hgb(random_state: int = 0) -> HistGradientBoostingRegressor:
    """Return a ``HistGradientBoostingRegressor`` with the project-standard settings.

    Uses 300 trees, learning rate 0.05, and ``categorical_features="from_dtype"``
    so that pandas ``CategoricalDtype`` columns are handled natively without
    any one-hot encoding overhead.

    Parameters
    ----------
    random_state : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    HistGradientBoostingRegressor
        Unfitted estimator ready to call ``.fit()``.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        categorical_features="from_dtype",
        random_state=random_state,
    )


def make_ridge() -> Pipeline:
    """Return a Ridge pipeline with imputation, scaling, and one-hot encoding.

    ``RidgeCV`` cannot handle missing values or categorical columns directly, so
    the pipeline applies:

    1. ``ColumnTransformer`` with two branches:

       * **numeric** — ``SimpleImputer(strategy="median", add_indicator=True)``
         followed by ``StandardScaler``.  The ``add_indicator=True`` flag adds a
         binary column for each imputed feature so the model can learn from the
         missingness pattern.
       * **categorical** — ``OneHotEncoder(handle_unknown="ignore")``.

    2. ``RidgeCV`` over 11 log-spaced alphas from 0.01 to 1000.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ``[ColumnTransformer, RidgeCV]``.
    """
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                _NUM,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CAT),
        ]
    )
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 11)))


def make_extra_trees(
    n_estimators: int = 200,
    random_state: int = 0,
    n_jobs: int = -1,
) -> Pipeline:
    """Return an ExtraTrees regression pipeline with categorical encoding.

    ExtraTrees does not support pandas ``CategoricalDtype`` natively, so
    categories are converted to integer ordinal codes via ``OrdinalEncoder``.
    Unknown categories and missing values are mapped to ``-1``.

    NaN handling for numeric columns depends on the scikit-learn version:

    * **≥ 1.6** — ``ExtraTreesRegressor`` supports NaN natively; numeric
      columns are passed through unchanged.
    * **< 1.6** — a ``SimpleImputer(strategy="median")`` is inserted.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of trees in the ensemble.
    random_state : int, default 0
        Random seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs (``-1`` = all available cores).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ``[ColumnTransformer, ExtraTreesRegressor]``.
    """
    num_step = (
        "passthrough"
        if _ET_NATIVE_NAN
        else SimpleImputer(strategy="median", add_indicator=True)
    )
    pre = ColumnTransformer(
        [
            ("num", num_step, _NUM),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                _CAT,
            ),
        ]
    )
    return make_pipeline(
        pre,
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features=0.5,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
    )


def make_stage2(
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    target_transform: _TargetTransform = "none",
    use_ridge: bool = True,
    use_extra_trees: bool = True,
    et_n_estimators: int = 200,
    random_state: int = 0,
    n_jobs: int = -1,
) -> StackingRegressor | TransformedTargetRegressor:
    """Build the stage-2 stacked regressor, optionally wrapped in a target transform.

    The stacking meta-model is a ``LinearRegression(positive=True)``, i.e. a
    non-negative weighted blend of the base predictions. This prevents the
    meta-model from shorting one base model against another, which would be
    numerically unstable with three highly correlated predictors.

    The stacking's internal CV uses ``inner_splits`` — patient-grouped folds
    passed in from the outer ``fit`` call. Using row-level ``KFold`` (the
    sklearn default) would put visits of the same patient on both sides of each
    fold, making tree-based models appear better than they are and over-weighting
    them in the final blend.

    Parameters
    ----------
    inner_splits : list of (train_idx, test_idx) pairs
        Pre-computed patient-grouped CV splits.  Passed directly to
        ``StackingRegressor(cv=inner_splits)`` so the stacking uses the same
        patient-disjoint folds as the OOF stage-1 predictions.
    target_transform : {"none", "log1p", "standard"}, default ``"none"``
        Wraps the stacked regressor in a ``TransformedTargetRegressor``:

        * ``"none"``     — no transform; returns the ``StackingRegressor`` directly.
        * ``"log1p"``    — ``func=np.log1p, inverse_func=np.expm1``; stabilises
          variance for skewed targets.
        * ``"standard"`` — ``transformer=StandardScaler()``; zero-mean, unit-variance.
    use_ridge : bool, default ``True``
        Include the Ridge pipeline as a base model.
    use_extra_trees : bool, default ``True``
        Include ExtraTrees as a third base model.
    et_n_estimators : int, default 200
        Number of trees in the ExtraTrees base model.
    random_state : int, default 0
        Passed to ``make_hgb`` and ``make_extra_trees``.
    n_jobs : int, default -1
        Parallelism for ExtraTrees.

    Returns
    -------
    StackingRegressor or TransformedTargetRegressor
        Unfitted stage-2 regressor.

    Raises
    ------
    ValueError
        If fewer than two base models are selected, or if ``target_transform``
        is not one of the three recognised values.
    """
    estimators = [("hgb", make_hgb(random_state))]
    if use_ridge:
        estimators.append(("ridge", make_ridge()))
    if use_extra_trees:
        estimators.append(
            ("et", make_extra_trees(et_n_estimators, random_state, n_jobs))
        )
    if len(estimators) < 2:
        raise ValueError("Stacking needs at least two base models.")
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(positive=True),
        cv=inner_splits,  # patient-grouped, never row KFold
        passthrough=False,
    )
    if target_transform == "none":
        return stack
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=stack, func=np.log1p, inverse_func=np.expm1
        )
    if target_transform == "standard":
        return TransformedTargetRegressor(regressor=stack, transformer=StandardScaler())
    raise ValueError(f"Unknown target_transform: {target_transform!r}")


class TwoStageStack(BaseEstimator, RegressorMixin):
    """Two-stage regressor: HGB stage 1, stacked + transformed + clipped stage 2.

    Architecture
    ------------
    **Stage 1** — a ``HistGradientBoostingRegressor`` trained on the feature
    matrix from ``build_features_fn``.  Out-of-fold (OOF) predictions are
    generated with an inner ``GroupKFold`` and used as extra features for
    stage 2.

    **Stage 2** — a ``StackingRegressor`` over three base models (HGB, Ridge,
    ExtraTrees) with a non-negative ``LinearRegression`` meta-model.  The
    stage-2 feature matrix is the stage-1 features plus the OOF prediction
    features from ``_build_pred_features`` (temporal shifts, rolling means,
    polynomial trends).  The stacking's internal CV reuses the same patient-
    grouped splits as the OOF generation, so patient identities never cross
    fold boundaries.

    **Target transform** — stage 2 is optionally wrapped in a
    ``TransformedTargetRegressor`` (``"log1p"``, ``"standard"`` or ``"none"``).

    **Clipping** — ``predict`` clips to ``[clip_lo_, clip_hi_]``, which is the
    union of ``clip_range`` and the training-target range.

    Parameters
    ----------
    build_features_fn : callable or None, default ``None``
        ``(raw_df: pd.DataFrame) -> feature_df: pd.DataFrame`` aligned to
        ``raw_df.index``.  See ``parkinsons.features.build_features``.
        Required; raises ``ValueError`` in ``fit`` if ``None``.
    cat_dtypes : dict[str, pd.CategoricalDtype] or None, default ``None``
        Shared category-dtype mapping; stored as a parameter so that
        ``sklearn.base.clone`` can reconstruct the estimator faithfully.
    inner_n_splits : int, default 5
        Inner ``GroupKFold`` folds, reused for both the OOF predictions and
        the stacking's internal CV.
    target_transform : {"none", "log1p", "standard"}, default ``"none"``
        Target transform applied to stage 2; see ``make_stage2``.
    use_ridge : bool, default ``True``
        Include Ridge as a base model in the stacking.
    use_extra_trees : bool, default ``True``
        Include ExtraTrees as a base model in the stacking.
    et_n_estimators : int, default 200
        Number of trees in the ExtraTrees base model.
    clip_range : tuple(float, float) or None, default ``(0.0, 132.0)``
        Clinical prediction bounds (MDS-UPDRS Part III: 0–132). Widened
        automatically if any training target falls outside the range.
        Pass ``None`` to disable clipping entirely.
    random_state : int, default 0
        Passed to ``make_hgb`` and ``make_extra_trees``.
    n_jobs : int, default -1
        Parallelism for ExtraTrees (``-1`` = all cores).

    Attributes
    ----------
    stage1_ : HistGradientBoostingRegressor
        Stage-1 model fitted on the full training set.
    stage2_ : StackingRegressor or TransformedTargetRegressor
        Stage-2 stacked model fitted on the augmented feature matrix.
    clip_lo_, clip_hi_ : float
        Effective lower and upper clipping bounds, set during ``fit``.
    n_features_in_ : int
        Number of features in the stage-1 feature matrix.

    Notes
    -----
    ``patient_id`` must be a column of ``X``; it is used for grouping only and
    is dropped by ``build_features_fn`` before any estimator sees the data.
    """

    def __init__(
        self,
        build_features_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        cat_dtypes: dict[str, pd.CategoricalDtype] | None = None,
        inner_n_splits: int = 5,
        target_transform: _TargetTransform = "none",
        use_ridge: bool = True,
        use_extra_trees: bool = True,
        et_n_estimators: int = 200,
        clip_range: tuple[float, float] | None = (0.0, 132.0),
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.build_features_fn = build_features_fn
        self.cat_dtypes = cat_dtypes
        self.inner_n_splits = inner_n_splits
        self.target_transform = target_transform
        self.use_ridge = use_ridge
        self.use_extra_trees = use_extra_trees
        self.et_n_estimators = et_n_estimators
        self.clip_range = clip_range
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    def _stage2_matrix(
        self,
        Xf: pd.DataFrame,
        pred: np.ndarray,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate stage-1 features with OOF prediction features for stage 2."""
        pred_feats = _build_pred_features(pred, X)
        return pd.concat(
            [Xf.reset_index(drop=True), pred_feats.reset_index(drop=True)], axis=1
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TwoStageStack":
        """Fit both stages on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features.  Must contain ``"patient_id"`` (for grouped
            splits) plus all columns expected by ``build_features_fn``.
        y : array-like of shape (n_samples,)
            Target UPDRS motor scores (continuous, ≥ 0).

        Returns
        -------
        self : TwoStageStack
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``build_features_fn`` is ``None``.
        """
        if self.build_features_fn is None:
            raise ValueError("build_features_fn must be provided.")
        y = np.asarray(y, dtype=float)
        groups = X["patient_id"].values

        # Stage-1 features
        Xf = self.build_features_fn(X)

        # One set of patient-grouped inner splits, reused by stage 1 OOF and
        # by the stacking CV (a list, so StackingRegressor accepts it).
        inner_splits = list(
            GroupKFold(n_splits=self.inner_n_splits).split(Xf, y, groups=groups)
        )

        # Stage 1: OOF predictions, then a fit on all rows for predict()
        oof_pred = np.full(len(y), np.nan)
        for tr, te in inner_splits:
            m = make_hgb(self.random_state).fit(Xf.iloc[tr], y[tr])
            oof_pred[te] = m.predict(Xf.iloc[te])
        self.stage1_ = make_hgb(self.random_state).fit(Xf, y)

        # Stage 2: stacked, transformed
        Xf2 = self._stage2_matrix(Xf, oof_pred, X)
        self.stage2_ = make_stage2(
            inner_splits,
            target_transform=self.target_transform,
            use_ridge=self.use_ridge,
            use_extra_trees=self.use_extra_trees,
            et_n_estimators=self.et_n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(Xf2, y)

        # Clipping bounds: clinical range, never tighter than the train target
        if self.clip_range is None:
            self.clip_lo_, self.clip_hi_ = -np.inf, np.inf
        else:
            lo, hi = map(float, self.clip_range)
            if y.min() < lo or y.max() > hi:
                warnings.warn(
                    f"Target range [{y.min():.2f}, {y.max():.2f}] exceeds "
                    f"clip_range {self.clip_range}; widening the bounds."
                )
            self.clip_lo_, self.clip_hi_ = min(lo, y.min()), max(hi, y.max())

        self.n_features_in_ = Xf.shape[1]
        return self

    def predict_unclipped(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw stage-2 predictions without clipping.

        Useful for diagnostics: comparing ``predict_unclipped`` with ``predict``
        quantifies how many predictions fall outside the clinical range and by
        how much.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features (same schema as ``fit``).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Stage-2 predictions before clipping.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        check_is_fitted(self, ["stage1_", "stage2_"])
        Xf = self.build_features_fn(X)
        pred1 = self.stage1_.predict(Xf)
        return self.stage2_.predict(self._stage2_matrix(Xf, pred1, X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict UPDRS motor scores, clipped to the clinical range.

        Calls ``predict_unclipped`` then clips to ``[clip_lo_, clip_hi_]``.
        Because the clipping is applied inside ``predict``, cross-validation
        scores (e.g. from ``skore.evaluate``) already include the clipping
        effect.

        Parameters
        ----------
        X : pd.DataFrame
            Raw input features (same schema as ``fit``).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Clipped predictions in ``[clip_lo_, clip_hi_]``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        return np.clip(self.predict_unclipped(X), self.clip_lo_, self.clip_hi_)

    def stack_weights(self) -> dict[str, float]:
        """Return the meta-model weight of each base model.

        The meta-model is a ``LinearRegression(positive=True)``, so weights
        are non-negative and sum to approximately 1.  Useful for the
        experiment journal and for sanity-checking that no base model
        dominates unexpectedly.

        Returns
        -------
        dict[str, float]
            ``{model_name: weight}`` mapping, e.g.
            ``{"hgb": 0.45, "ridge": 0.01, "et": 0.54}``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        check_is_fitted(self, ["stage2_"])
        stack = getattr(self.stage2_, "regressor_", self.stage2_)
        names = [name for name, _ in stack.estimators]
        coefs = stack.final_estimator_.coef_
        return {n: float(c) for n, c in zip(names, coefs)}
