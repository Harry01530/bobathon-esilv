"""Feature engineering for the Parkinson's UPDRS prediction project.

``build_features(raw, cat_dtypes)`` builds the full feature matrix
(04 base features + 05 extensions) from the raw X DataFrame.
``make_cat_dtypes(train_raw, test_raw)`` builds the shared
``CategoricalDtype`` dict that must be passed consistently across
train and test to keep category codes aligned.

Neither function uses ``target``, ``Index``, ``patient_id``, or
``rater_id`` as model inputs.  ``patient_id`` is used internally for
within-patient grouping and is never present in the returned feature
matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
CAT_COLS  = ["cohort", "gene"]
FORBIDDEN = {"Index", "patient_id", "target", "rater_id"}


def make_cat_dtypes(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> dict[str, pd.CategoricalDtype]:
    """Return a ``{col: CategoricalDtype}`` dict covering train + test categories.

    Parameters
    ----------
    train_raw :
        Raw training feature DataFrame (must contain the categorical columns).
    test_raw :
        Raw test feature DataFrame (must contain the same categorical columns).

    Returns
    -------
    dict[str, CategoricalDtype]
        One entry per column in ``CAT_COLS``.  Category codes are stable across
        train and test because both sets of categories are pooled.
    """
    return {
        c: pd.CategoricalDtype(
            sorted(
                pd.concat([train_raw[c], test_raw[c]]).dropna().unique()
            )
        )
        for c in CAT_COLS
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _poly_trend_col(
    vals: np.ndarray,
    ages: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Fit a polynomial of ``degree`` on non-NaN visits; return fitted values.

    Falls back to the raw values when there are too few non-NaN points or
    the age spread is zero (avoids ill-conditioned ``polyfit``).

    Parameters
    ----------
    vals :
        Observed values (may contain NaN).
    ages :
        Corresponding ages (same length as ``vals``).
    degree :
        Polynomial degree (1 or 2).

    Returns
    -------
    np.ndarray
        Fitted values at every age position (same length as ``vals``).
    """
    result = vals.copy().astype(float)
    mask = ~np.isnan(vals)
    min_pts = degree + 1
    if mask.sum() >= min_pts and ages[mask].max() - ages[mask].min() > 0:
        age_c  = ages[mask] - ages[mask].mean()
        poly   = np.polyfit(age_c, vals[mask], degree)
        result = np.polyval(poly, ages - ages[mask].mean())
    return result


def _loo_kernel(
    age: np.ndarray,
    z: np.ndarray,
    h: float,
) -> np.ndarray:
    """Leave-one-out Gaussian kernel smoother (vectorised per patient).

    Parameters
    ----------
    age :
        Age values for one patient's visits (1-D, sorted).
    z :
        Observed values for the same visits (may contain NaN).
    h :
        Bandwidth in the same units as ``age`` (years).

    Returns
    -------
    np.ndarray
        Smoothed values; NaN where the weight sum is below 1e-9.
    """
    ok  = ~np.isnan(z)
    W   = np.exp(-0.5 * ((age[:, None] - age[None, :]) / h) ** 2) * ok[None, :]
    np.fill_diagonal(W, 0.0)
    sw  = W.sum(1)
    num = (W * np.nan_to_num(z)[None, :]).sum(1)
    return np.where(sw > 1e-9, num / np.where(sw > 1e-9, sw, 1.0), np.nan)


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(
    raw: pd.DataFrame,
    cat_dtypes: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """Build the full feature matrix from raw X columns.

    Sorts by ``(patient_id, age)`` with a stable sort, computes all features
    on that ordered view, then returns the columns in the **original** row
    order of ``raw`` so alignment with ``y`` is preserved.

    Includes:
    - Base numeric columns and categoricals (with shared ``cat_dtypes``).
    - Simple derived features: ``years_dx``, missingness indicators.
    - Visit-order features: ``vis_rank``, ``since_first``, ``n_vis``.
    - Neighbour shifts (±1 and ±2) for ``off`` and ``on``.
    - Time deltas ``dt_prev``, ``dt_next``.
    - Shifted expanding cumulative mean and max (no same-row leakage).
    - Whole-patient aggregates (mean, max, deviation).
    - Per-patient polynomial trends degree 1 and 2 for ``off`` and ``on``.
    - Extended centred rolling means (window 3 and 5).
    - Difference and slope features.
    - Patient means of ``time_since_intake_*``.
    - ``ledd`` neighbours.
    - Cross/timing features.
    - LOO Gaussian kernel smoothing (bandwidths 0.5, 1.0, 2.0 years).

    Parameters
    ----------
    raw :
        Raw feature DataFrame.  Must include ``patient_id`` (used for
        grouping only; never in the output) and all columns in
        ``NUMERIC_COLS`` and ``CAT_COLS``.
    cat_dtypes :
        Shared ``{col: CategoricalDtype}`` from ``make_cat_dtypes``.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to ``raw.index``.  Contains no column from
        ``FORBIDDEN``.
    """
    order = raw.sort_values(["patient_id", "age"], kind="stable").index
    s     = raw.loc[order].copy()
    g     = s.groupby("patient_id", sort=False)

    feat = pd.DataFrame(index=s.index)

    # ── 1. Base numeric + categoricals ────────────────────────────────────
    for c in NUMERIC_COLS:
        feat[c] = s[c]
    for c in CAT_COLS:
        feat[c] = s[c].astype(cat_dtypes[c])

    # ── 2. Simple derived ─────────────────────────────────────────────────
    feat["years_dx"] = s["age"] - s["age_at_diagnosis"]
    feat["off_na"]   = s["off"].isna().astype(np.int8)
    feat["on_na"]    = s["on"].isna().astype(np.int8)
    feat["ledd_na"]  = s["ledd"].isna().astype(np.int8)

    # ── 3. Visit order ────────────────────────────────────────────────────
    feat["vis_rank"]    = g["age"].rank(method="first") - 1
    feat["since_first"] = s["age"] - g["age"].transform("min")
    feat["n_vis"]       = g["age"].transform("size")

    # ── 4. Neighbour shifts ±1 ────────────────────────────────────────────
    for c in ["off", "on"]:
        feat[f"{c}_prev"] = g[c].shift(1)
        feat[f"{c}_next"] = g[c].shift(-1)
    feat["dt_prev"] = s["age"] - g["age"].shift(1)
    feat["dt_next"] = g["age"].shift(-1) - s["age"]

    # ── 5. Cumulative (shifted to avoid same-row leakage) ─────────────────
    for c in ["off", "on", "ledd"]:
        exp_m = g[c].expanding().mean().shift(1)
        exp_m.index = exp_m.index.droplevel(0)
        feat[f"{c}_cmean"] = exp_m
        exp_x = g[c].expanding().max().shift(1)
        exp_x.index = exp_x.index.droplevel(0)
        feat[f"{c}_cmax"] = exp_x
    feat["off_dev_c"] = s["off"] - feat["off_cmean"]

    # ── 6. Whole-patient aggregates ───────────────────────────────────────
    for c in ["off", "on", "ledd"]:
        feat[f"{c}_pmean"] = g[c].transform("mean")
        feat[f"{c}_pmax"]  = g[c].transform("max")
    feat["off_dev"] = s["off"] - feat["off_pmean"]

    # ── 7. Per-patient polynomial trend (degree 1 and 2) ─────────────────
    pos_array = np.arange(len(s))
    for c in ["off", "on"]:
        fit1 = np.full(len(s), np.nan)
        fit2 = np.full(len(s), np.nan)
        for pid, idx in g.groups.items():
            pos  = pos_array[s.index.get_indexer(idx)]
            ages = s.loc[idx, "age"].values
            vals = s.loc[idx, c].values
            fit1[pos] = _poly_trend_col(vals, ages, 1)
            fit2[pos] = _poly_trend_col(vals, ages, 2)
        feat[f"{c}_fit1"] = fit1
        feat[f"{c}_fit2"] = fit2
        feat[f"{c}_res1"] = s[c].values - fit1
        feat[f"{c}_res2"] = s[c].values - fit2
    feat["gap1"] = feat["on_fit1"] - feat["off_fit1"]

    # ── 8. Neighbour shifts ±2 ────────────────────────────────────────────
    for c in ["off", "on"]:
        feat[f"{c}_prev2"] = g[c].shift(2)
        feat[f"{c}_next2"] = g[c].shift(-2)

    # ── 9. Centred rolling means (window 3 and 5, min_periods=1) ─────────
    for c in ["off", "on"]:
        for w, tag in [(3, "roll3"), (5, "roll5")]:
            rolled = (
                s.groupby("patient_id")[c]
                .apply(lambda x, w=w: x.rolling(w, center=True, min_periods=1).mean())
            )
            rolled.index = rolled.index.droplevel(0)
            feat[f"{c}_{tag}"] = rolled

    # ── 10. Differences and slope ─────────────────────────────────────────
    for c in ["off", "on"]:
        feat[f"{c}_dprev"]    = s[c].values - feat[f"{c}_prev"].values
        feat[f"{c}_dnext"]    = feat[f"{c}_next"].values - s[c].values
        denom = feat["dt_prev"].values + feat["dt_next"].values
        feat[f"{c}_slope_pn"] = np.where(
            denom > 0,
            (feat[f"{c}_next"].values - feat[f"{c}_prev"].values) / denom,
            np.nan,
        )

    # ── 11. Patient means of time_since_* ─────────────────────────────────
    for c in ["time_since_intake_off", "time_since_intake_on"]:
        feat[f"{c}_pmean"] = g[c].transform("mean")

    # ── 12. ledd neighbours ───────────────────────────────────────────────
    feat["ledd_prev"] = g["ledd"].shift(1)
    feat["ledd_next"] = g["ledd"].shift(-1)

    # ── 13. Cross / timing features ───────────────────────────────────────
    feat["tsi_gap"]   = (s["time_since_intake_off"] - s["time_since_intake_on"]).values
    feat["off_x_tsi"] = (s["off"] * s["time_since_intake_off"]).values
    feat["on_x_tsi"]  = (s["on"]  * s["time_since_intake_on"]).values

    # ── 14. LOO Gaussian kernel smoothing ─────────────────────────────────
    ages_arr = s["age"].values
    for c in ["off", "on"]:
        vals_arr = s[c].values
        for h, htag in [(0.5, "h05"), (1.0, "h10"), (2.0, "h20")]:
            ks = np.full(len(s), np.nan)
            for pid, grp_idx in g.groups.items():
                pos = s.index.get_indexer(grp_idx)
                ks[pos] = _loo_kernel(ages_arr[pos], vals_arr[pos], h)
            feat[f"k_{c}_{htag}"] = ks

    leaked = FORBIDDEN & set(feat.columns)
    assert not leaked, f"Forbidden columns in output: {leaked}"
    return feat.loc[raw.index]
