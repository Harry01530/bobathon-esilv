"""Feature engineering for the Parkinson's "true OFF" prediction project.

Public API (unchanged, so 05/06/07 and TwoStageHGB keep working):

- ``make_cat_dtypes(train_raw, test_raw)``: shared ``CategoricalDtype`` dict.
- ``build_features(raw, cat_dtypes)``: full feature matrix aligned to
  ``raw.index``.

Feature blocks
--------------
1. ``_base_features``      04-05 visit / patient features (unchanged).
2. ``latent_trajectory``   v1 latent "true OFF" curve (07).
3. ``visit_envelope``      v2 (08): ONE value per visit, per-patient drug
                           factor from the within-visit ON/OFF ratio, upper-
                           envelope (expectile) curves and running maximum.
4. ``lme_trajectory``      linear mixed-effects model fitted by EM. Off by
                           default: measured, it does not help (see 08 journal).

Key facts behind block 3 (official train; target used for analysis only):

* Within a visit, ON and OFF carry the SAME multiplicative error: their
  deviations from the target correlate at 0.977. It is a per-visit "rater"
  bias, independent from one visit to the next (lag-1 autocorrelation 0.01)
  and unrelated to any covariate. Consequences: ON and OFF of one visit are
  one observation, not two; and their ratio ON/OFF is almost bias-free, so it
  measures the drug effect, and each patient's drug response, precisely.
* That bias is very asymmetric: OFF / target has its mode at 1.02, 75 % of
  values <= 1.00 and 95 % <= 1.12, but a long low tail (25 % below 0.66).
  A measurement is "right or too low", rarely too high, so the latent curve
  lies near the UPPER ENVELOPE of the visit values, not their mean.

Facts behind block 2:

* The target never decreases within a patient (39 014 / 39 014 consecutive
  visit pairs) and a per-patient quadratic in age reproduces it with an RMSE
  of 0.33. The latent curve is smooth, monotone and low-dimensional.
* ON measurements are distorted by the drug: ON / true OFF falls from
  about 0.80 right after intake to about 0.44 after 1.5 h (absorption), then
  stays flat. The effect does not depend on the dose (LEDD).
* Measured OFF is about 0.81 x true OFF, with a little extra drug effect when
  the OFF exam is less than about 10 h after intake.
* Measurement noise is large (15-30 %). For ON, 42 % of the noise is a
  persistent per-patient factor (individual drug response); the rest, like
  the OFF noise, is independent from visit to visit.

How (all estimated from X only, without the target)
--------------------------------------------------
1. Drug correction: on visits where both ON and OFF were measured, the
   curve ``R(t) = sum(OFF) / sum(ON)`` per bin of ``time_since_intake_on``
   gives the factor that puts an ON score on the OFF scale. After a first
   latent fit, the factors are re-estimated against the latent itself
   (ON vs ``time_since_intake_on``, OFF vs ``time_since_intake_off``,
   with "time missing" as its own bin). Scale anchor: an OFF measured with
   unknown intake time has factor 1.
2. Joint smoothing: for each patient, the corrected OFF and ON
   observations are pooled into ONE series and fitted with a penalised
   quadratic in age (empirical-Bayes / mixed-model style shrinkage toward the
   population slope and curvature), with inverse-variance weights per
   measurement type and a per-patient ON response factor ``rho``.
3. Monotone projection: the fitted values are projected onto
   non-decreasing sequences with isotonic regression (PAVA), since true OFF
   never decreases.

``patient_id`` is only used for grouping; it never appears in the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.isotonic import isotonic_regression

NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CAT_COLS = ["cohort", "gene"]
FORBIDDEN = {"Index", "patient_id", "target", "rater_id"}

# Latent-trajectory settings (tuned by patient-grouped CV on the training set).
LATENT_PARAMS = {
    "lam_slope": 3.0,  # prior precision of the slope (prior std ~0.6 point/year)
    "lam_curv": 100.0,  # prior precision of the curvature (prior std ~0.1)
    "rho_prior_var": 0.02,  # prior variance of the per-patient ON factor (std ~0.14)
    "n_calib_iter": 2,  # re-calibration passes of the drug-correction curves
    "n_alt": 4,  # alternations curve <-> rho per patient
}
# Visit-envelope settings (block 3).
ENVELOPE_PARAMS = {
    "taus": (0.5, 0.8, 0.9),  # expectile levels (0.5 = mean, 0.9 = upper envelope)
    "lam_slope": 3.0,  # same shrinkage as block 2
    "lam_curv": 100.0,
    "kappa": 2.0,  # pseudo-visits shrinking the patient drug factor to 1
    "n_irls": 15,  # iterations of the asymmetric least squares
}
# Mixed-effects settings (block 4, off by default).
LME_PARAMS = {
    "n_calib_iter": 2,
    "em_iter_first": 40,
    "em_iter_next": 15,
    "x_scale": 5.0,
}
# Bias-likelihood settings (block 5, needs a calibration from calibrate_bias_model).
BIAS_PARAMS = {
    "lam_slope": 1.0,  # MLE block: prior precision of the slope (tuned)
    "lam_curv": 100.0,  # MLE block: prior precision of the curvature (tuned)
    "prior_scale": 0.5,  # MAP block: empirical-Bayes covariance x 0.5 (tuned)
    "n_newton": 15,  # Newton steps per pass
    "n_outer": 3,  # passes re-evaluating the level-dependent noise
}

_N_BINS = 16  # quantile bins for the correction curves
_MIN_COUNT = 30  # minimum visits per bin; below that the bin is skipped
_FALLBACK_ON_TO_OFF = 1.84  # plateau OFF/ON ratio, used only if no pairs exist


def make_cat_dtypes(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> dict[str, pd.CategoricalDtype]:
    """Return a ``{col: CategoricalDtype}`` dict covering train + test categories.

    Category codes are stable across train and test because both sets of
    categories are pooled.
    """
    return {
        c: pd.CategoricalDtype(
            sorted(pd.concat([train_raw[c], test_raw[c]]).dropna().unique())
        )
        for c in CAT_COLS
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part A — drug-effect correction curves (target-free)
# ══════════════════════════════════════════════════════════════════════════════


def _ratio_curve(t, num, den, fallback):
    """Ratio-of-sums ``num / den`` per quantile bin of ``t``.

    Returns ``(centers, ratios, nan_ratio)``: a piecewise-linear curve over
    ``t`` plus the ratio for rows whose ``t`` is missing. Ratio of sums (not
    mean of ratios) keeps it stable when ``den`` is small or zero (scores are
    integers and can be 0).
    """
    t = np.asarray(t, dtype=float)
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = np.isfinite(t)

    def _safe(n, d):
        return n.sum() / d.sum() if len(d) >= _MIN_COUNT and d.sum() > 0 else np.nan

    global_ratio = _safe(num, den)
    if not np.isfinite(global_ratio):
        global_ratio = fallback

    centers, ratios = [], []
    if ok.sum() >= _MIN_COUNT:
        edges = np.unique(np.quantile(t[ok], np.linspace(0, 1, _N_BINS + 1)))
        if len(edges) >= 2:
            idx = np.clip(
                np.searchsorted(edges, t[ok], side="right") - 1, 0, len(edges) - 2
            )
            for b in range(len(edges) - 1):
                m = idx == b
                r = _safe(num[ok][m], den[ok][m])
                if np.isfinite(r):
                    centers.append(np.median(t[ok][m]))
                    ratios.append(r)
    if not centers:  # not enough data: flat curve
        centers, ratios = [0.0], [global_ratio]

    nan_ratio = _safe(num[~ok], den[~ok])
    if not np.isfinite(nan_ratio):
        nan_ratio = global_ratio
    return np.asarray(centers), np.asarray(ratios), float(nan_ratio)


def _apply_curve(t, curve):
    """Evaluate a ``_ratio_curve`` result at ``t`` (flat beyond the ends)."""
    centers, ratios, nan_ratio = curve
    t = np.asarray(t, dtype=float)
    out = np.interp(t, centers, ratios)
    out[~np.isfinite(t)] = nan_ratio
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Part B — joint latent trajectory per patient
# ══════════════════════════════════════════════════════════════════════════════


def _safe_mse(resid: np.ndarray, floor: float = 1.0) -> float:
    """Mean squared residual over finite values; ``floor`` if none or too small."""
    r = resid[np.isfinite(resid)]
    return max(float(np.mean(r**2)), floor) if len(r) else floor


def _fit_patient(xa, zo, xo, zn, xn, s2_off, s2_on, prior, lam, rho_var, n_alt):
    """Penalised quadratic fit of one patient's pooled corrected series.

    Model (centred age ``x``):
        OFF_corr_i = L(x_i) + e_i,           e_i ~ N(0, s2_off)
        ON_corr_j  = rho * L(x_j) + u_j,     u_j ~ N(0, s2_on)
        L(x) = a + b x + c x^2,  (b, c) ~ N(prior, diag(1/lam)),  rho ~ N(1, rho_var)

    Solved by alternating weighted least squares (``theta | rho`` is linear,
    ``rho | theta`` is a 1-D shrunk regression).
    Returns ``(theta, rho)``.
    """
    Ao = np.c_[np.ones_like(xo), xo, xo**2]
    An = np.c_[np.ones_like(xn), xn, xn**2]
    Lam = np.diag([0.0, lam[0], lam[1]])  # intercept is not penalised
    rho = 1.0
    theta = np.r_[np.nan, prior[1], prior[2]]
    for _ in range(n_alt):
        M = (Ao.T @ Ao) / s2_off + (rho**2) * (An.T @ An) / s2_on + Lam
        v = (Ao.T @ zo) / s2_off + rho * (An.T @ zn) / s2_on + Lam @ prior
        M[0, 0] += 1e-9  # guard if a patient has no data
        theta = np.linalg.solve(M, v)
        if len(zn) == 0 or len(zo) == 0:
            break  # rho not identifiable: keep 1
        f = An @ theta
        rho = (f @ zn / s2_on + 1.0 / rho_var) / (f @ f / s2_on + 1.0 / rho_var)
    return theta, rho


def latent_trajectory(raw: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Estimate each patient's latent "true OFF" curve from X only.

    Parameters
    ----------
    raw :
        Raw X (must contain ``patient_id``, ``age``, ``on``, ``off``,
        ``time_since_intake_on``, ``time_since_intake_off``).
    params :
        Overrides for ``LATENT_PARAMS``.

    Returns
    -------
    pd.DataFrame aligned to ``raw.index`` with columns:
        lat_L       penalised-quadratic latent at the visit (OFF scale)
        lat_Liso    same, projected to be non-decreasing in age (isotonic)
        lat_slope   local slope dL/d(age) at the visit (points / year)
        lat_curv    curvature of the patient's curve
        lat_rho     per-patient ON response factor (1 = population average)
        lat_k_on    drug-correction factor applied to this visit's ON score
        lat_z_off   this visit's corrected OFF
        lat_z_on    this visit's corrected ON (divided by rho)
        lat_res_off lat_z_off - lat_L  (visit-level deviation)
        lat_res_on  lat_z_on  - lat_L
    """
    p = {**LATENT_PARAMS, **(params or {})}
    lam = (p["lam_slope"], p["lam_curv"])

    s = raw.sort_values(["patient_id", "age"], kind="stable")
    age = s["age"].to_numpy(dtype=float)
    on = s["on"].to_numpy(dtype=float)
    off = s["off"].to_numpy(dtype=float)
    t_on = s["time_since_intake_on"].to_numpy(dtype=float)
    t_off = s["time_since_intake_off"].to_numpy(dtype=float)
    pid = s["patient_id"].to_numpy()
    n = len(s)
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, n]

    # ── Initial drug correction from visits with both ON and OFF ─────────────
    both = np.isfinite(on) & np.isfinite(off)
    k_on = _apply_curve(
        t_on, _ratio_curve(t_on[both], off[both], on[both], _FALLBACK_ON_TO_OFF)
    )
    k_off = np.ones(n)
    s2_on = s2_off = 1.0  # equal weights in the first pass
    prior = np.zeros(3)  # population (., slope, curvature)

    for it in range(p["n_calib_iter"] + 1):
        z_on, z_off = on * k_on, off * k_off
        L = np.full(n, np.nan)
        L_iso = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        curv = np.full(n, np.nan)
        rho = np.full(n, np.nan)
        thetas = []

        for a, b in zip(starts[:-1], starts[1:]):
            ag = age[a:b]
            ok_age = np.isfinite(ag)
            if not ok_age.any():
                continue
            c0 = ag[ok_age].mean()
            x = ag - c0
            mo = ok_age & np.isfinite(z_off[a:b])
            mn = ok_age & np.isfinite(z_on[a:b])
            if not (mo.any() or mn.any()):
                continue
            theta, r = _fit_patient(
                x,
                z_off[a:b][mo],
                x[mo],
                z_on[a:b][mn],
                x[mn],
                s2_off,
                s2_on,
                prior,
                lam,
                p["rho_prior_var"],
                p["n_alt"],
            )
            thetas.append(theta)
            fit = theta[0] + theta[1] * x + theta[2] * x**2
            L[a:b] = fit
            # monotone projection (true OFF never decreases with age)
            iso = np.full(b - a, np.nan)
            okf = ok_age & np.isfinite(fit)
            if okf.any():
                iso[okf] = isotonic_regression(fit[okf], increasing=True)
            L_iso[a:b] = iso
            slope[a:b] = theta[1] + 2 * theta[2] * x
            curv[a:b] = theta[2]
            rho[a:b] = r

        if thetas:
            prior = np.r_[0.0, np.median(np.asarray(thetas)[:, 1:], axis=0)]
        if it == p["n_calib_iter"]:
            break

        # ── Re-calibrate the drug-correction curves against the latent ──────
        ok = np.isfinite(on) & np.isfinite(L) & (L > 1)
        k_on = _apply_curve(
            t_on, _ratio_curve(t_on[ok], rho[ok] * L[ok], on[ok], _FALLBACK_ON_TO_OFF)
        )
        ok = np.isfinite(off) & np.isfinite(L) & (L > 1)
        off_curve = _ratio_curve(t_off[ok], L[ok], off[ok], 1.0)
        k_off = _apply_curve(t_off, off_curve)
        anchor = off_curve[2]  # OFF with unknown intake time = scale reference
        k_off /= anchor
        k_on /= anchor
        # inverse-variance weights per measurement type (floored for safety)
        s2_on = _safe_mse(on * k_on - rho * L)
        s2_off = _safe_mse(off * k_off - L)

    z_on_scaled = on * k_on / rho
    z_off_final = off * k_off
    out = pd.DataFrame(
        {
            "lat_L": L,
            "lat_Liso": L_iso,
            "lat_slope": slope,
            "lat_curv": curv,
            "lat_rho": rho,
            "lat_k_on": k_on,
            "lat_z_off": z_off_final,
            "lat_z_on": z_on_scaled,
            "lat_res_off": z_off_final - L,
            "lat_res_on": z_on_scaled - L,
        },
        index=s.index,
    )
    return out.loc[raw.index]


# ══════════════════════════════════════════════════════════════════════════════
# Part C — visit values and upper-envelope curves (block 3)
# ══════════════════════════════════════════════════════════════════════════════


def _median_curve(t, r, fallback):
    """Median of ``r`` per quantile bin of ``t``; rows with missing ``t`` get
    the median of their own group. Robust version of ``_ratio_curve``."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    fin = np.isfinite(r)
    glob = float(np.median(r[fin])) if fin.sum() >= _MIN_COUNT else fallback
    ok = np.isfinite(t) & fin
    centers, vals = [], []
    if ok.sum() >= _MIN_COUNT:
        edges = np.unique(np.quantile(t[ok], np.linspace(0, 1, _N_BINS + 1)))
        if len(edges) >= 2:
            idx = np.clip(
                np.searchsorted(edges, t[ok], side="right") - 1, 0, len(edges) - 2
            )
            for b in range(len(edges) - 1):
                m = idx == b
                if m.sum() >= _MIN_COUNT:
                    centers.append(np.median(t[ok][m]))
                    vals.append(np.median(r[ok][m]))
    if not centers:
        centers, vals = [0.0], [glob]
    nan_m = ~np.isfinite(t) & fin
    nan_val = float(np.median(r[nan_m])) if nan_m.sum() >= _MIN_COUNT else glob
    return np.asarray(centers), np.asarray(vals), nan_val


def _expectile_quadratic(codes, slot, x, V, tau, lam, n_irls):
    """Per-patient penalised quadratic fitted with an asymmetric squared loss.

    Minimises, for each patient,
        sum_i  w_tau(r_i) * r_i^2 / s2  +  lam_b (b - b0)^2 + lam_c (c - c0)^2
    with r_i = V_i - (a + b x_i + c x_i^2) and w_tau(r) = tau if r > 0 else
    1 - tau (expectile regression; tau = 0.5 is ordinary least squares).
    Solved by iteratively re-weighted least squares, batched over patients.
    Returns per-patient coefficients ``theta`` (P x 3) and a has-data mask.
    """
    P = int(codes.max()) + 1
    Mv = int(slot.max()) + 1
    A = np.zeros((P, Mv, 3))
    z = np.zeros((P, Mv))
    mask = np.zeros((P, Mv), dtype=bool)
    ok = np.isfinite(V) & np.isfinite(x)
    A[codes, slot] = np.c_[np.ones_like(x), np.nan_to_num(x), np.nan_to_num(x) ** 2]
    z[codes[ok], slot[ok]] = V[ok]
    mask[codes[ok], slot[ok]] = True
    has_data = mask.any(axis=1)

    Lam = np.diag([0.0, lam[0], lam[1]])
    prior = np.zeros(3)
    s2 = 50.0
    theta = np.zeros((P, 3))
    for _ in range(2):  # 2 passes: update prior and scale
        w_asym = np.where(mask, 0.5, 0.0)
        for _ in range(n_irls):
            w = 2.0 * w_asym / s2  # tau = 0.5 -> weight 1 / s2
            M = np.einsum("pmi,pm,pmj->pij", A, w, A) + Lam[None]
            M[:, 0, 0] += 1e-9
            v = np.einsum("pmi,pm->pi", A, w * z) + (Lam @ prior)[None]
            theta = np.linalg.solve(M, v[..., None])[..., 0]
            r = z - np.einsum("pmi,pi->pm", A, theta)
            w_asym = np.where(mask, np.where(r > 0, tau, 1.0 - tau), 0.0)
        if has_data.any():
            prior = np.r_[0.0, np.median(theta[has_data, 1:], axis=0)]
        res = (z - np.einsum("pmi,pi->pm", A, theta))[mask]
        if len(res):
            s2 = max(float(np.median(np.abs(res))) ** 2 / 0.455, 1.0)  # MAD scale
    return theta, has_data


def visit_envelope(raw: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """One OFF-scale value per visit + upper-envelope curves (block 3).

    Steps (all target-free):
    1. Population drug curve ``m(t) = median(ON / OFF)`` per bin of
       ``time_since_intake_on``, over visits with both scores (the visit
       bias cancels in the ratio).
    2. Patient drug factor ``rho = median((ON/OFF) / m(t_on))`` over the
       patient's visits with both scores, shrunk to 1 with ``kappa``
       pseudo-visits.
    3. Visit value ``V``: OFF if measured, else ``ON / (m(t_on) * rho)``;
       mean of the two when both exist.
    4. Per patient, penalised quadratic fitted by expectile regression at
       several ``tau`` (0.9 = upper envelope), then projected to be
       non-decreasing (isotonic regression).
    5. Running maximum of ``V`` along the visits (true OFF never decreases
       and measurements are rarely too high, so it is a lower bound).

    Returns columns prefixed ``vis_`` (visit values, drug factor) and
    ``env_`` (envelope curves), aligned to ``raw.index``.
    """
    p = {**ENVELOPE_PARAMS, **(params or {})}
    s = raw.sort_values(["patient_id", "age"], kind="stable")
    n = len(s)
    if n == 0:
        return pd.DataFrame(index=raw.index)
    on = s["on"].to_numpy(dtype=float)
    off = s["off"].to_numpy(dtype=float)
    t_on = s["time_since_intake_on"].to_numpy(dtype=float)
    age = s["age"].to_numpy(dtype=float)
    pid = s["patient_id"].to_numpy()
    codes, _ = pd.factorize(pid)  # sorted by patient -> non-decreasing
    slot = np.arange(n) - np.searchsorted(codes, codes, side="left")

    # 1-2. drug curve and patient drug factor from the within-visit ratio
    both = np.isfinite(on) & np.isfinite(off) & (off > 0)
    q = np.full(n, np.nan)
    q[both] = on[both] / off[both]
    m_on = _apply_curve(
        t_on, _median_curve(t_on[both], q[both], 1.0 / _FALLBACK_ON_TO_OFF)
    )
    rel = q / m_on
    grp = pd.Series(rel).groupby(codes)
    med = grp.transform("median").to_numpy()
    nb = grp.transform("count").to_numpy().astype(float)
    rho = np.where(
        nb > 0, (nb * np.nan_to_num(med, nan=1.0) + p["kappa"]) / (nb + p["kappa"]), 1.0
    )
    rho = np.clip(rho, 0.3, 3.0)

    # 3. one value per visit, on the OFF scale
    on_scaled = on / (m_on * rho)
    V = np.where(np.isfinite(off), off, on_scaled)
    V = np.where(both, 0.5 * (off + on_scaled), V)

    out = {
        "vis_val": V,
        "vis_on_scaled": on_scaled,
        "vis_rho": rho,
        "vis_nboth": nb,
        "vis_m_on": m_on,
    }

    # 4. expectile curves + isotonic projection
    x = age - pd.Series(age).groupby(codes).transform("mean").to_numpy()
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, n]
    for tau in p["taus"]:
        theta, has_data = _expectile_quadratic(
            codes, slot, x, V, tau, (p["lam_slope"], p["lam_curv"]), p["n_irls"]
        )
        fit = theta[codes, 0] + theta[codes, 1] * x + theta[codes, 2] * x**2
        fit[~has_data[codes] | ~np.isfinite(x)] = np.nan
        iso = fit.copy()
        for a, b in zip(starts[:-1], starts[1:]):
            okf = np.isfinite(fit[a:b])
            if okf.any():
                seg = iso[a:b]
                seg[okf] = isotonic_regression(fit[a:b][okf], increasing=True)
                iso[a:b] = seg
        tag = f"env_e{int(round(tau * 100))}"
        out[f"{tag}_L"] = fit
        out[f"{tag}_Liso"] = iso
        out[f"{tag}_slope"] = theta[codes, 1] + 2 * theta[codes, 2] * x

    # 5. running maximum of the visit values (lower bound of true OFF)
    out["env_cummax"] = pd.Series(V).groupby(codes).cummax().to_numpy()
    top = f"env_e{int(round(max(p['taus']) * 100))}_L"
    out["vis_rel"] = np.where(
        out[top] > 0.5, V / np.where(out[top] > 0.5, out[top], 1.0), np.nan
    )

    return pd.DataFrame(out, index=s.index).loc[raw.index]


# ══════════════════════════════════════════════════════════════════════════════
# Part D — linear mixed-effects latent model, fitted by EM (block 4, optional)
# ══════════════════════════════════════════════════════════════════════════════


def _fixed_design(s: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Population curve: cubic in years since diagnosis + covariates."""
    age = s["age"].to_numpy(dtype=float)
    dx = s["age_at_diagnosis"].to_numpy(dtype=float)
    dx_na = ~np.isfinite(dx)
    first_age = (
        s.groupby("patient_id", sort=False)["age"]
        .transform("min")
        .to_numpy(dtype=float)
    )
    gap = float(np.nanmedian(first_age[~dx_na] - dx[~dx_na])) if (~dx_na).any() else 2.0
    dx = np.where(dx_na, first_age - gap, dx)  # impute missing diagnosis age
    t = np.clip(age - dx, 0, 30) / 10.0
    cols = {
        "one": np.ones(len(s)),
        "t": t,
        "t2": t**2,
        "t3": t**3,
        "dx": (dx - 60.0) / 10.0,
        "dx_t": (dx - 60.0) / 10.0 * t,
        "sex": s["sexM"].to_numpy(dtype=float),
        "dx_na": dx_na.astype(float),
    }
    for c in CAT_COLS:
        v = s[c].astype("object").where(s[c].notna(), "NA").astype(str).to_numpy()
        for lev in sorted(set(v))[1:]:  # first level = reference
            cols[f"{c}={lev}"] = (v == lev).astype(float)
    return np.column_stack(list(cols.values())), list(cols)


def _lme_em(Xp, Zp, zp, mask, typ, n_iter, init=None):
    """EM (maximum likelihood) for  z = X beta + Z b + e,
    b ~ N(0, G) per patient (full covariance), e ~ N(0, s2[type]).
    Batched over patients with padded arrays (P x M x .)."""
    P, M, Q = Zp.shape
    K = Xp.shape[2]
    if init is None:
        w = mask.reshape(-1)
        beta = np.linalg.lstsq(Xp.reshape(-1, K)[w], zp.reshape(-1)[w], rcond=None)[0]
        G = np.diag([100.0, 4.0, 1.0, 25.0][:Q])
        s2 = np.array([50.0, 50.0])
    else:
        beta, G, s2 = init[0].copy(), init[1].copy(), init[2].copy()
    n_t = np.array([(mask & (typ == k)).sum() for k in (0, 1)], dtype=float)
    b = np.zeros((P, Q))
    Vb = np.zeros((P, Q, Q))
    for _ in range(n_iter):
        rinv = np.where(mask, 1.0 / s2[typ], 0.0)
        Ginv = np.linalg.inv(G + 1e-8 * np.eye(Q))
        Vb = np.linalg.inv(np.einsum("pmi,pm,pmj->pij", Zp, rinv, Zp) + Ginv[None])
        r0 = np.where(mask, zp - np.einsum("pmk,k->pm", Xp, beta), 0.0)
        b = np.einsum("pij,pj->pi", Vb, np.einsum("pmj,pm->pj", Zp, rinv * r0))  # BLUP
        r1 = np.where(mask, zp - np.einsum("pmq,pq->pm", Zp, b), 0.0)
        beta = np.linalg.solve(
            np.einsum("pmk,pm,pml->kl", Xp, rinv, Xp) + 1e-8 * np.eye(K),
            np.einsum("pmk,pm->k", Xp, rinv * r1),
        )
        G = (np.einsum("pi,pj->ij", b, b) + Vb.sum(0)) / P
        e = np.where(
            mask,
            zp - np.einsum("pmk,k->pm", Xp, beta) - np.einsum("pmq,pq->pm", Zp, b),
            0.0,
        )
        quad = np.einsum("pmi,pij,pmj->pm", Zp, Vb, Zp)
        for k in (0, 1):
            if n_t[k] > 0:
                s2[k] = max(((e**2 + quad) * (mask & (typ == k))).sum() / n_t[k], 1.0)
    return beta, G, s2, b, Vb


def lme_trajectory(raw: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Linear mixed-effects latent "true OFF" (block 4).

    Model on drug-corrected ON/OFF measurements (long format):
        z_ij = X_ij beta + [1, x, x^2, is_on]_ij b_i + e_ij
    - fixed effects X: population curve (cubic in years since diagnosis,
      diagnosis age, sex, cohort, gene);
    - random effects b_i ~ N(0, G): patient intercept, slope, curvature and
      ON offset, with a full 4 x 4 covariance G;
    - e ~ N(0, s2_off) or N(0, s2_on): one residual variance per type.
    beta, G, s2 are estimated by EM; the latent is the BLUP
    ``X beta + [1, x, x^2] b_i``, with its posterior SD.
    """
    p = {**LME_PARAMS, **(params or {})}
    s = raw.sort_values(["patient_id", "age"], kind="stable")
    n = len(s)
    age = s["age"].to_numpy(dtype=float)
    on = s["on"].to_numpy(dtype=float)
    off = s["off"].to_numpy(dtype=float)
    t_on = s["time_since_intake_on"].to_numpy(dtype=float)
    t_off = s["time_since_intake_off"].to_numpy(dtype=float)
    pid = s["patient_id"].to_numpy()
    codes, uniq = pd.factorize(pid)
    P = len(uniq)
    x = (age - pd.Series(age).groupby(codes).transform("mean").to_numpy()) / p[
        "x_scale"
    ]
    Xv, _ = _fixed_design(s)
    K = Xv.shape[1]
    Zlat = np.column_stack([np.ones(n), x, x**2])
    Q = 4

    both = np.isfinite(on) & np.isfinite(off)
    k_on = _apply_curve(
        t_on, _ratio_curve(t_on[both], off[both], on[both], _FALLBACK_ON_TO_OFF)
    )
    k_off = np.ones(n)
    state = None
    for it in range(p["n_calib_iter"] + 1):
        z_off, z_on = off * k_off, on * k_on
        fo, fn = np.isfinite(z_off), np.isfinite(z_on)
        rv = np.r_[np.flatnonzero(fo), np.flatnonzero(fn)]
        rt = np.r_[np.zeros(fo.sum(), int), np.ones(fn.sum(), int)]
        rz = np.r_[z_off[fo], z_on[fn]]
        o = np.lexsort((rt, rv))
        rv, rt, rz = rv[o], rt[o], rz[o]
        rp = codes[rv]
        slot = np.arange(len(rp)) - np.searchsorted(rp, rp, side="left")
        M = int(slot.max()) + 1 if len(slot) else 1
        Xp = np.zeros((P, M, K))
        Zp = np.zeros((P, M, Q))
        zp = np.zeros((P, M))
        mask = np.zeros((P, M), dtype=bool)
        typ = np.zeros((P, M), dtype=int)
        Xp[rp, slot] = Xv[rv]
        Zp[rp, slot, :3] = Zlat[rv]
        Zp[rp, slot, 3] = rt == 1
        zp[rp, slot] = rz
        mask[rp, slot] = True
        typ[rp, slot] = rt
        n_iter = p["em_iter_first"] if state is None else p["em_iter_next"]
        beta, G, s2, b, Vb = _lme_em(Xp, Zp, zp, mask, typ, n_iter, state)
        state = (beta, G, s2)
        pop = Xv @ beta
        L = pop + np.einsum("vq,vq->v", Zlat, b[codes, :3])
        if it == p["n_calib_iter"]:
            break
        ok = np.isfinite(on) & (L > 1)
        k_on = _apply_curve(
            t_on,
            _ratio_curve(t_on[ok], (L + b[codes, 3])[ok], on[ok], _FALLBACK_ON_TO_OFF),
        )
        ok = np.isfinite(off) & (L > 1)
        oc = _ratio_curve(t_off[ok], L[ok], off[ok], 1.0)
        k_off = _apply_curve(t_off, oc) / oc[2]
        k_on = k_on / oc[2]

    sd = np.sqrt(
        np.maximum(np.einsum("vi,vij,vj->v", Zlat, Vb[codes][:, :3, :3], Zlat), 0)
    )
    L_iso = L.copy()
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, n]
    for a, bb in zip(starts[:-1], starts[1:]):
        L_iso[a:bb] = isotonic_regression(L[a:bb], increasing=True)
    bp = b[codes]
    out = pd.DataFrame(
        {
            "lme_L": L,
            "lme_Liso": L_iso,
            "lme_sd": sd,
            "lme_pop": pop,
            "lme_b0": bp[:, 0],
            "lme_slope": (bp[:, 1] + 2 * bp[:, 2] * x) / p["x_scale"],
            "lme_b2": bp[:, 2],
            "lme_on_bias": bp[:, 3],
            "lme_res_off": off * k_off - L,
            "lme_res_on": on * k_on - L - bp[:, 3],
        },
        index=s.index,
    )
    return out.loc[raw.index]


# ══════════════════════════════════════════════════════════════════════════════
# Part F — maximum-likelihood model of the asymmetric visit bias (block 5)
# ══════════════════════════════════════════════════════════════════════════════
#
# Measurement model, one value V per visit (block 3), visit type g:
#
#     V = c_g * T(x) + eps - D,   eps ~ N(0, sigma_g(T)^2),   D ~ Exp(mean mu_g(T))
#
# i.e. an exponentially-modified Gaussian (EMG): a symmetric noise eps plus an
# under-scoring D >= 0, both growing with the level T as power laws
#     sigma_g(T) = exp(s0_g + s1_g log T),   mu_g(T) = exp(m0_g + m1_g log T).
# Visit types g: 0 OFF only (untreated), 1 OFF only (treated), 2 ON and OFF,
# 3 ON only with intake time, 4 ON only without intake time.
#
# The EMG log-density is concave, so per-patient maximum likelihood has a
# unique optimum, reached by Newton's method (batched over patients):
#     log f(w) = -log mu + w/mu + sigma^2/(2 mu^2) + log Phi(-w/sigma - sigma/mu)
#     d(-log f)/dw  = -1/mu + lambda(a)/sigma,              a = -w/sigma - sigma/mu
#     d2(-log f)/dw2 = lambda(a) (a + lambda(a)) / sigma^2 >= 0,  lambda = phi/Phi
#
# Two estimators of the curve T(x) = a + b x + c x^2 (x = age - patient mean age):
#   mle_*  likelihood + independent Gaussian priors on b, c (lam_slope, lam_curv)
#   map_*  likelihood + empirical-Bayes prior (a, b, c) ~ N(Z_p B, Sigma), learned
#          from the TRUE training trajectories (slope and curvature correlate
#          at -0.86); Z_p = target-free patient covariates.
# The noise parameters, B and Sigma come from ``calibrate_bias_model(X, y)``,
# which must only ever see training targets (TwoStageMLE refits it per fold).

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_N_VISIT_GROUPS = 5
_DEFAULT_NOISE = np.array([0.0, -0.72, 0.72, -0.83, 0.76])  # used for tiny groups


def _emg_logpdf(w, sig, mu):
    r"""Log-density of the Exponentially-Modified Gaussian (EMG) distribution.

    The EMG arises from the measurement model::

        V = c * T + eps - D,   eps ~ N(0, sig^2),   D ~ Exp(mean=mu),   D >= 0

    where the under-scoring term ``D`` is always non-negative (physicians
    under-score, never over-score). Marginalising ``D`` over [0, +inf) yields
    the closed-form log-density for the residual ``w = V - c*T``:

    .. math::

        \log f(w \mid \sigma, \mu) =
            -\log\mu
            + \frac{w}{\mu}
            + \frac{\sigma^2}{2\mu^2}
            + \log\Phi\!\left(-\frac{w}{\sigma} - \frac{\sigma}{\mu}\right)

    where :math:`\Phi` is the standard normal CDF.

    Derivation sketch
    -----------------
    Condition on ``D = d >= 0``:

    .. math::

        f(w) = \int_0^{+\infty}
                 \frac{1}{\sigma}\phi\!\left(\frac{w+d}{\sigma}\right)
                 \frac{1}{\mu}e^{-d/\mu}\,\mathrm{d}d

    Complete the square in the exponent
    :math:`-(w+d)^2/(2\sigma^2) - d/\mu`, collect the constant, and
    recognise the remaining integral as :math:`\Phi(\cdot)`.

    Notes
    -----
    * ``scipy.special.log_ndtr`` is used for :math:`\log\Phi` to preserve
      accuracy in the left tail where ``Phi`` underflows to zero.
    * The function is **concave** in ``w`` (see ``_emg_psi``), which guarantees
      a unique maximum-likelihood optimum.

    Parameters
    ----------
    w : array_like
        Residuals ``V - c * T``.  May be negative (under-scoring) or mildly
        positive (symmetric noise wins).
    sig : array_like
        Standard deviation of the Gaussian noise component, ``sigma_g(T) > 0``.
    mu : array_like
        Mean of the exponential under-scoring component, ``mu_g(T) > 0``.

    Returns
    -------
    np.ndarray
        Element-wise log-density values (always finite for finite inputs with
        ``sig, mu > 0``).
    """
    return -np.log(mu) + w / mu + sig**2 / (2 * mu**2) + log_ndtr(-w / sig - sig / mu)


def _mills(a):
    r"""Inverse Mills ratio :math:`\lambda(a) = \phi(a) / \Phi(a)`, numerically stable.

    The standard formula ``scipy.stats.norm.pdf(a) / scipy.stats.norm.cdf(a)``
    loses all precision for ``a << 0`` because both numerator and denominator
    underflow independently.  This implementation stays in log-space::

        log lambda(a) = log phi(a) - log Phi(a)
                      = -a^2/2 - log(sqrt(2 pi)) - log_ndtr(a)

    and exponentiates only at the end, so it is accurate across the full range
    of ``a`` encountered in practice (roughly -50 to +10).

    Parameters
    ----------
    a : array_like
        Argument; typically ``a = -w/sigma - sigma/mu`` in the EMG context.

    Returns
    -------
    np.ndarray
        :math:`\phi(a)/\Phi(a) \geq 0`.  The ratio diverges as ``a -> -inf``
        (very negative residuals become increasingly unlikely under the Gaussian
        component alone) but stays finite for any finite ``a``.
    """
    return np.exp(-0.5 * a**2 - _LOG_SQRT_2PI - log_ndtr(a))


def _emg_psi(w, sig, mu):
    r"""First and second derivatives of the EMG negative log-density w.r.t. ``w``.

    Let :math:`f(w)` be the EMG density from ``_emg_logpdf``.  Define the
    composite argument:

    .. math::

        a = -\frac{w}{\sigma} - \frac{\sigma}{\mu}

    and the inverse Mills ratio :math:`\lambda(a) = \phi(a)/\Phi(a)`.

    First derivative (gradient of the negative log-likelihood)
    ----------------------------------------------------------
    Differentiating :math:`-\log f(w)` term by term:

    * :math:`-w/\mu`  contributes  :math:`-1/\mu`.
    * :math:`-\log\Phi(a)` contributes  :math:`-[\partial\log\Phi/\partial a]
      \cdot [\partial a/\partial w]`.

    Since :math:`\partial a/\partial w = -1/\sigma` and
    :math:`\partial\log\Phi(a)/\partial a = \phi(a)/\Phi(a) = \lambda(a)`:

    .. math::

        \psi_1(w) \triangleq \frac{\partial(-\log f)}{\partial w}
            = -\frac{1}{\mu} + \frac{\lambda(a)}{\sigma}

    Second derivative (Hessian element; proves concavity)
    -----------------------------------------------------
    Differentiating :math:`\psi_1` again:

    .. math::

        \psi_2(w) \triangleq \frac{\partial^2(-\log f)}{\partial w^2}
            = \frac{1}{\sigma} \cdot \frac{\partial\lambda(a)}{\partial w}
            = \frac{1}{\sigma} \cdot \frac{\partial\lambda}{\partial a}
              \cdot \left(-\frac{1}{\sigma}\right)

    Using the standard identity
    :math:`\partial\lambda/\partial a = -\lambda(a)\bigl(a + \lambda(a)\bigr)`:

    .. math::

        \psi_2(w) = \frac{\lambda(a)\,(a + \lambda(a))}{\sigma^2} \geq 0

    The inequality holds because :math:`a + \lambda(a) \geq 0` for all
    :math:`a` (a known property of the inverse Mills ratio), so the EMG
    negative log-density is **convex** in ``w``, guaranteeing a unique MLE.

    Parameters
    ----------
    w : array_like
        Residuals ``V - c * T``.
    sig : array_like
        Gaussian noise standard deviation, ``sigma_g(T) > 0``.
    mu : array_like
        Exponential mean (under-scoring), ``mu_g(T) > 0``.

    Returns
    -------
    psi1 : np.ndarray
        :math:`\partial(-\log f)/\partial w` — used as the per-visit
        contribution to the Newton gradient.
    psi2 : np.ndarray
        :math:`\partial^2(-\log f)/\partial w^2 \geq 0` — used as the
        per-visit contribution to the Newton Hessian.  Floored at ``1e-12``
        to prevent division issues in degenerate configurations.
    """
    a = -w / sig - sig / mu
    lam = _mills(a)
    return -1.0 / mu + lam / sig, np.maximum(lam * (a + lam) / sig**2, 1e-12)


def visit_groups(s: pd.DataFrame) -> np.ndarray:
    """Visit type (0-4) used by the noise model."""
    on = s["on"].notna().to_numpy()
    off = s["off"].notna().to_numpy()
    treated = (s["ledd"].notna() | s["on"].notna()).to_numpy()
    ton = s["time_since_intake_on"].notna().to_numpy()
    g = np.full(len(s), 4)
    g[on & ton] = 3
    g[on & off] = 2
    g[off & ~on & treated] = 1
    g[off & ~on & ~treated] = 0
    return g


def _group_noise(noise, g, T):
    lt = np.log(np.clip(T, 2.0, None))
    p = noise[g]
    return (
        np.exp(p[:, 0]),
        np.exp(p[:, 1] + p[:, 2] * lt),
        np.exp(p[:, 3] + p[:, 4] * lt),
    )


def _sorted_view(raw):
    s = raw.sort_values(["patient_id", "age"], kind="stable")
    codes, _ = pd.factorize(s["patient_id"].to_numpy())
    slot = np.arange(len(codes)) - np.searchsorted(codes, codes, side="left")
    age = s["age"].to_numpy(dtype=float)
    x = age - pd.Series(age).groupby(codes).transform("mean").to_numpy()
    return s, codes, slot, x


def _patient_covariates(s, codes):
    """Target-free covariates per patient for the prior mean (P x 10)."""
    age = s["age"].to_numpy(dtype=float)
    dx = s["age_at_diagnosis"].to_numpy(dtype=float)
    dx_na = ~np.isfinite(dx)
    first = pd.Series(age).groupby(codes).transform("min").to_numpy()
    gap = float(np.nanmedian(first[~dx_na] - dx[~dx_na])) if (~dx_na).any() else 2.0
    dx = np.where(dx_na, first - gap, dx)
    df = pd.DataFrame(
        {
            "code": codes,
            "t": np.clip(age - dx, 0, 40),
            "dx": dx,
            "dx_na": dx_na,
            "sex": s["sexM"].to_numpy(dtype=float),
            "age": age,
            "offna": s["off"].isna().to_numpy(dtype=float),
            "untreated": s["ledd"].isna().to_numpy(dtype=float),
        }
    )
    P = df.groupby("code").agg(
        t=("t", "mean"),
        dx=("dx", "first"),
        dx_na=("dx_na", "first"),
        sex=("sex", "first"),
        span=("age", lambda a: a.max() - a.min()),
        n=("age", "size"),
        offna=("offna", "mean"),
        untreated=("untreated", "mean"),
    )
    return np.column_stack(
        [
            np.ones(len(P)),
            P.t / 10,
            (P.t / 10) ** 2,
            (P.dx - 60) / 10,
            P.dx_na,
            P.sex,
            P.span / 10,
            P.n / 10,
            P.offna,
            P.untreated,
        ]
    )


def _design(codes, slot, x, V, g):
    P, Mv = int(codes.max()) + 1, int(slot.max()) + 1
    A = np.zeros((P, Mv, 3))
    z = np.zeros((P, Mv))
    mask = np.zeros((P, Mv), dtype=bool)
    G = np.zeros((P, Mv), dtype=int)
    ok = np.isfinite(V) & np.isfinite(x)
    A[codes, slot] = np.c_[np.ones_like(x), np.nan_to_num(x), np.nan_to_num(x) ** 2]
    z[codes[ok], slot[ok]] = V[ok]
    mask[codes[ok], slot[ok]] = True
    G[codes, slot] = g
    return A, z, mask, G


def _newton_emg(
    A,
    z,
    mask,
    G,
    noise,
    theta,
    prior_mean,
    prior_prec,
    n_newton,
    n_outer,
    update_prior=False,
):
    r"""Batched Newton-Raphson for the penalised EMG likelihood of every patient.

    Optimisation problem
    --------------------
    For each patient ``p`` with visits ``m = 0 … M-1``, minimise the penalised
    negative log-likelihood over the quadratic-curve parameters
    :math:`\theta_p = (a_p, b_p, c_p)^{\!\top}`:

    .. math::

        \mathcal{F}(\theta_p)
          = -\!\sum_{m:\,\text{mask}_{pm}}
              \log f\!\bigl(w_{pm} \mid \sigma_{pm}, \mu_{pm}\bigr)
          + \tfrac{1}{2}(\theta_p - \mu_p)^{\!\top}
            \Lambda\,(\theta_p - \mu_p)

    where:

    * :math:`T_{pm} = (A_p\,\theta_p)_m = a_p + b_p x_m + c_p x_m^2`
      is the latent curve evaluated at centred age :math:`x_m`.
    * :math:`w_{pm} = z_{pm} - c_{pm} \cdot T_{pm}` is the residual
      (observed visit value minus scaled curve); :math:`c_{pm}` is the
      level-scale factor for visit-type group ``G[p,m]``.
    * :math:`\sigma_{pm}`, :math:`\mu_{pm}` are the Gaussian noise std and
      exponential mean for visit ``(p, m)`` from ``_group_noise``.
    * :math:`\Lambda` is the prior precision matrix (diagonal, with
      ``prior_prec[0,0] = 0`` so the intercept is unpenalised).
    * :math:`\mu_p` is the prior mean vector for patient ``p``.

    Gradient
    --------
    By the chain rule, :math:`\partial w_{pm}/\partial\theta_p =
    -c_{pm}\,a_{pm}` where :math:`a_{pm} = [1,\,x_m,\,x_m^2]^{\!\top}`:

    .. math::

        \nabla_{\theta_p}\mathcal{F}
          = -\!\sum_m c_{pm}\,a_{pm}\,\psi_1(w_{pm})
            + \Lambda\,(\theta_p - \mu_p)

    In batched einsum notation over patients ``p`` and slots ``m``::

        grad = -einsum("pmi,pm->pi", A, psi * c)
               + einsum("pi,...ij->pj", theta - prior_mean, prior_prec)

    Hessian
    -------
    Differentiating the gradient once more, using
    :math:`\partial\psi_1/\partial\theta_p =
    -c_{pm}\,a_{pm}\,\psi_2(w_{pm})`:

    .. math::

        H_{\theta_p}
          = \sum_m c_{pm}^2\,a_{pm}a_{pm}^{\!\top}\,\psi_2(w_{pm})
            + \Lambda

    In batched einsum notation::

        H = einsum("pmi,pm,pmj->pij", A, dpsi * c**2, A) + prior_prec

    Because :math:`\psi_2 \geq 0` (see ``_emg_psi``), ``H`` is positive
    semi-definite.  The nugget ``H[:,0,0] += 1e-9`` guards the intercept
    component, which has :math:`\Lambda_{00} = 0` and could otherwise be
    singular for patients with no observed visits.

    Newton step
    -----------
    The full-step Newton update is :math:`\theta \leftarrow \theta - H^{-1}g`.
    Steps are clipped to :math:`[-20, 20]` per component to prevent divergence
    on the first iterations when the initialisation is far from the optimum.

    Outer loop (``n_outer``)
    ------------------------
    The noise parameters :math:`(c, \sigma, \mu)` depend on :math:`T` through
    the power-law scaling ``_group_noise``.  The outer loop re-evaluates them
    at the current curve estimate after each block of ``n_newton`` inner
    Newton steps, acting as a *majorisation–minimisation* (MM) outer iteration.

    Parameters
    ----------
    A : np.ndarray, shape (P, Mv, 3)
        Design matrix.  ``A[p, m] = [1, x_m, x_m^2]`` for visit slot ``m``
        of patient ``p``; padded with zeros for unused slots.
    z : np.ndarray, shape (P, Mv)
        Observed visit values (padded with zeros for unused slots).
    mask : np.ndarray of bool, shape (P, Mv)
        ``True`` where ``(p, m)`` is a real observation.
    G : np.ndarray of int, shape (P, Mv)
        Visit-type group index (0–4) for each slot; selects the noise row.
    noise : np.ndarray, shape (5, 5)
        Noise parameters ``[log_c, s0, s1, m0, m1]`` per visit-type group,
        as returned by ``calibrate_bias_model``.
    theta : np.ndarray, shape (P, 3)
        Initial curve parameters; modified in-place and returned.
    prior_mean : np.ndarray, shape (P, 3) or broadcastable
        Prior mean :math:`\mu_p` for each patient.
    prior_prec : np.ndarray, shape (3, 3) or (P, 3, 3)
        Prior precision matrix :math:`\Lambda`.  Diagonal with
        ``[0, lam_slope, lam_curv]`` for the MLE estimator, or the full
        empirical-Bayes precision ``Sigma^{-1}`` for the MAP estimator.
    n_newton : int
        Number of Newton steps in each inner loop.
    n_outer : int
        Number of outer MM iterations (noise re-evaluation passes).
    update_prior : bool, default False
        If ``True``, after each outer iteration update the population prior
        mean for ``b`` and ``c`` to the median of the current estimates.
        Used for the MLE estimator to adaptively shrink toward the
        population trend.

    Returns
    -------
    theta : np.ndarray, shape (P, 3)
        Converged curve parameters for every patient.
    H : np.ndarray, shape (P, 3, 3)
        Final Hessian for every patient (equals the observed Fisher information
        at convergence).  Used downstream to compute the MAP posterior
        covariance ``cov = inv(H)`` and the posterior standard deviation
        ``map_sd``.
    """
    P, Mv, _ = A.shape
    H = np.broadcast_to(prior_prec, (P, 3, 3)).copy()
    for _ in range(n_outer):
        # ── Re-evaluate level-dependent noise at the current curve ───────────
        # c   : scale factor   (visit value = c * T + noise)
        # sig : Gaussian std   sigma_g(T) = exp(s0 + s1 * log T)
        # mu  : Exponential mean mu_g(T)  = exp(m0 + m1 * log T)
        Tcur = np.einsum("pmi,pi->pm", A, theta)
        c, sig, mu = (
            v.reshape(P, Mv) for v in _group_noise(noise, G.ravel(), Tcur.ravel())
        )
        for _ in range(n_newton):
            # ── Residuals: w = z - c * T(theta) ─────────────────────────────
            w = z - c * np.einsum("pmi,pi->pm", A, theta)

            # ── Per-visit psi1 = d(-log f)/dw,  psi2 = d2(-log f)/dw2 ──────
            psi, dpsi = _emg_psi(w, sig, mu)
            psi = np.where(mask, psi, 0.0)  # zero out padding slots
            dpsi = np.where(mask, dpsi, 0.0)

            # ── Gradient: g = -sum_m c_m * a_m * psi1_m  +  Lambda(theta - mu_p)
            # Chain rule: d w_m / d theta = -c_m * a_m  =>  scale psi by c
            grad = -np.einsum("pmi,pm->pi", A, psi * c) + np.einsum(
                "pi,...ij->pj", theta - prior_mean, prior_prec
            )

            # ── Hessian: H = sum_m c_m^2 * a_m a_m^T * psi2_m  +  Lambda ───
            # Second chain-rule application; c^2 because dw/dtheta = -c * a
            H = np.einsum("pmi,pm,pmj->pij", A, dpsi * c**2, A) + prior_prec
            # Nugget on the intercept: Lambda[0,0]=0 and a patient with no
            # data at all would leave H[p,0,0]=0, making solve() singular.
            H[:, 0, 0] += 1e-9

            # ── Newton step: theta <- theta - H^{-1} g ───────────────────────
            # Clip to [-20, 20] to prevent divergence far from the optimum.
            step = np.linalg.solve(H, grad[..., None])[..., 0]
            theta = theta - np.clip(step, -20.0, 20.0)

        # ── Adaptive population prior (MLE estimator only) ───────────────────
        # After converging the inner loop, update the shrinkage target for b
        # and c to the current median across patients with data.  This is a
        # one-step empirical-Bayes update; the intercept a is left at 0.
        if update_prior and mask.any():
            prior_mean = np.broadcast_to(
                np.r_[0.0, np.median(theta[mask.any(1), 1:], axis=0)], theta.shape
            ).copy()
    return theta, H


def calibrate_bias_model(raw: pd.DataFrame, y) -> dict:
    """Fit every target-based quantity of block 5 on TRAINING data only.

    Returns ``{"noise": (5 x 5) per-group (log c, s0, s1, m0, m1),
               "B": (10 x 3) prior-mean regression, "Sigma": (3 x 3)}``.
    """
    y = np.asarray(y, dtype=float)
    env = visit_envelope(raw)
    s, codes, slot, x = _sorted_view(raw)
    ys = pd.Series(y, index=raw.index).loc[s.index].to_numpy()
    V = env.loc[s.index, "vis_val"].to_numpy(dtype=float)
    g = visit_groups(s)

    # 1. EMG noise per visit type, by maximum likelihood against the true T
    noise = np.tile(_DEFAULT_NOISE, (_N_VISIT_GROUPS, 1))
    lt_all = np.log(np.clip(ys, 2.0, None))
    for k in range(_N_VISIT_GROUPS):
        m = (g == k) & np.isfinite(V) & np.isfinite(ys)
        if m.sum() < 200:
            continue
        v, t, lt = V[m], ys[m], lt_all[m]

        def nll(p, v=v, t=t, lt=lt):
            return -np.mean(
                _emg_logpdf(
                    v - np.exp(p[0]) * t,
                    np.exp(p[1] + p[2] * lt),
                    np.exp(p[3] + p[4] * lt),
                )
            )

        noise[k] = minimize(
            nll,
            _DEFAULT_NOISE.copy(),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-8},
        ).x

    # 2. empirical-Bayes prior on the true per-patient quadratic coefficients
    A, zt, mask, _ = _design(codes, slot, x, ys, g)
    M = np.einsum("pmi,pm,pmj->pij", A, mask.astype(float), A) + 1e-6 * np.eye(3)
    theta = np.linalg.solve(M, np.einsum("pmi,pm->pi", A, zt * mask)[..., None])[..., 0]
    Z = _patient_covariates(s, codes)
    enough = mask.sum(1) >= 4
    B = np.linalg.lstsq(Z[enough], theta[enough], rcond=None)[0]
    Sigma = np.cov((theta[enough] - Z[enough] @ B).T) + 1e-6 * np.eye(3)
    return {"noise": noise, "B": B, "Sigma": Sigma}


def bias_likelihood(
    raw: pd.DataFrame,
    calibration: dict,
    env: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Block 5: per-patient MLE and MAP curves under the EMG bias model.

    Columns: ``mle_*`` (independent priors) and ``map_*`` (empirical-Bayes
    prior), plus the posterior under-scoring of each visit ``map_Dhat`` and the
    bias-corrected visit value ``map_Vcorr``.
    """
    p = {**BIAS_PARAMS, **(params or {})}
    env = visit_envelope(raw) if env is None else env
    s, codes, slot, x = _sorted_view(raw)
    E = env.loc[s.index]
    V = E["vis_val"].to_numpy(dtype=float)
    g = visit_groups(s)
    A, z, mask, G = _design(codes, slot, x, V, g)
    noise = np.asarray(calibration["noise"], dtype=float)
    has = mask.any(1)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]

    def _iso(T):
        out = T.copy()
        for a, b in zip(starts[:-1], starts[1:]):
            ok = np.isfinite(T[a:b])
            if ok.any():
                seg = out[a:b]
                seg[ok] = isotonic_regression(T[a:b][ok], increasing=True)
                out[a:b] = seg
        return out

    out = {}
    # ── MLE with independent priors (start from the expectile-0.8 curve) ────
    T0 = np.zeros(z.shape)
    T0[codes, slot] = np.nan_to_num(E["env_e80_L"].to_numpy(dtype=float))
    M = np.einsum("pmi,pm,pmj->pij", A, mask.astype(float), A) + 1e-6 * np.eye(3)
    theta0 = np.linalg.solve(M, np.einsum("pmi,pm->pi", A, T0 * mask)[..., None])[
        ..., 0
    ]
    prior = np.r_[0.0, np.median(theta0[has, 1:], axis=0)] if has.any() else np.zeros(3)
    Lam = np.diag([0.0, p["lam_slope"], p["lam_curv"]])
    theta_mle, _ = _newton_emg(
        A,
        z,
        mask,
        G,
        noise,
        theta0,
        np.broadcast_to(prior, theta0.shape).copy(),
        Lam,
        p["n_newton"],
        p["n_outer"],
        update_prior=True,
    )
    T_mle = np.einsum("pmi,pi->pm", A, theta_mle)[codes, slot]
    T_mle[~has[codes]] = np.nan
    out["mle_T"] = T_mle
    out["mle_Tiso"] = _iso(T_mle)
    out["mle_slope"] = theta_mle[codes, 1] + 2 * theta_mle[codes, 2] * x

    # ── MAP with the empirical-Bayes prior ───────────────────────────────────
    Z = _patient_covariates(s, codes)
    mu_p = Z @ np.asarray(calibration["B"], dtype=float)
    prec = np.linalg.inv(
        np.asarray(calibration["Sigma"], dtype=float) * p["prior_scale"]
    )
    theta_map, H = _newton_emg(
        A, z, mask, G, noise, mu_p.copy(), mu_p, prec, p["n_newton"], p["n_outer"]
    )
    T_map = np.einsum("pmi,pi->pm", A, theta_map)[codes, slot]
    Av = A[codes, slot]
    cov = np.linalg.inv(H)
    c, sg, mm = _group_noise(noise, g, T_map)
    w = V - c * T_map
    mpost = -w - sg**2 / mm  # posterior of D | w: N(mpost, sg^2) on D >= 0
    Dhat = mpost + sg * _mills(mpost / sg)
    out.update(
        {
            "map_T": T_map,
            "map_Tiso": _iso(T_map),
            "map_sd": np.sqrt(
                np.maximum(np.einsum("vi,vij,vj->v", Av, cov[codes], Av), 0.0)
            ),
            "map_prior": np.einsum("vi,vi->v", Av, mu_p[codes]),
            "map_slope": theta_map[codes, 1] + 2 * theta_map[codes, 2] * x,
            "map_curv": theta_map[codes, 2],
            "map_w": w,
            "map_Dhat": Dhat,
            "map_Vcorr": (V + Dhat) / c,
            "map_vgroup": g.astype(float),
        }
    )
    return pd.DataFrame(out, index=s.index).loc[raw.index]


# ══════════════════════════════════════════════════════════════════════════════
# Part E — visit / patient features from 04-05 (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


def _poly_trend_col(vals: np.ndarray, ages: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial of ``degree`` fitted on non-NaN visits; fitted values at all ages.

    Falls back to the raw values with too few points or no age spread.
    """
    result = vals.copy().astype(float)
    mask = ~np.isnan(vals)
    if mask.sum() >= degree + 1 and ages[mask].max() - ages[mask].min() > 0:
        age_c = ages[mask] - ages[mask].mean()
        poly = np.polyfit(age_c, vals[mask], degree)
        result = np.polyval(poly, ages - ages[mask].mean())
    return result


def _loo_kernel(age: np.ndarray, z: np.ndarray, h: float) -> np.ndarray:
    """Leave-one-out Gaussian kernel smoother for one patient (bandwidth h years)."""
    ok = ~np.isnan(z)
    W = np.exp(-0.5 * ((age[:, None] - age[None, :]) / h) ** 2) * ok[None, :]
    np.fill_diagonal(W, 0.0)
    sw = W.sum(1)
    num = (W * np.nan_to_num(z)[None, :]).sum(1)
    return np.where(sw > 1e-9, num / np.where(sw > 1e-9, sw, 1.0), np.nan)


def _base_features(
    raw: pd.DataFrame,
    cat_dtypes: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """The 04-05 feature set, computed on the (patient_id, age)-sorted view."""
    order = raw.sort_values(["patient_id", "age"], kind="stable").index
    s = raw.loc[order].copy()
    g = s.groupby("patient_id", sort=False)

    feat = pd.DataFrame(index=s.index)

    # 1. Base numeric + categoricals
    for c in NUMERIC_COLS:
        feat[c] = s[c]
    for c in CAT_COLS:
        feat[c] = s[c].astype(cat_dtypes[c])

    # 2. Simple derived
    feat["years_dx"] = s["age"] - s["age_at_diagnosis"]
    feat["off_na"] = s["off"].isna().astype(np.int8)
    feat["on_na"] = s["on"].isna().astype(np.int8)
    feat["ledd_na"] = s["ledd"].isna().astype(np.int8)

    # 3. Visit order
    feat["vis_rank"] = g["age"].rank(method="first") - 1
    feat["since_first"] = s["age"] - g["age"].transform("min")
    feat["n_vis"] = g["age"].transform("size")

    # 4. Neighbour shifts ±1
    for c in ["off", "on"]:
        feat[f"{c}_prev"] = g[c].shift(1)
        feat[f"{c}_next"] = g[c].shift(-1)
    feat["dt_prev"] = s["age"] - g["age"].shift(1)
    feat["dt_next"] = g["age"].shift(-1) - s["age"]

    # 5. Cumulative (shifted to avoid same-row leakage)
    for c in ["off", "on", "ledd"]:
        exp_m = g[c].expanding().mean().shift(1)
        exp_m.index = exp_m.index.droplevel(0)
        feat[f"{c}_cmean"] = exp_m
        exp_x = g[c].expanding().max().shift(1)
        exp_x.index = exp_x.index.droplevel(0)
        feat[f"{c}_cmax"] = exp_x
    feat["off_dev_c"] = s["off"] - feat["off_cmean"]

    # 6. Whole-patient aggregates
    for c in ["off", "on", "ledd"]:
        feat[f"{c}_pmean"] = g[c].transform("mean")
        feat[f"{c}_pmax"] = g[c].transform("max")
    feat["off_dev"] = s["off"] - feat["off_pmean"]

    # 7. Per-patient polynomial trend (degree 1 and 2), ON and OFF separately
    pos_array = np.arange(len(s))
    for c in ["off", "on"]:
        fit1 = np.full(len(s), np.nan)
        fit2 = np.full(len(s), np.nan)
        for _, idx in g.groups.items():
            pos = pos_array[s.index.get_indexer(idx)]
            ages = s.loc[idx, "age"].values
            vals = s.loc[idx, c].values
            fit1[pos] = _poly_trend_col(vals, ages, 1)
            fit2[pos] = _poly_trend_col(vals, ages, 2)
        feat[f"{c}_fit1"] = fit1
        feat[f"{c}_fit2"] = fit2
        feat[f"{c}_res1"] = s[c].values - fit1
        feat[f"{c}_res2"] = s[c].values - fit2
    feat["gap1"] = feat["on_fit1"] - feat["off_fit1"]

    # 8. Neighbour shifts ±2
    for c in ["off", "on"]:
        feat[f"{c}_prev2"] = g[c].shift(2)
        feat[f"{c}_next2"] = g[c].shift(-2)

    # 9. Centred rolling means (window 3 and 5)
    for c in ["off", "on"]:
        for w, tag in [(3, "roll3"), (5, "roll5")]:
            rolled = s.groupby("patient_id")[c].apply(
                lambda x, w=w: x.rolling(w, center=True, min_periods=1).mean()
            )
            rolled.index = rolled.index.droplevel(0)
            feat[f"{c}_{tag}"] = rolled

    # 10. Differences and slope
    for c in ["off", "on"]:
        feat[f"{c}_dprev"] = s[c].values - feat[f"{c}_prev"].values
        feat[f"{c}_dnext"] = feat[f"{c}_next"].values - s[c].values
        denom = feat["dt_prev"].values + feat["dt_next"].values
        feat[f"{c}_slope_pn"] = np.where(
            denom > 0,
            (feat[f"{c}_next"].values - feat[f"{c}_prev"].values) / denom,
            np.nan,
        )

    # 11. Patient means of time_since_*
    for c in ["time_since_intake_off", "time_since_intake_on"]:
        feat[f"{c}_pmean"] = g[c].transform("mean")

    # 12. LEDD neighbours
    feat["ledd_prev"] = g["ledd"].shift(1)
    feat["ledd_next"] = g["ledd"].shift(-1)

    # 13. Cross / timing features
    feat["tsi_gap"] = (s["time_since_intake_off"] - s["time_since_intake_on"]).values
    feat["off_x_tsi"] = (s["off"] * s["time_since_intake_off"]).values
    feat["on_x_tsi"] = (s["on"] * s["time_since_intake_on"]).values

    # 14. LOO Gaussian kernel smoothing (ON and OFF separately)
    ages_arr = s["age"].values
    for c in ["off", "on"]:
        vals_arr = s[c].values
        for h, htag in [(0.5, "h05"), (1.0, "h10"), (2.0, "h20")]:
            ks = np.full(len(s), np.nan)
            for _, grp_idx in g.groups.items():
                pos = s.index.get_indexer(grp_idx)
                ks[pos] = _loo_kernel(ages_arr[pos], vals_arr[pos], h)
            feat[f"k_{c}_{htag}"] = ks

    return feat.loc[raw.index]


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════


def build_features(
    raw: pd.DataFrame,
    cat_dtypes: dict[str, pd.CategoricalDtype],
    latent_params: dict | None = None,
    envelope_params: dict | None = None,
    lme_params: dict | None = False,
    bias_calibration: dict | None = None,
    bias_params: dict | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix from raw X columns.

    Parameters
    ----------
    raw :
        Raw X. Must include ``patient_id`` (grouping only, never in the
        output) and all columns in ``NUMERIC_COLS`` and ``CAT_COLS``.
    cat_dtypes :
        Shared ``{col: CategoricalDtype}`` from ``make_cat_dtypes``.
    latent_params, envelope_params, lme_params :
        Overrides for ``LATENT_PARAMS`` / ``ENVELOPE_PARAMS`` / ``LME_PARAMS``.
        ``False`` disables that block. Defaults: blocks 2 and 3 on, the
        mixed-effects block 4 off (``lme_params=None`` turns it on).
    bias_calibration :
        Output of ``calibrate_bias_model`` fitted on TRAINING data. When given,
        adds block 5 (EMG likelihood: ``mle_*`` and ``map_*`` columns). When
        ``None`` (default) the output is exactly the 08 feature set.
    bias_params :
        Overrides for ``BIAS_PARAMS``.

    Returns
    -------
    pd.DataFrame aligned to ``raw.index``, without any column from
    ``FORBIDDEN``. The target is never read here (block 5 only uses the
    already-fitted calibration).
    """
    parts = [_base_features(raw, cat_dtypes)]
    if latent_params is not False:
        parts.append(latent_trajectory(raw, latent_params or None))
    env = None
    if envelope_params is not False or bias_calibration is not None:
        env = visit_envelope(raw, envelope_params or None)
        if envelope_params is not False:
            parts.append(env)
    if lme_params is not False:
        parts.append(lme_trajectory(raw, lme_params or None))
    if bias_calibration is not None:
        parts.append(bias_likelihood(raw, bias_calibration, env, bias_params))
    feat = pd.concat(parts, axis=1)

    leaked = FORBIDDEN & set(feat.columns)
    assert not leaked, f"Forbidden columns in output: {leaked}"
    assert feat.index.equals(raw.index), "Row alignment lost"
    return feat
