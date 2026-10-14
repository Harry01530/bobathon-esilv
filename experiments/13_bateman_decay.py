# %% [markdown]
# # 13_bateman_decay
#
# Modèle inspiré de la pharmacocinétique (équation de Bateman) appliqué à
# l'effet du traitement dopaminergique sur le ratio ON/OFF.
#
# Motivation
# ----------
# Dans les séries temporelles financières, un choc (entrée d'information) génère
# une réponse de type : montée rapide (absorption) puis décroissance exponentielle
# (dissipation). La pharmacocinétique décrit exactement ce phénomène avec
# l'équation de Bateman :
#
#     Réponse(t) = F · (e^{-k_e·t} − e^{-k_a·t}) / (k_a − k_e)
#
# Pour le médicament dopaminergique, l'effet sur la mesure UPDRS se traduit par
# une réduction du ratio ON/OFF (ON < OFF quand le médicament est actif) :
#
#     ON(t)/OFF(t) = 1 − E_p · (e^{-k_e·t} − e^{-k_a·t}) / (k_a − k_e)
#
# où :
#   · t      = time_since_intake_on  (heures)
#   · k_a    = constante d'absorption  (h⁻¹) — universelle ?
#   · k_e    = constante d'élimination (h⁻¹) — universelle ?
#   · E_p    = efficacité max du patient (amplitude, [0, 1]) — individuelle
#
# Normalisation : on divise par (k_a − k_e) pour que le pic de la fonction
# Bateman vaille 1 quand E_p = 1 (le ratio tomberait à 0 au pic pour un
# patient parfaitement répondeur).
#
# Objectif : tester si k_a et k_e sont suffisamment stables pour être
# considérés comme des constantes de population, et estimer E_p par patient.
#
# Données : visites avec ON, OFF ET time_since_intake_on connus (21 % du train).

# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Kaggle_officiel"

# ── Load data ─────────────────────────────────────────────────────────────────
X = pd.read_csv(DATA_DIR / "X_train.csv")
y_arr = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X["Index"]]
    .values
)
X["target"] = y_arr

# ── Filtre : visites avec ON, OFF et t_on connus ──────────────────────────────
mask = X["on"].notna() & X["off"].notna() & X["time_since_intake_on"].notna()
s = X[mask].copy()
s["ratio"] = (s["on"] / s["off"]).clip(0.0, 1.5)  # quelques outliers > 1
# Supprimer les lignes avec off=0 (division → inf/nan) ou ratio non fini
s = s[np.isfinite(s["ratio"]) & (s["off"] > 0)].copy()
t = s["time_since_intake_on"].to_numpy(dtype=float)
ratio = s["ratio"].to_numpy(dtype=float)

log.info(
    "Visites avec ON+OFF+t_on : %d (%.1f%% du train)",
    len(s),
    len(s) / len(X) * 100,
)
log.info(
    "t_on : min=%.2fh  median=%.2fh  max=%.2fh",
    t.min(),
    np.median(t),
    t.max(),
)
log.info(
    "ratio ON/OFF : p5=%.3f  median=%.3f  p95=%.3f",
    np.quantile(ratio, 0.05),
    np.median(ratio),
    np.quantile(ratio, 0.95),
)

# ══════════════════════════════════════════════════════════════════════════════
# Étape 1 — Modèle de population : k_a et k_e universels, E_p médian = 1
# ══════════════════════════════════════════════════════════════════════════════

def bateman_ratio(t, ka, ke, E):
    """Ratio ON/OFF prédit par le modèle de Bateman.

    Paramètres
    ----------
    t  : temps depuis la prise (h)
    ka : constante d'absorption (h⁻¹), ka > ke
    ke : constante d'élimination (h⁻¹)
    E  : efficacité max (amplitude de la réduction du ratio)

    Retourne : ON/OFF(t) = 1 − E · (e^{-ke·t} − e^{-ka·t}) / (ka − ke)
    """
    # Garde numérique : si ka ≈ ke, la limite vaut t·exp(−ke·t)
    dka = ka - ke
    if abs(dka) < 1e-6:
        response = t * np.exp(-ke * t)
        # normaliser pour que le pic vaille 1
        t_peak = 1.0 / ke
        peak = t_peak * np.exp(-ke * t_peak)
        response = response / (peak + 1e-12)
    else:
        response = (np.exp(-ke * t) - np.exp(-ka * t)) / dka
        # normaliser : le max de Bateman est à t* = ln(ka/ke)/(ka-ke)
        t_star = np.log(ka / ke) / dka
        peak = (np.exp(-ke * t_star) - np.exp(-ka * t_star)) / dka
        response = response / (peak + 1e-12)
    return 1.0 - E * response


log.info("")
log.info("── Étape 1 : ajustement population (k_a, k_e, E) ──────────────────")

# Ajustement global sur toutes les visites (E = efficacité moyenne population)
p0 = [3.0, 0.5, 0.4]   # ka=3h⁻¹, ke=0.5h⁻¹, E=0.4 (réduction 40% au pic)
bounds = ([0.1, 0.01, 0.0], [50.0, 5.0, 1.0])

try:
    popt, pcov = curve_fit(
        bateman_ratio, t, ratio,
        p0=p0, bounds=bounds,
        maxfev=10_000,
        method="trf",
    )
    ka_pop, ke_pop, E_pop = popt
    perr = np.sqrt(np.diag(pcov))
    residuals_pop = ratio - bateman_ratio(t, *popt)
    rmse_pop = np.sqrt(np.mean(residuals_pop ** 2))
    log.info(
        "Population fit : ka=%.3f h⁻¹ (±%.3f)  ke=%.3f h⁻¹ (±%.3f)  "
        "E=%.3f (±%.3f)",
        ka_pop, perr[0], ke_pop, perr[1], E_pop, perr[2],
    )
    t_peak_pop = np.log(ka_pop / ke_pop) / (ka_pop - ke_pop)
    log.info(
        "Pic de l'effet à t*=%.2fh  (demi-vie élim.=%.2fh  demi-vie abs.=%.2fh)",
        t_peak_pop, np.log(2) / ke_pop, np.log(2) / ka_pop,
    )
    log.info("RMSE population fit : %.4f", rmse_pop)
except RuntimeError as e:
    log.error("curve_fit échoué : %s", e)
    ka_pop, ke_pop, E_pop = p0
    rmse_pop = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# Étape 2 — Modèle mixte : k_a, k_e fixés à la population, E_p par patient
# ══════════════════════════════════════════════════════════════════════════════

log.info("")
log.info("── Étape 2 : E_p par patient (k_a=%.3f, k_e=%.3f fixés) ──────────", ka_pop, ke_pop)

# Pour chaque patient avec >= 2 visites both, on estime E_p par OLS scalaire
patient_ids = s["patient_id"].to_numpy()

# La base Bateman évaluée aux temps de chaque visite (vecteur de 1 feature)
response_vec = bateman_ratio(t, ka_pop, ke_pop, 1.0)  # avec E=1
basis = 1.0 - response_vec   # = E * (ce que Bateman normalise)

# E_p = (1 - ratio) / basis  pour chaque visite, puis moyenne par patient
with np.errstate(divide="ignore", invalid="ignore"):
    ep_visit = np.where(np.abs(basis) > 0.01, (1.0 - ratio) / basis, np.nan)

df_ep = pd.DataFrame({"patient_id": patient_ids, "ep_visit": ep_visit})
patient_ep = (
    df_ep.groupby("patient_id")["ep_visit"]
    .agg(["median", "std", "count"])
    .rename(columns={"median": "E_p", "std": "E_p_std", "count": "n_visits"})
)
patient_ep = patient_ep[patient_ep["n_visits"] >= 2].copy()

log.info(
    "Patients avec E_p estimé (n>=2) : %d  E_p médian=%.3f  std inter-patient=%.3f",
    len(patient_ep),
    patient_ep["E_p"].median(),
    patient_ep["E_p"].std(),
)
log.info(
    "E_p : p5=%.3f  p25=%.3f  median=%.3f  p75=%.3f  p95=%.3f",
    *patient_ep["E_p"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values,
)

# ══════════════════════════════════════════════════════════════════════════════
# Étape 3 — Universalité de k_a et k_e : ajustement par patient
# ══════════════════════════════════════════════════════════════════════════════

log.info("")
log.info("── Étape 3 : k_a et k_e per-patient (patients avec >=5 visites) ───")

results_per_patient = []
patients_with_enough = s.groupby("patient_id").filter(lambda g: len(g) >= 5)

for pid, grp in patients_with_enough.groupby("patient_id"):
    t_p = grp["time_since_intake_on"].to_numpy(dtype=float)
    r_p = grp["ratio"].to_numpy(dtype=float)
    if len(np.unique(t_p)) < 3:  # pas assez de diversité temporelle
        continue
    try:
        pp, _ = curve_fit(
            bateman_ratio, t_p, r_p,
            p0=[ka_pop, ke_pop, E_pop],
            bounds=bounds,
            maxfev=2000,
        )
        resid = r_p - bateman_ratio(t_p, *pp)
        results_per_patient.append({
            "patient_id": pid,
            "ka": pp[0], "ke": pp[1], "E_p": pp[2],
            "rmse": np.sqrt(np.mean(resid ** 2)),
            "n": len(t_p),
        })
    except RuntimeError:
        pass

df_pp = pd.DataFrame(results_per_patient)
log.info("Patients fittés individuellement : %d", len(df_pp))

if len(df_pp) > 0:
    log.info(
        "k_a per-patient : median=%.3f  std=%.3f  p5=%.3f  p95=%.3f",
        df_pp["ka"].median(), df_pp["ka"].std(),
        df_pp["ka"].quantile(0.05), df_pp["ka"].quantile(0.95),
    )
    log.info(
        "k_e per-patient : median=%.3f  std=%.3f  p5=%.3f  p95=%.3f",
        df_pp["ke"].median(), df_pp["ke"].std(),
        df_pp["ke"].quantile(0.05), df_pp["ke"].quantile(0.95),
    )
    log.info(
        "E_p per-patient : median=%.3f  std=%.3f  p5=%.3f  p95=%.3f",
        df_pp["E_p"].median(), df_pp["E_p"].std(),
        df_pp["E_p"].quantile(0.05), df_pp["E_p"].quantile(0.95),
    )
    log.info("RMSE médian per-patient fit : %.4f", df_pp["rmse"].median())

    # Coefficient de variation : si CV < 50% => paramètre "universel"
    for param in ["ka", "ke"]:
        cv = df_pp[param].std() / df_pp[param].median()
        verdict = "UNIVERSEL (CV<50%)" if cv < 0.5 else "HÉTÉROGÈNE (CV>=50%)"
        log.info("  CV(%s) = %.1f%%  → %s", param, cv * 100, verdict)

# ══════════════════════════════════════════════════════════════════════════════
# Étape 4 — E_p prédit les mesures ON manquantes ?
# ══════════════════════════════════════════════════════════════════════════════

log.info("")
log.info("── Étape 4 : E_p corrèle-t-il avec le target y ? ──────────────────")

# Joindre E_p à y (moyenne des targets du patient)
patient_y = X.groupby("patient_id")["target"].median().rename("y_median")
joined = patient_ep.join(patient_y, how="inner")

if len(joined) > 10:
    r_ep_y, p_ep_y = pearsonr(joined["E_p"], joined["y_median"])
    log.info(
        "Corrélation(E_p, y_median_patient) = %.3f  (p=%.3e, n=%d)",
        r_ep_y, p_ep_y, len(joined),
    )
    verdict = (
        "E_p est un prédicteur UTILE du niveau UPDRS"
        if abs(r_ep_y) > 0.15
        else "E_p n'est pas corrélé au niveau UPDRS (confirmation : biais visit-level)"
    )
    log.info("→ %s", verdict)

# Corrélation de E_p avec la pente per-patient (progression de la maladie)
if len(df_pp) > 10:
    patient_slope = (
        X.groupby("patient_id")
        .apply(lambda g: np.polyfit(g["age"], g["target"], 1)[0] if len(g) >= 3 else np.nan)
        .dropna()
        .rename("slope")
    )
    joined2 = df_pp.set_index("patient_id").join(patient_slope, how="inner")
    if len(joined2) > 10:
        r_ep_sl, p_ep_sl = pearsonr(joined2["E_p"], joined2["slope"])
        log.info(
            "Corrélation(E_p, pente UPDRS/an) = %.3f  (p=%.3e, n=%d)",
            r_ep_sl, p_ep_sl, len(joined2),
        )

# ══════════════════════════════════════════════════════════════════════════════
# Étape 5 — Courbe Bateman moyennée par bin de t_on
# ══════════════════════════════════════════════════════════════════════════════

log.info("")
log.info("── Étape 5 : courbe Bateman vs données binned ──────────────────────")

t_grid = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
pred_curve = bateman_ratio(t_grid, ka_pop, ke_pop, E_pop)

# Statistiques observées par bin
bins_obs = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0]
s["t_bin"] = pd.cut(t, bins=bins_obs)
binned = s.groupby("t_bin", observed=False)["ratio"].agg(["median", "count", "std"])
binned["t_mid"] = [(a.left + a.right) / 2 for a in binned.index]
binned["pred"] = bateman_ratio(binned["t_mid"].to_numpy(), ka_pop, ke_pop, E_pop)
binned["resid"] = binned["median"] - binned["pred"]

log.info("")
log.info(
    "  %-15s  %6s  %6s  %6s  %6s  %6s",
    "bin t_on (h)", "n", "obs", "pred", "resid", "σ_obs",
)
for idx, row in binned.iterrows():
    log.info(
        "  %-15s  %6d  %6.3f  %6.3f  %+6.3f  %6.3f",
        str(idx), int(row["count"]), row["median"],
        row["pred"], row["resid"], row["std"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# Résumé
# ══════════════════════════════════════════════════════════════════════════════

log.info("")
log.info("=" * 66)
log.info("RÉSUMÉ — Modèle de Bateman sur le ratio ON/OFF")
log.info("=" * 66)
log.info("Constantes de population :")
log.info("  k_a = %.3f h⁻¹  (absorption,  t½ = %.1f min)", ka_pop, np.log(2)/ka_pop*60)
log.info("  k_e = %.3f h⁻¹  (élimination, t½ = %.1f min)", ke_pop, np.log(2)/ke_pop*60)
log.info("  E   = %.3f        (efficacité pop.  = réduction max du ratio)", E_pop)
log.info("  t*  = %.2f h      (pic de l'effet médicament)", t_peak_pop)
log.info("  RMSE population fit = %.4f (sur ratio ON/OFF ∈ [0,1])", rmse_pop)
if len(df_pp) > 0:
    log.info(
        "Universalité : CV(k_a)=%.0f%%  CV(k_e)=%.0f%%",
        df_pp["ka"].std() / df_pp["ka"].median() * 100,
        df_pp["ke"].std() / df_pp["ke"].median() * 100,
    )
log.info("=" * 66)
log.info("Script terminé — aucun upload Hub (analyse exploratoire).")
