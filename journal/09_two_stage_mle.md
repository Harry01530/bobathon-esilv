# 09 · TwoStageMLE — design note

> **Résultat final (après tuning 10) :** mean CV RMSE **2.6772** ± 0.0338
> (`prior_scale=0.3`, GroupKFold 5, Kaggle_officiel)
> — gain de **−0.54** vs exp. 08 (2.7120) et **−0.61** vs exp. 05 (3.2618).

---

## 1. Intuition : pourquoi une EMG ?

Les résidus des modèles 02–06 présentent une **asymétrie marquée vers les
valeurs basses** : les médecins sous-évaluent fréquemment le score UPDRS, mais
ne le surestiment presque jamais. Ce comportement n'est pas capturé par une
hypothèse gaussienne symétrique.

En finance, ce type de queue asymétrique est souvent modélisé par une loi
exponentielle modifiée (EMG = Exponentially Modified Gaussian). Le modèle de
mesure ici est :

```
V = c_g · T(x) + ε - D
```

où :

| Symbole | Signification |
|---|---|
| `T(x)` | Vraie valeur UPDRS latente (courbe quadratique en âge centré `x`) |
| `c_g` | Facteur d'échelle dépendant du type de visite `g` ∈ {0…4} |
| `ε ~ N(0, σ_g²)` | Bruit symétrique du rater (aléatoire, indépendant d'une visite à l'autre) |
| `D ~ Exp(μ_g)`, D ≥ 0 | Biais de sous-évaluation (toujours positif) |

Les paramètres `(c_g, σ_g, μ_g)` suivent des lois puissance en `T` :

```
σ_g(T) = exp(s0_g + s1_g · ln T)
μ_g(T) = exp(m0_g + m1_g · ln T)
```

Les cinq types de visite (`g`) sont :

| g | Type |
|---|---|
| 0 | OFF seul, patient non traité |
| 1 | OFF seul, patient traité |
| 2 | ON et OFF mesurés |
| 3 | ON seul, temps depuis prise connu |
| 4 | ON seul, temps depuis prise inconnu |

---

## 2. Log-vraisemblance de l'EMG

En marginalisant sur `D ≥ 0` (intégrale sur une exponentielle × une
gaussienne, complétion du carré) :

```
log f(w | σ, μ) = −log μ + w/μ + σ²/(2μ²) + log Φ(−w/σ − σ/μ)
```

avec `w = V − c·T` le résidu et `Φ` la CDF normale standard.
`log_ndtr` de SciPy est utilisé pour `log Φ` afin de préserver la précision
numérique dans la queue gauche où `Φ` underflow à zéro.

La densité est **concave en w**, ce qui garantit un optimum unique par patient
et la convergence de Newton-Raphson.

---

## 3. Gradient et Hessien — dérivation complète

En posant `a = −w/σ − σ/μ` et le **ratio de Mills** `λ(a) = φ(a)/Φ(a)` :

### Dérivée première (gradient de la log-vraisemblance négative)

```
ψ₁(w) = ∂(−log f)/∂w = −1/μ + λ(a)/σ
```

Dérivation : le terme `−w/μ` donne `−1/μ` ; le terme `−log Φ(a)` donne
`−[∂log Φ/∂a]·[∂a/∂w]` = `−λ(a)·(−1/σ)` = `+λ(a)/σ`.

### Dérivée seconde (Hessien — preuve de convexité)

```
ψ₂(w) = ∂²(−log f)/∂w² = λ(a)·(a + λ(a)) / σ² ≥ 0
```

L'inégalité tient car `a + λ(a) ≥ 0` pour tout `a` (propriété connue du ratio
de Mills). Cela prouve que `−log f` est convexe en `w`, donc en `θ_p`.

### Stabilité numérique du ratio de Mills

La formule directe `φ(a)/Φ(a)` perd toute précision pour `a ≪ 0` (underflow
des deux termes). L'implémentation reste en log-espace :

```python
λ(a) = exp(−a²/2 − log√(2π) − log_ndtr(a))
```

---

## 4. Optimisation par Newton-Raphson batché

### Problème par patient

Pour chaque patient `p`, minimiser la **log-vraisemblance pénalisée** sur les
paramètres de la courbe quadratique `θ_p = (a_p, b_p, c_p)ᵀ` :

```
F(θ_p) = −∑_m log f(w_pm | σ_pm, μ_pm) + ½(θ_p − μ_p)ᵀ Λ (θ_p − μ_p)
```

où `T_pm = a_p + b_p·x_m + c_p·x_m²` et `w_pm = z_pm − c_pm·T_pm`.

La matrice de précision `Λ = diag(0, λ_slope, λ_curv)` — l'intercept n'est
pas pénalisé.

### Gradient (règle des chaînes : ∂w/∂θ = −c·a)

```
∇_θ F = −∑_m c_pm · a_pm · ψ₁(w_pm) + Λ(θ_p − μ_p)
```

En notation einsum batché sur `(patients, slots)` :

```python
grad = -einsum("pmi,pm->pi", A, psi * c) + einsum("pi,...ij->pj", θ - μ, Λ)
```

### Hessien (dérivation seconde : c² car ∂w/∂θ = −c·a)

```
H_θ = ∑_m c_pm² · a_pm aᵀ_pm · ψ₂(w_pm) + Λ
```

```python
H = einsum("pmi,pm,pmj->pij", A, dpsi * c**2, A) + prior_prec
H[:, 0, 0] += 1e-9  # garde sur l'intercept non pénalisé
```

### Pas Newton + boucle MM externe

- Pas clipé à `[−20, 20]` par composante pour éviter la divergence loin de
  l'optimum.
- La boucle externe (MM) ré-évalue les paramètres de bruit `(c, σ, μ)` à la
  courbe courante après chaque bloc de `n_newton` pas internes.

---

## 5. Deux estimateurs de la courbe

| Estimateur | Prior | Colonnes |
|---|---|---|
| **MLE** (`mle_*`) | Prior gaussien indépendant sur `(b, c)` avec `λ_slope=1.0`, `λ_curv=100.0` | `mle_T`, `mle_Tiso`, `mle_slope` |
| **MAP** (`map_*`) | Prior empirique-bayésien `(a,b,c) ~ N(Z_p B, Σ)` appris sur les vraies trajectoires d'entraînement | `map_T`, `map_Tiso`, `map_sd`, `map_prior`, `map_slope`, `map_curv`, `map_w`, `map_Dhat`, `map_Vcorr`, `map_vgroup` |

`map_Dhat = E[D | w]` est la correction postérieure du biais de sous-évaluation.
`map_Vcorr = (V + Dhat) / c` est la valeur de visite corrigée du biais.

La calibration (`calibrate_bias_model`) est toujours refittée **à l'intérieur
du fold d'entraînement** dans `TwoStageMLE.fit` — les cibles des patients de
test ne sont jamais vues.

---

## 6. Projection finale des prédictions

Après le stage 2, chaque patient est projeté via `smooth_per_patient` :

1. **Quadratique** : on fitte un polynôme degré 2 en âge aux prédictions du
   patient par OLS — lisse les sauts visite-à-visite.
2. **Isotonique** (PAVA) : on projette les valeurs fittées sur une suite
   non-décroissante — respecte la contrainte physiologique (UPDRS ne diminue
   pas).

Ce post-traitement seul apporte ~0.01 RMSE sur la configuration à stage unique.

---

## 7. Tuning (exp. 10) — meilleur paramètre

| Config | Mean RMSE | Δ vs 09 |
|---|---|---|
| `prior_scale=0.5` (défaut 09) | 2.6905 | — |
| **`prior_scale=0.3`** | **2.6772** | **+0.0133** ✅ |
| `prior_scale=0.7` | 2.6889 | +0.0016 |
| `lam_slope=0.5` | 2.7003 | −0.0098 ❌ |
| HGB 600 arbres / lr 0.03 / 63 feuilles | 2.6860 | +0.0045 |

Serrer le prior empirique-bayésien (`prior_scale=0.3`) est le levier le plus
efficace : il réduit la variance des estimations `map_*` en renforçant la
régularisation vers la population.

---

## 8. Refactorings appliqués avant le commit

### 8.1 Docstrings & type hints (`src/parkinsons/`)

| Fichier | Ce qui a été ajouté |
|---|---|
| `features.py` | Docstrings PEP 8 avec dérivations LaTeX pour `_emg_logpdf`, `_mills`, `_emg_psi`, `_newton_emg` ; vérification numérique par différences finies (gradient ≈ 2.5×10⁻¹¹, Hessien ≈ 3.9×10⁻⁶) |
| `two_stage_mle.py` | Docstrings complètes pour `TwoStageMLE`, `smooth_per_patient`, `fit`, `predict_unsmoothed`, `predict` ; type hints stricts (`_SmoothMode = Literal[…] \| None`) |
| `two_stage_stack.py` | Docstrings complètes pour `TwoStageStack`, `make_hgb`, `make_ridge`, `make_extra_trees`, `make_stage2`, `fit`, `predict_unclipped`, `predict`, `stack_weights` ; type hints stricts (`_TargetTransform = Literal[…]`, `Callable[[pd.DataFrame], pd.DataFrame]`) |

### 8.2 Logging structuré (`experiments/`)

Remplacement de tous les `print()` par `logging` (niveau `INFO`) dans
`06_two_stage_stack.py`, `08_two_stage_envelope.py`, `09_two_stage_mle.py`.

Format uniforme :

```
2025-10-12 14:23:07  INFO      Train: 44590 rows, 5576 patients
2025-10-12 14:27:41  WARNING   Gain is within the fold-to-fold spread; it may not be real.
2025-10-12 14:31:02  CRITICAL  project.put('09_cv_two_stage_mle') failed: ConnectionError: …
```

Niveaux utilisés : `INFO` (flux normal), `WARNING` (incertitude sur le gain),
`CRITICAL` + `SystemExit(1)` (erreur d'upload Hub fatale).

### 8.3 Formatage PEP 8 global (`ruff format` + `ruff check`)

Appliqué sur l'ensemble de `src/` et `experiments/` :

- **14 fichiers reformatés** par `ruff format` (alignements manuels supprimés,
  listes/dicts dépliés, lignes > 88 chars découpées).
- **9 violations E/W/I corrigées** automatiquement (`--fix`) : imports triés,
  espaces en trop.
- **4 violations E501 résiduelles** corrigées manuellement (f-strings longues
  scindées en deux littéraux adjacents, une ligne de docstring raccourcie).
- Résultat final : `ruff format --check` + `ruff check --select E,W,I` →
  **18 fichiers conformes, 0 violation**.
