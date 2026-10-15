# Rapport final — Bobathon ESILV
# Équipe : Julien Gouel

---

## 1. Problème

La compétition Kaggle [bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)
demande de prédire, pour chaque visite médicale, le score moteur MDS-UPDRS III
(« true OFF ») non biaisé d'un patient atteint de la maladie de Parkinson.
La variable cible est continue, entre 0 et 132 ; la métrique est la RMSE.

Les données d'entraînement contiennent 44 590 visites pour 5 576 patients
(~8 visites par patient). Colonnes disponibles : `Index`, `patient_id`,
`cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`,
`time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
Trois colonnes présentent des valeurs manquantes structurelles importantes
(`time_since_intake_off` 79 %, `ledd` 37 %, `time_since_intake_on` 46 %).

Le score public Kaggle est calculé sur un sous-ensemble des 11 013 visites de test.

---

## 2. Chronologie des expériences

| # | Nom | CV RMSE moyen | Écart-type | RMSE Kaggle public | Rapport Skore |
|---|-----|:---:|:---:|:---:|---|
| 01 | `01_dummy` — DummyRegressor(mean), holdout 0.2 aléatoire | 16.477937 | non relevé | 16.421 | [Est. 49769](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/49769) |
| 02 | `02_day2_models` — hgbr_mixed, GroupKFold(5) officiel | 7.4270 | 0.1192 | 7.142 | [CV 50335](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50335) · [Est. 50341](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50341) |
| 04 | `04_hgb_rich_patient_feats` — HGB + features temporelles riches | 3.4831 | 0.0167 | non relevé | [CV 04_cv_hgb_rich](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60938) |
| 05 | `05_two_stage` — TwoStageHGB (OOF preds stage 2) | 3.2618 | 0.0255 | non relevé | [CV 60514](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60514) · [Est. 60520](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60520) |
| 06 | `06_two_stage_stack` — log1p + stack HGB+Ridge+ET | 3.2583 | non relevé | non relevé | [CV 60813](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60813) · [Est. 60819](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60819) |
| 08 | `08_two_stage_envelope` — TwoStageHGB + features enveloppe (bloc 3) | 2.7129 | 0.0270 | non relevé | [CV 60938](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/60938) · [Est. 60944](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/60944) |
| 09 | `09_two_stage_mle` — TwoStageMLE (EMG bloc 5, prior_scale=0.5) | 2.6905 | 0.0449 | non relevé | [CV 61428](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/61428) · [Est. 61434](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/61434) |
| 10 | `10_mle_tuning` — TwoStageMLE prior_scale=0.3 (meilleur tuning) | 2.6772 | 0.0338 | non relevé | [CV 61684](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/61684) · [Est. 61690](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/61690) |
| 11 | `11_target_vcorr` — stage 1 sur map_Vcorr, stage 2 sur résidu | 2.8127 | 0.0445 | non relevé | non relevé |
| 12 | `12_voting_ensemble` — VotingRegressor(model_a, model_b) | 2.6784 | 0.0297 | **2.708** | [CV 83280](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/83280) · [Est. 82849](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/82849) |
| 13 | `13_hybrid_gru` — HybridModel HGB + BiGRU bidirectionnel | **2.5598** | **0.0117** | **2.642** | [CV 85779](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/85779) · [Est. 85398](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/85398) |

Note : les expériences 03 et 07 sont présentes dans le dépôt mais leur RMSE de
validation n'a pas été consigné dans les notes disponibles.

---

## 3. Ce qui a fonctionné

### Features temporelles par patient (exps. 04-08)
Trier les visites par `(patient_id, age)` et construire des features de
voisinage (décalages, moyennes cumulatives, tendances polynomiales, maximum
courant) réduit la RMSE de 7.43 à 3.26, soit −56 %. La vraie trajectoire UPDRS
est monotone et quadratique en âge ; les modèles arborescents en profitent dès
qu'on leur expose cette structure.

### Enveloppe supérieure des visites (exp. 08 → gain −0.55)
Le biais de mesure est asymétrique : les médecins sous-évaluent plus souvent
qu'ils ne surestiment. La vraie valeur latente se trouve donc près de
l'**enveloppe supérieure** des mesures brutes, pas de leur moyenne. Construire
une enveloppe par patient via régression par expectiles (tau=0.5/0.8/0.9)
incorpore cette contrainte directement dans les features.

### Modèle de biais EMG et prior empirique-bayésien (exps. 09-10 → gain −0.035)
Modéliser explicitement la distribution asymétrique du biais (loi exponentielle
modifiée gaussienne : ε − D, D ≥ 0) permet d'estimer la vraie trajectoire
latente par maximum de vraisemblance par patient. La calibration (paramètres de
bruit + prior MAP sur (niveau, pente, courbure)) est refittée sur les lignes
d'entraînement de chaque pli. Resserrer le prior (`prior_scale=0.3` au lieu de
0.5) donne le meilleur résultat en CV à 2.6772.

### BiGRU sur les séquences de visites (exp. 13 → gain −0.12 vs 12)
La trajectoire d'un patient est une série temporelle irrégulière. Le GRU
bidirectionnel à 2 couches (hidden=64, dropout=0.1, 60 epochs) lit les visites
triées par âge et prédit un résidu par rapport aux prédictions HGB. La
combinaison `0,5 × HGB + 0,5 × GRU` suivie de `smooth_per_patient("quad_iso")`
donne la meilleure RMSE CV à **2.5598** et le meilleur score Kaggle public à
**2.642**, soit une amélioration de +2.8 % sur l'exp. 12.

### Projection monotone finale (exps. 09-13)
`smooth_per_patient(X, pred, "quad_iso")` : fit quadratique en âge puis
régression isotonique (PAVA) par patient. Exploite la contrainte physiologique
(le score UPDRS ne diminue jamais pour un patient donné). Apporte environ
0.01 RMSE de gain en isolation.

---

## 4. Ce qui n'a pas fonctionné

**Stacking HGB+Ridge+ExtraTrees avec log1p (exp. 06) :** gain de seulement
−0.0045 de RMSE, inférieur à la variabilité inter-plis (~0.025). Le stage 2 du
modèle à deux étapes est déjà quasi-optimal à ce niveau ; l'erreur résiduelle
vient de l'estimation de la trajectoire latente, pas du régresseur final.

**Stage 1 entraîné sur map_Vcorr (exp. 11, RMSE 2.8127 ± 0.0445) :** régresser
d'abord sur la valeur de visite corrigée du biais (V + D̂)/c (estimée via EMG)
puis apprendre le résidu y−ŷ₁ en stage 2 **dégrade** le score de −0.135 vs
exp. 10. Hypothèse : map_Vcorr contient des valeurs aberrantes pour les patients
au prior mal conditionné, et le HGB stage 1 overfit ces outliers.

**Ensemble vote simple (exp. 12, RMSE CV 2.6784 ± 0.0297) :** légèrement
inférieur au meilleur modèle seul (2.6772 de exp. 10) en CV, mais soumet à
Kaggle à 2.708. Néanmoins utile comme rapport unifié pour le jury.

---

## 5. Étanchéité de l'évaluation — absence de fuite de données

### GroupKFold sur patient_id
L'évaluation utilise `GroupKFold(n_splits=5, groups=patient_id)`. Les cinq plis
sont précalculés une seule fois avant l'évaluation et réutilisés identiques pour
toutes les expériences. Un patient n'apparaît jamais à la fois dans le pli
d'entraînement et dans le pli de test. Vérifié par assertion dans chaque script.

### Calibration et statistiques de standardisation ajustées sur les lignes
d'entraînement uniquement
La fonction `calibrate_bias_model(X_train_fold, y_train_fold)` est appelée
dans `TwoStageMLE.fit` sur les seules lignes d'entraînement du pli courant.
Elle n'a jamais accès aux cibles des patients du pli de test. De même, les
statistiques de z-score du GRU (moyenne et écart-type des colonnes
longitudinales) sont calculées dans `HybridModel.fit` sur les lignes reçues.

### OOF HGB pour le GRU
Les prédictions HGB utilisées comme entrée du GRU sont obtenues par
validation croisée interne (`GroupKFold(5)` imbriqué dans `fit`), de sorte que
le GRU ne voit jamais les vraies cibles via les prédictions HGB.

### Pas de prétraitement global
Aucune normalisation, aucune calibration, aucune statistique issue de
l'ensemble complet d'entraînement n'est injectée dans un pli de test.

---

## 6. Conformité aux règles et provenance honnête

### Règles du hackathon
- L'évaluation passe par **skore** (`skore.evaluate`, `project.put`).
- Chaque soumission Kaggle contient l'URL du rapport `EstimatorReport` Skore
  Hub correspondant dans sa description.
- Les CSV de soumission sont générés dans `submissions/` par les scripts eux-
  mêmes ; aucun CSV externe n'a été utilisé.

### Provenance du code de l'expérience 12
Le code de `TwoStageMLE` et des features a été extrait du commit
[a608a6f86a](https://github.com/Harry01530/bobathon-esilv/commit/a608a6f86a0f7886f46982bf19535fcfb8b99cd6)
du dépôt `Harry01530/bobathon-esilv` (Stein). Un correctif MRO a été appliqué
par Bob dans le terminal intégré (`RegressorMixin` avant `BaseEstimator` pour
sklearn 1.9). Le script `12_voting_ensemble.py` a été exécuté dans le terminal
Bob.

### Provenance du code de l'expérience 13
- Le module séquentiel (`seq_gru.py`) de l'expérience 13 a été intégré à partir d'un fichier préparé en amont, puis adapté, exécuté et évalué dans Bob (CV, ré-entraînement, envois skore Hub). Détail de la provenance : `journal/12_13_provenance_bob.md`.

### Ce que Bob n'a pas fait
Bob n'a pas inventé de chiffre : tous les RMSE, SHA256 et URLs ci-dessus sont
copiés tels quels depuis `12_voting_ensemble_meta.json`,
`13_hybrid_gru_meta.json` et `journal/12_13_provenance_bob.md`.

---

## 7. Équipe

Julien Gouel, Othmane Hassani, Stein Lawson et Yassine. Le travail a été mené en équipe sur un dépôt commun : solution Day 2 (Othmane), variables par patient et deux étages (Julien), modèle TwoStageMLE et réglages (Stein), ensemble final et hybride (équipe). Historique détaillé : git log et journal/.

---

## 8. Reproduire les résultats

### Prérequis
1. Cloner le dépôt et suivre [docs/SETUP.md](../docs/SETUP.md).
2. Télécharger les données officielles depuis [Kaggle — Data](https://www.kaggle.com/competitions/bobathon-esilv/data)
   et les placer dans `data/Kaggle_officiel/` :
   `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`.
3. Installer PyTorch CPU : `.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Expériences 01-11
```bash
.venv/Scripts/python experiments/01_dummy.py
.venv/Scripts/python experiments/02_day2_models.py
# … (04, 05, 06, 08, 09, 10, 11)
```

### Expérience 12
Le script de référence est `experiments/12_voting_ensemble.py` (non conservé
dans le dépôt principal ; sa provenance est documentée dans
`journal/12_13_provenance_bob.md`).
Le CSV final est `submissions/12_voting_ensemble.csv`
(SHA-256 : `a622204e5d732f54732153094ba38042813074a265a52314970a391a4a6c8c06`).

### Expérience 13
Le script est `experiments/13_hybrid_gru.py`. Le package isolé utilisé lors
de l'exécution réelle est dans `scratch/test13_bob/parkinsons13/`. Pour
relancer depuis zéro (environ 15-20 min sur CPU) :
```bash
.venv/Scripts/python experiments/13_hybrid_gru.py
```
Le CSV final est `submissions/13_fold5_hybrid_gru.csv`
(SHA-256 : `f82943a73f76053bca34e397f56e42663c1617300efcfe57a1221bb0ed966dc7`).

**Ne pas relancer** si les pickles `scratch/test13_bob/13_cv_report.pkl` et
`scratch/test13_bob/13_final_model.pkl` sont présents : utiliser l'option
`--resume` du script pour régénérer le CSV et les rapports Hub sans refaire
la CV.

---

## 9. Pistes non explorées

- **Modèle de visite mixte (LME)** : le bloc 4 de `features.py` (`lme_trajectory`)
  est implémenté mais n'améliore pas la RMSE HGB (2.7176 vs 2.7129 pour 08).
  Une architecture de mélange gaussien ou un LGBM avec interactions croisées
  pourrait mieux exploiter les effets aléatoires per-patient.
- **Équation de Bateman (exp. 13_bateman_decay)** : l'analyse pharmacocinétique
  du ratio ON/OFF a confirmé que les constantes d'absorption et d'élimination
  sont hétérogènes entre patients (CV > 50 %). Incorporer les paramètres
  individuels `E_p` comme features pourrait aider, mais nécessite des visites
  avec ON et OFF simultanément mesurés (21 % du train seulement).
- **Séquences plus longues pour le GRU** : seuls 60 epochs ont été utilisés
  faute de temps. Un entraînement plus long, un scheduler cosine ou un GRU
  avec attention pourrait encore réduire le résidu.
- **Post-traitement de l'ensemble par patient** : pondérer HGB et GRU
  différemment selon le nombre de visites du patient (patients à une seule
  visite bénéficient moins du GRU).
- **Données externes / transfer learning** : l'utilisation d'embeddings pré-
  entraînés sur d'autres cohortes Parkinson n'a pas été explorée.
