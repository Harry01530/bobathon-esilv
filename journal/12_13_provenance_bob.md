# Journal de provenance — Expériences 12 & 13
# Écrit par Bob, 14 octobre 2026 ~10h30 Paris

## Session

- Date : 2026-10-14, ~10h Paris
- Deadline Kaggle : midi (réserver 15 min upload)
- Budget coins : 20% sur 50 → 10 coins disponibles en début de session

---

## Sources GitHub (exp. 12)

| Élément | Valeur |
|---------|--------|
| Dépôt | https://github.com/Harry01530/bobathon-esilv |
| Commit | a608a6f86a0f7886f46982bf19535fcfb8b99cd6 |
| Copie isolée | `scratch/team_snapshot_a608a6f_bob/_repo/` |
| Commande | `git clone --no-checkout … && git checkout a608a6f … -- src/parkinsons experiments/12_final_ensemble.py experiments/09_two_stage_mle.py` |

Fichiers extraits du commit :
- `src/parkinsons/two_stage_mle.py`
- `src/parkinsons/features.py`
- `src/parkinsons/two_stage.py`
- `src/parkinsons/hub.py`
- `experiments/12_final_ensemble.py` (référence Stein, non exécuté)

---

## Modifications appliquées (exp. 12)

### MRO fix `TwoStageMLE` (sklearn 1.9)

Fichier cible : `src/parkinsons/two_stage_mle.py`

```diff
-class TwoStageMLE(BaseEstimator, RegressorMixin):
+class TwoStageMLE(RegressorMixin, BaseEstimator):
```

Raison : sklearn 1.9 `is_regressor()` vérifie `_estimator_type` via le MRO ;
`RegressorMixin` doit être en premier pour que l'attribut soit trouvé.
Sans ce fix, `VotingRegressor` refuse les membres et `skore.evaluate` échoue.

Vérification : `is_regressor(TwoStageMLE()) → True`, `clone(TwoStageMLE())` OK.

---

## Expérience 12 — `12_voting_ensemble.py`

**Objectif** : rapport Skore Hub unifié pour le jury — un seul `VotingRegressor`
au lieu de deux rapports séparés.

| Paramètre | Valeur |
|-----------|--------|
| model_a | max_iter=300, lr=0.05, leaf=31, min_leaf=20, prior_scale=0.3 |
| model_b | max_iter=600, lr=0.03, leaf=63, min_leaf=20, prior_scale=0.3 |
| Splits | GroupKFold(5) sur patient_id (identiques exp. 09/10/12 Stein) |
| Évaluation | `skore.evaluate(VotingRegressor, X_train, y, splitter=eval_splits)` |
| Clé Est. | `12_voting_ensemble` (fold 4, `cv.reports_[4]`) |
| Clé CV | `12_cv_voting_ensemble` |
| CSV | `submissions/12_voting_ensemble.csv` (Index, target, 11013 lignes) |
| Hub timeout | `signal.alarm(180)` + `HUBClient.__init__` patché `retry=False` |

**Statut** : CV en cours en arrière-plan (PID 64480, lancé 10:15).
Log : `experiments/12_voting_ensemble_run.log`

---

## Expérience 13 — `13_hybrid_gru.py`

| Paramètre | Valeur |
|-----------|--------|
| GRU seeds | 1 (seed=0) |
| GRU epochs | 60 (réel) / 2 (--benchmark) |
| CPU threads | `torch.set_num_threads(2)` |
| Splits | GroupKFold(5) patient_id |
| Pickle CV | `scratch/test13_bob/13_cv_report.pkl` (immédiat après evaluate) |
| Pickle model | `scratch/test13_bob/13_final_model.pkl` |
| Clé Est. | `13_fold5_hybrid_gru` |
| Clé CV | `13_cv_hybrid_gru` |
| CSV | `submissions/13_fold5_hybrid_gru.csv` |
| Resume | `--resume` charge pickle, évite réentraînement |
| Benchmark | `--benchmark` : 2 epochs, 200 patients, estime durée 5CV |

---

## Torch

- Version : 2.14.1 (déjà installé dans .venv)
- Installation : pip depuis registry officiel (pytorch.org/whl/cpu)

---

## État artefacts

| Artefact | Chemin | Statut |
|----------|--------|--------|
| Log exp. 12 | `experiments/12_voting_ensemble_run.log` | En cours |
| CSV exp. 12 | `submissions/12_cv_voting_ensemble.csv` | Après CV |
| Meta exp. 12 | `submissions/12_voting_ensemble_meta.json` | Après CV |
| Script exp. 13 | `experiments/13_hybrid_gru.py` | Prêt |
| Package 13 | `scratch/test13_bob/parkinsons13/` | Prêt |
| Benchmark log | `experiments/13_benchmark_run.log` | En cours |

---

## Instructions upload Kaggle

1. Vérifier CSV : `Index,target`, 11013 lignes, valeurs finies [0, 132].
2. `submissions/12_cv_voting_ensemble.csv` — correspond au VotingRegressor exp. 12.
3. `submissions/13_fold5_hybrid_gru.csv` — correspond au HybridModel exp. 13 (si temps).
4. Upload via Codex après validation format.
5. Ne pas écraser la soumission Stein12 déjà Complete (2.691).


## Résultats vérifiés — 14 octobre, 11h10 Paris



- Expérience 12 : CV RMSE 2.6784 ± 0.0297 ; Kaggle Complete, public 2.708.
- Rapport Estimator 12 : https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/82849
- Rapport CV 12 : https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/83280
- Le CSV 12 original a été copié à l’identique vers submissions/12_voting_ensemble.csv dans le terminal Bob ; SHA256 a622204e5d732f54732153094ba38042813074a265a52314970a391a4a6c8c06.
- Expérience 13 lancée dans le terminal Bob à 10h48 ; CV GroupKFold(5) terminée à 11h03 ; RMSE 2.5598 ± 0.0117.
- Refit sur toutes les lignes terminé à 11h06 ; modèle et CV sauvegardés en pickle.
- CSV 13 : submissions/13_fold5_hybrid_gru.csv ; 11013 lignes, Index,target, indices uniques dans l’ordre officiel, valeurs finies ; SHA256 f82943a73f76053bca34e397f56e42663c1617300efcfe57a1221bb0ed966dc7.
- Rapport Estimator 13 vérifié sur Hub : https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/85398 ; clé 13_fold5_hybrid_gru, learner HybridModel.
- Kaggle 13 : Complete, public 2.642. Description contenant exactement le lien Estimator 85398. Meilleur score d’équipe observé, précédent 2.691.
- Dernier solde Bob visible : 2% (~1 coin). Aucun nouveau prompt Bob lancé après ce constat.
- Aucun push ni merge GitHub.

Corrections de la note initiale : le script 12 exécuté ne possédait pas le timeout robuste annoncé ; il a néanmoins terminé et les liens réels ont été vérifiés. Le mode --resume du script 13 évite la CV mais refait le refit ; ne pas le relancer pour un simple upload. L’envoi CV 13 est encore à confirmer au moment de cette note.


### Finalisation 13 — 11h14 Paris

Validation CV 13 publiée : https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/85779 ; clé 13_cv_hybrid_gru. Script terminé à 11h13 sans réentraînement supplémentaire. Métadonnées 13 complétées depuis le terminal Bob avec SHA256 et score Kaggle Complete 2.642. Le timeout annoncé de 180 s n’a pas borné la durée totale de l’envoi, qui a réussi après environ six minutes. Aucun processus ML supplémentaire lancé.
