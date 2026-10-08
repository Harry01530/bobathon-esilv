# Solution Othmane — relais pour l'équipe

Résultat vérifié le 7 octobre 2026 : **RMSE public Kaggle 7.142**, contre **16.421** pour la soumission dummy de Julien. Plus le RMSE est bas, mieux c'est. Le code a été écrit et exécuté dans **IBM Bob**, avec évaluation **skore**.

## À reprendre

- Code : [experiments/02_day2_models.py](../experiments/02_day2_models.py).
- Résultats détaillés, versions et empreintes des données : [othmane_results.json](othmane_results.json).
- [Rapport Skore associé à la soumission Kaggle](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50341).
- [Validation croisée associée](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50335).
- [Soumissions Kaggle de l'équipe](https://www.kaggle.com/competitions/bobathon-esilv/submissions).

La soumission existante est `02_fold5_hgbr_mixed_othmane_2.csv`, avec l'URL du rapport 50341 dans sa description. Il n'est pas nécessaire de la soumettre à nouveau.

## Comparaison déjà réalisée

Les quatre modèles ont été évalués sur les mêmes cinq plis, en séparant les patients avec `GroupKFold(n_splits=5)`. `patient_id` sert uniquement à construire les groupes ; `Index` et la cible sont exclus des variables explicatives.

| Modèle | RMSE moyen de validation |
|---|---:|
| Dummy : moyenne | 16.499184 |
| Ridge : imputation médiane, alpha=1 | 10.436547 |
| HistGradientBoosting : variables numériques | 7.434293 |
| TableVectorizer + HistGradientBoosting : numériques, cohort, gene | **7.427018** |

Le modèle mixte est celui soumis. Il est réentraîné sur toutes les lignes d'entraînement pour prédire les 11 013 lignes de test. Les colonnes et l'ordre des `Index` ont été vérifiés contre `sample_submission.csv` ; toutes les prédictions sont finies.

L'écart entre les deux modèles HistGradientBoosting est très faible : 0.0073 de RMSE moyen. Cela ne démontre pas un gain robuste des catégories. Les RMSE des cinq plis du modèle mixte sont 7.484926, 7.550232, 7.258698, 7.310903 et 7.530330.

## Données à utiliser

Télécharger les quatre CSV depuis [Kaggle — Data](https://www.kaggle.com/competitions/bobathon-esilv/data) et les placer dans **`data/kaggle_official/`** : `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`.

Les CSV hérités déjà présents à la racine de `data/` diffèrent des fichiers officiels, notamment pour `on` et `off`. Le script utilise explicitement `data/kaggle_official/`. Les empreintes SHA-256 des fichiers utilisés sont conservées dans le JSON pour vérifier que l'équipe travaille sur la même version. Les données officielles, les prédictions et les identifiants de connexion ne sont pas publiés dans cette contribution.

## Continuer sans répéter le travail

1. Reprendre cette contribution depuis la branche **`othmane`** dans la branche personnelle du prochain membre. Travailler successivement, comme convenu dans le groupe, et noter la branche ou le commit repris.
2. Lire les résultats et les rapports existants avant d'utiliser Bob. Ces quatre modèles constituent déjà le point de comparaison ; ne pas les relancer simplement pour retrouver leur score.
3. Faire écrire et exécuter dans Bob une seule amélioration ciblée, par exemple un réglage de HistGradientBoosting, en conservant les mêmes patients dans les cinq plis et les mêmes données officielles. Évaluer avec skore et comparer au RMSE 7.427018.
4. Consigner le changement, les résultats par pli et le lien Skore dans un nouveau journal. Toute nouvelle soumission Kaggle doit avoir un nouveau rapport, un CSV portant sa clé et son URL EstimatorReport dans la description.

Pour reproduire volontairement tout le script, suivre d'abord [SETUP.md](../docs/SETUP.md), installer les données ci-dessus et utiliser l'environnement du projet dans Bob. **L'exécution du script relance les quatre validations, entraîne le modèle final et ajoute deux rapports au Hub** : ce n'est pas nécessaire pour consulter ou reprendre cette solution. Les clés libres sont choisies avec un suffixe ; les sorties restent dans `submissions/`.

## Traçabilité et avancement

Le code publié est la version finale déjà exécutée par Bob. Il extrait les cinq scores avec `metrics.get("rmse", data_source="test", aggregate=None)` puis calcule leur moyenne. Il ne moyenne pas une moyenne et un écart-type.

La dernière exécution de vérification a créé les rapports [CV 50437](https://skore.probabl.ai/julien.gouel/bobathon-esilv/cross-validations/50437) et [EstimatorReport 50443](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/50443), avec le suffixe `_3`. Son CSV est identique octet par octet au `_2` déjà soumis. La soumission Kaggle reste liée au rapport **50341** ; les deux exécutions sont distinguées dans le JSON.

La baseline personnelle du Day 1 est disponible dans le [rapport 49769](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/49769), RMSE 16.477937 sur une séparation aléatoire de lignes. Ce score n'est pas directement comparable au protocole de validation par patients ci-dessus. Une partie du Day 2 a été réalisée : Ridge, HistGradientBoosting, catégories, validation par patients et soumission. Le workbook complet, une recherche de paramètres et skrub DataOps ne sont pas terminés.

Kaggle a confirmé le succès technique et le score public. L'éligibilité à l'événement reste à confirmer avec les organisateurs, car Overview et Rules affichent des fenêtres de soumission contradictoires.
