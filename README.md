<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="60">
</p>
# Help science build better Parkinson's evaluation

*Predicting and correcting bias in motor score evaluation — for each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score.*

## Getting started

```bash
git clone https://github.com/probabl-ai/bobathon-esilv.git
cd bobathon-esilv
```

The lab has three steps. Do them **in order**: each one assumes the previous one is finished.


| Step | Guide                     | What you do                                                                                                |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | [SETUP.md](docs/SETUP.md) | Join Kaggle, form a team, install Python, create the shared Skore Hub workspace, install skore and sign in |
| 2    | [DAY1.md](docs/DAY1.md)   | Download the data, explore it (EDA), push a dummy baseline to Skore Hub                                    |
| 3    | [DAY2.md](docs/DAY2.md)   | Build real models, evaluate them with patient-grouped cross-validation, submit to Kaggle                   |


### Step 1 — Setup ([SETUP.md](docs/SETUP.md))

Follow every section of [SETUP.md](docs/SETUP.md):

1. **Join Kaggle and form a team** (maximum four people). Remember the exact team name.
2. **Install Python 3.12 or newer** and check it with `python --version` (or `python3 --version`).
3. **Create one Skore Hub workspace per team** on [skore.probabl.ai](https://skore.probabl.ai), named like the Kaggle team. One teammate creates it and invites the others.
4. **Allow Bob to use the lab skills** (tick **Skill** in Bob's Permissions menu), then **install skore and sign in** from the repo root.

#### Check your setup before moving on

Run the setup tests, with the virtual environment still active:

```bash
python -m pytest
```

See [SETUP.md step 5](docs/SETUP.md#5-install-skore-and-check-your-setup) for details.

**You cannot proceed to Day 1 until the last line says** `all passed`**.** Without a working `.skore`, you cannot push reports to Skore Hub, and every Kaggle Submission needs a Hub report URL.

### Step 2 — Day 1 ([DAY1.md](docs/DAY1.md))

1. Download the CSVs from the [competition Data tab](https://www.kaggle.com/competitions/bobathon-esilv/data) into `data/`.
2. Explore the data in Bob (EDA) and go through the workbook questions.
3. Build a **dummy mean** baseline, evaluate it with `skore.evaluate`, and `put` it to Skore Hub as `01_dummy`. Every later model must beat this score.

### Step 3 — Day 2 ([DAY2.md](docs/DAY2.md))

1. Train a **Ridge** linear model, compare it with the dummy, and make your **first Kaggle Submission**.
2. Switch to **patient-grouped cross-validation** (`GroupKFold`) so the score matches the Kaggle split.
3. Try **HistGradientBoosting** (missing values as signal), skrub `**tabular_pipeline`** (categorical columns), and skrub **DataOps** (groups baked into the graph).
4. Fit your best model on all training visits and **submit again**.

## Documents

- [SETUP.md](docs/SETUP.md) - Kaggle team, Python, Skore Hub workspace, skore install and sign-in
- [DAY1.md](docs/DAY1.md) - data, exploratory data analysis (EDA), dummy baseline
- [DAY2.md](docs/DAY2.md) - models, grouped cross-validation, Kaggle Submissions
- [CONTEXT.md](docs/CONTEXT.md) - scientific background, goals, and modelling pitfalls for the Parkinson's dataset
- Kaggle competition: [bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)

## Submission rules

All code must be written and run in **Bob** (IDE or CLI), with evaluation through **skore**. Every Kaggle upload must put a Skore Hub EstimatorReport URL (`https://skore.probabl.ai/…`) in the Submission Description, and each Submission needs a **new** report key (`01_dummy`, `02_ridge`, …). Each experiment writes its file to `submissions/<report-key>.csv` (e.g. `submissions/02_ridge.csv`), so the filename matches the report key and a later model never overwrites an earlier submission. See the [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules) for eligibility and what makes a Submission valid.

## Solution finale

Le meilleur résultat de l'équipe est l'expérience **13_hybrid_gru** :
CV RMSE **2.5598 ± 0.0117** (GroupKFold 5, patient_id) —
Kaggle public **2.642** (top 16 au moment de la soumission).

Le rapport complet (chronologie de toutes les expériences, ce qui a fonctionné,
étanchéité de l'évaluation, provenance du code, comment reproduire) est dans
[**journal/FINAL_REPORT.md**](journal/FINAL_REPORT.md).

| Soumission | CV RMSE | Kaggle public | Rapport Hub |
|---|:---:|:---:|---|
| `12_voting_ensemble` | 2.6784 | 2.708 | [Est. 82849](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/82849) |
| `13_fold5_hybrid_gru` ✅ | **2.5598** | **2.642** | [Est. 85398](https://skore.probabl.ai/julien.gouel/bobathon-esilv/estimators/85398) |
