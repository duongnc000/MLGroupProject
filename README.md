# CreditSense — Loan Risk Assessment

> **AI1215 — Introduction to Machine Learning** · Kaggle: [CreditSense competition](https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04)

Two coupled supervised-learning tasks on a 35,000-row tabular dataset of personal-loan applicants:

| Task | Target | Type | Metric |
|------|--------|------|--------|
| A | `RiskTier` ∈ {0, 1, 2, 3, 4} | 5-class classification | Accuracy |
| B | `InterestRate` ∈ [4.99, 35.99] | Regression (APR %) | R² |

Kaggle's combined leaderboard score is `0.5 × Accuracy + 0.5 × R²`.

**Final submission:** `submission_tabm_v2.csv` — Kaggle public LB **0.88110** (vs ~0.51 baseline).

---

## Quick start

```bash
# 1. Create a Python 3.10+ environment, then:
pip install -r requirements.txt

# 2. Reproduce the final submission (TabM v2)
python train_tabm_kfold_submission.py
# → writes submission_tabm_v2.csv at the repo root
```

That's it for the final pipeline. Tuned hyperparameters are loaded from `artifacts_hpo/tabm_v2_best_params_*.json` — no need to re-run the Optuna search.

---

## Repository layout

```
.
├── credit_train.csv, credit_test.csv      # input data (35k train, 15k test)
│
├── src/creditsense/                       # packaged sklearn pipeline
│   ├── config.py                          #   seeds, target/ID names, missing-flag list
│   ├── features.py                        #   add_engineered_features, build_data_profile
│   ├── modeling.py                        #   preprocessor + model factories
│   └── training.py                        #   train_all_models orchestrator
│
├── train.py                               # entry point for the sklearn pipeline
├── train_classic_submission.py            # CatBoost-only Optuna-tuned submission
├── train_tabm_submission.py               # TabM v1 submission (LB 0.87417)
├── train_tabm_kfold_submission.py         # TabM v2 submission (LB 0.88110, FINAL)
│
├── tune_tabm_v2.py                        # Optuna search → produced final TabM params
├── credit_optuna_search.py                # Optuna search → produced CatBoost params
│
├── artifacts/
│   ├── best_params_classification.json    # CatBoost cls hyperparams
│   ├── best_params_regression.json        # CatBoost reg hyperparams
│   └── imputer_search_*.csv               # CV scores from CatBoost imputer search
│
├── artifacts_hpo/
│   ├── tabm_best_params_*.json            # TabM v1 hyperparams
│   └── tabm_v2_best_params_*.json         # TabM v2 hyperparams (used by final)
│
├── outputs/                               # written by train.py
│   ├── data_profile.json                  #   missingness / sanity-check summary
│   ├── experiment_results.csv             #   holdout metrics for every tried model
│   └── run_summary.json                   #   best models + combined local score
│
├── notebooks/eda_starter.ipynb            # exploratory notebook
├── submission*.csv                        # committed Kaggle submissions (see below)
└── requirements.txt                       # all Python deps (classic + deep-learning)
```

---

## Modeling tracks

The repo holds three independent modeling tracks that share inputs but otherwise duplicate preprocessing on purpose. Each writes to a distinct submission filename.

| Track | Entry point | Submission file | Public LB |
|-------|-------------|-----------------|-----------|
| Sklearn pipeline (Logistic / XGB / LGB / weighted ensemble) | `python train.py` | `submission.csv` | 0.82717 |
| CatBoost (Optuna-tuned) | `python train_classic_submission.py` | `submission_classic.csv` | 0.84443 |
| TabM v1 (neural ensemble of MLPs) | `python train_tabm_submission.py` | `submission.csv` (overwrites) | 0.87417 |
| **TabM v2 (final)** | `python train_tabm_kfold_submission.py` | `submission_tabm_v2.csv` | **0.88110** |

### Reproducing each submission

```bash
# Sklearn pipeline (writes outputs/ + submissions/submission.csv)
python train.py --data-dir . --output-dir outputs --submission-dir submissions

# Tuned CatBoost (reads artifacts/best_params_*.json)
python train_classic_submission.py

# TabM v1 (reads artifacts_hpo/tabm_best_params_*.json)
python train_tabm_submission.py

# TabM v2 — the final approach (reads artifacts_hpo/tabm_v2_best_params_*.json)
python train_tabm_kfold_submission.py
```

`InterestRate` predictions are always clipped to `[4.99, 35.99]` and rounded to 2 decimals before export.

---

## Hyperparameter tuning

All tuning uses [Optuna](https://optuna.org/) with the **TPE sampler** and **Median pruner** (unpromising trials are killed mid-CV).

```bash
# Re-run the TabM search (overwrites artifacts_hpo/tabm_v2_best_params_*.json)
python tune_tabm_v2.py

# Re-run the sklearn / CatBoost search
python credit_optuna_search.py
```

| Search | CV strategy | Trials per task | Best score |
|--------|-------------|-----------------|------------|
| TabM v2 (RiskTier) | 3-fold StratifiedKFold | 60 | Acc 0.8895 |
| TabM v2 (InterestRate) | 3-fold KFold | 60 | R² 0.8494 |

---

## Data and validation

- **35,000 train rows × 55 features**, plus two target columns; **15,000 test rows** without targets.
- **Validation:** the sklearn pipeline uses an 80/20 stratified holdout (seed `1215`); the Optuna searches use 3-fold CV (seed `42`).
- **Meaningful-missing columns** (`PropertyValue`, `MortgageOutstandingBalance`, `StudentLoanOutstandingBalance`, `CollateralValue`, `SecondaryMonthlyIncome`) get explicit `*_missing_flag` indicators rather than being imputed away. TabM achieves the same effect by reserving index 0 in its categorical embedding for missing/unknown.

---

## Notes

- A GPU is preferred for the TabM track (`tune_tabm_v2.py` auto-selects CUDA if available); CPU works but is significantly slower.
- The `train_tabm_kfold_submission.py` script reads `artifacts_hpo/tabm_v2_best_params_*.json` first and falls back to the v1 JSONs if the v2 files are missing.
- See `Placeholder_Report.pdf` for the written project report.
