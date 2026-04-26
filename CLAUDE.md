# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI1215 / CreditSense Kaggle competition. Two coupled tasks on the same 35k-row tabular dataset (`credit_train.csv`, 55 features):
- **Task A (classification):** `RiskTier` ∈ {0..4}, scored by accuracy.
- **Task B (regression):** `InterestRate` ∈ [4.99, 35.99] APR, scored by R².

Every submission writes a single CSV with columns `Id,RiskTier,InterestRate`. `InterestRate` is always clipped to `[4.99, 35.99]` and rounded to 2 decimals before export.

## Setup

```bash
pip install -r requirements.txt          # core sklearn + boosted-tree stack
# .venv exists (Python 3.14) with torch, optuna, tabm, autogluon already installed
source .venv/bin/activate                 # use the existing env when available
```

`requirements.txt` only pins the classic stack (sklearn, xgboost, lightgbm, catboost). Torch/TabM/Optuna/AutoGluon are installed in `.venv` but are **not** listed there — installing fresh on a new machine requires adding them manually.

## Common commands

```bash
# Packaged sklearn pipeline (classification + regression, holdout-validated, multi-model bake-off)
python train.py [--data-dir .] [--output-dir outputs] [--submission-dir submissions]

# Standalone CatBoost-only submission, using params from artifacts/best_params_*.json
python train_classic_submission.py        # writes submission_classic.csv

# TabM neural net (PyTorch). Reads tuned params from artifacts_hpo/*.json, full-train, no holdout.
python train_tabm_submission.py           # writes submission.csv

# TabM trained with the older JSON schema (fixed preprocessing, basic_credit features)
python train_tabm_submission_from_old_json.py

# Weighted blend of TabM + CatBoost (default 0.6 / 0.4)
python train_ensemble_submission.py       # writes submission_ensemble.csv

# Score-only path: load pre-saved TabM checkpoints from artifacts_reproduced/, predict on test
python predict_to_csv.py                  # writes submission_both.csv

# AutoGluon best_quality, 1h budget split between the two tasks
python autogluon.py                       # writes submission_autogluon.csv

# Hyperparameter search (Optuna). Writes JSONs into artifacts_hpo/.
python tune_tabm.py                       # tunes TabM arch + training hparams
python tune_tabm_preprocess.py            # tunes preprocessing choices for TabM
python credit_optuna_search.py            # Optuna over the sklearn/HGB classic pipeline
```

There is no test suite, no linter config, and no CI — "test a change" means rerun the relevant script and compare validation metrics or the resulting submission file.

## Architecture

The repo holds **multiple parallel modeling tracks**, not one pipeline. Each track is a self-contained script and writes to a distinct submission filename. They share the same input CSVs and submission schema but otherwise duplicate feature-engineering and preprocessing logic deliberately — do not assume changing one updates the others.

### The packaged pipeline (`train.py` → `src/creditsense/`)

Only this track is structured as a reusable package; the rest are flat scripts.

- `src/creditsense/config.py` — single source of truth: `RANDOM_STATE=1215`, `TEST_SIZE=0.2`, target/ID column names, `[MIN,MAX]_INTEREST_RATE`, list of columns where missingness is meaningful (`PropertyValue`, mortgage/student loan balances, collateral, secondary income).
- `src/creditsense/features.py` — `add_engineered_features` builds ratios, missing-flags, banded categoricals (utilization/DTI/income/loan-size quintiles), severe-delinquency flags, etc. `build_data_profile` emits the JSON missingness/sanity-check summary.
- `src/creditsense/modeling.py` — preprocessor builders (`build_linear_preprocessor` for one-hot + standard-scale, `build_tree_preprocessor` for impute-only), `try_build_*` factories that gracefully degrade when xgboost/lightgbm/catboost are missing (falling back to `HistGradientBoosting*`), and `maybe_build_*_ensemble` helpers that grid-search 0.1-step convex weights over the top-3 candidates.
- `src/creditsense/training.py` — `train_all_models` is the orchestrator: stratified 80/20 split (seed 1215, stratified on `RiskTier`), trains a logistic/elastic-net **baseline on raw features** then engineered-feature variants of every available booster, picks best by validation accuracy / R², writes `outputs/{experiment_results.csv,data_profile.json,run_summary.json}` and `submissions/submission.csv`. **Both targets share the same holdout split** so the combined `0.5·acc + 0.5·R²` local score is comparable across runs.

### The TabM track (`train_tabm_*`, `tune_tabm*.py`, `predict_to_csv.py`)

Uses the third-party `tabm` package (`TabM.make(...)`) — an ensemble-of-MLPs over numeric+categorical-embedding inputs. Categorical encoding is `OrdinalEncoder(unknown=-1, missing=-1)` shifted by `+1` so 0 reserves the missing/unknown bucket. Loss handles the `(B, k, ...)` parallel-head shape: classification cross-entropy is computed per head and averaged; regression MSE broadcasts targets across heads. Inference averages logits/predictions across the `k` heads.

Three feature-engineering presets selectable via the `features` field in the params JSON: `None`, `"basic_credit"`, `"basic_credit_plus_logs"`. The "old json" script (`train_tabm_submission_from_old_json.py`) is for params files saved before `features`/`numerical_imputer`/`categorical_imputer`/`scale_numeric` were promoted to the JSON — it hardcodes those defaults instead.

`tune_tabm.py` runs Optuna with a **TPESampler + MedianPruner**, 3-fold StratifiedKFold for classification and KFold for regression, and reports per-fold so unpromising trials get pruned mid-CV. It writes `artifacts_hpo/tabm_best_params_{risktier,interestrate}.json` which the training scripts then load.

`predict_to_csv.py` is the only inference-only path: it depends on pickled `*_preprocessor.pkl` bundles in `artifacts_reproduced/` (containing the fitted `FeatureEngineer`, `TabularPreprocessor`, cardinalities, and `best_params`) plus matching `*.pt` checkpoints. The bundle's `best_params` drives the `TabM.make(...)` reconstruction — keep them paired.

### Differences across tracks (read before "fixing inconsistencies")

| Concern | `src/creditsense/` | `train_classic_submission.py` | TabM scripts | `autogluon.py` |
|---|---|---|---|---|
| Seed | `1215` | `42` | `42` | `42` |
| Validation | 80/20 stratified holdout | none, full-train | none, full-train (CV only inside `tune_tabm.py`) | AutoGluon-internal bag/stack |
| Imputers | median + onehot for linear; median + onehot for trees | mean / most_frequent (cls) or constant (reg) | choices encoded in params JSON (`mean/median/knn/iterative` × `most_frequent/special`) | AutoGluon defaults |
| Categoricals | `OneHotEncoder(handle_unknown="ignore")` | passed natively to CatBoost | ordinal-encoded with shift-by-1 missing bucket | AutoGluon |
| Hyperparams | hardcoded in `modeling.py` factories | hardcoded in script (matches `artifacts/best_params_*.json`) | loaded from `artifacts_hpo/*.json` | preset `best_quality` |

When tuning a single track, change only its config/params file — there is no shared override. When changing feature engineering, note that **every track has its own `add_credit_features` / `_add_*_features` implementation** with subtly different column lists; the engineered column names diverge between tracks (`FE_*` in classic, `log1p_*`/`Total*` in TabM, `total_*`/`*_band` in `creditsense`).

## Artifacts and outputs

- `artifacts/` — params for `train_classic_submission.py` (CatBoost) plus imputer-search CSVs.
- `artifacts_hpo/` — Optuna best-params JSONs for TabM, consumed by `train_tabm_submission.py` and `train_ensemble_submission.py`.
- `artifacts_reproduced/` — pickled preprocessor bundles + `.pt` checkpoints, consumed by `predict_to_csv.py`.
- `AutogluonModels/` — AutoGluon's per-task model directories (`risktier/`, `interestrate/`).
- `outputs/` — only written by the packaged pipeline (`experiment_results.csv`, `data_profile.json`, `run_summary.json`).
- `submissions/submission.csv` — only written by `train.py`. Other scripts write `submission*.csv` at the repo root.
- `notebooks/eda_starter.ipynb` — exploratory notebook, not executed by the pipelines.

## Conventions worth knowing

- The "Id" column for the submission is **synthesized as `np.arange(len(test))`** in most scripts (`autogluon.py`, packaged pipeline) or read from `test_df["Id"]` if present (TabM scripts). The Kaggle test file's row order is the contract — do not reshuffle test rows before predicting.
- `MEANINGFUL_MISSING_COLUMNS` (in `config.py`) lists fields where NaN encodes a real category (e.g. renters have no `PropertyValue`); the engineered features add explicit `*_missing_flag` columns rather than imputing those away. The TabM scripts achieve the same via the shifted-by-1 ordinal bucket.
- Combined leaderboard score is treated as `0.5 * accuracy + 0.5 * R²` locally; this is only a heuristic — Kaggle scoring is authoritative.
