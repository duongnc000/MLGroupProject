# CreditSense — Presentation Brief

**Course:** AI1215 — Introduction to Machine Learning
**Competition:** CreditSense Loan Risk Assessment Challenge (Kaggle)
**Format target:** 10-minute oral presentation + 5-minute Q&A
**Audience:** Course staff and classmates

This brief is the source-of-truth content for the slides. Each top-level section ≈ one slide block. Anything in `[brackets like this]` is a placeholder I (Duong) still need to fill in once TabM-v2 tuning finishes — please leave them visible in the deck for now and I'll send you the numbers when they're ready.

---

## Slide 1 — Title

- **Project:** CreditSense — Loan Risk Assessment
- **Team name:** [team name]
- **Members:** [your name + Kaggle handle], [my name + Kaggle handle]
- **Final Kaggle combined score:** **0.88110** (TabM-v2)
- **Public leaderboard rank:** [rank if we want to flex]

> Speaker note (1 sentence): "We trained two models on the same 35k-row dataset — one to predict the borrower's risk tier, one to predict their interest rate."

---

## Slide 2 — Problem Overview (≈ 1 min mark)

Two coupled tasks on the same applicant:

| Task | Target | Type | Metric |
|---|---|---|---|
| A | `RiskTier` ∈ {0,1,2,3,4} | 5-class classification | Accuracy |
| B | `InterestRate` ∈ [4.99, 35.99] | Regression (APR %) | R² |

- **Leaderboard score = 0.5 × Accuracy + 0.5 × R²**
- Random baseline: ~0.10. Logistic+Linear baseline: ~0.51.
- Submission file = single CSV with columns `Id, RiskTier, InterestRate`. Every interest-rate prediction is clipped to `[4.99, 35.99]` and rounded to 2 decimals before export.

> Speaker note: emphasise that the two tasks share the same features and very likely share the same signal — a borrower's risk drives their offered rate.

---

## Slide 3 — Dataset At a Glance

- **Train:** 35,000 applicants × 55 features + 2 targets
- **Test:** 15,000 applicants × 55 features (no targets)
- **Feature groups:**
  - Demographics (8): Age, Education, MaritalStatus, HomeOwnership, ...
  - Employment & Income (10): AnnualIncome, EmploymentStatus, ...
  - Assets & Liabilities (9): SavingsBalance, PropertyValue, MortgageOutstandingBalance, ...
  - Credit History (18): RevolvingUtilizationRate, DebtToIncomeRatio, NumberOfLatePayments30/60/90, NumberOfBankruptcies, ...
  - Loan Request (10): RequestedLoanAmount, LoanPurpose, RequestedTermMonths, ...

> Speaker note: 18 of the 55 features are credit-history features — that group is where we expected most of the signal, and that's what the data showed.

---

## Slide 4 — Data Exploration: Key Findings (≈ 3 min mark)

Suggested visual: a single bar chart of missing-value % per column, with the "meaningful-missing" columns colour-coded. (We have `outputs/data_profile.json` to source the numbers.)

Talking points:

- **Missingness is not random.** Several columns are missing *because the borrower doesn't have that thing*:
  - `PropertyValue` is missing for renters
  - `StudentLoanOutstandingBalance` is missing for older applicants
  - `MortgageOutstandingBalance` is missing for non-homeowners
  - `CollateralValue` is missing for unsecured loans
  - `SecondaryMonthlyIncome` is missing for people with no second income
  - → Imputing these with mean/median would destroy real information.
- **Targets are well-distributed.** RiskTier is roughly balanced across the 5 classes (no severe imbalance — accuracy is a fair metric).
- **Strong target correlations.** `RevolvingUtilizationRate`, `DebtToIncomeRatio`, `NumberOfLatePayments30/60/90` and `CreditHistoryLengthMonths` correlate strongly with both targets. (Confirms the financial intuition.)
- **`InterestRate` and `RiskTier` are tightly coupled** in the training set — high-risk tiers get the high APRs. This is what makes the dual-task setup work: the same features drive both.
- **No data leakage / duplicates** found. Train and test feature distributions match.

---

## Slide 5 — Missing-Value & Encoding Strategy

Two strategies, applied differently per model family:

| Family | Numeric missing | Categorical missing | Encoding |
|---|---|---|---|
| **Linear baselines** | Median impute | Most-frequent impute | One-hot |
| **Tree boosters (XGB / LightGBM / CatBoost)** | Median impute (XGB/LGB handle NaN natively in some configs) | Most-frequent / native CatBoost handling | One-hot or native categorical |
| **TabM (neural)** | Tunable: mean / median / KNN / iterative (chosen by Optuna) | Tunable: most-frequent / **`special` (own bucket)** | OrdinalEncoder shifted by +1, so index 0 = "missing/unknown" |

**Key idea:** for the columns where missing means something (PropertyValue, StudentLoan, CollateralValue, etc.) we add an explicit `*_missing_flag` indicator before imputing. For TabM we get the same effect "for free" by using the dedicated 0-bucket in the categorical embedding.

---

## Slide 6 — Feature Engineering: What and Why (≈ 5 min mark)

We hand-crafted a small set of ratio/aggregate features grounded in credit-scoring intuition. Roughly 15 new columns on top of the 55 raw ones.

| New feature | Definition | Financial intuition |
|---|---|---|
| `LiquidAssets` | SavingsBalance + CheckingBalance | Cash a borrower can actually access |
| `TotalDebtApprox` | Mortgage + AutoLoan + StudentLoan balances | Sum of outstanding obligations |
| `TotalDebtToIncome` | TotalDebtApprox / AnnualIncome | A cleaner DTI than the raw column |
| `RequestedToAnnualIncome` | RequestedLoanAmount / AnnualIncome | How big the ask is vs. income |
| `PaymentToIncomeRatio_FE` | MonthlyPaymentEstimate / TotalMonthlyIncome | Affordability of the new loan |
| `CollateralCoverage` | CollateralValue / RequestedLoanAmount | Lender-side risk cushion |
| `TotalLatePayments` | Late30 + Late60 + Late90 | Recency-blind delinquency count |
| `WeightedLatePayments` | Late30 + 2·Late60 + 3·Late90 | Penalises the more severe lates |
| `AnyDerogatoryFlag` | (Bankruptcies + PublicRecords + Collections + ChargeOffs) > 0 | Single binary "ever defaulted" |
| `CreditHistoryYears` | CreditHistoryLengthMonths / 12 | Same info, more readable scale |
| `log1p_<money_col>` | log(1 + value) for all $ columns | Compresses the long right tail of income/balance distributions — helps neural & linear models |
| `*_missing_flag` | binary indicator per meaningful-missing column | Preserves the "borrower doesn't have this" signal |
| Banded categoricals | Quintile bins for utilization, DTI, income, loan-size | Lets linear models capture non-linear thresholds |

Three preset bundles in the TabM track — picked by Optuna:
- `None` — raw features only
- `basic_credit` — ratio + behaviour + time features
- `basic_credit_plus_logs` — adds log-transformed money columns

Optuna picked **`basic_credit_plus_logs`** for the RiskTier model and **`None`** for the InterestRate model — interesting result, see Slide 11.

---

## Slide 7 — Feature Engineering: Impact Table

Holdout validation, 80/20 stratified split, seed 1215 (from `outputs/experiment_results.csv`):

| Configuration | Task A Acc | Task B R² | Combined |
|---|---|---|---|
| Logistic / Elastic-Net on raw features (baseline) | 0.634 | 0.618 | 0.626 |
| XGBoost + engineered features | 0.792 | 0.840 | 0.816 |
| LightGBM + engineered features | 0.791 | 0.839 | 0.815 |
| CatBoost + engineered features | 0.816 | 0.846 | 0.831 |
| Tuned weighted ensemble (engineered) | **0.798** | **0.847** | **0.822** |

Headline: **engineered features alone moved us from ~0.63 → ~0.82** combined. Baseline → boosters is the single biggest jump in the project.

> Speaker note: the bulk of the improvement came from feature engineering + switching to gradient boosting, not from any single fancy model.

---

## Slide 8 — Model Development Journey (≈ 6 min mark)

Tell this as a 4-step story, not a list:

1. **Baselines (sanity check).** Logistic Regression for RiskTier, ElasticNet for InterestRate, on raw features. → Combined ≈ 0.63. Confirms the assignment's reported baseline (~0.51) and shows our preprocessing is at least sane.
2. **Gradient boosting.** Drop in XGBoost / LightGBM / CatBoost with our engineered features. → Combined ≈ 0.82. Big jump — confirms tabular tree models are the right family.
3. **Hyperparameter search with Optuna.** We weren't going to hand-tune ~10 hyperparams across three boosters and a neural model. Optuna with TPE sampler + Median pruner found materially better configs than our defaults (next slide).
4. **Going deeper — TabM (neural ensemble).** Once we had tuned boosters, we tried a deep model designed for tabular data to see if we could squeeze out more. Final approach.

Flow diagram suggestion for the slide:

```
Logistic / Linear baselines
        │
        ▼
XGBoost  →  +LightGBM  →  +CatBoost   (with engineered features)
        │
        ▼
Optuna tuning (TPE + MedianPruner, 3-fold CV)
        │
        ▼
TabM (parallel ensemble of MLPs, Optuna-tuned)  →  final submission
```

> Speaker note: we also tried weighting TabM with CatBoost as a final blend, but the blended submission scored *worse* than pure TabM on the public LB — see Slide 13.

---

## Slide 9 — Models Compared

A clean comparison table (one row per family, best variant only):

| Model family | Task A Acc (val) | Task B R² (val) | Combined (val) | Notes |
|---|---|---|---|---|
| Logistic / ElasticNet (raw) | 0.634 | 0.618 | 0.626 | Baseline |
| **XGBoost** (engineered) | 0.792 | 0.840 | 0.816 | First strong model |
| **LightGBM** (engineered) | 0.791 | 0.839 | 0.815 | Comparable to XGB |
| **CatBoost** (Optuna-tuned) | 0.816 | 0.837 | 0.826 | Best single classic model |
| **Weighted booster ensemble** | 0.798 | 0.847 | 0.822 | XGB+LGB+LR / CatBoost+XGB+LGB grid-searched weights |
| **TabM** (Optuna-tuned, final) | 0.890 | 0.849 | 0.870 | **Final model. Public LB = 0.88110** |

---

## Slide 10 — Hyperparameter Tuning with Optuna

Why we used it:
- **Search space too large to grid-search.** TabM alone has 7+ structural hyperparams + 4 preprocessing choices.
- **Optuna prunes bad trials early.** MedianPruner stops a CV trial after the first fold if it's already worse than the median trial — saves ~50% of compute.
- **TPE sampler** (Bayesian) converges to good regions much faster than random search.

What we tuned:

| Track | Search space |
|---|---|
| **CatBoost** (`credit_optuna_search.py`) | iterations, depth, learning rate, L2 leaf reg, random strength, bagging temperature, border count |
| **TabM** (`tune_tabm.py` → `tune_tabm_v2.py`) | preprocessing (imputers, scaling, feature bundle), `k` heads (8–48), `n_blocks` (3–6), `d_block` (128–512), dropout, learning rate, weight decay, batch size, n_epochs, patience |

Settings:
- 3-fold StratifiedKFold (classification) / KFold (regression)
- 60 trials per task per study
- TPESampler(multivariate=True) + MedianPruner

Best CatBoost config (from `artifacts/best_params_*.json`):
- Classification: 2,812 iter, depth 6, lr 0.0366
- Regression: 3,800 iter, depth 6, lr 0.0167

Best TabM config (from `artifacts_hpo/tabm_best_params_*.json`):
- **RiskTier:** k=48 heads, 5 blocks of width 192, dropout 0.04, lr 2.7e-4, features = `basic_credit_plus_logs`
- **InterestRate:** k=32 heads, 5 blocks of width 512, dropout 0.15, lr 2.0e-3, features = none (raw)

> Speaker note: it's interesting that the regression model preferred raw features and the classification model wanted engineered ones — likely because the boosters captured the ratios anyway and TabM's MLP+embedding can re-derive them from raw inputs given enough capacity, while the categorical RiskTier benefits from explicit thresholding.

---

## Slide 11 — Final Approach: TabM (≈ 8 min mark)

What TabM is, in 2 lines:
- A **neural ensemble of `k` parallel MLPs** sharing input embeddings (numeric + categorical-embedding) and a feature backbone, but with `k` independent prediction heads.
- Effectively in-model bagging: each forward pass produces `k` predictions; we average them at inference.

Why it fit the problem:
- Tabular dataset → neural-net-on-tabular usually loses to gradient boosting. TabM is one of the few architectures that competes (and sometimes beats) GBDTs on tabular benchmarks.
- It handles **mixed numeric + categorical** natively via embeddings — no one-hot blow-up.
- The ensemble-of-heads structure is **regularising** — single seeds are noisy, k=32–48 heads smooths it.

Training recipe:
- Categorical encoder: `OrdinalEncoder(unknown=-1, missing=-1) + 1` so index 0 = unknown bucket
- Loss: per-head cross-entropy (cls) / per-head MSE (reg), averaged across heads
- Optimizer: AdamW, with the Optuna-tuned lr / weight-decay
- Early stopping with `patience` chosen by Optuna
- Trained on the **full training set** (no holdout) once tuned — Optuna already gave us a CV-validated config

Final submission pipeline:
1. Load tuned TabM params from `artifacts_hpo/`
2. Train two TabM models: one for RiskTier, one for InterestRate
3. Predict on test, average the `k` heads
4. Clip InterestRate to [4.99, 35.99], round to 2 decimals
5. Submit. (We also tried a 0.6/0.4 weighted blend with CatBoost — it scored worse than pure TabM, so we kept TabM as the final model.)

---

## Slide 12 — Final Results

| Submission | Task A Acc (val) | Task B R² (val) | Combined (val) | Kaggle public LB |
|---|---|---|---|---|
| Baseline (template) | ~0.53 | ~0.50 | ~0.51 | ~0.51 (provided benchmark) |
| Best XGBoost / sklearn-ensemble submission | 0.798 | 0.847 | 0.822 | **0.82717** |
| Tuned CatBoost (`submission_classic.csv`) | 0.816 | 0.837 | 0.826 | **0.84443** |
| TabM, first tuned version (`submission.csv`) | — | — | — | 0.87417 |
| **TabM-v2 (`submission_tabm_v2.csv`) — final** | **0.890** | **0.849** | **0.870** | **0.88110** |
| TabM + CatBoost blend (tried, not used) | — | — | — | < 0.87417 (worse than pure TabM) |

> Headline number for the title slide: **0.88110** on the public leaderboard, vs ~0.51 baseline. ~73% above baseline.

---

## Slide 13 — What Worked / What Didn't / Lessons Learned (≈ 10 min mark)

**Worked:**
- Treating the meaningful-missing columns as signal (flags + dedicated TabM bucket) instead of imputing them away.
- Hand-crafted credit ratios — biggest single jump after switching to boosters (~0.63 → ~0.82 combined).
- Optuna with pruning — let us search a 7-d space for TabM in a reasonable time.
- TabM as a deep-tabular alternative — it actually beat our tuned boosters on this dataset (0.881 vs 0.844 on the public LB), which is rare. The v2 search (piecewise-linear numeric embeddings, cosine LR + warmup, label smoothing) gave us another ~0.7-point bump on top of the original TabM (0.874 → 0.881).

**Didn't work / didn't help much:**
- **Blending TabM with CatBoost.** The 0.6/0.4 weighted blend scored *below* pure TabM on Kaggle. Most likely cause: TabM is itself a k-head ensemble (k=32–48 parallel MLPs), so adding a less-accurate CatBoost on top diluted it rather than diversifying it.
- One-hot + linear baselines plateau hard around R² ≈ 0.62 even with engineered features — confirms the targets aren't linearly separable.
- Adding more aggressive feature engineering for the regression task (logs, banded ratios) — Optuna ended up picking the raw-feature bundle for InterestRate.
- More TabM heads beyond k≈48 — diminishing returns, just slower training.

**Lessons:**
- Domain features matter more than fancy models: ratio features + missing-flags moved us further than any single architecture switch.
- Validate locally before every Kaggle submission — we have only 5/day, and the local 80/20 holdout (seed 1215) tracked the public LB closely enough to trust.
- The same features didn't matter equally for both tasks — Task A wanted explicit credit ratios, Task B was happy with raw inputs + a deep enough model. Useful to remember even though we ended up using the same model family for both.

---

## Slide 14 (backup, only if time) — Reproducibility

- Fixed seeds (1215 for sklearn, 42 for TabM/CatBoost)
- Stratified 80/20 holdout, same split for both tasks → directly comparable local combined score
- All tuned hyperparameters dumped to JSON (`artifacts/`, `artifacts_hpo/`) and reloaded by training scripts — no copy-paste of magic numbers
- Submission scripts are deterministic given the JSON params

---

# Q&A Cheat Sheet (not slides — keep on phone/laptop)

**Q: Why TabM and not just XGBoost?**
A: We tried XGBoost / LightGBM / CatBoost first — best public LB was 0.844 (tuned CatBoost). TabM is one of the few neural architectures that's competitive with GBDTs on tabular data, and it pushed us to 0.881 — a ~3.7-point jump. We also tried blending TabM with CatBoost expecting the usual ensemble win, but the blend was *worse* than pure TabM, probably because TabM's k=48 parallel-head ensemble already provides the diversity a second model would add.

**Q: How did you handle missing values?**
A: Two complementary tricks. (1) Add explicit `*_missing_flag` columns for fields where "missing" means something — renter ⇒ no PropertyValue, etc. (2) For TabM, shift the ordinal encoding by +1 so index 0 is reserved for missing/unknown, giving the embedding a dedicated bucket.

**Q: How many features in the end?**
A: 55 raw + ~15 engineered = ~70. TabM picked the bundle that uses all of them for RiskTier; for InterestRate Optuna actually preferred raw features only (the deep model can re-derive ratios from inputs).

**Q: Why didn't you use AutoGluon / [other AutoML]?**
A: We did experiment with it as a sanity check, but our hand-tuned TabM + CatBoost stack beat it locally and we wanted a model we could explain end-to-end. Happy to share details if useful.

**Q: How did you prevent overfitting?**
A: (1) 80/20 stratified holdout, single fixed seed. (2) Optuna's CV with MedianPruner keeps us honest — the score we tune against is the mean of 3 folds, not a single train/val split. (3) TabM's k-head ensemble is itself a form of bagging. (4) Early stopping on patience for TabM, max iterations capped for CatBoost.

**Q: Did the same features matter for both tasks?**
A: No — that's actually one of our findings. RiskTier loved the engineered ratios (DTI, weighted late payments, missing flags). InterestRate cared more about raw money columns, especially when we let TabM handle them with embeddings. Both relied heavily on RevolvingUtilizationRate and the late-payment counts, but the *useful representation* of those features differed.

---

# Notes For You (Teammate)

- Speaker timing: 1 / 2 / 2 / 3 / 2 minutes for sections 1–5 of the slide deck (matches the assignment brief).
- The assignment grades content 50% / presentation skills 30% / Q&A 20%, so the cheat sheet matters.
- I'll send you the `[bracketed]` numbers (final TabM-v2 results + Kaggle scores) the moment tuning finishes — should be tonight.
- I left AutoGluon out of the deck on purpose — it adds explanation overhead and didn't beat TabM, so it's a Q&A liability, not an asset. If you want to slide it into the appendix as a "we also tried this" footnote, that's fine, but don't put it in the main flow.
- One visualisation is required for the report; the missing-value bar chart on Slide 4 is the obvious candidate. A correlation heatmap of the credit-history features against `RiskTier` / `InterestRate` would also work well.
