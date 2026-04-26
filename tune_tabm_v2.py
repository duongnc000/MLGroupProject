from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype

import rtdl_num_embeddings as rne
from tabm import TabM


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    train_path: str = "credit_train.csv"
    out_dir: str = "artifacts_hpo"

    n_trials_cls: int = 60
    n_trials_reg: int = 60
    n_splits_cls: int = 3
    n_splits_reg: int = 3
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    # When True, classifier objective trains 2 seeds per fold and reports the
    # mean. Halves trial count but reduces score noise; recommended on small
    # datasets where a single seed makes Optuna chase noise.
    multi_seed_eval: bool = False


CFG = Config()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Column selectors
# =========================================================
def get_numeric_column_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def get_categorical_column_names(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if (
            is_string_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or is_bool_dtype(df[c])
            or df[c].dtype == "object"
        )
    ]


# =========================================================
# Feature engineering bundles (mirrors tune_tabm_preprocess.py)
# =========================================================
def safe_div(a, b, eps=1e-6):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / (b + eps)


def _normalize_blanks_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for col in get_categorical_column_names(X):
        s = X[col].astype("string")
        s = s.mask(s.str.strip() == "", np.nan)
        X[col] = s.astype("object")
    return X


def _drop_fully_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    to_drop = [c for c in X.columns if X[c].isna().all()]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X


def _add_basic_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if {"SavingsBalance", "CheckingBalance"}.issubset(X.columns):
        X["LiquidAssets"] = (
            pd.to_numeric(X["SavingsBalance"], errors="coerce").fillna(0)
            + pd.to_numeric(X["CheckingBalance"], errors="coerce").fillna(0)
        )
    debt_cols = [
        "MortgageOutstandingBalance",
        "AutoLoanOutstandingBalance",
        "StudentLoanOutstandingBalance",
    ]
    existing = [c for c in debt_cols if c in X.columns]
    if existing:
        X["TotalDebtApprox"] = sum(pd.to_numeric(X[c], errors="coerce").fillna(0) for c in existing)
    if {"RequestedLoanAmount", "AnnualIncome"}.issubset(X.columns):
        X["RequestedToAnnualIncome"] = safe_div(X["RequestedLoanAmount"], X["AnnualIncome"])
    if {"MonthlyPaymentEstimate", "TotalMonthlyIncome"}.issubset(X.columns):
        X["PaymentToIncomeRatio_FE"] = safe_div(X["MonthlyPaymentEstimate"], X["TotalMonthlyIncome"])
    if {"TotalDebtApprox", "AnnualIncome"}.issubset(X.columns):
        X["TotalDebtToIncome"] = safe_div(X["TotalDebtApprox"], X["AnnualIncome"])
    if {"CollateralValue", "RequestedLoanAmount"}.issubset(X.columns):
        X["CollateralCoverage"] = safe_div(X["CollateralValue"], X["RequestedLoanAmount"])
    return X


def _add_credit_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    def _num(col):
        return pd.to_numeric(X[col], errors="coerce").fillna(0) if col in X.columns else 0
    late30, late60, late90 = (
        _num("NumberOfLatePayments30Days"),
        _num("NumberOfLatePayments60Days"),
        _num("NumberOfLatePayments90Days"),
    )
    X["TotalLatePayments"] = late30 + late60 + late90
    X["WeightedLatePayments"] = late30 + 2 * late60 + 3 * late90
    bk, pr, co, ch = (
        _num("NumberOfBankruptcies"),
        _num("NumberOfPublicRecords"),
        _num("NumberOfCollections"),
        _num("NumberOfChargeOffs"),
    )
    X["AnyDerogatoryFlag"] = ((bk + pr + co + ch) > 0).astype(int)
    return X


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if "CreditHistoryLengthMonths" in X.columns:
        X["CreditHistoryYears"] = pd.to_numeric(X["CreditHistoryLengthMonths"], errors="coerce") / 12.0
    return X


def _add_log_money_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    log_cols = [
        "AnnualIncome", "MonthlyGrossIncome", "TotalMonthlyIncome",
        "SavingsBalance", "CheckingBalance", "TotalAssets",
        "RequestedLoanAmount", "CollateralValue", "MonthlyPaymentEstimate",
    ]
    for col in log_cols:
        if col in X.columns:
            vals = pd.to_numeric(X[col], errors="coerce")
            X[f"log1p_{col}"] = np.log1p(np.clip(vals, 0, None))
    return X


def _coerce_pdna_to_npnan(df: pd.DataFrame) -> pd.DataFrame:
    X = df.replace({pd.NA: np.nan})
    for col in X.columns:
        dt = X[col].dtype
        if (
            dt == "object"
            or is_string_dtype(X[col])
            or isinstance(dt, pd.CategoricalDtype)
            or str(dt).lower().startswith("boolean")
        ):
            s = X[col].astype("object")
            mask_blank = s.notna() & (s.astype(str).str.strip() == "")
            s.loc[mask_blank] = np.nan
            X[col] = s.astype("object")
    return X


def _apply_features(df: pd.DataFrame, features: str | None) -> pd.DataFrame:
    out = _normalize_blanks_to_nan(df)
    out = _drop_fully_empty_columns(out)
    if features == "basic_credit":
        out = _add_basic_credit_features(out)
        out = _add_credit_behavior_features(out)
        out = _add_time_features(out)
    elif features == "basic_credit_plus_logs":
        out = _add_basic_credit_features(out)
        out = _add_credit_behavior_features(out)
        out = _add_time_features(out)
        out = _add_log_money_features(out)
    return _coerce_pdna_to_npnan(out)


# =========================================================
# Preprocessing
# =========================================================
def _make_num_imputer(name: str):
    if name in {"mean", "median", "most_frequent"}:
        return SimpleImputer(strategy=name)
    if name == "constant":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    if name == "knn":
        return KNNImputer(n_neighbors=5, weights="distance")
    if name == "iterative":
        return IterativeImputer(max_iter=10, random_state=CFG.random_state, initial_strategy="mean")
    raise ValueError(f"Unsupported numerical_imputer: {name}")


def _make_cat_imputer(name: str):
    if name in {"most_frequent", "mode"}:
        return SimpleImputer(strategy="most_frequent")
    if name in {"special", "__MISSING__"}:
        return SimpleImputer(strategy="constant", fill_value="__MISSING__")
    raise ValueError(f"Unsupported categorical_imputer: {name}")


def preprocess_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    numerical_imputer: str,
    categorical_imputer: str,
    features: str | None,
    scale_numeric: bool,
):
    X_tr = _apply_features(train_df, features)
    X_va = _apply_features(val_df, features)

    num_cols = get_numeric_column_names(X_tr)
    cat_cols = get_categorical_column_names(X_tr)

    num_imp = _make_num_imputer(numerical_imputer)
    if num_cols:
        num_tr = num_imp.fit_transform(X_tr[num_cols])
        num_va = num_imp.transform(X_va[num_cols])
    else:
        num_tr = np.zeros((len(X_tr), 0))
        num_va = np.zeros((len(X_va), 0))

    if scale_numeric and num_cols:
        scaler = StandardScaler()
        num_tr = scaler.fit_transform(num_tr)
        num_va = scaler.transform(num_va)

    cat_imp = _make_cat_imputer(categorical_imputer)
    if cat_cols:
        cat_tr_raw = cat_imp.fit_transform(X_tr[cat_cols])
        cat_va_raw = cat_imp.transform(X_va[cat_cols])
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        cat_tr = enc.fit_transform(cat_tr_raw).astype(np.int64) + 1
        cat_va = enc.transform(cat_va_raw).astype(np.int64) + 1
        cat_card = [int(cat_tr[:, i].max()) + 1 for i in range(cat_tr.shape[1])]
    else:
        cat_tr = np.zeros((len(X_tr), 0), dtype=np.int64)
        cat_va = np.zeros((len(X_va), 0), dtype=np.int64)
        cat_card = []

    return (
        num_tr.astype(np.float32),
        num_va.astype(np.float32),
        cat_tr,
        cat_va,
        cat_card,
    )


# =========================================================
# Numerical embeddings
# =========================================================
def build_num_embedding(
    *,
    kind: str,
    n_num_features: int,
    d_embedding: int,
    n_bins: int,
    x_num_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
):
    """Construct the embedding module, computing PLR bins on the fold's training split."""
    if kind == "none" or n_num_features == 0:
        return None
    if kind == "linear-relu":
        return rne.LinearReLUEmbeddings(n_features=n_num_features, d_embedding=d_embedding)
    if kind == "periodic":
        return rne.PeriodicEmbeddings(
            n_features=n_num_features,
            d_embedding=d_embedding,
            lite=False,
        )
    if kind == "piecewise-linear":
        x_t = torch.as_tensor(np.ascontiguousarray(x_num_train), dtype=torch.float32)
        tree_kwargs = {"min_samples_leaf": 64, "min_impurity_decrease": 1e-4}
        if task == "classification":
            y_t = torch.as_tensor(np.ascontiguousarray(y_train), dtype=torch.long)
            bins = rne.compute_bins(
                x_t, n_bins=n_bins, tree_kwargs=tree_kwargs, regression=False, y=y_t,
            )
        else:
            y_t = torch.as_tensor(np.ascontiguousarray(y_train), dtype=torch.float32)
            bins = rne.compute_bins(
                x_t, n_bins=n_bins, tree_kwargs=tree_kwargs, regression=True, y=y_t,
            )
        return rne.PiecewiseLinearEmbeddings(
            bins=bins,
            d_embedding=d_embedding,
            activation=True,
            version="B",
        )
    raise ValueError(f"Unknown num embedding kind: {kind}")


# =========================================================
# Dataset / loss / predict
# =========================================================
class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, x_num, x_cat, y=None, task="classification"):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.task = task
        if y is None:
            self.y = None
        elif task == "classification":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def tabm_classification_loss(logits, y, label_smoothing=0.0):
    b, k, c = logits.shape
    logits = logits.reshape(b * k, c)
    y_rep = y.unsqueeze(1).repeat(1, k).reshape(b * k)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, y_rep)


def tabm_regression_loss(preds, y):
    y = y.view(-1, 1, 1).expand(-1, preds.shape[1], -1)
    return nn.MSELoss()(preds, y)


@torch.no_grad()
def predict_classification(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        x_num, x_cat = batch[0].to(device), batch[1].to(device)
        logits = model(x_num, x_cat)
        out.append(torch.softmax(logits, dim=-1).mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_regression(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        x_num, x_cat = batch[0].to(device), batch[1].to(device)
        out.append(model(x_num, x_cat).mean(dim=1).squeeze(-1).cpu().numpy())
    return np.concatenate(out, axis=0)


# =========================================================
# Scheduler
# =========================================================
def make_scheduler(optimizer, *, use_cosine: bool, warmup_epochs: int, n_epochs: int):
    if not use_cosine:
        return None

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================
# Training
# =========================================================
def train_one_epoch(model, loader, optimizer, device, task, *, label_smoothing, grad_clip):
    model.train()
    total_loss, total_count = 0.0, 0
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_num, x_cat)
        if task == "classification":
            loss = tabm_classification_loss(out, y, label_smoothing=label_smoothing)
        else:
            loss = tabm_regression_loss(out, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = x_num.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


def fit_tabm(
    model, train_loader, valid_loader, y_valid, optimizer, device, task,
    *, n_epochs, patience, scheduler=None, label_smoothing=0.0, grad_clip=0.0,
):
    best_score = -np.inf
    wait = 0
    for _epoch in range(1, n_epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, device, task,
            label_smoothing=label_smoothing, grad_clip=grad_clip,
        )
        if scheduler is not None:
            scheduler.step()

        if task == "classification":
            probs = predict_classification(model, valid_loader, device)
            score = accuracy_score(y_valid, probs.argmax(axis=1))
        else:
            score = r2_score(y_valid, predict_regression(model, valid_loader, device))

        if score > best_score:
            best_score = score
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_score


# =========================================================
# Search space
# =========================================================
def suggest_params(trial: optuna.Trial, n_features: int, n_rows: int, task: str):
    if n_rows < 10000:
        epoch_low, epoch_high = 60, 180
    elif n_rows < 100000:
        epoch_low, epoch_high = 50, 140
    else:
        epoch_low, epoch_high = 40, 100

    width_choices = [128, 192, 256, 320, 384, 512]
    if n_features > 60:
        width_choices += [640, 768]

    k_choices = [8, 16, 24, 32]
    if n_features > 50:
        k_choices += [48]

    p = {
        # preprocessing
        "numerical_imputer": trial.suggest_categorical(
            "numerical_imputer", ["median", "mean", "constant", "knn"]
        ),
        "categorical_imputer": trial.suggest_categorical(
            "categorical_imputer", ["special", "most_frequent"]
        ),
        "features": trial.suggest_categorical(
            "features", [None, "basic_credit", "basic_credit_plus_logs"]
        ),
        "scale_numeric": True,

        # numerical embedding
        "num_embedding": trial.suggest_categorical(
            "num_embedding", ["none", "linear-relu", "piecewise-linear", "periodic"]
        ),
        "d_embedding": trial.suggest_categorical("d_embedding", [8, 16, 24, 32]),
        "n_bins": trial.suggest_categorical("n_bins", [16, 32, 48, 64]),

        # architecture
        "arch_type": trial.suggest_categorical("arch_type", ["tabm", "tabm-mini"]),
        "k": trial.suggest_categorical("k", k_choices),
        "n_blocks": trial.suggest_int("n_blocks", 2, 5),
        "d_block": trial.suggest_categorical("d_block", width_choices),
        "dropout": trial.suggest_float("dropout", 0.02, 0.30),

        # optimization
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "n_epochs": trial.suggest_int("n_epochs", epoch_low, epoch_high),
        "patience": trial.suggest_int("patience", 8, 25),
        "use_cosine": trial.suggest_categorical("use_cosine", [False, True]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0]),
    }
    if task == "classification":
        p["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.10)
    else:
        p["label_smoothing"] = 0.0
    return p


# =========================================================
# Single fold train+eval
# =========================================================
def train_eval_fold(p, X_tr_df, X_va_df, y_tr, y_va, task, n_classes, *, seed):
    seed_everything(seed)

    num_tr, num_va, cat_tr, cat_va, cat_card = preprocess_fold(
        X_tr_df, X_va_df,
        numerical_imputer=p["numerical_imputer"],
        categorical_imputer=p["categorical_imputer"],
        features=p["features"],
        scale_numeric=p["scale_numeric"],
    )

    train_ds = TabularDataset(num_tr, cat_tr, y_tr, task=task)
    valid_ds = TabularDataset(num_va, cat_va, y_va, task=task)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=CFG.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=p["batch_size"], shuffle=False, num_workers=CFG.num_workers
    )

    num_emb = build_num_embedding(
        kind=p["num_embedding"],
        n_num_features=num_tr.shape[1],
        d_embedding=p["d_embedding"],
        n_bins=p["n_bins"],
        x_num_train=num_tr,
        y_train=y_tr,
        task=task,
    )

    model = TabM.make(
        n_num_features=num_tr.shape[1],
        cat_cardinalities=cat_card,
        d_out=(n_classes if task == "classification" else 1),
        num_embeddings=num_emb,
        k=p["k"],
        n_blocks=p["n_blocks"],
        d_block=p["d_block"],
        dropout=p["dropout"],
        arch_type=p["arch_type"],
    ).to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = make_scheduler(
        optimizer,
        use_cosine=p["use_cosine"],
        warmup_epochs=p["warmup_epochs"],
        n_epochs=p["n_epochs"],
    )

    score = fit_tabm(
        model, train_loader, valid_loader, y_va, optimizer, CFG.device, task,
        n_epochs=p["n_epochs"], patience=p["patience"], scheduler=scheduler,
        label_smoothing=p["label_smoothing"], grad_clip=p["grad_clip"],
    )

    del model, optimizer, scheduler, num_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(score)


# =========================================================
# Objectives
# =========================================================
def classification_objective(train_df: pd.DataFrame):
    y = train_df["RiskTier"].astype(int).values
    X = train_df.drop(columns=["RiskTier", "InterestRate", "Id"], errors="ignore")
    n_rows, n_features = len(X), X.shape[1]
    n_classes = int(np.unique(y).size)

    def objective(trial: optuna.Trial):
        p = suggest_params(trial, n_features=n_features, n_rows=n_rows, task="classification")

        skf = StratifiedKFold(
            n_splits=CFG.n_splits_cls, shuffle=True, random_state=CFG.random_state,
        )
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr_df = X.iloc[tr_idx].reset_index(drop=True)
            X_va_df = X.iloc[va_idx].reset_index(drop=True)
            y_tr, y_va = y[tr_idx], y[va_idx]

            seeds = [CFG.random_state, CFG.random_state + 7] if CFG.multi_seed_eval else [CFG.random_state]
            seed_scores = [
                train_eval_fold(p, X_tr_df, X_va_df, y_tr, y_va, "classification", n_classes, seed=s)
                for s in seeds
            ]
            fold_scores.append(float(np.mean(seed_scores)))

            trial.report(float(np.mean(fold_scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def regression_objective(train_df: pd.DataFrame):
    y = train_df["InterestRate"].astype(float).values
    X = train_df.drop(columns=["RiskTier", "InterestRate", "Id"], errors="ignore")
    n_rows, n_features = len(X), X.shape[1]

    def objective(trial: optuna.Trial):
        p = suggest_params(trial, n_features=n_features, n_rows=n_rows, task="regression")

        kf = KFold(n_splits=CFG.n_splits_reg, shuffle=True, random_state=CFG.random_state)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr_df = X.iloc[tr_idx].reset_index(drop=True)
            X_va_df = X.iloc[va_idx].reset_index(drop=True)
            y_tr, y_va = y[tr_idx], y[va_idx]

            seeds = [CFG.random_state, CFG.random_state + 7] if CFG.multi_seed_eval else [CFG.random_state]
            seed_scores = [
                train_eval_fold(p, X_tr_df, X_va_df, y_tr, y_va, "regression", 0, seed=s)
                for s in seeds
            ]
            fold_scores.append(float(np.mean(seed_scores)))

            trial.report(float(np.mean(fold_scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# =========================================================
# Main
# =========================================================
def _run_study(direction, study_name, sampler, pruner, objective, n_trials, out_path):
    study = optuna.create_study(
        direction=direction, study_name=study_name, sampler=sampler, pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    payload = {**study.best_params, "_best_value": float(study.best_value)}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[{study_name}]")
    print("Best score:", study.best_value)
    print(json.dumps(payload, indent=2))
    return study


def main():
    seed_everything(CFG.random_state)
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(CFG.train_path)
    if "RiskTier" not in train_df.columns or "InterestRate" not in train_df.columns:
        raise ValueError("credit_train.csv must contain RiskTier and InterestRate")

    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=1)

    cls_study = _run_study(
        direction="maximize",
        study_name="tabm_v2_risktier_hpo",
        sampler=TPESampler(seed=CFG.random_state, multivariate=True, group=True),
        pruner=pruner,
        objective=classification_objective(train_df),
        n_trials=CFG.n_trials_cls,
        out_path=out_dir / "tabm_v2_best_params_risktier.json",
    )

    reg_study = _run_study(
        direction="maximize",
        study_name="tabm_v2_interestrate_hpo",
        sampler=TPESampler(seed=CFG.random_state + 1, multivariate=True, group=True),
        pruner=pruner,
        objective=regression_objective(train_df),
        n_trials=CFG.n_trials_reg,
        out_path=out_dir / "tabm_v2_best_params_interestrate.json",
    )

    print("\nEstimated combined CV score:",
          0.5 * cls_study.best_value + 0.5 * reg_study.best_value)


if __name__ == "__main__":
    main()
