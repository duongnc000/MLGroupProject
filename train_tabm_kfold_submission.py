"""
K-fold + multi-seed TabM submission.

Reads tuned params from artifacts_hpo/tabm_v2_best_params_*.json (preferred)
or falls back to the v1 files in the same dir. For each task:

  1. K-fold split of train.
  2. Per fold, fit preprocessing on the fold's train split, transform
     val and test under it. Compute PLR bins (if used) on the fold's train.
  3. Train M seed-different TabM models per fold with early stopping on
     the fold's holdout, predict on val (OOF) and test.
  4. Average all K * M test predictions; report OOF score.

Writes submission_tabm_v2.csv at repo root.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    test_path: str = "credit_test.csv"

    cls_params_paths: tuple[str, ...] = (
        "artifacts_hpo/tabm_v2_best_params_risktier.json",
        "artifacts_hpo/tabm_best_params_risktier.json",
    )
    reg_params_paths: tuple[str, ...] = (
        "artifacts_hpo/tabm_v2_best_params_interestrate.json",
        "artifacts_hpo/tabm_best_params_interestrate.json",
    )

    output_path: str = "submission_tabm_v2.csv"

    n_folds_cls: int = 5
    n_folds_reg: int = 5
    seeds: tuple[int, ...] = (1215, 42, 7)

    early_stop_patience: int = 15
    max_epochs_cap: int = 200

    random_state: int = 1215
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


CFG = Config()


# =========================================================
# Defaults that fill in for v1 param files
# =========================================================
V2_DEFAULTS = {
    "num_embedding": "none",
    "d_embedding": 16,
    "n_bins": 48,
    "arch_type": "tabm",
    "use_cosine": False,
    "warmup_epochs": 0,
    "grad_clip": 0.0,
    "label_smoothing": 0.0,
}


def load_params(paths: tuple[str, ...]) -> dict:
    for p in paths:
        path = Path(p)
        if path.exists():
            data = json.loads(path.read_text())
            data.pop("_best_value", None)
            merged = {**V2_DEFAULTS, **data}
            print(f"Loaded params from {path}")
            return merged
    raise FileNotFoundError(f"None of these param files exist: {paths}")


# =========================================================
# Reproducibility
# =========================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Column selectors / FE / preprocessing (mirrored from tune_tabm_v2.py)
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


def preprocess_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    numerical_imputer: str,
    categorical_imputer: str,
    features: str | None,
    scale_numeric: bool,
):
    """Fit preprocessing on train_df only, transform val_df and test_df."""
    X_tr = _apply_features(train_df, features)
    X_va = _apply_features(val_df, features)
    X_te = _apply_features(test_df, features)

    num_cols = get_numeric_column_names(X_tr)
    cat_cols = get_categorical_column_names(X_tr)

    # Align val/test columns with train (rare cols-on-val might exist after FE)
    for X in (X_va, X_te):
        for c in num_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in cat_cols:
            if c not in X.columns:
                X[c] = np.nan

    num_imp = _make_num_imputer(numerical_imputer)
    if num_cols:
        num_tr = num_imp.fit_transform(X_tr[num_cols])
        num_va = num_imp.transform(X_va[num_cols])
        num_te = num_imp.transform(X_te[num_cols])
    else:
        num_tr = np.zeros((len(X_tr), 0))
        num_va = np.zeros((len(X_va), 0))
        num_te = np.zeros((len(X_te), 0))

    if scale_numeric and num_cols:
        scaler = StandardScaler()
        num_tr = scaler.fit_transform(num_tr)
        num_va = scaler.transform(num_va)
        num_te = scaler.transform(num_te)

    cat_imp = _make_cat_imputer(categorical_imputer)
    if cat_cols:
        cat_tr_raw = cat_imp.fit_transform(X_tr[cat_cols])
        cat_va_raw = cat_imp.transform(X_va[cat_cols])
        cat_te_raw = cat_imp.transform(X_te[cat_cols])
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        cat_tr = enc.fit_transform(cat_tr_raw).astype(np.int64) + 1
        cat_va = enc.transform(cat_va_raw).astype(np.int64) + 1
        cat_te = enc.transform(cat_te_raw).astype(np.int64) + 1
        # Cardinality must cover test/val unknowns (encoded as 0 after the +1 shift).
        all_cat = np.vstack([cat_tr, cat_va, cat_te])
        cat_card = [int(all_cat[:, i].max()) + 1 for i in range(all_cat.shape[1])]
    else:
        cat_tr = np.zeros((len(X_tr), 0), dtype=np.int64)
        cat_va = np.zeros((len(X_va), 0), dtype=np.int64)
        cat_te = np.zeros((len(X_te), 0), dtype=np.int64)
        cat_card = []

    return (
        num_tr.astype(np.float32),
        num_va.astype(np.float32),
        num_te.astype(np.float32),
        cat_tr, cat_va, cat_te,
        cat_card,
    )


def build_num_embedding(*, kind, n_num_features, d_embedding, n_bins,
                        x_num_train, y_train, task):
    if kind == "none" or n_num_features == 0:
        return None
    if kind == "linear-relu":
        return rne.LinearReLUEmbeddings(n_features=n_num_features, d_embedding=d_embedding)
    if kind == "periodic":
        return rne.PeriodicEmbeddings(
            n_features=n_num_features, d_embedding=d_embedding, lite=False,
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
            bins=bins, d_embedding=d_embedding, activation=True, version="B",
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
        out.append(torch.softmax(model(x_num, x_cat), dim=-1).mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_regression(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        x_num, x_cat = batch[0].to(device), batch[1].to(device)
        out.append(model(x_num, x_cat).mean(dim=1).squeeze(-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def make_scheduler(optimizer, *, use_cosine, warmup_epochs, n_epochs):
    if not use_cosine:
        return None

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, device, task, *, label_smoothing, grad_clip):
    model.train()
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


def fit_with_early_stop(
    model, train_loader, valid_loader, y_valid, optimizer, device, task,
    *, n_epochs, patience, scheduler=None, label_smoothing=0.0, grad_clip=0.0,
):
    best_score = -np.inf
    best_state = None
    wait = 0
    for _epoch in range(1, n_epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, device, task,
            label_smoothing=label_smoothing, grad_clip=grad_clip,
        )
        if scheduler is not None:
            scheduler.step()
        if task == "classification":
            score = accuracy_score(y_valid, predict_classification(model, valid_loader, device).argmax(axis=1))
        else:
            score = r2_score(y_valid, predict_regression(model, valid_loader, device))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_score)


# =========================================================
# Per-task K-fold + multi-seed driver
# =========================================================
def run_kfold(
    X_train_df: pd.DataFrame,
    y: np.ndarray,
    X_test_df: pd.DataFrame,
    p: dict,
    *,
    task: str,
    n_classes: int,
    n_folds: int,
    seeds: tuple[int, ...],
):
    """Returns averaged test predictions and OOF predictions on train."""
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CFG.random_state)
        splits = list(kf.split(X_train_df, y))
        test_pred_sum = np.zeros((len(X_test_df), n_classes), dtype=np.float64)
        oof_pred = np.zeros((len(X_train_df), n_classes), dtype=np.float64)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=CFG.random_state)
        splits = list(kf.split(X_train_df))
        test_pred_sum = np.zeros((len(X_test_df),), dtype=np.float64)
        oof_pred = np.zeros((len(X_train_df),), dtype=np.float64)

    n_models_per_test_row = n_folds * len(seeds)
    fold_scores = []

    epochs = min(p["n_epochs"], CFG.max_epochs_cap)

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr_df = X_train_df.iloc[tr_idx].reset_index(drop=True)
        X_va_df = X_train_df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        num_tr, num_va, num_te, cat_tr, cat_va, cat_te, cat_card = preprocess_split(
            X_tr_df, X_va_df, X_test_df,
            numerical_imputer=p["numerical_imputer"],
            categorical_imputer=p["categorical_imputer"],
            features=p["features"],
            scale_numeric=p.get("scale_numeric", True),
        )

        seed_val_score = []
        if task == "classification":
            fold_oof_sum = np.zeros((len(X_va_df), n_classes), dtype=np.float64)
        else:
            fold_oof_sum = np.zeros((len(X_va_df),), dtype=np.float64)

        for seed in seeds:
            seed_everything(seed)

            train_ds = TabularDataset(num_tr, cat_tr, y_tr, task=task)
            valid_ds = TabularDataset(num_va, cat_va, y_va, task=task)
            test_ds = TabularDataset(num_te, cat_te, task=task)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=CFG.num_workers
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_ds, batch_size=p["batch_size"], shuffle=False, num_workers=CFG.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=p["batch_size"], shuffle=False, num_workers=CFG.num_workers
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

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
            )
            scheduler = make_scheduler(
                optimizer,
                use_cosine=p["use_cosine"],
                warmup_epochs=p["warmup_epochs"],
                n_epochs=epochs,
            )

            model, val_score = fit_with_early_stop(
                model, train_loader, valid_loader, y_va, optimizer, CFG.device, task,
                n_epochs=epochs, patience=CFG.early_stop_patience, scheduler=scheduler,
                label_smoothing=p["label_smoothing"], grad_clip=p["grad_clip"],
            )
            seed_val_score.append(val_score)

            if task == "classification":
                test_probs = predict_classification(model, test_loader, CFG.device)
                val_probs = predict_classification(model, valid_loader, CFG.device)
                test_pred_sum += test_probs
                fold_oof_sum += val_probs
            else:
                test_preds = predict_regression(model, test_loader, CFG.device)
                val_preds = predict_regression(model, valid_loader, CFG.device)
                test_pred_sum += test_preds
                fold_oof_sum += val_preds

            del model, optimizer, scheduler, num_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_val = float(np.mean(seed_val_score))
        fold_scores.append(mean_val)
        print(f"[{task}] fold {fold_idx}/{n_folds}  val={mean_val:.5f}  "
              f"seed_vals={[round(s, 5) for s in seed_val_score]}")

        oof_pred[va_idx] = fold_oof_sum / len(seeds)

    test_pred_avg = test_pred_sum / n_models_per_test_row
    print(f"[{task}] mean fold val = {np.mean(fold_scores):.5f}  "
          f"std = {np.std(fold_scores):.5f}")
    return test_pred_avg, oof_pred, fold_scores


# =========================================================
# Main
# =========================================================
def main():
    seed_everything(CFG.random_state)
    print("Using device:", CFG.device)

    train_df = pd.read_csv(CFG.train_path)
    test_df = pd.read_csv(CFG.test_path)

    if "RiskTier" not in train_df.columns or "InterestRate" not in train_df.columns:
        raise ValueError("credit_train.csv must contain RiskTier and InterestRate.")

    cls_params = load_params(CFG.cls_params_paths)
    reg_params = load_params(CFG.reg_params_paths)

    test_ids = (
        test_df["Id"].astype(int)
        if "Id" in test_df.columns
        else pd.Series(np.arange(len(test_df)), name="Id")
    )

    y_cls = train_df["RiskTier"].astype(int).values
    y_reg = train_df["InterestRate"].astype(float).values
    n_classes = int(np.unique(y_cls).size)

    X_train_df = train_df.drop(columns=["RiskTier", "InterestRate", "Id"], errors="ignore")
    X_test_df = test_df.drop(columns=["Id"], errors="ignore")

    print(f"\n=== Classification: {CFG.n_folds_cls} folds × {len(CFG.seeds)} seeds ===")
    test_probs, oof_probs, _ = run_kfold(
        X_train_df, y_cls, X_test_df, cls_params,
        task="classification", n_classes=n_classes,
        n_folds=CFG.n_folds_cls, seeds=CFG.seeds,
    )
    test_pred_cls = test_probs.argmax(axis=1).astype(int)
    oof_acc = accuracy_score(y_cls, oof_probs.argmax(axis=1))
    print(f"[classification] OOF accuracy = {oof_acc:.5f}")

    print(f"\n=== Regression: {CFG.n_folds_reg} folds × {len(CFG.seeds)} seeds ===")
    test_pred_reg, oof_reg, _ = run_kfold(
        X_train_df, y_reg, X_test_df, reg_params,
        task="regression", n_classes=0,
        n_folds=CFG.n_folds_reg, seeds=CFG.seeds,
    )
    oof_r2 = r2_score(y_reg, oof_reg)
    print(f"[regression] OOF R² = {oof_r2:.5f}")

    test_pred_reg = np.clip(test_pred_reg, 4.99, 35.99)
    test_pred_reg = np.round(test_pred_reg, 2)

    submission = pd.DataFrame({
        "Id": test_ids.astype(int),
        "RiskTier": test_pred_cls,
        "InterestRate": test_pred_reg,
    })
    submission.to_csv(CFG.output_path, index=False)

    print(f"\nSaved submission to: {CFG.output_path}")
    print(submission.head())
    print(f"\nLocal heuristic combined score = {0.5 * oof_acc + 0.5 * oof_r2:.5f}")


if __name__ == "__main__":
    main()
