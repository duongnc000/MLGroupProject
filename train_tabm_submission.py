from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from pandas.api.types import (
    is_numeric_dtype,
    is_string_dtype,
    is_bool_dtype,
    # is_categorical_dtype,
)

from tabm import TabM


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    train_path: str = "credit_train.csv"
    test_path: str = "credit_test.csv"

    cls_params_path: str = "artifacts_hpo/tabm_best_params_risktier.json"
    reg_params_path: str = "artifacts_hpo/tabm_best_params_interestrate.json"

    output_path: str = "submission_tabm_rerun.csv"

    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


CFG = Config()


# =========================================================
# Reproducibility
# =========================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Column selectors
# =========================================================
def get_numeric_column_names(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if is_numeric_dtype(df[col])]


def get_categorical_column_names(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if (
            is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or is_bool_dtype(df[col])
            or df[col].dtype == "object"
        )
    ]


# =========================================================
# Feature engineering wrapper
# =========================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, funcs=None):
        self.funcs = funcs

    def fit(self, X, y=None):
        self.funcs_ = list(self.funcs) if self.funcs is not None else []
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineer expects a pandas DataFrame.")
        X = X.copy()
        for fn in self.funcs_:
            try:
                X = fn(X)
            except Exception as e:
                print(f"[FeatureEngineer] Skipped {getattr(fn, '__name__', fn)}: {e}")
        return X


# =========================================================
# Small disciplined feature bundles
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
    existing_debt_cols = [c for c in debt_cols if c in X.columns]
    if existing_debt_cols:
        X["TotalDebtApprox"] = sum(
            pd.to_numeric(X[c], errors="coerce").fillna(0)
            for c in existing_debt_cols
        )

    if {"RequestedLoanAmount", "AnnualIncome"}.issubset(X.columns):
        X["RequestedToAnnualIncome"] = safe_div(
            X["RequestedLoanAmount"], X["AnnualIncome"]
        )

    if {"MonthlyPaymentEstimate", "TotalMonthlyIncome"}.issubset(X.columns):
        X["PaymentToIncomeRatio_FE"] = safe_div(
            X["MonthlyPaymentEstimate"], X["TotalMonthlyIncome"]
        )

    if {"TotalDebtApprox", "AnnualIncome"}.issubset(X.columns):
        X["TotalDebtToIncome"] = safe_div(
            X["TotalDebtApprox"], X["AnnualIncome"]
        )

    if {"CollateralValue", "RequestedLoanAmount"}.issubset(X.columns):
        X["CollateralCoverage"] = safe_div(
            X["CollateralValue"], X["RequestedLoanAmount"]
        )

    return X


def _add_credit_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    late30 = (
        pd.to_numeric(X["NumberOfLatePayments30Days"], errors="coerce").fillna(0)
        if "NumberOfLatePayments30Days" in X.columns else 0
    )
    late60 = (
        pd.to_numeric(X["NumberOfLatePayments60Days"], errors="coerce").fillna(0)
        if "NumberOfLatePayments60Days" in X.columns else 0
    )
    late90 = (
        pd.to_numeric(X["NumberOfLatePayments90Days"], errors="coerce").fillna(0)
        if "NumberOfLatePayments90Days" in X.columns else 0
    )

    X["TotalLatePayments"] = late30 + late60 + late90
    X["WeightedLatePayments"] = late30 + 2 * late60 + 3 * late90

    bk = (
        pd.to_numeric(X["NumberOfBankruptcies"], errors="coerce").fillna(0)
        if "NumberOfBankruptcies" in X.columns else 0
    )
    pr = (
        pd.to_numeric(X["NumberOfPublicRecords"], errors="coerce").fillna(0)
        if "NumberOfPublicRecords" in X.columns else 0
    )
    co = (
        pd.to_numeric(X["NumberOfCollections"], errors="coerce").fillna(0)
        if "NumberOfCollections" in X.columns else 0
    )
    ch = (
        pd.to_numeric(X["NumberOfChargeOffs"], errors="coerce").fillna(0)
        if "NumberOfChargeOffs" in X.columns else 0
    )

    X["AnyDerogatoryFlag"] = ((bk + pr + co + ch) > 0).astype(int)
    return X


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if "CreditHistoryLengthMonths" in X.columns:
        X["CreditHistoryYears"] = (
            pd.to_numeric(X["CreditHistoryLengthMonths"], errors="coerce") / 12.0
        )
    return X


def _add_log_money_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    log_cols = [
        "AnnualIncome",
        "MonthlyGrossIncome",
        "TotalMonthlyIncome",
        "SavingsBalance",
        "CheckingBalance",
        "TotalAssets",
        "RequestedLoanAmount",
        "CollateralValue",
        "MonthlyPaymentEstimate",
    ]
    for col in log_cols:
        if col in X.columns:
            vals = pd.to_numeric(X[col], errors="coerce")
            X[f"log1p_{col}"] = np.log1p(np.clip(vals, 0, None))
    return X

def _coerce_pdna_to_npnan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas nullable missing values (pd.NA) into plain np.nan,
    especially in categorical/object columns, so sklearn imputers work.
    """
    X = df.copy()

    # Replace pd.NA globally first
    X = X.replace({pd.NA: np.nan})

    for col in X.columns:
        dt = X[col].dtype

        # String-like / object / categorical / boolean-like columns
        if (
            dt == "object"
            or is_string_dtype(X[col])
            or isinstance(dt, pd.CategoricalDtype)
            or str(dt).lower().startswith("boolean")
        ):
            s = X[col].astype("object")
            s = pd.Series(s, index=X.index)
            mask_blank = s.notna() & (s.astype(str).str.strip() == "")
            s.loc[mask_blank] = np.nan
            s = s.where(~s.isna(), np.nan)
            X[col] = s.astype("object")

    return X

# =========================================================
# Preprocessing for TabM
# =========================================================
def preprocessing_tabm_fit_transform(
    train_df: pd.DataFrame,
    *,
    numerical_imputer: str = "median",
    categorical_imputer: str = "special",
    features: str | None = None,
    scale_numeric: bool = True,
):
    feature_funcs = [_normalize_blanks_to_nan, _drop_fully_empty_columns]

    if features == "basic_credit":
        feature_funcs += [
            _add_basic_credit_features,
            _add_credit_behavior_features,
            _add_time_features,
        ]
    elif features == "basic_credit_plus_logs":
        feature_funcs += [
            _add_basic_credit_features,
            _add_credit_behavior_features,
            _add_time_features,
            _add_log_money_features,
        ]

    fe_pipe = Pipeline([
        ("feature_engineering", FeatureEngineer(funcs=feature_funcs)),
    ])

    X_train_fe = fe_pipe.fit_transform(train_df.copy())
    X_train_fe = _coerce_pdna_to_npnan(X_train_fe)

    num_cols = get_numeric_column_names(X_train_fe)
    cat_cols = get_categorical_column_names(X_train_fe)

    if numerical_imputer in {"mean", "median", "most_frequent"}:
        num_imputer = SimpleImputer(strategy=numerical_imputer)
    elif numerical_imputer == "constant":
        num_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    elif numerical_imputer == "knn":
        num_imputer = KNNImputer(n_neighbors=5, weights="distance")
    elif numerical_imputer == "iterative":
        num_imputer = IterativeImputer(max_iter=10, random_state=CFG.random_state, initial_strategy="mean")
    else:
        raise ValueError(f"Unsupported numerical_imputer: {numerical_imputer}")

    num_train = num_imputer.fit_transform(X_train_fe[num_cols]) if num_cols else np.zeros((len(X_train_fe), 0))

    scaler = None
    if scale_numeric and num_cols:
        scaler = StandardScaler()
        num_train = scaler.fit_transform(num_train)

    if categorical_imputer in {"most_frequent", "mode"}:
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif categorical_imputer in {"special", "__MISSING__"}:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")
    else:
        raise ValueError(f"Unsupported categorical_imputer: {categorical_imputer}")

    if cat_cols:
        cat_train_raw = cat_imputer.fit_transform(X_train_fe[cat_cols])

        ord_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        cat_train = ord_encoder.fit_transform(cat_train_raw).astype(np.int64) + 1
        cat_cardinalities = [int(cat_train[:, i].max()) + 1 for i in range(cat_train.shape[1])]
    else:
        cat_train = np.zeros((len(X_train_fe), 0), dtype=np.int64)
        cat_cardinalities = []

    meta = {
        "feature_pipe": fe_pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "scaler": scaler,
        "cat_imputer": cat_imputer,
        "ord_encoder": ord_encoder if cat_cols else None,
        "cat_cardinalities": cat_cardinalities,
    }

    return num_train.astype(np.float32), cat_train, meta


def preprocessing_tabm_transform(
    test_df: pd.DataFrame,
    meta: dict,
):
    X_test_fe = meta["feature_pipe"].transform(test_df.copy())
    X_test_fe = _coerce_pdna_to_npnan(X_test_fe)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    num_test = meta["num_imputer"].transform(X_test_fe[num_cols]) if num_cols else np.zeros((len(X_test_fe), 0))
    if meta["scaler"] is not None and num_cols:
        num_test = meta["scaler"].transform(num_test)

    if cat_cols:
        cat_test_raw = meta["cat_imputer"].transform(X_test_fe[cat_cols])
        cat_test = meta["ord_encoder"].transform(cat_test_raw).astype(np.int64) + 1
    else:
        cat_test = np.zeros((len(X_test_fe), 0), dtype=np.int64)

    return num_test.astype(np.float32), cat_test


# =========================================================
# Dataset
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


# =========================================================
# TabM train / infer
# =========================================================
def tabm_classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, k, c = logits.shape
    logits = logits.reshape(b * k, c)
    y_rep = y.unsqueeze(1).repeat(1, k).reshape(b * k)
    return nn.CrossEntropyLoss()(logits, y_rep)


def tabm_regression_loss(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.view(-1, 1, 1).expand(-1, preds.shape[1], -1)
    return nn.MSELoss()(preds, y)


@torch.no_grad()
def predict_classification(model, loader, device):
    model.eval()
    probs_all = []
    for batch in loader:
        if len(batch) == 3:
            x_num, x_cat, _ = batch
        else:
            x_num, x_cat = batch

        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        logits = model(x_num, x_cat)
        probs = torch.softmax(logits, dim=-1).mean(dim=1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


@torch.no_grad()
def predict_regression(model, loader, device):
    model.eval()
    preds_all = []
    for batch in loader:
        if len(batch) == 3:
            x_num, x_cat, _ = batch
        else:
            x_num, x_cat = batch

        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        preds = model(x_num, x_cat).mean(dim=1).squeeze(-1)
        preds_all.append(preds.cpu().numpy())
    return np.concatenate(preds_all, axis=0)


def train_one_epoch(model, loader, optimizer, device, task):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x_num, x_cat)

        if task == "classification":
            loss = tabm_classification_loss(out, y)
        else:
            loss = tabm_regression_loss(out, y)

        loss.backward()
        optimizer.step()

        bs = x_num.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def fit_tabm_full(
    model,
    train_loader,
    optimizer,
    device,
    task,
    n_epochs,
):
    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, task)
        print(f"[{task}] epoch={epoch:03d}/{n_epochs:03d} loss={loss:.6f}")
    return model


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

    with open(CFG.cls_params_path, "r") as f:
        cls_params = json.load(f)

    with open(CFG.reg_params_path, "r") as f:
        reg_params = json.load(f)

    test_ids = test_df["Id"].astype(int) if "Id" in test_df.columns else pd.Series(np.arange(len(test_df)), name="Id")

    y_cls = train_df["RiskTier"].astype(int).values
    y_reg = train_df["InterestRate"].astype(float).values

    X_train_df = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")
    X_test_df = test_df.copy()

    X_train_df = X_train_df.drop(columns=["Id"], errors="ignore")
    X_test_df = X_test_df.drop(columns=["Id"], errors="ignore")

    # -------------------------
    # Classification pipeline
    # -------------------------
    X_cls_num, X_cls_cat, cls_meta = preprocessing_tabm_fit_transform(
        X_train_df,
        numerical_imputer=cls_params["numerical_imputer"],
        categorical_imputer=cls_params["categorical_imputer"],
        features=cls_params["features"],
        scale_numeric=cls_params["scale_numeric"],
    )
    X_test_cls_num, X_test_cls_cat = preprocessing_tabm_transform(X_test_df, cls_meta)

    cls_train_ds = TabularDataset(X_cls_num, X_cls_cat, y_cls, task="classification")
    cls_test_ds = TabularDataset(X_test_cls_num, X_test_cls_cat, y=None)

    cls_train_loader = torch.utils.data.DataLoader(
        cls_train_ds,
        batch_size=cls_params["batch_size"],
        shuffle=True,
        num_workers=CFG.num_workers,
    )
    cls_test_loader = torch.utils.data.DataLoader(
        cls_test_ds,
        batch_size=cls_params["batch_size"],
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    n_classes = len(np.unique(y_cls))

    clf = TabM.make(
        n_num_features=X_cls_num.shape[1],
        cat_cardinalities=cls_meta["cat_cardinalities"],
        d_out=n_classes,
        k=cls_params["k"],
        n_blocks=cls_params["n_blocks"],
        d_block=cls_params["d_block"],
        dropout=cls_params["dropout"],
        arch_type="tabm",
    ).to(CFG.device)

    opt_clf = torch.optim.AdamW(
        clf.parameters(),
        lr=cls_params["lr"],
        weight_decay=cls_params["weight_decay"],
    )

    clf = fit_tabm_full(
        model=clf,
        train_loader=cls_train_loader,
        optimizer=opt_clf,
        device=CFG.device,
        task="classification",
        n_epochs=cls_params["n_epochs"],
    )

    test_probs = predict_classification(clf, cls_test_loader, CFG.device)
    test_pred_cls = test_probs.argmax(axis=1).astype(int)

    # -------------------------
    # Regression pipeline
    # -------------------------
    X_reg_num, X_reg_cat, reg_meta = preprocessing_tabm_fit_transform(
        X_train_df,
        numerical_imputer=reg_params["numerical_imputer"],
        categorical_imputer=reg_params["categorical_imputer"],
        features=reg_params["features"],
        scale_numeric=reg_params["scale_numeric"],
    )
    X_test_reg_num, X_test_reg_cat = preprocessing_tabm_transform(X_test_df, reg_meta)

    reg_train_ds = TabularDataset(X_reg_num, X_reg_cat, y_reg, task="regression")
    reg_test_ds = TabularDataset(X_test_reg_num, X_test_reg_cat, y=None)

    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_ds,
        batch_size=reg_params["batch_size"],
        shuffle=True,
        num_workers=CFG.num_workers,
    )
    reg_test_loader = torch.utils.data.DataLoader(
        reg_test_ds,
        batch_size=reg_params["batch_size"],
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    reg = TabM.make(
        n_num_features=X_reg_num.shape[1],
        cat_cardinalities=reg_meta["cat_cardinalities"],
        d_out=1,
        k=reg_params["k"],
        n_blocks=reg_params["n_blocks"],
        d_block=reg_params["d_block"],
        dropout=reg_params["dropout"],
        arch_type="tabm",
    ).to(CFG.device)

    opt_reg = torch.optim.AdamW(
        reg.parameters(),
        lr=reg_params["lr"],
        weight_decay=reg_params["weight_decay"],
    )

    reg = fit_tabm_full(
        model=reg,
        train_loader=reg_train_loader,
        optimizer=opt_reg,
        device=CFG.device,
        task="regression",
        n_epochs=reg_params["n_epochs"],
    )

    test_pred_reg = predict_regression(reg, reg_test_loader, CFG.device)
    test_pred_reg = np.clip(test_pred_reg, 4.99, 35.99)
    test_pred_reg = np.round(test_pred_reg, 2)

    # -------------------------
    # Submission
    # -------------------------
    submission = pd.DataFrame({
        "Id": test_ids.astype(int),
        "RiskTier": test_pred_cls,
        "InterestRate": test_pred_reg,
    })

    submission.to_csv(CFG.output_path, index=False)

    print(f"\nSaved submission to: {CFG.output_path}")
    print(submission.head())


if __name__ == "__main__":
    main()