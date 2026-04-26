from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pandas.api.types import (
    is_numeric_dtype,
    is_string_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_extension_array_dtype,
    is_integer_dtype,
)

warnings.filterwarnings("ignore")

SEED = 42
TRAIN_PATH = "credit_train.csv"
TEST_PATH = "credit_test.csv"


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
            or is_categorical_dtype(df[col])
            or is_bool_dtype(df[col])
            or df[col].dtype == "object"
        )
    ]


# =========================================================
# Safe feature engineering wrapper
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
# Cleaning helpers copied/adapted from your old code
# =========================================================
def _normalize_blanks_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for col in get_categorical_column_names(X):
        s_str = X[col].astype("string")
        mask_blank = s_str.notna() & (s_str.str.strip() == "")
        s_str = s_str.mask(mask_blank, np.nan)
        s_str = s_str.where(~s_str.isna(), np.nan)
        X[col] = s_str.astype("object")
    return X


class CoerceToNumpyNa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.replace({pd.NA: np.nan})

        for col in X.columns:
            dt = X[col].dtype

            if is_string_dtype(dt):
                s = X[col].astype("object")
                mask_blank = pd.notna(s) & (pd.Series(s).astype(str).str.strip() == "")
                s[mask_blank] = np.nan
                X[col] = s

            elif str(dt).lower().startswith("boolean"):
                s = X[col].astype("object")
                s = s.where(pd.notna(s), np.nan)
                X[col] = s

            elif is_extension_array_dtype(dt) and is_integer_dtype(dt):
                X[col] = X[col].astype("float")

        return X


def _drop_fully_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    to_drop = [c for c in X.columns if pd.isna(X[c]).all()]
    return X.drop(columns=to_drop) if to_drop else X


# =========================================================
# Credit-specific feature engineering
# =========================================================
def safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a.astype(float) / (b.astype(float) + eps)


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # Ratios aligned with your feature groups
    if {"RequestedLoanAmount", "AnnualIncome"}.issubset(X.columns):
        X["FE_LoanToIncome"] = safe_ratio(X["RequestedLoanAmount"], X["AnnualIncome"])

    if {"MonthlyPaymentEstimate", "TotalMonthlyIncome"}.issubset(X.columns):
        X["FE_PaymentToIncome"] = safe_ratio(X["MonthlyPaymentEstimate"], X["TotalMonthlyIncome"])

    if {"TotalAssets", "RequestedLoanAmount"}.issubset(X.columns):
        X["FE_AssetsToLoan"] = safe_ratio(X["TotalAssets"], X["RequestedLoanAmount"])

    if {"SavingsBalance", "RequestedLoanAmount"}.issubset(X.columns):
        X["FE_SavingsToLoan"] = safe_ratio(X["SavingsBalance"], X["RequestedLoanAmount"])

    if {"PropertyValue", "MortgageOutstandingBalance"}.issubset(X.columns):
        X["FE_MortgageToProperty"] = safe_ratio(X["MortgageOutstandingBalance"], X["PropertyValue"])

    if {"CreditHistoryLengthMonths", "Age"}.issubset(X.columns):
        X["FE_CreditHistoryPerAge"] = safe_ratio(X["CreditHistoryLengthMonths"], X["Age"])

    if {"NumberOfLatePayments30Days", "NumberOfLatePayments60Days", "NumberOfLatePayments90Days"}.issubset(X.columns):
        X["FE_TotalLatePayments"] = (
            X["NumberOfLatePayments30Days"].fillna(0)
            + X["NumberOfLatePayments60Days"].fillna(0)
            + X["NumberOfLatePayments90Days"].fillna(0)
        )

    if {"NumberOfChargeOffs", "NumberOfCollections", "NumberOfBankruptcies"}.issubset(X.columns):
        X["FE_SevereDerogatoryCount"] = (
            X["NumberOfChargeOffs"].fillna(0)
            + X["NumberOfCollections"].fillna(0)
            + X["NumberOfBankruptcies"].fillna(0)
        )

    if {"NumberOfOpenAccounts", "NumberOfCreditCards"}.issubset(X.columns):
        X["FE_CreditCardsShare"] = safe_ratio(X["NumberOfCreditCards"], X["NumberOfOpenAccounts"])

    # Log transforms for skewed money features
    log_cols = [
        "AnnualIncome",
        "MonthlyGrossIncome",
        "SecondaryMonthlyIncome",
        "TotalMonthlyIncome",
        "SavingsBalance",
        "CheckingBalance",
        "InvestmentPortfolioValue",
        "PropertyValue",
        "VehicleValue",
        "TotalAssets",
        "MortgageOutstandingBalance",
        "AutoLoanOutstandingBalance",
        "StudentLoanOutstandingBalance",
        "TotalCreditLimit",
        "RequestedLoanAmount",
        "CollateralValue",
        "MonthlyPaymentEstimate",
    ]
    for col in log_cols:
        if col in X.columns:
            X[f"FE_log1p_{col}"] = np.log1p(np.clip(pd.to_numeric(X[col], errors="coerce"), 0, None))

    # Interactions with missingness meaning
    if {"HomeOwnership", "PropertyValue"}.issubset(X.columns):
        X["FE_HasPropertyValue"] = X["PropertyValue"].notna().astype(int)

    if {"Age", "StudentLoanOutstandingBalance"}.issubset(X.columns):
        X["FE_HasStudentLoan"] = X["StudentLoanOutstandingBalance"].notna().astype(int)

    return X


# =========================================================
# Preprocessing function adapted from your old file
# =========================================================
def preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numerical_imputer: str,
    categorical_imputer: str,
    is_catboost: bool,
    features: str = "base",
):
    # Imputer selection
    if numerical_imputer in {"mean", "median", "most_frequent"}:
        num_imputer = SimpleImputer(strategy=numerical_imputer)
    elif numerical_imputer == "constant":
        num_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    elif numerical_imputer == "knn":
        num_imputer = KNNImputer(weights="distance", n_neighbors=5)
    elif numerical_imputer == "iterative":
        num_imputer = IterativeImputer(max_iter=10, random_state=SEED, initial_strategy="mean")
    else:
        raise ValueError(f"Unsupported numerical_imputer: {numerical_imputer}")

    numeric_pipeline = Pipeline([
        ("impute", num_imputer),
    ])

    cat_steps = []
    if categorical_imputer in {"most_frequent", "mode"}:
        cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    elif categorical_imputer in {"special", "__MISSING__"}:
        cat_steps.append(("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")))
    else:
        raise ValueError(f"Unsupported categorical_imputer: {categorical_imputer}")

    if not is_catboost:
        try:
            cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)))
        except TypeError:
            cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)))

    categorical_pipeline = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, get_numeric_column_names),
            ("cat", categorical_pipeline, get_categorical_column_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    feature_funcs = [_normalize_blanks_to_nan, _drop_fully_empty_columns]
    if features == "credit_expert":
        feature_funcs.append(add_credit_features)

    pipe = Pipeline([
        ("feature_engineering", FeatureEngineer(funcs=feature_funcs)),
        ("coerce_to_numpy_na", CoerceToNumpyNa()),
        ("preprocess", preprocessor),
    ])

    X_train = pipe.fit_transform(train_df)
    X_val = pipe.transform(val_df)

    cols = pipe.named_steps["preprocess"].get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=cols)
    X_val = pd.DataFrame(X_val, columns=cols)

    if is_catboost:
        ct = pipe.named_steps["preprocess"]
        fe = pipe.named_steps["feature_engineering"]

        def _resolve_cols(sel, X_in):
            return list(sel(X_in)) if callable(sel) else list(sel)

        X_fe_schema = fe.transform(train_df.copy())
        num_cols = set(_resolve_cols(ct.transformers_[0][2], X_fe_schema))
        cat_cols = set(_resolve_cols(ct.transformers_[1][2], X_fe_schema))

        for col in X_train.columns:
            if col in num_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
            elif col in cat_cols:
                X_train[col] = X_train[col].astype("string").fillna("__MISSING__")
                X_val[col] = X_val[col].astype("string").fillna("__MISSING__")

    return X_train, X_val


# =========================================================
# Helpers
# =========================================================
def cat_feature_indices(df: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(df.columns) if str(df[c].dtype) in ("string", "object", "category")]


def make_stratified_folds(y: np.ndarray, n_splits: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def make_regression_folds(n_samples: int, n_splits: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n_samples)))


# =========================================================
# Model runners
# =========================================================
def run_xgb_classifier_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        early_stopping_rounds=200,
    )

    y = train_df["RiskTier"].astype(int).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_stratified_folds(y, n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        pred = model.predict(X_va)
        acc = accuracy_score(y_va, pred)
        scores.append(acc)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_lgbm_classifier_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from lightgbm import LGBMClassifier, early_stopping

    params = dict(
        objective="multiclass",
        num_class=5,
        n_estimators=trial.suggest_int("n_estimators", 300, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        verbosity=-1,
        random_state=SEED,
    )

    y = train_df["RiskTier"].astype(int).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_stratified_folds(y, n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=200, verbose=False)],
        )

        pred = model.predict(X_va)
        acc = accuracy_score(y_va, pred)
        scores.append(acc)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_catboost_classifier_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from catboost import CatBoostClassifier

    params = dict(
        loss_function="MultiClass",
        iterations=trial.suggest_int("iterations", 500, 4000),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        border_count=trial.suggest_int("border_count", 64, 254),
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
    )

    y = train_df["RiskTier"].astype(int).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_stratified_folds(y, n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=True,
            features=features,
        )

        cat_idx = cat_feature_indices(X_tr)

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=False,
        )

        pred = model.predict(X_va).ravel().astype(int)
        acc = accuracy_score(y_va, pred)
        scores.append(acc)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_hgb_classifier_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_iter=trial.suggest_int("max_iter", 200, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 0.0, 3.0),
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=trial.suggest_int("n_iter_no_change", 20, 200),
        random_state=SEED,
    )

    y = train_df["RiskTier"].astype(int).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_stratified_folds(y, n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)
        acc = accuracy_score(y_va, pred)
        scores.append(acc)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_xgb_regressor_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from xgboost import XGBRegressor

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 4000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        early_stopping_rounds=200,
    )

    y = train_df["InterestRate"].astype(float).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_regression_folds(len(train_df), n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        pred = model.predict(X_va)
        score = r2_score(y_va, pred)
        scores.append(score)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_lgbm_regressor_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from lightgbm import LGBMRegressor, early_stopping

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 4000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        verbosity=-1,
        random_state=SEED,
    )

    y = train_df["InterestRate"].astype(float).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_regression_folds(len(train_df), n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=200, verbose=False)],
        )

        pred = model.predict(X_va)
        score = r2_score(y_va, pred)
        scores.append(score)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_catboost_regressor_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    from catboost import CatBoostRegressor

    params = dict(
        loss_function="RMSE",
        iterations=trial.suggest_int("iterations", 500, 4000),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        border_count=trial.suggest_int("border_count", 64, 254),
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
    )

    y = train_df["InterestRate"].astype(float).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_regression_folds(len(train_df), n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=True,
            features=features,
        )

        cat_idx = cat_feature_indices(X_tr)

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=False,
        )

        pred = model.predict(X_va)
        score = r2_score(y_va, pred)
        scores.append(score)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_hgb_regressor_cv(trial, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_iter=trial.suggest_int("max_iter", 200, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 0.0, 3.0),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=trial.suggest_int("n_iter_no_change", 20, 200),
        random_state=SEED,
    )

    y = train_df["InterestRate"].astype(float).values
    X_all = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore")

    folds = make_regression_folds(len(train_df), n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr_raw = X_all.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_all.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr, X_va = preprocessing(
            X_tr_raw, X_va_raw,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            is_catboost=False,
            features=features,
        )

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)
        score = r2_score(y_va, pred)
        scores.append(score)

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


CLASSIFICATION_RUNNERS = {
    "xgb": run_xgb_classifier_cv,
    "lgbm": run_lgbm_classifier_cv,
    "catboost": run_catboost_classifier_cv,
    "hgb": run_hgb_classifier_cv,
}

REGRESSION_RUNNERS = {
    "xgb": run_xgb_regressor_cv,
    "lgbm": run_lgbm_regressor_cv,
    "catboost": run_catboost_regressor_cv,
    "hgb": run_hgb_regressor_cv,
}


# =========================================================
# Imputer search
# =========================================================
def evaluate_fixed_config(task, model_name, train_df, n_splits, numerical_imputer, categorical_imputer, features):
    dummy_trial = optuna.trial.FixedTrial({})

    if task == "classification":
        if model_name == "xgb":
            return run_xgb_classifier_cv(
                optuna.trial.FixedTrial({
                    "n_estimators": 1200,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "min_child_weight": 2,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "lgbm":
            return run_lgbm_classifier_cv(
                optuna.trial.FixedTrial({
                    "n_estimators": 1200,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "min_child_samples": 30,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "catboost":
            return run_catboost_classifier_cv(
                optuna.trial.FixedTrial({
                    "iterations": 2000,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 5.0,
                    "random_strength": 1.0,
                    "bagging_temperature": 1.0,
                    "border_count": 128,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "hgb":
            return run_hgb_classifier_cv(
                optuna.trial.FixedTrial({
                    "learning_rate": 0.05,
                    "max_iter": 600,
                    "max_depth": 8,
                    "min_samples_leaf": 30,
                    "l2_regularization": 0.1,
                    "n_iter_no_change": 30,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )

    if task == "regression":
        if model_name == "xgb":
            return run_xgb_regressor_cv(
                optuna.trial.FixedTrial({
                    "n_estimators": 1500,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 2,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "lgbm":
            return run_lgbm_regressor_cv(
                optuna.trial.FixedTrial({
                    "n_estimators": 1500,
                    "learning_rate": 0.03,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "min_child_samples": 30,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "catboost":
            return run_catboost_regressor_cv(
                optuna.trial.FixedTrial({
                    "iterations": 2500,
                    "depth": 6,
                    "learning_rate": 0.04,
                    "l2_leaf_reg": 5.0,
                    "random_strength": 1.0,
                    "bagging_temperature": 1.0,
                    "border_count": 128,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )
        if model_name == "hgb":
            return run_hgb_regressor_cv(
                optuna.trial.FixedTrial({
                    "learning_rate": 0.05,
                    "max_iter": 700,
                    "max_depth": 8,
                    "min_samples_leaf": 30,
                    "l2_regularization": 0.1,
                    "n_iter_no_change": 30,
                }),
                train_df, n_splits, numerical_imputer, categorical_imputer, features
            )

    raise ValueError(f"Unsupported task/model: {task}/{model_name}")


def search_imputers(train_df, task="classification", model_name="catboost", n_splits=5, features="credit_expert"):
    num_imputers = ["median", "mean", "knn", "iterative"]
    cat_imputers = ["most_frequent", "special"]

    rows = []
    for num_imp in num_imputers:
        for cat_imp in cat_imputers:
            score = evaluate_fixed_config(
                task=task,
                model_name=model_name,
                train_df=train_df,
                n_splits=n_splits,
                numerical_imputer=num_imp,
                categorical_imputer=cat_imp,
                features=features,
            )
            rows.append({
                "task": task,
                "model": model_name,
                "numerical_imputer": num_imp,
                "categorical_imputer": cat_imp,
                "features": features,
                "cv_score": score,
            })
            print(rows[-1])

    return pd.DataFrame(rows).sort_values("cv_score", ascending=False).reset_index(drop=True)


# =========================================================
# Optuna study
# =========================================================
def tune_model(train_df, task, model_name, n_trials, n_splits, numerical_imputer, categorical_imputer, features):
    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=max(5, n_trials // 10))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    if task == "classification":
        runner = CLASSIFICATION_RUNNERS[model_name]
    else:
        runner = REGRESSION_RUNNERS[model_name]

    def objective(trial):
        return runner(
            trial=trial,
            train_df=train_df,
            n_splits=n_splits,
            numerical_imputer=numerical_imputer,
            categorical_imputer=categorical_imputer,
            features=features,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# =========================================================
# Example main
# =========================================================
if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)

    # print("=== Stage 1: imputer search for RiskTier with CatBoost ===")
    # imp_cls = search_imputers(
    #     train_df,
    #     task="classification",
    #     model_name="catboost",
    #     n_splits=5,
    #     features="credit_expert",
    # )
    # print(imp_cls.head())

    # best_num_cls = imp_cls.iloc[0]["numerical_imputer"]
    # best_cat_cls = imp_cls.iloc[0]["categorical_imputer"]

    print("=== Stage 2: tune classification model ===")
    study_cls = tune_model(
        train_df=train_df,
        task="classification",
        model_name="catboost",
        n_trials=40,
        n_splits=5,
        numerical_imputer="mean",
        categorical_imputer="most_frequent",
        features="credit_expert",
    )
    print("Best classification score:", study_cls.best_value)
    print("Best classification params:", study_cls.best_params)

    print("=== Stage 1: imputer search for InterestRate with CatBoost ===")
    # imp_reg = search_imputers(
    #     train_df,
    #     task="regression",
    #     model_name="catboost",
    #     n_splits=5,
    #     features="credit_expert",
    # )
    # print(imp_reg.head())

    # best_num_reg = imp_reg.iloc[0]["numerical_imputer"]
    # best_cat_reg = imp_reg.iloc[0]["categorical_imputer"]

    print("=== Stage 2: tune regression model ===")
    study_reg = tune_model(
        train_df=train_df,
        task="regression",
        model_name="catboost",
        n_trials=40,
        n_splits=5,
        numerical_imputer="mean",
        categorical_imputer="special",
        features="credit_expert",
    )
    print("Best regression score:", study_reg.best_value)
    print("Best regression params:", study_reg.best_params)

    Path("artifacts").mkdir(exist_ok=True)
    # imp_cls.to_csv("artifacts/imputer_search_classification.csv", index=False)
    # imp_reg.to_csv("artifacts/imputer_search_regression.csv", index=False)

    with open("artifacts/best_params_classification.json", "w") as f:
        json.dump(study_cls.best_params, f, indent=2)

    with open("artifacts/best_params_regression.json", "w") as f:
        json.dump(study_reg.best_params, f, indent=2)