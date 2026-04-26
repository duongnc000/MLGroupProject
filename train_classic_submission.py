from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pandas.api.types import (
    is_numeric_dtype,
    is_string_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_extension_array_dtype,
    is_integer_dtype,
)

from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

SEED = 42
TRAIN_PATH = Path("credit_train.csv")
TEST_PATH = Path("credit_test.csv")
SUBMISSION_PATH = Path("submission_classic_3.csv")


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
# Cleaning helpers
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

    if {"HomeOwnership", "PropertyValue"}.issubset(X.columns):
        X["FE_HasPropertyValue"] = X["PropertyValue"].notna().astype(int)

    if {"Age", "StudentLoanOutstandingBalance"}.issubset(X.columns):
        X["FE_HasStudentLoan"] = X["StudentLoanOutstandingBalance"].notna().astype(int)

    return X


# =========================================================
# Preprocessing
# =========================================================
def preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_imputer: str,
    categorical_imputer: str,
    is_catboost: bool,
    features: str = "credit_expert",
):
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
    X_test = pipe.transform(test_df)

    cols = pipe.named_steps["preprocess"].get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)

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
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            elif col in cat_cols:
                X_train[col] = X_train[col].astype("string").fillna("__MISSING__")
                X_test[col] = X_test[col].astype("string").fillna("__MISSING__")

    return X_train, X_test


def cat_feature_indices(df: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(df.columns) if str(df[c].dtype) in ("string", "object", "category")]


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "Id" in test_df.columns:
        test_ids = test_df["Id"].astype(int)
    else:
        test_ids = pd.Series(np.arange(len(test_df)), name="Id")

    y_cls = train_df["RiskTier"].astype(int)
    y_reg = train_df["InterestRate"].astype(float)

    X_train_raw = train_df.drop(columns=["RiskTier", "InterestRate"], errors="ignore").copy()
    X_test_raw = test_df.copy()

    X_train_raw = X_train_raw.drop(columns=["Id"], errors="ignore")
    X_test_raw = X_test_raw.drop(columns=["Id"], errors="ignore")

    # Classification preprocessing
    X_cls_train, X_cls_test = preprocessing(
        train_df=X_train_raw,
        test_df=X_test_raw,
        numerical_imputer="mean",
        categorical_imputer="most_frequent",
        is_catboost=True,
        features="credit_expert",
    )

    # Regression preprocessing
    X_reg_train, X_reg_test = preprocessing(
        train_df=X_train_raw,
        test_df=X_test_raw,
        numerical_imputer="mean",
        categorical_imputer="special",
        is_catboost=True,
        features="credit_expert",
    )

    cat_idx_cls = cat_feature_indices(X_cls_train)
    cat_idx_reg = cat_feature_indices(X_reg_train)

    clf = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2812,
        depth=6,
        learning_rate=0.03662928494042786,
        l2_leaf_reg=5.525816417960988,
        random_strength=2.37414761418501,
        bagging_temperature=0.40811313140481587,
        border_count=220,
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
    )

    reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3800,
        depth=6,
        learning_rate=0.01673125446804648,
        l2_leaf_reg=1.4014395435401745,
        random_strength=1.114459109165391,
        bagging_temperature=0.6371505534654706,
        border_count=183,
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
    )

    clf.fit(X_cls_train, y_cls, cat_features=cat_idx_cls, verbose=False)
    reg.fit(X_reg_train, y_reg, cat_features=cat_idx_reg, verbose=False)

    pred_cls = clf.predict(X_cls_test).ravel().astype(int)
    pred_reg = reg.predict(X_reg_test).astype(float)

    pred_reg = np.clip(pred_reg, 4.99, 35.99)
    pred_reg = np.round(pred_reg, 2)

    submission = pd.DataFrame({
        "Id": test_ids,
        "RiskTier": pred_cls,
        "InterestRate": pred_reg,
    })

    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved: {SUBMISSION_PATH.resolve()}")
    print(submission.head())


if __name__ == "__main__":
    main()