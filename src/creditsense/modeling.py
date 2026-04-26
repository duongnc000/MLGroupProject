"""Model construction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MAX_INTEREST_RATE, MIN_INTEREST_RATE


@dataclass
class ModelResult:
    name: str
    feature_set: str
    estimator: Any
    validation_metric: float
    train_metric: float
    task: str
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_probabilities: np.ndarray | None
    test_probabilities: np.ndarray | None
    details: dict[str, Any]


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in df.columns if column not in categorical]
    return numeric, categorical


def build_linear_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = split_feature_types(df)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_tree_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = split_feature_types(df)
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def try_build_xgboost_classifier(random_state: int) -> Any | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def try_build_xgboost_regressor(random_state: int) -> Any | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def try_build_lightgbm_classifier(random_state: int) -> Any | None:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    return LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def try_build_lightgbm_regressor(random_state: int) -> Any | None:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None

    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.04,
        max_depth=-1,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def try_build_catboost_classifier(random_state: int) -> Any | None:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=4.0,
        random_seed=random_state,
        verbose=False,
    )


def try_build_catboost_regressor(random_state: int) -> Any | None:
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=4.0,
        random_seed=random_state,
        verbose=False,
    )


def train_classification_model(
    name: str,
    feature_set: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
) -> ModelResult:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    train_predictions = pipeline.predict(x_train)
    valid_predictions = pipeline.predict(x_valid)
    test_predictions = pipeline.predict(x_test)
    valid_probabilities = pipeline.predict_proba(x_valid) if hasattr(pipeline, "predict_proba") else None
    test_probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
    return ModelResult(
        name=name,
        feature_set=feature_set,
        estimator=pipeline,
        validation_metric=float(accuracy_score(y_valid, valid_predictions)),
        train_metric=float(accuracy_score(y_train, train_predictions)),
        task="classification",
        validation_predictions=np.asarray(valid_predictions, dtype=int),
        test_predictions=np.asarray(test_predictions, dtype=int),
        validation_probabilities=None if valid_probabilities is None else np.asarray(valid_probabilities, dtype=float),
        test_probabilities=None if test_probabilities is None else np.asarray(test_probabilities, dtype=float),
        details={"confusion_matrix": confusion_matrix(y_valid, valid_predictions).tolist()},
    )


def train_regression_model(
    name: str,
    feature_set: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
) -> ModelResult:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    train_predictions = pipeline.predict(x_train)
    valid_predictions = pipeline.predict(x_valid)
    test_predictions = pipeline.predict(x_test)
    return ModelResult(
        name=name,
        feature_set=feature_set,
        estimator=pipeline,
        validation_metric=float(r2_score(y_valid, valid_predictions)),
        train_metric=float(r2_score(y_train, train_predictions)),
        task="regression",
        validation_predictions=np.asarray(valid_predictions, dtype=float),
        test_predictions=np.asarray(test_predictions, dtype=float),
        validation_probabilities=None,
        test_probabilities=None,
        details={
            "prediction_min": float(np.min(valid_predictions)),
            "prediction_max": float(np.max(valid_predictions)),
        },
    )


def build_classification_candidates(
    feature_frame: pd.DataFrame, random_state: int, feature_set: str
) -> list[tuple[str, str, ColumnTransformer, Any]]:
    candidates = [
        (
            "logistic_regression",
            feature_set,
            build_linear_preprocessor(feature_frame),
            LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    ]

    xgb = try_build_xgboost_classifier(random_state)
    if xgb is not None:
        candidates.append(("xgboost_classifier", feature_set, build_tree_preprocessor(feature_frame), xgb))
    lgbm = try_build_lightgbm_classifier(random_state)
    if lgbm is not None:
        candidates.append(("lightgbm_classifier", feature_set, build_tree_preprocessor(feature_frame), lgbm))
    catboost = try_build_catboost_classifier(random_state)
    if catboost is not None:
        candidates.append(("catboost_classifier", feature_set, build_tree_preprocessor(feature_frame), catboost))
    if xgb is None and lgbm is None and catboost is None:
        candidates.append(
            (
                "hist_gradient_boosting_classifier",
                feature_set,
                build_tree_preprocessor(feature_frame),
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=350,
                    min_samples_leaf=30,
                    random_state=random_state,
                ),
            )
        )

    return candidates


def build_regression_candidates(
    feature_frame: pd.DataFrame, random_state: int, feature_set: str
) -> list[tuple[str, str, ColumnTransformer, Any]]:
    candidates = [
        (
            "elastic_net",
            feature_set,
            build_linear_preprocessor(feature_frame),
            ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=5000, random_state=random_state),
        )
    ]

    xgb = try_build_xgboost_regressor(random_state)
    if xgb is not None:
        candidates.append(("xgboost_regressor", feature_set, build_tree_preprocessor(feature_frame), xgb))
    lgbm = try_build_lightgbm_regressor(random_state)
    if lgbm is not None:
        candidates.append(("lightgbm_regressor", feature_set, build_tree_preprocessor(feature_frame), lgbm))
    catboost = try_build_catboost_regressor(random_state)
    if catboost is not None:
        candidates.append(("catboost_regressor", feature_set, build_tree_preprocessor(feature_frame), catboost))
    if xgb is None and lgbm is None and catboost is None:
        candidates.append(
            (
                "hist_gradient_boosting_regressor",
                feature_set,
                build_tree_preprocessor(feature_frame),
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=500,
                    min_samples_leaf=30,
                    random_state=random_state,
                ),
            )
        )

    return candidates


def _generate_weight_combinations(count: int, step: float = 0.1) -> list[tuple[float, ...]]:
    units = int(round(1 / step))
    combos = []
    for raw_weights in product(range(units + 1), repeat=count):
        if sum(raw_weights) == units:
            combos.append(tuple(weight / units for weight in raw_weights))
    return combos


def maybe_build_regression_ensemble(results: list[ModelResult], y_valid: pd.Series) -> ModelResult | None:
    eligible = sorted(results, key=lambda result: result.validation_metric, reverse=True)[:3]
    if len(eligible) < 2:
        return None

    best_metric = float("-inf")
    best_weights: tuple[float, ...] | None = None
    best_valid = None
    best_test = None
    for weights in _generate_weight_combinations(len(eligible), step=0.1):
        if max(weights) == 1.0:
            continue
        combined_valid = np.zeros_like(eligible[0].validation_predictions, dtype=float)
        combined_test = np.zeros_like(eligible[0].test_predictions, dtype=float)
        for weight, result in zip(weights, eligible):
            combined_valid += weight * result.validation_predictions
            combined_test += weight * result.test_predictions
        metric = float(r2_score(y_valid, combined_valid))
        if metric > best_metric:
            best_metric = metric
            best_weights = weights
            best_valid = combined_valid
            best_test = combined_test

    if best_weights is None or best_valid is None or best_test is None:
        return None

    return ModelResult(
        name="tuned_weighted_regression_ensemble",
        feature_set="engineered_v1",
        estimator=None,
        validation_metric=best_metric,
        train_metric=float("nan"),
        task="regression",
        validation_predictions=best_valid,
        test_predictions=best_test,
        validation_probabilities=None,
        test_probabilities=None,
        details={
            "members": [result.name for result in eligible],
            "weights": list(best_weights),
        },
    )


def maybe_build_classification_ensemble(results: list[ModelResult], y_valid: pd.Series) -> ModelResult | None:
    eligible = [result for result in sorted(results, key=lambda result: result.validation_metric, reverse=True) if result.validation_probabilities is not None][:3]
    if len(eligible) < 2:
        return None

    best_metric = float("-inf")
    best_weights: tuple[float, ...] | None = None
    best_valid_probs = None
    best_test_probs = None
    best_valid_predictions = None
    best_test_predictions = None
    for weights in _generate_weight_combinations(len(eligible), step=0.1):
        if max(weights) == 1.0:
            continue
        combined_valid_probs = np.zeros_like(eligible[0].validation_probabilities, dtype=float)
        combined_test_probs = np.zeros_like(eligible[0].test_probabilities, dtype=float)
        for weight, result in zip(weights, eligible):
            combined_valid_probs += weight * result.validation_probabilities
            combined_test_probs += weight * result.test_probabilities
        valid_predictions = np.argmax(combined_valid_probs, axis=1).astype(int)
        test_predictions = np.argmax(combined_test_probs, axis=1).astype(int)
        metric = float(accuracy_score(y_valid, valid_predictions))
        if metric > best_metric:
            best_metric = metric
            best_weights = weights
            best_valid_probs = combined_valid_probs
            best_test_probs = combined_test_probs
            best_valid_predictions = valid_predictions
            best_test_predictions = test_predictions

    if (
        best_weights is None
        or best_valid_probs is None
        or best_test_probs is None
        or best_valid_predictions is None
        or best_test_predictions is None
    ):
        return None

    return ModelResult(
        name="tuned_weighted_classification_ensemble",
        feature_set="engineered_v1",
        estimator=None,
        validation_metric=best_metric,
        train_metric=float("nan"),
        task="classification",
        validation_predictions=best_valid_predictions,
        test_predictions=best_test_predictions,
        validation_probabilities=best_valid_probs,
        test_probabilities=best_test_probs,
        details={
            "members": [result.name for result in eligible],
            "weights": list(best_weights),
        },
    )


def clip_interest_rates(values: np.ndarray) -> np.ndarray:
    return np.clip(values, MIN_INTEREST_RATE, MAX_INTEREST_RATE)
