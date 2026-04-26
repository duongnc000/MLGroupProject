"""End-to-end training entrypoint helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUBMISSION_DIR,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_CLASS,
    TARGET_REGRESSION,
    TEST_SIZE,
)
from .features import add_engineered_features, build_data_profile, save_json
from .modeling import (
    ModelResult,
    build_classification_candidates,
    build_regression_candidates,
    clip_interest_rates,
    maybe_build_classification_ensemble,
    maybe_build_regression_ensemble,
    train_classification_model,
    train_regression_model,
)


def load_datasets(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "credit_train.csv")
    test_df = pd.read_csv(data_dir / "credit_test.csv")
    return train_df, test_df


def validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    expected_test_columns = [column for column in train_df.columns if column not in [TARGET_CLASS, TARGET_REGRESSION]]
    if list(test_df.columns) != expected_test_columns:
        raise ValueError("Train/test feature schemas do not match exactly.")


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_features = train_df.drop(columns=[TARGET_CLASS, TARGET_REGRESSION]).copy()
    test_features = test_df.copy()
    train_engineered = add_engineered_features(train_features)
    test_engineered = add_engineered_features(test_features)
    profile = {
        "raw_train": build_data_profile(train_features),
        "engineered_train": build_data_profile(train_engineered),
    }
    return train_features, test_features, train_engineered, test_engineered, profile


def result_to_record(result: ModelResult) -> dict:
    metric_key = "validation_accuracy" if result.task == "classification" else "validation_r2"
    train_metric_key = "train_accuracy" if result.task == "classification" else "train_r2"
    record = {
        "task": result.task,
        "model": result.name,
        "feature_set": result.feature_set,
        metric_key: result.validation_metric,
        train_metric_key: result.train_metric,
        "overfit_gap": result.train_metric - result.validation_metric if np.isfinite(result.train_metric) else np.nan,
        "details": result.details,
    }
    return record


def train_all_models(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    submission_dir: Path = DEFAULT_SUBMISSION_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_datasets(data_dir)
    validate_schema(train_df, test_df)

    x_raw_all, x_raw_test, x_all, x_test, profile = prepare_features(train_df, test_df)
    y_class = train_df[TARGET_CLASS].astype(int)
    y_reg = train_df[TARGET_REGRESSION].astype(float)

    x_train_raw, x_valid_raw, y_class_train, y_class_valid, y_reg_train, y_reg_valid = train_test_split(
        x_raw_all,
        y_class,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    x_train = x_all.loc[x_train_raw.index].copy()
    x_valid = x_all.loc[x_valid_raw.index].copy()

    classification_results = []
    baseline_class_candidates = build_classification_candidates(x_train_raw, RANDOM_STATE, feature_set="raw_baseline")
    baseline_class_name, baseline_class_feature_set, baseline_class_preprocessor, baseline_class_estimator = baseline_class_candidates[0]
    classification_results.append(
        train_classification_model(
            name=baseline_class_name,
            feature_set=baseline_class_feature_set,
            estimator=baseline_class_estimator,
            preprocessor=baseline_class_preprocessor,
            x_train=x_train_raw,
            y_train=y_class_train,
            x_valid=x_valid_raw,
            y_valid=y_class_valid,
            x_test=x_raw_test,
        )
    )
    for name, feature_set, preprocessor, estimator in build_classification_candidates(
        x_train, RANDOM_STATE, feature_set="engineered_v1"
    )[1:]:
        classification_results.append(
            train_classification_model(
                name=name,
                feature_set=feature_set,
                estimator=estimator,
                preprocessor=preprocessor,
                x_train=x_train,
                y_train=y_class_train,
                x_valid=x_valid,
                y_valid=y_class_valid,
                x_test=x_test,
            )
        )
    classification_ensemble = maybe_build_classification_ensemble(classification_results, y_class_valid)
    if classification_ensemble is not None:
        classification_results.append(classification_ensemble)

    regression_results = []
    baseline_reg_candidates = build_regression_candidates(x_train_raw, RANDOM_STATE, feature_set="raw_baseline")
    baseline_reg_name, baseline_reg_feature_set, baseline_reg_preprocessor, baseline_reg_estimator = baseline_reg_candidates[0]
    regression_results.append(
        train_regression_model(
            name=baseline_reg_name,
            feature_set=baseline_reg_feature_set,
            estimator=baseline_reg_estimator,
            preprocessor=baseline_reg_preprocessor,
            x_train=x_train_raw,
            y_train=y_reg_train,
            x_valid=x_valid_raw,
            y_valid=y_reg_valid,
            x_test=x_raw_test,
        )
    )
    for name, feature_set, preprocessor, estimator in build_regression_candidates(
        x_train, RANDOM_STATE, feature_set="engineered_v1"
    )[1:]:
        regression_results.append(
            train_regression_model(
                name=name,
                feature_set=feature_set,
                estimator=estimator,
                preprocessor=preprocessor,
                x_train=x_train,
                y_train=y_reg_train,
                x_valid=x_valid,
                y_valid=y_reg_valid,
                x_test=x_test,
            )
        )
    regression_ensemble = maybe_build_regression_ensemble(regression_results, y_reg_valid)
    if regression_ensemble is not None:
        regression_results.append(regression_ensemble)

    best_classification = max(classification_results, key=lambda result: result.validation_metric)
    best_regression = max(regression_results, key=lambda result: result.validation_metric)
    best_interest_predictions = clip_interest_rates(best_regression.test_predictions)

    submission = pd.DataFrame(
        {
            ID_COLUMN: np.arange(len(x_test), dtype=int),
            TARGET_CLASS: best_classification.test_predictions.astype(int),
            TARGET_REGRESSION: np.round(best_interest_predictions, 2),
        }
    )
    submission_path = submission_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    experiment_rows = [result_to_record(result) for result in classification_results + regression_results]
    experiments_df = pd.DataFrame(experiment_rows)
    best_class_metric = best_classification.validation_metric
    best_reg_metric = best_regression.validation_metric
    experiments_df["selected_best_model"] = False
    experiments_df.loc[
        (experiments_df["task"] == "classification") & (experiments_df["model"] == best_classification.name), "selected_best_model"
    ] = True
    experiments_df.loc[
        (experiments_df["task"] == "regression") & (experiments_df["model"] == best_regression.name), "selected_best_model"
    ] = True
    experiments_df["best_combined_local_score"] = 0.5 * best_class_metric + 0.5 * best_reg_metric
    experiments_df.to_csv(output_dir / "experiment_results.csv", index=False)

    save_json(profile, output_dir / "data_profile.json")
    save_json(
        {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "best_classification_model": best_classification.name,
            "best_classification_validation_accuracy": best_classification.validation_metric,
            "best_regression_model": best_regression.name,
            "best_regression_validation_r2": best_regression.validation_metric,
            "combined_local_score": 0.5 * best_class_metric + 0.5 * best_reg_metric,
            "submission_path": str(submission_path),
        },
        output_dir / "run_summary.json",
    )

    return {
        "submission_path": submission_path,
        "best_classification": asdict(best_classification),
        "best_regression": asdict(best_regression),
        "experiment_results_path": output_dir / "experiment_results.csv",
    }
