"""Feature engineering and data profiling helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MEANINGFUL_MISSING_COLUMNS


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy(deep=True)

    for column in MEANINGFUL_MISSING_COLUMNS:
        if column in engineered.columns:
            engineered.loc[:, f"{column}_missing_flag"] = engineered[column].isna().astype(int)

    numeric_fill_zero = [
        "MortgageOutstandingBalance",
        "AutoLoanOutstandingBalance",
        "StudentLoanOutstandingBalance",
        "NumberOfLatePayments30Days",
        "NumberOfLatePayments60Days",
        "NumberOfLatePayments90Days",
        "NumberOfChargeOffs",
        "NumberOfCollections",
        "NumberOfBankruptcies",
        "NumberOfHardInquiries12Mo",
        "NumberOfHardInquiries24Mo",
    ]
    zero_filled = {}
    for column in numeric_fill_zero:
        if column in engineered.columns:
            zero_filled[column] = engineered[column].fillna(0)

    if {"MortgageOutstandingBalance", "AutoLoanOutstandingBalance", "StudentLoanOutstandingBalance"} <= set(engineered.columns):
        engineered.loc[:, "total_outstanding_debt"] = (
            zero_filled["MortgageOutstandingBalance"]
            + zero_filled["AutoLoanOutstandingBalance"]
            + zero_filled["StudentLoanOutstandingBalance"]
        )
    if {"total_outstanding_debt", "TotalAssets"} <= set(engineered.columns):
        engineered.loc[:, "debt_to_assets"] = safe_divide(engineered["total_outstanding_debt"], engineered["TotalAssets"])
    if {"TotalCreditLimit", "RevolvingUtilizationRate"} <= set(engineered.columns):
        available_credit = engineered["TotalCreditLimit"] * (1 - engineered["RevolvingUtilizationRate"].fillna(0))
        engineered.loc[:, "unused_credit_ratio"] = safe_divide(available_credit, engineered["TotalCreditLimit"])
    if {
        "NumberOfLatePayments30Days",
        "NumberOfLatePayments60Days",
        "NumberOfLatePayments90Days",
    } <= set(engineered.columns):
        engineered.loc[:, "total_late_payments"] = (
            zero_filled["NumberOfLatePayments30Days"]
            + zero_filled["NumberOfLatePayments60Days"]
            + zero_filled["NumberOfLatePayments90Days"]
        )
    if {
        "NumberOfLatePayments90Days",
        "NumberOfBankruptcies",
        "NumberOfChargeOffs",
        "NumberOfCollections",
    } <= set(engineered.columns):
        engineered.loc[:, "severe_delinquency_flag"] = (
            (
                zero_filled["NumberOfLatePayments90Days"] > 0
            )
            | (zero_filled["NumberOfBankruptcies"] > 0)
            | (zero_filled["NumberOfChargeOffs"] > 0)
            | (zero_filled["NumberOfCollections"] > 0)
        ).astype(int)
    if {"NumberOfHardInquiries12Mo", "NumberOfHardInquiries24Mo"} <= set(engineered.columns):
        engineered.loc[:, "hard_inquiry_intensity"] = (
            zero_filled["NumberOfHardInquiries12Mo"] + 0.5 * zero_filled["NumberOfHardInquiries24Mo"]
        )
    if {"TotalMonthlyIncome", "NumberOfDependents"} <= set(engineered.columns):
        engineered.loc[:, "income_per_dependent"] = safe_divide(
            engineered["TotalMonthlyIncome"], engineered["NumberOfDependents"].fillna(0) + 1
        )
    if {"RequestedLoanAmount", "TotalAssets"} <= set(engineered.columns):
        engineered.loc[:, "loan_to_assets"] = safe_divide(engineered["RequestedLoanAmount"], engineered["TotalAssets"])
    if {"MonthlyPaymentEstimate", "TotalMonthlyIncome"} <= set(engineered.columns):
        engineered.loc[:, "requested_payment_burden"] = safe_divide(
            engineered["MonthlyPaymentEstimate"], engineered["TotalMonthlyIncome"]
        )
    if {"EmploymentLengthYears", "YearsAtCurrentEmployer"} <= set(engineered.columns):
        engineered.loc[:, "employment_stability_gap"] = (
            engineered["EmploymentLengthYears"] - engineered["YearsAtCurrentEmployer"]
        )
    if {"AverageAccountAgeMonths", "OldestAccountAgeMonths"} <= set(engineered.columns):
        engineered.loc[:, "credit_age_ratio"] = safe_divide(
            engineered["AverageAccountAgeMonths"], engineered["OldestAccountAgeMonths"]
        )
    if {"HomeOwnership", "PropertyValue"} <= set(engineered.columns):
        engineered.loc[:, "homeowner_missing_property_flag"] = (
            engineered["HomeOwnership"].fillna("Missing").isin(["Own", "Mortgage"])
            & engineered["PropertyValue"].isna()
        ).astype(int)
    if "StudentLoanOutstandingBalance" in engineered.columns:
        engineered.loc[:, "student_debt_present_flag"] = (
            engineered["StudentLoanOutstandingBalance"].fillna(0).gt(0).astype(int)
        )
    if "CollateralValue" in engineered.columns:
        engineered.loc[:, "collateral_present_flag"] = engineered["CollateralValue"].fillna(0).gt(0).astype(int)
    if "RevolvingUtilizationRate" in engineered.columns:
        engineered.loc[:, "utilization_band"] = pd.cut(
            engineered["RevolvingUtilizationRate"],
            bins=[-np.inf, 0.1, 0.3, 0.6, 0.9, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype("object")
    if "DebtToIncomeRatio" in engineered.columns:
        engineered.loc[:, "dti_band"] = pd.cut(
            engineered["DebtToIncomeRatio"],
            bins=[-np.inf, 0.15, 0.3, 0.45, 0.6, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype("object")
    if "AnnualIncome" in engineered.columns:
        engineered.loc[:, "income_band"] = pd.qcut(
            engineered["AnnualIncome"].rank(method="first"),
            q=5,
            labels=["very_low", "low", "mid", "high", "very_high"],
        ).astype("object")
    if "RequestedLoanAmount" in engineered.columns:
        engineered.loc[:, "loan_size_band"] = pd.qcut(
            engineered["RequestedLoanAmount"].rank(method="first"),
            q=5,
            labels=["very_low", "low", "mid", "high", "very_high"],
        ).astype("object")

    return engineered


def build_data_profile(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]

    suspicious_ratios = {}
    for column in ["RevolvingUtilizationRate", "DebtToIncomeRatio", "LoanToIncomeRatio", "PaymentToIncomeRatio"]:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            suspicious_ratios[column] = {
                "lt_0": int((series < 0).sum()),
                "gt_1": int((series > 1).sum()),
                "missing": int(series.isna().sum()),
            }

    negative_checks = {}
    for column in ["AnnualIncome", "MonthlyGrossIncome", "TotalMonthlyIncome", "SavingsBalance", "CheckingBalance"]:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            negative_checks[column] = int((series < 0).sum())

    income_gap = None
    if {"AnnualIncome", "TotalMonthlyIncome"} <= set(df.columns):
        annualized_monthly = pd.to_numeric(df["TotalMonthlyIncome"], errors="coerce") * 12
        annual_income = pd.to_numeric(df["AnnualIncome"], errors="coerce")
        relative_gap = (annualized_monthly - annual_income).abs() / annual_income.replace(0, np.nan)
        income_gap = {
            "rows_gt_25pct_gap": int((relative_gap > 0.25).sum()),
            "median_relative_gap": float(relative_gap.median(skipna=True)),
        }

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_counts": {k: int(v) for k, v in df.isna().sum().sort_values(ascending=False).items()},
        "suspicious_ratios": suspicious_ratios,
        "negative_value_counts": negative_checks,
        "income_consistency": income_gap,
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
