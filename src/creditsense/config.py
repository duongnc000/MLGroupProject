"""Project-wide configuration."""

from pathlib import Path

RANDOM_STATE = 1215
TEST_SIZE = 0.2
TARGET_CLASS = "RiskTier"
TARGET_REGRESSION = "InterestRate"
MIN_INTEREST_RATE = 4.99
MAX_INTEREST_RATE = 35.99
ID_COLUMN = "Id"

MEANINGFUL_MISSING_COLUMNS = [
    "PropertyValue",
    "MortgageOutstandingBalance",
    "StudentLoanOutstandingBalance",
    "CollateralValue",
    "SecondaryMonthlyIncome",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SUBMISSION_DIR = PROJECT_ROOT / "submissions"

