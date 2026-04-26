#!/usr/bin/env python3
"""Train CreditSense models and generate Kaggle submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT, help="Directory containing credit_train.csv and credit_test.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for experiment artifacts")
    parser.add_argument("--submission-dir", type=Path, default=PROJECT_ROOT / "submissions", help="Directory for submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from creditsense.training import train_all_models
    except ImportError as exc:
        print(
            "Missing dependency while importing the training pipeline. "
            "Install requirements with `pip install -r requirements.txt` and rerun.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    result = train_all_models(data_dir=args.data_dir, output_dir=args.output_dir, submission_dir=args.submission_dir)
    best_classification = result["best_classification"]
    best_regression = result["best_regression"]
    combined_local_score = 0.5 * best_classification["validation_metric"] + 0.5 * best_regression["validation_metric"]
    print(
        json.dumps(
            {
                "submission_path": str(result["submission_path"]),
                "experiment_results_path": str(result["experiment_results_path"]),
                "best_classification_model": best_classification["name"],
                "validation_accuracy": round(best_classification["validation_metric"], 6),
                "best_regression_model": best_regression["name"],
                "validation_r2": round(best_regression["validation_metric"], 6),
                "combined_local_score": round(combined_local_score, 6),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
