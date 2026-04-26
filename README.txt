CreditSense Loan Risk Assessment Project

Overview
This project trains two models on the provided credit dataset:
- Task A: classify RiskTier (0-4)
- Task B: regress InterestRate (APR)

The pipeline uses shared feature engineering, compares baseline linear models against boosted-tree models, tracks validation results, and writes a Kaggle-ready submission file.

Project Layout
- train.py: main entrypoint for training and submission generation
- src/creditsense/: reusable project code
- outputs/: experiment tables and metadata
- submissions/: generated submission.csv files
- notebooks/: reserved for EDA/report notebooks

Recommended Colab Workflow
1. Upload this project folder plus the CSV files to Colab or Google Drive.
2. Open a terminal cell or notebook cell and install dependencies:
   pip install -r requirements.txt
3. Run training:
   python train.py --data-dir . --output-dir outputs --submission-dir submissions
4. Check generated artifacts:
   - outputs/experiment_results.csv
   - outputs/data_profile.json
   - outputs/run_summary.json
   - submissions/submission.csv
5. Submit submissions/submission.csv to Kaggle.

Local Workflow
1. Create a Python 3.10+ environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python train.py

Reproducibility Notes
- Fixed split seed: 1215
- Fixed holdout size: 20%
- The same holdout is used for both tasks for comparable local scoring.
- The training script chooses the best classification model by validation accuracy and the best regression model by validation R2.
- Final InterestRate predictions are clipped to [4.99, 35.99] and rounded to 2 decimals only for submission export.

Expected Outputs
- outputs/experiment_results.csv: validation metrics for all tried models
- outputs/data_profile.json: missingness and data-quality summary
- outputs/run_summary.json: final selected models and combined local score
- submissions/submission.csv: Kaggle upload file with columns Id,RiskTier,InterestRate

Notes For Report / Presentation
- Use outputs/data_profile.json plus your own plots for the data exploration section.
- Use outputs/experiment_results.csv as the base table for feature/model comparison.
- Add EDA visuals in notebooks/ or your report workflow as needed.

