# Project Architecture & File Structure

This document details the modular layout of the **Smart Electricity Consumption Predictor** codebase.

```text
smart-electricity-prediction/
├── app/
│   └── app.py                  # Streamlit Web application UI & solver interface
├── dataset/
│   ├── processed/              # Preprocessed splits (X_train, y_train, etc.)
│   └── electricity_consumption_3000.csv  # Raw dataset
├── models/
│   └── linear_regression_model.pkl       # Saved trained model weight binary
├── notebooks/
│   └── 1.0-eda-and-modeling.ipynb        # Jupyter notebook detailing EDA
├── reports/
│   ├── figures/                # EDA graphs & distributions
│   ├── model_results/          # Evaluation plots (Actual vs Predicted)
│   └── branding_assets.md      # Marketing descriptions & pitch assets
├── src/
│   ├── __init__.py             # Identifies src as a package
│   ├── data_preprocessing.py   # Modular cleaning and train-test splits
│   └── train.py                # Model fitting, evaluation, and saving
├── .gitignore                  # Git untracked pattern filters
├── CHANGELOG.md                # Project release record
├── CONTRIBUTING.md             # Code contribution guidelines
├── LICENSE                     # MIT License details
├── PROJECT_STRUCTURE.md        # File guide (this file)
├── README.md                   # Repository landing homepage
└── requirements.txt            # Package dependency definitions
```
