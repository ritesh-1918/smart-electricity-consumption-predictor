# ⚡ Smart Electricity Consumption Predictor

[![License: MIT](https://img.shields.Format/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.Format/badge/Python-3.8%2B-blue)](requirements.txt)
[![Streamlit App](https://img.shields.Format/badge/Streamlit-App-brightgreen)](app/app.py)
[![R2 Accuracy](https://img.shields.Format/badge/Model%20Accuracy%20(R2)-94.95%25-neon)](src/train.py)

> **"Demystifying Machine Learning Predictions, One Equation at a Time."**

An end-to-end, production-ready Explainable AI (XAI) pipeline and interactive dashboard that predicts daily household electricity consumption using **Linear Regression**. Built explicitly as a teaching baseline for engineering students and workshops, it transforms a "black-box" model into a visual, live mathematical solver.

---

## 📈 Model Performance & Parameters

The model was trained on 3,000 household samples, achieving an **$R^2$ accuracy score of 94.95%** on testing data. 

### Model Coefficients (Weights)
*   **Intercept (Baseline Constant)**: `0.4721`
*   **Occupancy**: `+8.0050` *(Most influential feature)*
*   **Day_Type**: `+5.1341`
*   **AC_Hours**: `+4.4470`
*   **Appliance_Hours**: `+2.1748`
*   **Temperature**: `+1.8166`
*   **Humidity**: `+0.2958` *(Least influential feature)*

---

## 📂 Project Architecture

```text
smart-electricity-prediction/
├── app/
│   └── app.py                  # Streamlit Web dashboard & live solver
├── dataset/
│   ├── processed/              # Cleaned splits (X_train, y_train, etc.)
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
├── requirements.txt            # Package dependency definitions
└── LICENSE                     # MIT License
```

For a detailed file guide, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure Python 3.8 or higher is installed.

### 2. Install Dependencies
Clone the repository and install packages:
```bash
pip install -r requirements.txt
```

### 3. Preprocess Dataset & Split
Run the cleaning pipeline to segment raw data:
```bash
python src/data_preprocessing.py
```

### 4. Train Model
Fit weights and generate evaluation graphs:
```bash
python src/train.py
```

### 5. Run Web Dashboard
Serve the interactive app locally:
```bash
python -m streamlit run app/app.py
```
Open `http://localhost:8501` to view your dashboard.

---

## 📱 User Interface Highlights

*   **Live Parameter Simulator**: Real-time sliders adjusting climate parameters and appliance runtimes.
*   **Explainable AI Solver**: Prints the live algebra calculations on screen as sliders are dragged.
*   **Live Feature Contributions**: A dynamic bar plot showing feature contributions (`Weight * Input Value`) updating instantly.
*   **Workshop Syllabus Sidebar**: Built-in terminology cheat-sheets explaining Supervised Learning and Regression.

---

## 🚀 Future Roadmap
- [ ] Add Ridge & Lasso regularization support for teaching.
- [ ] Add comparative metrics tab (Linear Regression vs. Decision Trees).
- [ ] Integrate database storage to save prediction logs.

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
