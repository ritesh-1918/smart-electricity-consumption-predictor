---
title: Volterra Energy Analytics
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ Volterra: Intelligent Energy Analytics Engine

[![License: MIT](https://img.shields.Format/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.Format/badge/Python-3.8%2B-blue)](requirements.txt)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ritesh19180/smart-electricity-consumption-predictor)
[![R2 Accuracy](https://img.shields.Format/badge/Model%20Accuracy%20(R2)-94.95%25-neon)](src/train.py)

> **"Demystifying Power Demands, Optimizing Consumption Footprints."**

Volterra is a production-ready, containerized energy forecasting and load-attribution engine. It uses a Linear Regression model with an **$R^2$ accuracy score of 94.95%** to predict daily electricity consumption (kWh), calculate financial operating costs, forecast carbon footprint impact, and trace real-time feature attribution impacts.

---

## 📈 Platform Metrics & Coefficients

The core predictive engine operates on weights optimized over historical load configurations:

*   **Intercept (Baseline Constant)**: `0.4721`
*   **Occupancy Coefficient**: `+8.0050` *(Primary driver)*
*   **Weekend Profile Offset**: `+5.1341`
*   **AC Operating Hour Coefficient**: `+4.4470`
*   **Appliance Operating Hour Coefficient**: `+2.1748`
*   **Temperature Coefficient**: `+1.8166`
*   **Humidity Coefficient**: `+0.2958` *(Baseline driver)*

---

## 📂 System Architecture

```text
smart-electricity-prediction/
├── app/
│   └── app.py                  # Volterra Streamlit analytics dashboard
├── dataset/
│   ├── processed/              # Preprocessed data splits
│   └── electricity_consumption_3000.csv  # Raw load telemetry data
├── models/
│   └── linear_regression_model.pkl       # Serialized predictive weights
├── notebooks/
│   └── 1.0-eda-and-modeling.ipynb        # Data exploration and statistics
├── reports/
│   ├── figures/                # Visual distributions and correlations
│   ├── model_results/          # Regression diagnostics plots
│   └── portfolio_upgrade_audit.md # Portfolio alignment audit logs
├── src/
│   ├── __init__.py             # Package descriptor
│   ├── data_preprocessing.py   # Preprocessing and partition pipeline
│   └── train.py                # Regression training and diagnostics
├── requirements.txt            # Package specifications
└── LICENSE                     # MIT License
```

---

## 🛠️ Deploying & Running Volterra Locally

### 1. Prerequisite Packages
Ensure Python 3.8+ is installed on your local host.

### 2. Dependency Setup
Install required system packages:
```bash
pip install -r requirements.txt
```

### 3. Initialize Preprocessing
Run the telemetry partitioning pipeline to format arrays:
```bash
python src/data_preprocessing.py
```

### 4. Train Prediction Engine
Optimize feature attribution weights and write diagnostic reports:
```bash
python src/train.py
```

### 5. Launch Local Dashboard
Serve the interactive analytics panel locally:
```bash
python -m streamlit run app/app.py
```
Open `http://localhost:8501` to access the simulator dashboard.

---

## 🌐 Live Production Deployment

Volterra is compiled and deployed live in production on **Hugging Face Spaces**:
*   **Active Application URL**: [huggingface.co/spaces/ritesh19180/smart-electricity-consumption-predictor](https://huggingface.co/spaces/ritesh19180/smart-electricity-consumption-predictor)

---

## 📄 License
This codebase is distributed open-source under the terms of the [MIT License](LICENSE).
