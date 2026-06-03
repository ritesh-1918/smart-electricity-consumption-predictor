# Volterra: Project Overview & Mathematics

This document details the underlying math and data flows powering the **Volterra** energy forecasting dashboard.

---

## 📐 The Linear Regression Equation

The prediction engine models daily electricity consumption ($y$) as a linear combination of environmental and behavioral features ($X_i$) multiplied by their respective slopes (coefficients $\beta_i$):

$$\text{Forecasted Consumption (kWh)} = \beta_0 + \beta_1(\text{Temp}) + \beta_2(\text{Humidity}) + \beta_3(\text{Occupancy}) + \beta_4(\text{AC\_Hours}) + \beta_5(\text{Appliance\_Hours}) + \beta_6(\text{Day\_Type})$$

### Ingesting Parameter Metrics:
*   **$\beta_0$ (Intercept)**: `0.4721` (The baseline consumption when all inputs are 0).
*   **$\beta_1$ (Temperature)**: `+1.8166` (For every 1°C increase, load rises by ~1.82 kWh).
*   **$\beta_2$ (Humidity)**: `+0.2958` (Very weak positive driver).
*   **$\beta_3$ (Occupancy)**: `+8.0050` (Each occupant increases daily load by 8.01 kWh).
*   **$\beta_4$ (AC Operating Hours)**: `+4.4470` (Each hour of active cooling consumes 4.45 kWh).
*   **$\beta_5$ (Appliance Operating Hours)**: `+2.1748` (Each active hour adds 2.17 kWh).
*   **$\beta_6$ (Day Type Offset)**: `+5.1341` (Weekend profiles add 5.13 kWh baseline overhead compared to weekdays).

---

## 🔄 Core Pipeline Data Flow

```text
+-------------------+      +-------------------------+      +-------------------------+
|  dataset/raw.csv  | ---> | src/data_preprocessing.py| ---> |  dataset/processed/     |
+-------------------+      +-------------------------+      +-------------------------+
                                                                         |
                                                                         v
+-------------------+      +-------------------------+      +-------------------------+
|  models/*.pkl     | <--- | src/train.py            | <--- |  X_train, y_train, etc. |
+-------------------+      +-------------------------+      +-------------------------+
          |
          v
+-------------------+      +-------------------------+
|  app/app.py       | ---> |  Hugging Face Container |
+-------------------+      +-------------------------+
```

1.  **Preprocessing**: Encodes categorical values (`Weekday` -> 0, `Weekend` -> 1) and separates labels from independent variables.
2.  **Training**: Fits a Scikit-Learn `LinearRegression` class, saves diagnostic visualizations to `reports/model_results/`, and serializes the model to `models/linear_regression_model.pkl`.
3.  **Deployment**: Streamlit loads the model binary, receives real-time user inputs, computes predictions, and displays attribution bar charts.
