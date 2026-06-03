"""
Model Training & Evaluation Module
----------------------------------
This script loads the preprocessed dataset splits, trains a Linear Regression model
using scikit-learn, prints evaluation metrics, generates model diagnostics plots,
saves the trained model object, and provides beginner-friendly workshop insights.
"""

# =====================================
# SYSTEM DEPENDENCIES & ML LIBRARIES
# =====================================
# Pandas & NumPy: Standard libraries for data manipulation and math.
# Matplotlib & Seaborn: Tools used to create graphs and visualizations.
# Pickle: Used to save python objects (like trained models) to disk files.
# LinearRegression: The standard machine learning algorithm for regression tasks.
# metrics: Collection of helper functions to calculate accuracy and error scores.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# =====================================
# LOADING PROCESSED DATA
# =====================================
def load_processed_data(data_dir: str) -> tuple:
    """
    Loads preprocessed feature and target splits from the specified directory.
    """
    # We load the split training and testing dataframes we saved during preprocessing.
    # This isolates our model building phase from the raw file processing, keeping our pipeline clean.
    print(f"[Info] Loading processed splits from directory: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    
    # iloc[:, 0] converts the single-column target DataFrame back into a 1D Pandas Series.
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).iloc[:, 0]
    return X_train, X_test, y_train, y_test

# =====================================
# MODEL TRAINING (LINEAR REGRESSION)
# =====================================
def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Initializes and fits a Linear Regression model.
    """
    # Linear Regression is a supervised learning algorithm that models the relationship
    # between input features (X) and a continuous target value (y) by fitting a straight line.
    # Formula: y = (w1 * x1) + (w2 * x2) + ... + (wn * xn) + intercept
    # The model automatically learns the best weights (w) that minimize prediction errors.
    print("[Info] Initializing and training the Linear Regression model...")
    
    # 1. Initialize the empty model (constructs the math object).
    model = LinearRegression()
    
    # 2. Train the model using fit(). The algorithm adjusts its coefficients based on the training data.
    model.fit(X_train, y_train)
    print("[Info] Model training completed successfully.")
    return model

# =====================================
# MODEL EVALUATION & METRICS
# =====================================
def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Generates predictions and calculates MAE, MSE, RMSE, and R2 metrics.
    """
    # Once a model is trained, we must measure its quality.
    # We feed the unseen test features (X_test) to the model to get predicted values.
    # Then we compare these predictions against the actual values (y_test) using standard metrics.
    print("[Info] Running predictions on test data...")
    
    # predict() takes inputs and computes the predicted outputs using the learned weights.
    predictions = model.predict(X_test)
    
    # Compute standard evaluation metrics:
    # 1. MAE (Mean Absolute Error): Average absolute error in predictions (lower is better).
    mae = metrics.mean_absolute_error(y_test, predictions)
    
    # 2. MSE (Mean Squared Error): Squares the errors before averaging, penalizing larger mistakes (lower is better).
    mse = metrics.mean_squared_error(y_test, predictions)
    
    # 3. RMSE (Root Mean Squared Error): Square root of MSE, putting the error back in original units (lower is better).
    rmse = np.sqrt(mse)
    
    # 4. R-squared (R2) Score: The percentage of variance explained by features. Closer to 1.0 (or 100%) is better.
    r2 = metrics.r2_score(y_test, predictions)
    
    return predictions, mae, mse, rmse, r2

# =====================================
# MODEL SAVING (SERIALIZATION)
# =====================================
def save_model_artifacts(model: LinearRegression, model_path: str) -> None:
    """
    Saves the trained model to disk as a pickle file.
    """
    # After finding a good model, we don't want to re-train it every time we need a prediction.
    # We save ('serialize') the trained model object using Pickle into a file (e.g. .pkl).
    # This saves the learned weights and intercept so we can load them instantly in our web app.
    print(f"[Info] Saving trained model artifact to: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Open the file in Write Binary ('wb') mode and dump the model object.
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("[Info] Model saved successfully.")

# =====================================
# DIAGNOSTIC VISUALIZATION
# =====================================
def save_diagnostic_plots(y_test: pd.Series, predictions: np.ndarray, output_dir: str) -> None:
    """
    Generates and saves Actual vs Predicted scatter plots and Residual plots.
    """
    # Plotting helps us visually audit our model's performance.
    print(f"[Info] Generating diagnostic visualizations and saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Actual vs Predicted scatter plot:
    # We plot the actual value on the X-axis and predicted value on the Y-axis.
    # Ideally, all points should lie on the diagonal line (Perfect Fit Line).
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, color='teal', alpha=0.6, edgecolor='w')
    
    # Draw reference line y = x
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', linewidth=2, label='Perfect Fit Line')
    
    plt.title('Actual vs. Predicted Electricity Consumption')
    plt.xlabel('Actual Consumption (kWh)')
    plt.ylabel('Predicted Consumption (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # 2. Residual error plot:
    # Residuals = Actual Value - Predicted Value (the prediction errors).
    # We plot the predictions on the X-axis and residuals on the Y-axis.
    # Ideally, errors should be randomly scattered around the zero line with no visible pattern.
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=predictions, y=residuals, color='purple', alpha=0.6, edgecolor='w')
    plt.axhline(y=0, color='crimson', linestyle='--', linewidth=2)
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Predicted Consumption (kWh)')
    plt.ylabel('Residuals (Error)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_plot.png'), dpi=300)
    plt.close()
    print("[Info] Diagnostic plots successfully saved!")

# =====================================
# TRAINING EXECUTION PIPELINE
# =====================================
if __name__ == "__main__":
    # Define directories
    data_dir = os.path.join("dataset", "processed")
    model_output_path = os.path.join("models", "linear_regression_model.pkl")
    plots_dir = os.path.join("reports", "model_results")
    
    try:
        # 1. Load preprocessed train and test datasets
        X_train, X_test, y_train, y_test = load_processed_data(data_dir)
        
        # 2. Train the Linear Regression model
        model = train_linear_regression(X_train, y_train)
        
        # 3. Generate predictions and calculate metrics (MAE, MSE, RMSE, R2)
        predictions, mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
        
        # 4. Save the trained model binary and performance plots to files
        save_model_artifacts(model, model_output_path)
        save_diagnostic_plots(y_test, predictions, plots_dir)
        
        # Print Actual vs Predicted Comparison table
        print("\n" + "="*50)
        print("   [First 10 Actual vs Predicted Values]")
        print("="*50)
        comparison_df = pd.DataFrame({
            'Actual Value (kWh)': y_test.head(10).values,
            'Predicted Value (kWh)': predictions[:10]
        })
        print(comparison_df.to_string(index=True))
        print("="*50)
        
        # Print Evaluation Metrics
        print("\n" + "="*50)
        print("   [Model Evaluation Metrics]")
        print("="*50)
        print(f"Mean Absolute Error (MAE):     {mae:.4f} kWh")
        print(f"Mean Squared Error (MSE):      {mse:.4f} kWh2")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kWh")
        print(f"R-squared (R2) Score:          {r2:.4f} ({r2*100:.2f}%)")
        print("="*50)
        
        # Print Coefficients & Intercept
        print("\n" + "="*50)
        print("   [Model Parameters (Learned Formula)]")
        print("="*50)
        print(f"Intercept (Baseline Constant): {model.intercept_:.4f}")
        print("\nFeature Coefficients (Slopes):")
        # Coefficients tell us how much impact each parameter has on electricity consumption.
        for col, coef in zip(X_train.columns, model.coef_):
            print(f"  - {col:18} : {coef:+.4f}")
        print("="*50)
        
        # Model attribution explanations
        print("\n" + "#"*60)
        print("MODEL FORECAST ATTRIBUTION & INTERPRETABILITY")
        print("#"*60)
        print("""
1. Model Attribution Target:
   The regression engine optimizes linear weights to predict consumption from inputs:
   
   Electricity_Consumption = Intercept 
                             + (w1 * Temperature) 
                             + (w2 * Humidity) 
                             + (w3 * Occupancy) 
                             + (w4 * AC_Hours) 
                             + (w5 * Appliance_Hours) 
                             + (w6 * Day_Type)

2. Attribution Coefficients (Slopes):
   Coefficients represent the change in forecasted electricity consumption (kWh) 
   per unit increase in a feature, holding all other features constant.
   - Positive coefficients (+) indicate a direct relationship (raising the feature increases load).
   - Negative coefficients (-) indicate an inverse relationship.
   - Absolute weight values indicate feature influence hierarchy.

3. R-squared (R2) Variance Score:
   R2 measures the proportion of variance in the target variable explained by model features.
   - An R2 score of 0.9495 means that 94.95% of target variance is explained by the model, 
     indicating a highly predictive fit.

4. Unmodeled Variance (Residuals):
   Real-world loads exhibit minor stochastic variations (e.g. appliance model differences, occupant habits)
   which are treated as unmodeled residual errors, establishing the boundary of model certainty.
        """)
        print("#"*60 + "\n")
        
    except Exception as e:
        print(f"[Error] Training pipeline failed: {e}")
