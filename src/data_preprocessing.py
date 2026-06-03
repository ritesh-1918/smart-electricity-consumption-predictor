"""
Data Preprocessing Pipeline
--------------------------
This module contains clean, production-grade functions for loading, cleaning, and preprocessing 
the Smart Electricity Consumption dataset. It prepares the data for model training by 
encoding categorical features and performing a train-test split.
"""

# =====================================
# SYSTEM DEPENDENCIES & LIBRARIES
# =====================================
# Pandas: Used for loading and manipulating tabular datasets (like Excel/CSVs).
# NumPy: Used for efficient mathematical arrays and numerical computations.
# OS: Standard Python library for navigating system file folders and paths.
# Train_test_split: Core tool to divide dataset into training and testing sets.
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# =====================================
# DATA LOADING FUNCTION
# =====================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    """
    # We load our dataset from a CSV (Comma Separated Values) file.
    # In ML, datasets are the foundation. Before building any model, we must load the raw data
    # into memory (a Pandas DataFrame) so we can analyze, clean, and process it.
    print(f"[Info] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    # read_csv reads tabular data and constructs a structured data table (DataFrame).
    df = pd.read_csv(filepath)
    return df

# =====================================
# DATA CLEANING FUNCTION
# =====================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values and removes duplicate records.
    """
    # Data cleaning is one of the most critical steps in Machine Learning.
    # Real-world data is often messy, containing missing values or duplicate records.
    # Models cannot train properly if there are gaps (nulls) or duplicate entries,
    # which can bias the model or cause mathematical errors during training.
    
    # 1. Check for missing values (nulls)
    # isnull().sum() counts how many empty values exist in each column.
    missing_counts = df.isnull().sum()
    print("[Info] Missing values check:")
    for col, count in missing_counts.items():
        print(f"  - {col}: {count} missing values")
        
    # 2. Check and remove duplicates
    # duplicated().sum() checks if any row is an exact duplicate of another.
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[Warning] Found {duplicate_count} duplicate rows. Removing duplicates...")
        # drop_duplicates() removes redundant rows so the model doesn't over-weight repeated data.
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[Info] No duplicate rows found.")
        
    return df

# =====================================
# CATEGORICAL VARIABLE ENCODING
# =====================================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features (specifically Day_Type) using binary mapping.
    Weekday -> 0
    Weekend -> 1
    """
    # Machine Learning models are mathematical equations that only understand numbers.
    # If we have text columns (like 'Weekday' or 'Weekend'), we must convert them into numbers.
    # Here, we map 'Weekday' to 0 and 'Weekend' to 1. This process is called Label Encoding/Binary Mapping,
    # enabling our regression algorithm to perform mathematical matrix calculations on this feature.
    print("[Info] Encoding categorical 'Day_Type' column (Weekday = 0, Weekend = 1)...")
    df_encoded = df.copy()
    
    # Mapping dictionary defines what each text value converts to.
    day_mapping = {
        'Weekday': 0,
        'Weekend': 1
    }
    
    # Map applies this replacement rule across the entire column.
    df_encoded['Day_Type'] = df_encoded['Day_Type'].map(day_mapping)
    return df_encoded

# =====================================
# FEATURE & TARGET SELECTION
# =====================================
def split_features_target(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Separates independent feature columns from the target variable.
    """
    # In Supervised Learning, we must separate our dataset into two parts:
    # 1. Features (X): The inputs or predictors (e.g., temperature, occupancy, AC hours) that we use to predict.
    # 2. Target (y): The label or outcome we want to predict (e.g., electricity consumption).
    # This separation is essential so we can tell the model: "Here are the inputs (X), learn how they produce output (y)".
    print(f"[Info] Separating features (X) and target variable (y: {target_column})...")
    
    # X contains all inputs (we drop the target column to isolate predictors).
    X = df.drop(columns=[target_column])
    
    # y contains the target label only.
    y = df[target_column]
    return X, y

# =====================================
# TRAIN-TEST SPLITTING
# =====================================
def perform_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split variables into train and test groups.
    """
    # To evaluate how well our model performs, we must test it on data it has never seen before.
    # We split our data into:
    # - Training Set (typically 80%): Used by the model to learn relationships.
    # - Testing Set (typically 20%): Kept hidden from the model during training, used later to test its accuracy.
    # This prevents 'overfitting' (where a model memorizes the training data but fails on new, unseen data).
    print(f"[Info] Splitting data into train and test splits (test_size={test_size}, random_state={random_state})...")
    
    # train_test_split shuffles the rows randomly and splits them based on the test_size percentage.
    # random_state acts as a random seed, ensuring the split is reproducible (same split every time).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# =====================================
# SAVING PREPROCESSED DATA
# =====================================
def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, output_dir: str) -> None:
    """
    Saves processed splits as CSV files into the specified directory.
    """
    # We save our split datasets as separate files.
    # In production ML workflows, separating preprocessing from training ensures consistency.
    # It allows us to load the exact same training and testing splits anytime, preventing data leakage.
    print(f"[Info] Saving split dataframes to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # to_csv writes the pandas DataFrame structures out to physical file assets.
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("[Info] Data splits saved successfully!")

# =====================================
# PIPELINE EXECUTION
# =====================================
if __name__ == "__main__":
    # Define filepaths relative to project root
    dataset_path = os.path.join("dataset", "electricity_consumption_3000.csv")
    output_directory = os.path.join("dataset", "processed")
    
    try:
        # 1. Load the dataset
        raw_df = load_data(dataset_path)
        
        # 2. Clean the dataset (check nulls, remove duplicates)
        cleaned_df = clean_data(raw_df)
        
        # 3. Encode categorical variables (convert text columns to numeric codes)
        encoded_df = encode_categorical_features(cleaned_df)
        
        # 4. Separate features (X) and target variable (y)
        X, y = split_features_target(encoded_df, "Electricity_Consumption")
        
        # 5. Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 6. Save processed splits to disk for training
        save_splits(X_train, X_test, y_train, y_test, output_directory)
        
        # Print summaries to screen
        print("\n" + "="*40 + "\n[Summary of Preprocessing Pipeline]\n" + "="*40)
        print("Feature Columns: ", list(X.columns))
        print("X Shape (Features): ", X.shape)
        print("y Shape (Target): ", y.shape)
        print("X_train Shape: ", X_train.shape)
        print("X_test Shape: ", X_test.shape)
        print("y_train Shape: ", y_train.shape)
        print("y_test Shape: ", y_test.shape)
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"[Error] Pipeline execution failed: {e}")
