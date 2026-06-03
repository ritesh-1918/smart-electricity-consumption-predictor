"""
Data Preprocessing Pipeline
--------------------------
This module contains clean, production-grade functions for loading, cleaning, and preprocessing 
the Smart Electricity Consumption dataset. It prepares the data for model training by 
encoding categorical features and performing a train-test split.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.

    Parameters:
    -----------
    filepath : str
        The path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        Loaded pandas DataFrame.
    """
    print(f"[Info] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values and removes duplicate records.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # 1. Check for missing values
    missing_counts = df.isnull().sum()
    print("[Info] Missing values check:")
    for col, count in missing_counts.items():
        print(f"  - {col}: {count} missing values")
        
    # 2. Check and remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[Warning] Found {duplicate_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[Info] No duplicate rows found.")
        
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features (specifically Day_Type) using binary mapping.
    Weekday -> 0
    Weekend -> 1

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded features.
    """
    print("[Info] Encoding categorical 'Day_Type' column (Weekday = 0, Weekend = 1)...")
    df_encoded = df.copy()
    
    # Mapping dictionary
    day_mapping = {
        'Weekday': 0,
        'Weekend': 1
    }
    
    df_encoded['Day_Type'] = df_encoded['Day_Type'].map(day_mapping)
    return df_encoded

def split_features_target(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Separates independent feature columns from the target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        The input preprocessed DataFrame.
    target_column : str
        Name of the label column.

    Returns:
    --------
    tuple (X, y)
        X: features DataFrame, y: target Series.
    """
    print(f"[Info] Separating features (X) and target variable (y: {target_column})...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def perform_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split variables into train and test groups.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of dataset to include in the test split.
    random_state : int
        Controls shuffling.

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) split matrices.
    """
    print(f"[Info] Splitting data into train and test splits (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, output_dir: str) -> None:
    """
    Saves processed splits as CSV files into the specified directory.

    Parameters:
    -----------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        Data splits.
    output_dir : str
        Path to output folder.
    """
    print(f"[Info] Saving split dataframes to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("[Info] Data splits saved successfully!")

if __name__ == "__main__":
    # Define filepaths relative to project root
    dataset_path = os.path.join("dataset", "electricity_consumption_3000.csv")
    output_directory = os.path.join("dataset", "processed")
    
    # Run the full pipeline
    try:
        # Load
        raw_df = load_data(dataset_path)
        
        # Clean
        cleaned_df = clean_data(raw_df)
        
        # Encode
        encoded_df = encode_categorical_features(cleaned_df)
        
        # Separate Features & Target
        X, y = split_features_target(encoded_df, "Electricity_Consumption")
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save processed splits
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
