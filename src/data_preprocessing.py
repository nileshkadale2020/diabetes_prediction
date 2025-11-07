"""
Data Preprocessing Script for Diabetes Prediction Project
Handles data cleaning, missing value imputation, and feature scaling

This script is responsible for cleaning and preparing the raw dataset for model training.
Good preprocessing is crucial - garbage in, garbage out! We need to handle missing values,
normalize features, and ensure the data is in the right format for our ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(file_path='data/diabetes.csv'):
    """
    Load the diabetes dataset from CSV file
    
    Simple function to load the data. We check if the file exists first
    to give a helpful error message if the dataset hasn't been downloaded yet.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download from Kaggle.")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """
    Preprocess the diabetes dataset
    
    This is where we clean up the data and get it ready for machine learning.
    The preprocessing steps are:
    1. Replace zero values in key features with NaN (zeros often mean missing data)
    2. Impute missing values with median (robust to outliers)
    3. Normalize features with StandardScaler (important for many ML algorithms)
    
    We save the scaler and imputer so we can use them later when making predictions
    on new data. This ensures consistency between training and inference.
    """
    df_processed = df.copy()  # Work on a copy to avoid modifying the original
    
    # Replace zero values in key features with NaN
    # This is important because in medical data, zeros often represent missing values
    # For example, a glucose level of 0 doesn't make biological sense
    # These features cannot realistically be zero
    features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for feature in features_to_check:
        if feature in df_processed.columns:
            # Replace 0 with NaN so we can properly handle them as missing values
            df_processed[feature] = df_processed[feature].replace(0, np.nan)
            print(f"Replaced zeros in {feature}: {df_processed[feature].isna().sum()} missing values")
    
    # Separate features (X) and target variable (y)
    # The target is 'Outcome' - whether the person has diabetes (1) or not (0)
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # Impute missing values with median
    # We use median instead of mean because it's more robust to outliers
    # For example, if we have one person with BMI 50, the mean would be skewed
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print(f"\nMissing values after imputation: {X_imputed.isna().sum().sum()}")
    
    # Normalize features using StandardScaler
    # This transforms features to have mean=0 and std=1
    # Many ML algorithms (like neural networks, SVM) work much better with normalized data
    # It also helps when features have very different scales (e.g., Age vs Glucose)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    
    # Save the scaler and imputer for later use
    # This is critical! When we make predictions on new data, we need to use
    # the SAME scaler and imputer that were fitted on the training data
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    
    print("\nData preprocessing completed!")
    print(f"Final shape: {X_scaled.shape}")
    
    return X_scaled, y, scaler, imputer

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    We use an 80-20 split (80% training, 20% testing) which is a common practice.
    The stratify parameter ensures that the class distribution is maintained in both
    training and testing sets - this is important when dealing with imbalanced data.
    
    random_state=42 ensures reproducibility - we'll get the same split every time.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
    print(f"Class distribution in testing: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Main preprocessing pipeline
    
    This function orchestrates the entire preprocessing workflow:
    1. Load the raw data
    2. Display some basic information about it
    3. Clean and preprocess the data
    4. Split into training and testing sets
    5. Save everything for later use
    
    You can run this script standalone to preprocess the data before training models.
    """
    print("=" * 50)
    print("Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Load the raw dataset
    df = load_data()
    
    # Display basic info about the dataset
    # This helps us understand what we're working with
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    
    # Preprocess the data (cleaning, imputation, scaling)
    X, y, scaler, imputer = preprocess_data(df)
    
    # Split into training and testing sets
    # We'll use training set to train models, testing set to evaluate them
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save the processed data to CSV files
    # This way we don't have to preprocess every time we want to train a model
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("\nProcessed data saved to data/ directory")
    print("=" * 50)

if __name__ == "__main__":
    main()

