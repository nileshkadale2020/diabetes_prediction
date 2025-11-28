"""
Data preprocessing for diabetes dataset
Handles cleaning, imputation, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(file_path='data/diabetes.csv'):
    """Load diabetes dataset from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download from Kaggle.")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    df_processed = df.copy()
    
    # Replace zeros with NaN (these features can't be zero)
    features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for feature in features_to_check:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].replace(0, np.nan)
            print(f"Replaced zeros in {feature}: {df_processed[feature].isna().sum()} missing values")
    
    # Separate features and target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # Impute missing values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print(f"\nMissing values after imputation: {X_imputed.isna().sum().sum()}")
    
    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    
    # Save scaler and imputer for predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    
    print("\nData preprocessing completed!")
    print(f"Final shape: {X_scaled.shape}")
    
    return X_scaled, y, scaler, imputer

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets (80-20 split)"""
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
    """Run the complete preprocessing pipeline"""
    print("=" * 50)
    print("Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Load the raw dataset
    df = load_data()
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    
    # Preprocess the data
    X, y, scaler, imputer = preprocess_data(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("\nProcessed data saved to data/ directory")
    print("=" * 50)

if __name__ == "__main__":
    main()

