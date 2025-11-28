"""
Model training script for diabetes prediction
Trains Logistic Regression, Random Forest, XGBoost, and Neural Network
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load preprocessed train and test data"""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()  # Convert to 1D array
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning"""
    print("\n" + "=" * 50)
    print("Training Logistic Regression...")
    print("=" * 50)
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    joblib.dump(best_model, 'models/logistic_regression.pkl')
    print("Model saved to models/logistic_regression.pkl")
    
    return best_model

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning"""
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    print("=" * 50)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    joblib.dump(best_model, 'models/random_forest.pkl')
    print("Model saved to models/random_forest.pkl")
    
    return best_model

def train_xgboost(X_train, y_train):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "=" * 50)
    print("Training XGBoost...")
    print("=" * 50)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    joblib.dump(best_model, 'models/xgboost.pkl')
    print("Model saved to models/xgboost.pkl")
    
    return best_model

def train_neural_network(X_train, y_train, X_test, y_test):
    """Train TensorFlow Neural Network"""
    print("\n" + "=" * 50)
    print("Training TensorFlow Neural Network...")
    print("=" * 50)
    
    X_train_np = X_train.values
    X_test_np = X_test.values
    
    # Build neural network (64 -> 32 -> 16 -> 1)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_np.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train_np, y_train,
        validation_data=(X_test_np, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/neural_network.h5')
    print("Model saved to models/neural_network.h5")
    
    return model, history

def main():
    """Run the complete model training pipeline"""
    print("=" * 50)
    print("Model Training Pipeline")
    print("=" * 50)
    
    os.makedirs('models', exist_ok=True)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train all models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    nn_model, nn_history = train_neural_network(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 50)
    print("All models trained successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()

