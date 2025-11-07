"""
Model Training Script for Diabetes Prediction Project
Trains multiple models: Logistic Regression, Random Forest, XGBoost, TensorFlow NN

This script trains all four machine learning models with hyperparameter tuning.
We use GridSearchCV to find the best hyperparameters for each model through
cross-validation. This takes a while but gives us the best possible models.
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
    """
    Load preprocessed training and testing data
    
    This loads the data that was already preprocessed by data_preprocessing.py.
    We use the preprocessed data to save time - no need to preprocess again.
    """
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()  # Convert to 1D array
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model with hyperparameter tuning
    
    Logistic Regression is our baseline model - it's simple, fast, and interpretable.
    We tune the regularization strength (C) and penalty type (L1/L2) to find the best
    balance between fitting the data and avoiding overfitting.
    """
    print("\n" + "=" * 50)
    print("Training Logistic Regression...")
    print("=" * 50)
    
    # Hyperparameter tuning using GridSearchCV
    # C controls regularization strength - smaller C = stronger regularization
    # L1 penalty (Lasso) can zero out features, L2 (Ridge) shrinks them
    # Note: liblinear supports both l1 and l2, saga supports both but is slower
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],  # L1 (Lasso) or L2 (Ridge) regularization
        'solver': ['liblinear', 'saga']  # Different optimization algorithms
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    # GridSearchCV tries all combinations and picks the best one
    # cv=5 means 5-fold cross-validation, scoring='roc_auc' uses ROC-AUC as the metric
    # Use error_score='raise' to catch any solver/penalty incompatibility
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Save the best model so we can use it later
    joblib.dump(best_model, 'models/logistic_regression.pkl')
    print("Model saved to models/logistic_regression.pkl")
    
    return best_model

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model with hyperparameter tuning
    
    Random Forest is an ensemble method that combines multiple decision trees.
    It's great for handling non-linear relationships and feature interactions.
    We tune the number of trees, tree depth, and splitting criteria.
    """
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    print("=" * 50)
    
    # Hyperparameter tuning
    # n_estimators: more trees = better but slower
    # max_depth: controls how deep trees can grow (None = no limit)
    # min_samples_split: minimum samples needed to split a node
    # min_samples_leaf: minimum samples in a leaf node
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None],  # Max depth of trees (None = unlimited)
        'min_samples_split': [2, 5, 10],  # Min samples to split a node
        'min_samples_leaf': [1, 2, 4]  # Min samples in a leaf
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # verbose=1 shows progress during training (helpful for long runs)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'models/random_forest.pkl')
    print("Model saved to models/random_forest.pkl")
    
    return best_model

def train_xgboost(X_train, y_train):
    """
    Train XGBoost model with hyperparameter tuning
    
    XGBoost is a gradient boosting algorithm that's often the best performer
    on structured data. It builds trees sequentially, each one correcting
    the errors of the previous ones. We tune learning rate, tree depth, and
    subsampling to find the optimal configuration.
    """
    print("\n" + "=" * 50)
    print("Training XGBoost...")
    print("=" * 50)
    
    # Hyperparameter tuning
    # n_estimators: number of boosting rounds
    # max_depth: depth of trees (deeper = more complex, risk of overfitting)
    # learning_rate: how much each tree contributes (lower = more trees needed)
    # subsample: fraction of samples used for each tree (helps prevent overfitting)
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
    
    # Save the best model
    joblib.dump(best_model, 'models/xgboost.pkl')
    print("Model saved to models/xgboost.pkl")
    
    return best_model

def train_neural_network(X_train, y_train, X_test, y_test):
    """
    Train TensorFlow Neural Network
    
    Neural networks can capture complex non-linear patterns in the data.
    We use a multi-layer architecture with dropout to prevent overfitting.
    Early stopping helps us stop training when the model stops improving.
    """
    print("\n" + "=" * 50)
    print("Training TensorFlow Neural Network...")
    print("=" * 50)
    
    # Convert to numpy arrays (TensorFlow/Keras prefers numpy over pandas)
    X_train_np = X_train.values
    X_test_np = X_test.values
    
    # Build the neural network model
    # We use a sequential model (layers stacked one after another)
    # Architecture: 64 -> 32 -> 16 -> 1 neurons
    # Dropout layers randomly turn off neurons during training to prevent overfitting
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_np.shape[1],)),  # First hidden layer
        layers.Dropout(0.3),  # Drop 30% of neurons randomly during training
        layers.Dense(32, activation='relu'),  # Second hidden layer
        layers.Dropout(0.3),  # More dropout
        layers.Dense(16, activation='relu'),  # Third hidden layer
        layers.Dropout(0.2),  # Less dropout in later layers
        layers.Dense(1, activation='sigmoid')  # Output layer (sigmoid for binary classification)
    ])
    
    # Compile the model - this sets up the training process
    # Adam optimizer is a good default choice (adaptive learning rate)
    # binary_crossentropy is the right loss function for binary classification
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    # This stops training if validation loss doesn't improve for 10 epochs
    # restore_best_weights=True means we keep the best model, not the last one
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    # We train for up to 100 epochs, but early stopping will likely stop earlier
    # batch_size=32 is a good default (processes 32 samples at a time)
    history = model.fit(
        X_train_np, y_train,
        validation_data=(X_test_np, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the trained model
    # .h5 is the standard format for saving Keras models
    os.makedirs('models', exist_ok=True)
    model.save('models/neural_network.h5')
    print("Model saved to models/neural_network.h5")
    
    return model, history

def main():
    """
    Main training pipeline
    
    This function orchestrates the entire model training process:
    1. Creates the models directory
    2. Loads preprocessed training and testing data
    3. Trains all four models (Logistic Regression, Random Forest, XGBoost, Neural Network)
    4. Saves all trained models to disk
    
    This can take a while (30-60 minutes) because of hyperparameter tuning,
    but it's worth it to get the best possible models.
    """
    print("=" * 50)
    print("Model Training Pipeline")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    # This is where we'll save all trained models
    os.makedirs('models', exist_ok=True)
    
    # Load the preprocessed data
    # This data was already cleaned and scaled by data_preprocessing.py
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train all four models
    # Each model training function handles its own hyperparameter tuning
    # We save each model as we go so we don't lose progress if something fails
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    nn_model, nn_history = train_neural_network(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 50)
    print("All models trained successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()

