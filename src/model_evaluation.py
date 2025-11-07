"""
Model Evaluation Script for Diabetes Prediction Project
Evaluates and compares all trained models

This script evaluates all trained models on the test set and creates
comparison visualizations. We use multiple metrics to get a comprehensive
view of each model's performance.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow import keras
import os

def load_data_and_models():
    """
    Load test data and all trained models
    
    We load the test set (data we held out during training) and all
    the trained models. This allows us to evaluate how well each
    model generalizes to unseen data.
    """
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    models = {}
    
    # Load sklearn models (saved as .pkl files)
    models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
    models['Random Forest'] = joblib.load('models/random_forest.pkl')
    models['XGBoost'] = joblib.load('models/xgboost.pkl')
    
    # Load TensorFlow model (saved as .h5 file)
    models['Neural Network'] = keras.models.load_model('models/neural_network.h5')
    
    return X_test, y_test, models

def evaluate_model(model, X_test, y_test, model_name, is_nn=False):
    """
    Evaluate a single model on the test set
    
    This function gets predictions from a model and calculates various metrics.
    We need to handle TensorFlow models differently from sklearn models because
    they have different APIs for getting predictions.
    """
    if is_nn:
        # Neural network predictions work differently
        # TensorFlow models return probabilities directly
        y_pred_proba = model.predict(X_test.values, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probability to binary prediction
    else:
        # Sklearn models have predict and predict_proba methods
        y_pred = model.predict(X_test)  # Binary predictions (0 or 1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (diabetic)
    
    # Calculate various metrics to evaluate model performance
    # Each metric tells us something different about how well the model performs
    accuracy = accuracy_score(y_test, y_pred)  # Overall correctness
    precision = precision_score(y_test, y_pred)  # Of predicted positives, how many are actually positive
    recall = recall_score(y_test, y_pred)  # Of actual positives, how many did we catch
    f1 = f1_score(y_test, y_pred)  # Harmonic mean of precision and recall
    roc_auc = roc_auc_score(y_test, y_pred_proba)  # Area under ROC curve (great for imbalanced data)
    
    # Confusion matrix shows us where the model makes mistakes
    # It's a 2x2 matrix: True Negatives, False Positives, False Negatives, True Positives
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrices(results):
    """
    Plot confusion matrices for all models
    
    Confusion matrices are great for understanding where models make mistakes.
    They show us: True Negatives, False Positives, False Negatives, True Positives.
    This helps us see if models are biased toward one class or the other.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, result in enumerate(results):
        cm = result['confusion_matrix']
        # Create a heatmap to visualize the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{result["model_name"]} - Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices saved to models/confusion_matrices.png")
    plt.close()

def plot_roc_curves(results, y_test):
    """
    Plot ROC curves for all models
    
    ROC curves show the trade-off between True Positive Rate and False Positive Rate
    at different classification thresholds. The area under the curve (AUC) tells us
    how well the model can distinguish between classes. Higher AUC = better model.
    """
    plt.figure(figsize=(10, 8))
    
    for result in results:
        # Calculate ROC curve points
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['roc_auc']:.4f})")
    
    # Plot diagonal line representing random classifier (AUC = 0.5)
    # Any model below this line is worse than random guessing
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('models/roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC curves saved to models/roc_curves.png")
    plt.close()

def compare_models(results):
    """
    Create comparison table of all models
    
    This function creates a nice comparison table showing all metrics side-by-side.
    We sort by ROC-AUC because it's the best metric for imbalanced classification problems.
    The best model is the one with the highest ROC-AUC score.
    """
    comparison_df = pd.DataFrame({
        'Model': [r['model_name'] for r in results],
        'Accuracy': [r['accuracy'] for r in results],
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-Score': [r['f1_score'] for r in results],
        'ROC-AUC': [r['roc_auc'] for r in results]
    })
    
    # Round to 4 decimal places for readability
    comparison_df = comparison_df.round(4)
    
    # Sort by ROC-AUC (descending) - best model first
    # ROC-AUC is our primary metric because it works well with imbalanced data
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV for easy reference later
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("\nComparison saved to models/model_comparison.csv")
    
    # Find and display the best model
    # The best model is the one with the highest ROC-AUC
    best_model = comparison_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']} (ROC-AUC: {best_model['ROC-AUC']:.4f})")
    
    return comparison_df, best_model

def print_detailed_reports(results, y_test):
    """Print detailed classification reports"""
    print("\n" + "=" * 80)
    print("Detailed Classification Reports")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print("-" * 80)
        print(classification_report(y_test, result['y_pred'], 
                                    target_names=['Non-Diabetic', 'Diabetic']))

def main():
    """
    Main evaluation pipeline
    
    This function orchestrates the entire evaluation process:
    1. Loads test data and all trained models
    2. Evaluates each model on the test set
    3. Creates visualizations (confusion matrices, ROC curves)
    4. Compares all models and identifies the best one
    5. Saves the best model name for use in the Flask app
    """
    print("=" * 80)
    print("Model Evaluation Pipeline")
    print("=" * 80)
    
    # Load test data and all trained models
    X_test, y_test, models = load_data_and_models()
    
    # Evaluate all models on the test set
    # We evaluate each model separately because they have different APIs
    results = []
    for model_name, model in models.items():
        is_nn = model_name == 'Neural Network'  # Check if it's a TensorFlow model
        result = evaluate_model(model, X_test, y_test, model_name, is_nn=is_nn)
        results.append(result)
        
        # Print results for each model
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    
    # Create visualizations to help us understand model performance
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    
    # Compare all models and find the best one
    comparison_df, best_model = compare_models(results)
    
    # Print detailed classification reports for each model
    print_detailed_reports(results, y_test)
    
    # Save the best model name to a text file
    # The Flask app will read this to know which model to use for predictions
    with open('models/best_model.txt', 'w') as f:
        f.write(best_model['Model'])
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()

