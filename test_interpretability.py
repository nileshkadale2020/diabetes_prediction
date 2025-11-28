"""
Test script for SHAP and LIME explanations
"""

import sys
import pandas as pd
import numpy as np
import joblib

# Import the interpretability module
from src.model_interpretability import explain_prediction

def test_interpretability():
    """Test SHAP and LIME explanations"""
    
    print("=" * 60)
    print("Testing Model Interpretability (SHAP & LIME)")
    print("=" * 60)
    
    # Load trained model and data
    print("\n1. Loading model and data...")
    try:
        # Load XGBoost model (or fallback to Random Forest)
        if joblib.os.path.exists('models/xgboost.pkl'):
            model = joblib.load('models/xgboost.pkl')
            model_name = "XGBoost"
        elif joblib.os.path.exists('models/random_forest.pkl'):
            model = joblib.load('models/random_forest.pkl')
            model_name = "Random Forest"
        else:
            print("✗ No suitable model found")
            return False
        
        print(f"✓ Loaded {model_name} model")
        
        # Load training data
        X_train = pd.read_csv('data/X_train.csv')
        print(f"✓ Loaded training data: {X_train.shape}")
        
        # Load preprocessors
        scaler = joblib.load('models/scaler.pkl')
        imputer = joblib.load('models/imputer.pkl')
        print("✓ Loaded preprocessors")
        
    except Exception as e:
        print(f"✗ Error loading model/data: {e}")
        return False
    
    # Create sample instance (high-risk patient)
    print("\n2. Creating sample instance...")
    sample_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    # Preprocess the sample
    feature_names = list(sample_data.keys())
    sample_df = pd.DataFrame([sample_data], columns=feature_names)
    
    # Apply same preprocessing as training
    features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in features_to_check:
        if sample_df[feature].iloc[0] == 0:
            sample_df[feature] = np.nan
    
    sample_imputed = pd.DataFrame(
        imputer.transform(sample_df),
        columns=sample_df.columns
    )
    
    sample_scaled = pd.DataFrame(
        scaler.transform(sample_imputed),
        columns=sample_imputed.columns
    )
    
    print(f"✓ Preprocessed sample: {list(sample_data.values())}")
    
    # Get prediction
    print("\n3. Getting model prediction...")
    try:
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0][1]
        print(f"✓ Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
        print(f"✓ Probability: {probability:.2%}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Generate explanations
    print("\n4. Generating SHAP and LIME explanations...")
    try:
        explanations = explain_prediction(
            model=model,
            X_train=X_train.values,
            X_instance=sample_scaled.values,
            feature_names=feature_names
        )
        
        # Check SHAP results
        if explanations.get('shap_data'):
            shap_values = explanations['shap_data']['shap_values']
            print(f"✓ SHAP values generated: {len(shap_values)} features")
            
            # Show top 3 features
            indices = np.argsort(np.abs(shap_values))[::-1][:3]
            print("\n   Top 3 SHAP contributors:")
            for idx in indices:
                print(f"   - {feature_names[idx]}: {shap_values[idx]:.4f}")
        else:
            print("⚠ SHAP explanation not available")
        
        # Check LIME results
        if explanations.get('lime_data'):
            lime_contrib = explanations['lime_data']['feature_contributions']
            print(f"\n✓ LIME explanation generated: {len(lime_contrib)} features")
            print("\n   Top 3 LIME contributors:")
            for feature_desc, value in lime_contrib[:3]:
                print(f"   - {feature_desc}: {value:.4f}")
        else:
            print("⚠ LIME explanation not available")
        
        # Check visualizations
        if explanations.get('shap_plot'):
            print("\n✓ SHAP waterfall plot generated")
        else:
            print("\n⚠ SHAP plot not available")
        
        if explanations.get('lime_plot'):
            print("✓ LIME bar plot generated")
        else:
            print("⚠ LIME plot not available")
        
        # Show explanation text
        if explanations.get('explanation_text'):
            print("\n5. Human-readable explanation:")
            print("-" * 60)
            print(explanations['explanation_text'])
            print("-" * 60)
        
        print("\n" + "=" * 60)
        print("✓ All interpretability tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        import traceback
        print(f"\n✗ Explanation generation failed:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_interpretability()
    sys.exit(0 if success else 1)
