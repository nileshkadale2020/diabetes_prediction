"""
Flask Web Application for Diabetes Prediction
Main application file with routes for Homepage, Resume, Projects, and Diabetes Prediction

This is the main entry point for our web application. It handles all the routing,
model loading, and prediction logic. The app serves multiple pages including
a homepage, resume, projects showcase, and the main diabetes prediction interface.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras

# Initialize Flask app with custom template and static folders
# This allows us to organize our HTML templates and CSS files in a separate directory
app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
app.config['SECRET_KEY'] = 'diabetes-prediction-app-2024'

# Global variables to store loaded models and preprocessors
# We load these once at startup to avoid reloading on every prediction (much faster!)
models = {}
scaler = None
imputer = None

def load_models():
    """
    Load all trained models and preprocessors from disk
    
    This function runs once when the Flask app starts up. It loads:
    - The scaler and imputer used during training (critical for preprocessing new data)
    - All trained ML models (Logistic Regression, Random Forest, XGBoost, Neural Network)
    - Determines which model performed best during evaluation
    
    We check if files exist before loading because XGBoost might not be available
    if libomp isn't installed on the system.
    """
    global models, scaler, imputer
    
    try:
        # First, load the preprocessors - these are essential!
        # We need the same scaler and imputer that were used during training
        # to ensure new data is processed the same way
        scaler = joblib.load('models/scaler.pkl')
        imputer = joblib.load('models/imputer.pkl')
        
        # Load sklearn models (these are saved as .pkl files)
        # We check if each file exists because some models might not be available
        if os.path.exists('models/logistic_regression.pkl'):
            models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        if os.path.exists('models/random_forest.pkl'):
            models['Random Forest'] = joblib.load('models/random_forest.pkl')
        if os.path.exists('models/xgboost.pkl'):
            models['XGBoost'] = joblib.load('models/xgboost.pkl')
        
        # Load TensorFlow model (saved as .h5 file, different format)
        if os.path.exists('models/neural_network.h5'):
            models['Neural Network'] = keras.models.load_model('models/neural_network.h5')
        
        # Figure out which model performed best during evaluation
        # This info is saved in a text file after model evaluation
        if os.path.exists('models/best_model.txt'):
            with open('models/best_model.txt', 'r') as f:
                best_model_name = f.read().strip()
        else:
            # Fallback: use XGBoost if available, otherwise just use the first model we have
            best_model_name = 'XGBoost' if 'XGBoost' in models else list(models.keys())[0] if models else None
        
        print(f"Models loaded: {list(models.keys())}")
        print(f"Best model: {best_model_name}")
        
    except Exception as e:
        # If something goes wrong, print a helpful error message
        print(f"Error loading models: {e}")
        print("Please train models first by running src/model_training.py")

load_models()

def preprocess_input(data):
    """
    Preprocess user input for prediction
    
    This is crucial! We need to apply the EXACT same preprocessing steps
    that were used during training. Otherwise, the model will make incorrect predictions.
    
    Steps:
    1. Convert input dictionary to DataFrame
    2. Replace unrealistic zeros with NaN (same as training)
    3. Impute missing values using the saved imputer
    4. Scale features using the saved scaler
    
    The order matters - we must do it in the same sequence as during training.
    """
    # Create DataFrame from the input dictionary
    # The order of features must match what the model expects
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    input_data = pd.DataFrame([data], columns=features)
    
    # Replace zeros with NaN for key features
    # In the original dataset, zeros often represented missing values
    # We need to handle this the same way we did during training
    features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in features_to_check:
        if feature in input_data.columns and input_data[feature].iloc[0] == 0:
            input_data[feature] = np.nan
    
    # Impute missing values using the same imputer from training
    # This ensures we use the same median values that were calculated during training
    input_imputed = pd.DataFrame(
        imputer.transform(input_data),
        columns=input_data.columns
    )
    
    # Scale features using the same scaler from training
    # This normalizes the data to the same scale the model was trained on
    input_scaled = pd.DataFrame(
        scaler.transform(input_imputed),
        columns=input_imputed.columns
    )
    
    return input_scaled

@app.route('/')
def index():
    """Homepage route"""
    return render_template('index.html')

@app.route('/resume')
def resume():
    """Resume page route"""
    return render_template('resume.html')

@app.route('/projects')
def projects():
    """Projects page route"""
    return render_template('projects.html')

@app.route('/diabetes')
def diabetes():
    """Diabetes prediction page route"""
    return render_template('diabetes.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction API endpoint
    
    This is where the magic happens! When a user submits the form, this function:
    1. Extracts the input values from the form
    2. Preprocesses them (same as training)
    3. Gets predictions from all available models
    4. Returns the best model's prediction with probability
    
    We return predictions from all models for transparency, but use the best one
    for the final assessment.
    """
    try:
        # Extract input data from the HTML form
        # We convert everything to float and provide defaults of 0
        # The form validation on the frontend should catch invalid inputs, but
        # this provides an extra layer of safety
        data = {
            'Pregnancies': float(request.form.get('pregnancies', 0)),
            'Glucose': float(request.form.get('glucose', 0)),
            'BloodPressure': float(request.form.get('blood_pressure', 0)),
            'SkinThickness': float(request.form.get('skin_thickness', 0)),
            'Insulin': float(request.form.get('insulin', 0)),
            'BMI': float(request.form.get('bmi', 0)),
            'DiabetesPedigreeFunction': float(request.form.get('diabetes_pedigree', 0)),
            'Age': float(request.form.get('age', 0))
        }
        
        # Preprocess the input using the same pipeline as training
        # This is critical - without proper preprocessing, predictions will be wrong
        input_scaled = preprocess_input(data)
        
        # Get predictions from all available models
        # We run all models and compare their outputs
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            if model_name == 'Neural Network':
                # TensorFlow models work a bit differently
                # They return probabilities directly, and we need to convert to binary prediction
                prob = float(model.predict(input_scaled.values, verbose=0)[0][0])
                pred = 1 if prob >= 0.5 else 0  # Threshold at 0.5 for binary classification
            else:
                # Sklearn models have predict_proba method which gives us probabilities
                prob = float(model.predict_proba(input_scaled)[0][1])  # Probability of class 1 (diabetic)
                pred = int(model.predict(input_scaled)[0])  # Binary prediction
            
            predictions[model_name] = pred
            probabilities[model_name] = prob
        
        # Determine which model to use for the final prediction
        # We prefer XGBoost if available (usually performs best), otherwise use first available
        best_model_name = 'XGBoost' if 'XGBoost' in models else list(models.keys())[0] if models else None
        best_prediction = predictions.get(best_model_name, 0)
        best_probability = probabilities.get(best_model_name, 0.0)
        
        # Prepare the response with all the information
        # We include predictions from all models for transparency
        result = {
            'prediction': int(best_prediction),  # 0 or 1
            'probability': round(best_probability * 100, 2),  # Convert to percentage
            'risk_level': 'High' if best_probability >= 0.5 else 'Low',  # Human-readable risk level
            'all_predictions': predictions,  # Show what all models predicted
            'all_probabilities': {k: round(v * 100, 2) for k, v in probabilities.items()},  # All probabilities as percentages
            'best_model': best_model_name  # Which model we used for final prediction
        }
        
        return jsonify(result)
        
    except Exception as e:
        # If something goes wrong, return an error message
        # This helps with debugging and provides user feedback
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    # debug=True enables auto-reload on code changes (useful during development)
    # host='0.0.0.0' makes it accessible from other devices on the network
    # port=5000 is the default Flask port
    app.run(debug=True, host='0.0.0.0', port=5000)

