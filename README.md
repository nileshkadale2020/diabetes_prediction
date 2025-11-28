# Diabetes Prediction Machine Learning Project

## Project Overview

This project aims to develop a machine learning model that predicts the likelihood of diabetes in individuals based on key medical parameters such as glucose level, BMI, blood pressure, age, and insulin levels. The final deliverable is an end-to-end machine learning pipeline integrated into a Flask web application.

## Project Goals

- Perform exploratory data analysis (EDA) to understand data characteristics and relationships
- Handle missing values, outliers, and feature scaling for optimal model input
- Train and fine-tune models including Logistic Regression, Random Forest, XGBoost, and TensorFlow Neural Network
- Compare model performance using multiple evaluation metrics
- Deploy the model via a Flask web application with multiple pages (Homepage, Resume, Projects, Diabetes Model)

## Dataset

- **Dataset**: PIMA Indians Diabetes Database (Kaggle)
- **Size**: 768 records and 9 columns (8 features and 1 target variable)
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target**: Outcome (1 = Diabetic, 0 = Non-Diabetic)
- **Source**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Project Structure

```
DIABETES_PREDICTION/
├── data/
│   └── diabetes.csv              # Dataset (to be downloaded)
├── notebooks/
│   └── diabetes_prediction_complete.ipynb  # Includes EDA, Model Training and Evaluation
├── src/
│   ├── data_preprocessing.py     # Individually test Data cleaning and preprocessing
│   ├── model_training.py         # Individually test Model training scripts
│   ├── model_evaluation.py       # Individually test  Model evaluation and comparison
│   └── utils.py                  # Utility functions
├── models/
│   └── (saved models)            # Trained model files
├── app/
│   ├── __init__.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html            # Homepage
│   │   ├── resume.html           # Resume page
│   │   ├── projects.html         # Projects page
│   │   └── diabetes.html         # Diabetes prediction page
│   └── static/
│       └── css/
│           └── style.css         # CSS styles
├── app.py                        # Main Flask application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

**📖 For complete setup instructions, see [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**

This guide includes step-by-step instructions from cloning to running the application.

### Quick Start

1. **Clone the repository:**
```bash
git clone git@github.com:nileshkadale2020/diabetes_prediction.git
cd diabetes_prediction
```

2. **Create a project environment (recommended: conda)**

Option A — recommended :
```bash
# create and activate a Python 3.11 environment (name is arbitrary)
conda create -n diab-py311 python=3.11 -y
conda activate diab-py311
```

Option B — using venv:
```bash
python3 -m venv dtsc691
source dtsc691/bin/activate  # macOS/Linux
# OR
dtsc691\Scripts\activate  # Windows
```

3. **Install dependencies:**

If you used the recommended conda env above, run:
```bash
python -m pip install -r requirements.txt
```

If pip install fails on macOS because of missing system libs (e.g., XGBoost or libomp), see the Complete Setup Guide for troubleshooting steps.


4. **Preprocess data:**
```bash
python src/data_preprocessing.py
```

5. **Train models:**
```bash
python src/model_training.py
```

6. **Run Flask application:**

If you used the recommended conda environment (`diab-py311`):
```bash
# either activate the environment first
conda activate diab-py311
python app.py

# or run without activating
conda run -n diab-py311 python app.py
```

If you used a venv, activate it then run `python app.py` as before.

Then open: http://localhost:5000

**Note**: Model training takes 30-60 minutes. The Flask app requires trained models to make predictions.

## Usage

### Training Models

1. Run data preprocessing:
```bash
python src/data_preprocessing.py
```

2. Train models:
```bash
python src/model_training.py
```

3. Evaluate models:
```bash
python src/model_evaluation.py
```

### Running the Flask Application

```bash
python app.py
```

Then navigate to `http://localhost:5000` in your web browser.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, TensorFlow Neural Network
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Interpretability**: SHAP and LIME integration
- **Web Interface**: Flask-based web application with multiple pages
- **Real-time Predictions**: Interactive form for diabetes risk prediction

## Technologies Used

- Python 3.11
- Pandas & NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- Matplotlib & Seaborn
- Flask
- SHAP & LIME

## Author

Nilesh Kadale

## License

This project is part of a Master's in Data Science capstone project.


