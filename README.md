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
Cursor_diabtestPred/
├── data/
│   └── diabetes.csv              # Dataset (to be downloaded)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── model_training.py         # Model training scripts
│   ├── model_evaluation.py       # Model evaluation and comparison
│   └── utils.py                  # Utility functions
├── models/
│   └── (saved models)            # Trained model files
├── app/
│   ├── __init__.py
│   ├── routes.py                 # Flask routes
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

1. Clone the repository or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` directory as `diabetes.csv`

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

