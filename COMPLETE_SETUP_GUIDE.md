# Complete Setup Guide: Clone, Install, and Run

This guide will walk you through cloning the repository, setting up the environment, and running the Flask application step-by-step. Follow these instructions in order, and the application will be up and running.

## Prerequisites

Before starting, make sure you have:
- **Python 3.8 or higher** installed on your system
- **pip** (Python package manager) installed
- **Git** installed
- **Internet connection** (for downloading dependencies and dataset)

### Check Prerequisites

**Check Python version:**
```bash
python3 --version
# Should show Python 3.8 or higher
```

**Check pip:**
```bash
pip3 --version
# Should show pip version
```

**Check Git:**
```bash
git --version
# Should show git version
```

If any of these are missing, install them first:
- **Python**: https://www.python.org/downloads/
- **Git**: https://git-scm.com/downloads

---

## Step 1: Clone the Repository

### Option A: Using SSH (Recommended if you have SSH keys set up)

```bash
git clone git@github.com:nileshkadale2020/diabetes_prediction.git
cd diabetes_prediction
```

### Option B: Using HTTPS

```bash
git clone https://github.com/nileshkadale2020/diabetes_prediction.git
cd diabetes_prediction
```

**Note**: If the repository is private, you'll need to authenticate. Use SSH keys or a Personal Access Token.

### Verify Clone

After cloning, you should see the project structure:
```bash
ls -la
```

You should see directories like: `src/`, `app/`, `notebooks/`, `data/`, `models/`, etc.

---

## Step 2: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python.

### On macOS/Linux:

```bash
python3 -m venv dtsc691
source dtsc691/bin/activate
```

### On Windows:

```bash
python -m venv dtsc691
dtsc691\Scripts\activate
```

**Verify activation:**
You should see `(dtsc691)` at the beginning of your terminal prompt:
```
(dtsc691) user@computer:~/diabetes_prediction$
```

**Important**: Keep the terminal open and the virtual environment activated for all subsequent steps.

---

## Step 3: Install Dependencies

Install all required Python packages:

```bash
# Make sure virtual environment is activated (you should see (dtsc691) in prompt)
pip install --upgrade pip
pip install -r requirements.txt
```

**This will install:**
- pandas, numpy (data manipulation)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- tensorflow (deep learning)
- matplotlib, seaborn (visualization)
- flask (web framework)
- joblib (model persistence)
- shap, lime (model interpretability)

**Installation time**: 5-10 minutes depending on your internet speed.

### Troubleshooting: XGBoost on macOS

If you get an XGBoost error on macOS, install OpenMP first:

```bash
brew install libomp
```

Then reactivate the virtual environment:
```bash
deactivate
source dtsc691/bin/activate  # macOS/Linux
# OR
dtsc691\Scripts\activate  # Windows
```

### Verify Installation

Check that packages are installed:
```bash
python -c "import pandas, numpy, sklearn, tensorflow, flask; print('All packages installed successfully!')"
```

---

## Step 4: Download the Dataset

The dataset is not included in the repository (for privacy and size reasons). Download it:

### Option A: Using the Download Script (Recommended)

```bash
python download_data.py
```

This will:
- Download the PIMA Indians Diabetes dataset
- Save it to `data/diabetes.csv`
- Verify the download

### Option B: Manual Download

1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download the dataset
3. Place it in the `data/` directory as `diabetes.csv`

**Verify dataset:**
```bash
ls -lh data/diabetes.csv
# Should show the file exists
```

---

## Step 5: Preprocess the Data

Clean and prepare the data for model training:

```bash
python src/data_preprocessing.py
```

**This will:**
- Load the dataset
- Replace zeros with NaN (missing values)
- Impute missing values with median
- Scale features using StandardScaler
- Split data into training (80%) and testing (20%) sets
- Save preprocessors (`scaler.pkl`, `imputer.pkl`) to `models/` directory

**Output**: You should see messages about:
- Dataset loaded
- Missing values replaced
- Data preprocessing completed
- Data split information

**Time**: 1-2 minutes

---

## Step 6: Train the Models

Train all machine learning models:

```bash
python src/model_training.py
```

**This will train:**
1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **XGBoost** - Gradient boosting model (if available)
4. **Neural Network** - Deep learning model (TensorFlow)

**⚠️ Important**: This step takes **30-60 minutes** depending on your system.

**What's happening:**
- Each model is trained with hyperparameter tuning
- Models are saved to `models/` directory
- Best hyperparameters are selected using GridSearchCV

**Output**: You should see:
- Training progress for each model
- Best parameters for each model
- Models saved to disk

**Note**: If XGBoost is not available (e.g., on macOS without libomp), the script will continue with other models.

---

## Step 7: Evaluate the Models (Optional but Recommended)

Evaluate and compare all trained models:

```bash
python src/model_evaluation.py
```

**This will:**
- Load all trained models
- Evaluate on test set
- Calculate metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Create visualizations (confusion matrices, ROC curves)
- Save comparison table
- Identify best model

**Output**: 
- Model comparison table in console
- Visualizations saved to `models/` directory
- Best model name saved to `models/best_model.txt`

**Time**: 2-3 minutes

---

## Step 8: Run the Flask Application

Start the web application:

```bash
python app.py
```

**Output**: You should see:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

**Access the application:**
1. Open your web browser
2. Go to: **http://localhost:5000**
3. You should see the homepage

**Application pages:**
- **Homepage**: http://localhost:5000/
- **Resume**: http://localhost:5000/resume
- **Projects**: http://localhost:5000/projects
- **Diabetes Prediction**: http://localhost:5000/diabetes

**To stop the application:**
Press `Ctrl + C` in the terminal

---

## Step 9: Test the Application

### Test the Diabetes Prediction Page

1. Navigate to: http://localhost:5000/diabetes
2. Fill in the form with example values:

**Example 1: High Risk Patient**
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
```

3. Click "Predict Diabetes Risk"
4. You should see:
   - Probability percentage
   - Risk level (High/Low)
   - Interpretation message
   - Predictions from all models

**If you see an error**: Make sure you completed Steps 4-6 (download data, preprocess, train models).

---

## Quick Setup Script (Alternative)

If you prefer to run everything at once:

```bash
# Make script executable (macOS/Linux)
chmod +x run_project.sh

# Run the script
./run_project.sh
```

**Note**: This script will run all steps automatically. It may take 30-60 minutes.

---

## Troubleshooting

### Error: "Models not found"

**Solution**: Make sure you completed:
1. Step 4: Download dataset
2. Step 5: Preprocess data
3. Step 6: Train models

Check that model files exist:
```bash
ls -lh models/*.pkl models/*.h5
```

### Error: "Dataset not found"

**Solution**: Run Step 4 again:
```bash
python download_data.py
```

### Error: "Import errors"

**Solution**: 
1. Make sure virtual environment is activated (you should see `(dtsc691)` in prompt)
2. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Error: "XGBoost not available"

**Solution** (macOS):
```bash
brew install libomp
deactivate
source dtsc691/bin/activate
pip install xgboost
```

**Note**: The application will work without XGBoost - it will use other models.

### Error: "TensorFlow errors"

**Solution**:
```bash
pip install tensorflow --upgrade
```

### Flask app won't start

**Solution**:
1. Make sure port 5000 is not in use
2. Check that models are trained (Step 6)
3. Verify virtual environment is activated

### Virtual environment not activating

**Solution**:
```bash
# Remove and recreate
rm -rf dtsc691  # macOS/Linux
# OR
rmdir /s dtsc691  # Windows

# Recreate
python3 -m venv dtsc691
source dtsc691/bin/activate  # macOS/Linux
# OR
dtsc691\Scripts\activate  # Windows
```

---

## Project Structure

After setup, your project should look like:

```
diabetes_prediction/
├── data/
│   └── diabetes.csv              # Dataset (downloaded in Step 4)
├── models/
│   ├── scaler.pkl               # Feature scaler (created in Step 5)
│   ├── imputer.pkl              # Missing value imputer (created in Step 5)
│   ├── logistic_regression.pkl  # Trained model (created in Step 6)
│   ├── random_forest.pkl        # Trained model (created in Step 6)
│   ├── xgboost.pkl              # Trained model (created in Step 6, if available)
│   ├── neural_network.h5        # Trained model (created in Step 6)
│   ├── best_model.txt           # Best model name (created in Step 7)
│   └── *.png                    # Visualizations (created in Step 7)
├── notebooks/
│   └── diabetes_prediction_complete.ipynb  # Jupyter notebook
├── src/
│   ├── data_preprocessing.py    # Data cleaning script
│   ├── model_training.py        # Model training script
│   ├── model_evaluation.py      # Model evaluation script
│   └── utils.py                 # Utility functions
├── app/
│   ├── templates/               # HTML templates
│   └── static/                  # CSS files
├── dtsc691/                     # Virtual environment (created in Step 2)
├── app.py                       # Flask application
├── requirements.txt             # Python dependencies
├── download_data.py             # Dataset download script
└── README.md                    # Project documentation
```

---

## Summary Checklist

Follow these steps in order:

- [ ] **Step 1**: Clone repository
- [ ] **Step 2**: Create virtual environment
- [ ] **Step 3**: Install dependencies
- [ ] **Step 4**: Download dataset
- [ ] **Step 5**: Preprocess data
- [ ] **Step 6**: Train models (30-60 minutes)
- [ ] **Step 7**: Evaluate models (optional)
- [ ] **Step 8**: Run Flask application
- [ ] **Step 9**: Test the application

**Total setup time**: Approximately 40-70 minutes (mostly waiting for model training)

---

## Next Steps

After the application is running:

1. **Explore the Jupyter notebook**: `notebooks/diabetes_prediction_complete.ipynb`
   - Contains complete EDA and analysis
   - Run: `jupyter notebook notebooks/diabetes_prediction_complete.ipynb`

2. **Customize the application**:
   - Edit templates in `app/templates/`
   - Modify styles in `app/static/css/style.css`
   - Update routes in `app.py`

3. **Review model performance**:
   - Check `models/model_comparison.csv`
   - View visualizations in `models/` directory

4. **Deploy to cloud** (optional):
   - Deploy to Render, Railway, or AWS
   - See deployment documentation

---

## Support

If you encounter any issues:

1. Check the **Troubleshooting** section above
2. Review error messages carefully
3. Verify all prerequisites are installed
4. Make sure you followed all steps in order
5. Check that virtual environment is activated

**Common issues are usually:**
- Virtual environment not activated
- Missing dataset (Step 4)
- Models not trained (Step 6)
- Dependencies not installed (Step 3)

---

## Contact

For questions or issues, refer to:
- **README.md**: Project overview
- **QUICKSTART.md**: Quick reference
- **PRESENTATION_SCRIPT.md**: Project presentation

---

**Good luck! The application should be up and running after completing all steps.**

