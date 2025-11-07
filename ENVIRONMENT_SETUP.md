# Virtual Environment Setup

## Virtual Environment: `dtsc691`

A virtual environment named `dtsc691` has been created with all required libraries installed.

## How to Activate the Virtual Environment

### Option 1: Using the activation script (macOS/Linux)
```bash
source activate_env.sh
```

### Option 2: Manual activation (macOS/Linux)
```bash
source dtsc691/bin/activate
```

### Option 3: Manual activation (Windows)
```bash
dtsc691\Scripts\activate
```

## Installed Packages

The following packages have been installed in the virtual environment:

- **pandas** (2.0.3) - Data manipulation and analysis
- **numpy** (1.24.3) - Numerical computing
- **scikit-learn** (1.3.2) - Machine learning algorithms
- **xgboost** (2.1.4) - Gradient boosting framework
- **tensorflow** (2.13.1) - Deep learning framework
- **matplotlib** (3.7.5) - Data visualization
- **seaborn** (0.13.2) - Statistical data visualization
- **flask** (3.0.3) - Web framework
- **joblib** (1.4.2) - Model persistence
- **shap** (0.44.1) - Model interpretability
- **lime** (0.2.0.1) - Model interpretability
- **pypdf** (5.9.0) - PDF processing

## Verify Installation

To verify all packages are installed correctly:

```bash
source dtsc691/bin/activate
python -c "import pandas, numpy, sklearn, xgboost, tensorflow, matplotlib, seaborn, flask, joblib, shap, lime, pypdf; print('All packages imported successfully!')"
```

## Using the Virtual Environment

### Running the Jupyter Notebook
```bash
source dtsc691/bin/activate
jupyter notebook notebooks/diabetes_prediction_complete.ipynb
```

### Running the Flask Application
```bash
source dtsc691/bin/activate
python app.py
```

### Running Data Preprocessing
```bash
source dtsc691/bin/activate
python src/data_preprocessing.py
```

### Running Model Training
```bash
source dtsc691/bin/activate
python src/model_training.py
```

## Deactivating the Virtual Environment

To deactivate the virtual environment:

```bash
deactivate
```

## Notes

- The virtual environment uses Python 3.8.1
- All packages are compatible with this Python version
- The virtual environment is located in the `dtsc691/` directory
- Make sure to activate the environment before running any Python scripts

## Troubleshooting

If you encounter any issues:

1. **Activation fails**: Make sure you're in the project root directory
2. **Import errors**: Activate the virtual environment first
3. **Package not found**: Reinstall packages using `pip install -r requirements.txt` (with environment activated)
4. **XGBoost error on macOS**: If you see an error about `libomp.dylib`, install OpenMP runtime:
   ```bash
   brew install libomp
   ```
   Then reactivate the virtual environment.

