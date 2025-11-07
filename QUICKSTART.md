# Quick Start Guide

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   ```bash
   python download_data.py
   ```
   Or manually download from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in `data/diabetes.csv`

3. **Preprocess Data**
   ```bash
   python src/data_preprocessing.py
   ```

4. **Train Models**
   ```bash
   python src/model_training.py
   ```
   Note: This may take 30-60 minutes depending on your system

5. **Evaluate Models**
   ```bash
   python src/model_evaluation.py
   ```

6. **Run Flask Application**
   ```bash
   python app.py
   ```
   Then open http://localhost:5000 in your browser

## Alternative: Run All Steps at Once

```bash
./run_project.sh
```

## Project Structure

```
Cursor_diabtestPred/
├── data/                    # Dataset files
├── models/                  # Trained models and results
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app/                     # Flask application
│   ├── templates/           # HTML templates
│   └── static/              # CSS and static files
├── app.py                   # Main Flask app
├── download_data.py         # Dataset download script
└── requirements.txt         # Python dependencies
```

## Troubleshooting

### Models Not Found Error
If you see "Models not found" error when running the Flask app, make sure you've:
1. Downloaded the dataset
2. Run data preprocessing
3. Trained the models

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### TensorFlow Issues
If you encounter TensorFlow installation issues, try:
```bash
pip install tensorflow --upgrade
```

## Next Steps

1. Explore the Jupyter notebooks in `notebooks/` for detailed EDA
2. Customize the Flask templates in `app/templates/`
3. Add more features or models as needed
4. Deploy to cloud platforms like Render or Railway

## Support

For issues or questions, please refer to the main README.md file.

