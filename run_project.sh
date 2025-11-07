#!/bin/bash
# Script to run the complete project pipeline

echo "=========================================="
echo "Diabetes Prediction Project - Setup & Run"
echo "=========================================="

# Step 1: Download dataset
echo ""
echo "Step 1: Downloading dataset..."
python3 download_data.py

# Step 2: Preprocess data
echo ""
echo "Step 2: Preprocessing data..."
python3 src/data_preprocessing.py

# Step 3: Train models
echo ""
echo "Step 3: Training models..."
echo "This may take a while..."
python3 src/model_training.py

# Step 4: Evaluate models
echo ""
echo "Step 4: Evaluating models..."
python3 src/model_evaluation.py

# Step 5: Run Flask app
echo ""
echo "Step 5: Starting Flask application..."
echo "Open http://localhost:5000 in your browser"
python3 app.py

