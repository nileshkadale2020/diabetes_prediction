"""
Script to download the PIMA Indians Diabetes Dataset
This script provides instructions and attempts to download the dataset
"""

import os
import urllib.request
import ssl
import pandas as pd

def download_dataset():
    """Download the PIMA Indians Diabetes Dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    # Column names for the dataset
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    print("Downloading PIMA Indians Diabetes Dataset...")
    print(f"Source: {url}")
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Try downloading with SSL verification first
        try:
            urllib.request.urlretrieve(url, 'data/diabetes.csv')
        except urllib.error.URLError as e:
            # If SSL verification fails, try without verification
            print("SSL verification failed, trying without verification...")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open('data/diabetes.csv', 'wb') as out_file:
                    out_file.write(response.read())
            print("Downloaded without SSL verification")
        
        # Read and verify the dataset
        df = pd.read_csv('data/diabetes.csv', names=column_names)
        
        print(f"\nDataset downloaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Save with proper column names
        df.to_csv('data/diabetes.csv', index=False)
        
        print(f"\nDataset saved to data/diabetes.csv")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Please download the dataset manually from:")
        print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print("\nAnd place it in the 'data/' directory as 'diabetes.csv'")
        return False

if __name__ == "__main__":
    download_dataset()

