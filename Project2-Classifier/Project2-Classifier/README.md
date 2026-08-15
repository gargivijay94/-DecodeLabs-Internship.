# Iris Flower Classifier (Supervised Learning)

## Description
A supervised machine learning project built in Python as part of DecodeLabs' AI Engineering Training Kit (Project 2). It trains a K-Nearest Neighbors (KNN) model on the classic Iris dataset to classify flowers into one of three species based on their sepal and petal measurements.

**Key features:**
- Loads and explores the Iris dataset (150 samples, 4 features, 3 balanced classes)
- Stratified train/test split to preserve class balance
- Feature scaling with `StandardScaler` (important for distance-based models like KNN)
- Automatic hyperparameter tuning — sweeps K from 1–20 to find the best-performing value
- Full evaluation: accuracy, F1 score, confusion matrix, and classification report
- Demonstrates prediction on a brand-new, unseen data point

## How to Run
1. Make sure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install scikit-learn numpy
   ```
3. Run the script:
   ```bash
   python3 classifier.py
   ```
4. The script will print dataset info, the best K value found, evaluation metrics, and a sample prediction.

## Tech Stack
- Python 3
- scikit-learn (dataset, model, metrics)
- NumPy
