"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit — Batch 2026

Pipeline (IPO Framework):
  INPUT   -> Load the Iris dataset (150 samples, 4 features, 3 classes)
  PROCESS -> Scale features, split train/test, train a K-Nearest Neighbors model
  OUTPUT  -> Predict on the test set, report accuracy, confusion matrix, F1 score

Key skills demonstrated: data handling, supervised learning basics, model training.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)


def load_data():
    """INPUT: Load and understand the dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target
    print("Dataset loaded: Iris Benchmark")
    print(f"  Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(iris.target_names)}")
    print(f"  Classes: {list(iris.target_names)}")
    return X, y, iris.target_names


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """PROCESS (part 1): Shuffle + split into train/test, then scale features.

    Scaling matters here because KNN is distance-based — unscaled features
    with larger numeric ranges would dominate the distance calculation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only — no leakage from test set

    print(f"\nSplit complete: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """Tune K by testing a range of neighbor counts and picking the best accuracy."""
    best_k, best_acc = 1, 0.0
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_k, best_acc = k, acc
    print(f"\nBest K found: {best_k} (test accuracy: {best_acc:.2%})")
    return best_k


def train_and_evaluate(X_train, X_test, y_train, y_test, target_names, k=5):
    """PROCESS (part 2) + OUTPUT: Instantiate, fit, predict, and validate."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print(f"\n--- Model Evaluation (K={k}) ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"F1 Score (macro): {f1:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    return model


def classify_new_sample(model, scaler, target_names, sample):
    """Demonstrate the trained model on a brand-new, unseen data point."""
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0]
    print(f"New sample {sample} -> Predicted class: {target_names[prediction]}")


def main():
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    best_k = find_best_k(X_train, y_train, X_test, y_test)
    model = train_and_evaluate(X_train, X_test, y_train, y_test, target_names, k=best_k)

    # Try the model on a new, unseen flower measurement
    classify_new_sample(model, scaler, target_names, sample=[5.9, 3.0, 5.1, 1.8])


if __name__ == "__main__":
    main()
