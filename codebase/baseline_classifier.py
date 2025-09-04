"""
baseline_classifier.py
----------------------
Implements scikit-learn baseline models for music mood classification.
Uses features provided by data_loader.py.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset loader
from codebase.data_loader import load_deam_dataset_mfcc


def train_baseline_classifier(data_dir, annotations_file):
    # 1. Load dataset
    print("Loading dataset...")
    X, y = load_deam_dataset_mfcc(data_dir, annotations_file, n_samples=200)  # adjust n_samples later

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Accuracy: {acc:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Baseline Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return clf


if __name__ == "__main__":
    # Example run
    DATA_DIR = "data/audio/DEAM/audio"  # adjust if needed
    ANNOTATIONS_FILE = "data/DEAM/annotations.csv"

    train_baseline_classifier(DATA_DIR, ANNOTATIONS_FILE)
