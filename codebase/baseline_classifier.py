"""
baseline_classifier.py
----------------------
Baseline models for valence/arousal prediction.
Includes Logistic Regression as a starting point.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_logistic_regression(dataset: pd.DataFrame, label_col: str, n_splits=5, random_state=42):
    """
    Trains and evaluates a logistic regression classifier with k-fold cross-validation.

    Args:
        dataset: DataFrame containing features + labels
        label_col: column name of the binary target label
        n_splits: number of folds for cross-validation
        random_state: reproducibility

    Returns:
        model, scaler, dataset with predictions
    """

    # Define the feature columns (exclude non-numeric / label cols)
    feature_cols = [
        col for col in dataset.columns
        if col not in [
            "song_id", "filename",
            "valence_mean", "arousal_mean",
            "valence_label", "arousal_label"
        ]
    ]

    # Copy dataset to avoid modifying in place
    dataset = dataset.copy()

    X = dataset[feature_cols]
    y = dataset[label_col]

    # Build pipeline (imputer -> scaler -> logistic regression)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="accuracy")

    print(f"🔎 {label_col} Logistic Regression CV Accuracy: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train final model on full dataset
    pipeline.fit(X, y)

    # Add predictions back into dataset
    dataset[f"{label_col.replace('_label','')}_pred"] = pipeline.predict(X)

    # Extract model + scaler separately for return (useful later)
    model = pipeline.named_steps["logreg"]
    scaler = pipeline.named_steps["scaler"]

    return model, scaler, dataset