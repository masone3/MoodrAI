"""
baseline_classifier.py
----------------------
Baseline models for valence/arousal prediction.
Includes Logistic Regression as a starting point.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def train_logistic_regression(
    dataset: pd.DataFrame, 
    label_col: str, 
    n_splits=5, 
    random_state=42, 
    visualize=True,
    save_plots_dir=None
):
    """
    Trains and evaluates a logistic regression classifier with k-fold cross-validation,
    and optionally generates feature importance + evaluation visualizations.

    Args:
        dataset: DataFrame containing features + labels
        label_col: column name of the binary target label
        n_splits: number of folds for cross-validation
        random_state: reproducibility
        visualize: whether to show plots (default=True)
        save_plots_dir: folder path to save plots (default=None means no saving)

    Returns:
        model, scaler, dataset with predictions
    """

    if save_plots_dir:
        os.makedirs(save_plots_dir, exist_ok=True)

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
    
    # Visualizing the information
    if visualize or save_plots_dir:
        # --- CV Accuracy per Fold ---
        plt.figure()
        plt.plot(range(1, len(cv_scores)+1), cv_scores, marker="o")
        plt.title(f"{label_col} Accuracy per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        if save_plots_dir:
            plt.savefig(os.path.join(save_plots_dir, f"{label_col}_cv_accuracy_folds.png"))
        if visualize:
            plt.show()
        plt.close()

        # --- CV Accuracy Distribution ---
        plt.figure()
        plt.hist(cv_scores, bins=n_splits)
        plt.title(f"{label_col} Accuracy Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        if save_plots_dir:
            plt.savefig(os.path.join(save_plots_dir, f"{label_col}_cv_accuracy_distribution.png"))
        if visualize:
            plt.show()
        plt.close()

    # Train final model on full dataset
    pipeline.fit(X, y)
    dataset[f"{label_col.replace('_label','')}_pred"] = pipeline.predict(X)

    if visualize or save_plots_dir:
        # --- Confusion Matrix ---
        y_pred = cross_val_predict(pipeline, X, y, cv=kf)
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred)
        plt.title(f"{label_col} Confusion Matrix (Cross-Val)")
        if save_plots_dir:
            plt.savefig(os.path.join(save_plots_dir, f"{label_col}_confusion_matrix.png"))
        if visualize:
            plt.show()
        plt.close()

        # --- Feature Importance ---
        model = pipeline.named_steps["logreg"]
        scaler = pipeline.named_steps["scaler"]
        feature_importance = pd.Series(model.coef_[0], index=feature_cols)
        top_features = feature_importance.abs().sort_values(ascending=False).head(15)

        plt.figure()
        top_features.plot(kind="barh")
        plt.title(f"Top Features for {label_col} (LogReg Importance)")
        plt.xlabel("Coefficient Magnitude")
        if save_plots_dir:
            plt.savefig(os.path.join(save_plots_dir, f"{label_col}_feature_importance.png"))
        if visualize:
            plt.show()
        plt.close()

    return model, scaler, dataset