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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def train_baseline_model(
    dataset: pd.DataFrame, 
    label_col: str, 
    model_type: str = "logreg",   # "logreg" or "svm",
    n_splits=5, 
    random_state=42, 
    visualize=True,
    save_plots_dir=None
):
    """
    Trains and evaluates a baseline model (Logistic Regression or SVM) with k-fold cross-validation.

    Args:
        dataset: DataFrame containing features + labels
        label_col: column name of the binary target label
        model_type: "logreg" for Logistic Regression, "svm" for Support Vector Machine, "randomforest" for Random Forest Classifier
        n_splits: number of folds for cross-validation
        random_state: reproducibility
        visualize: whether to show/save evaluation plots
        save_dir: optional directory path to save plots

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

    # Choose model
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=random_state)
    elif model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=100,       # number of trees (default=100 is usually fine)
            max_depth=None,         # allow trees to expand fully
            random_state=random_state,
            n_jobs=-1               # use all CPU cores
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build pipeline (imputer -> scaler -> logistic regression)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="accuracy")

    print(f"🔎 {model_type.upper()} {label_col} CV Accuracy: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
    # Train final model on full dataset
    pipeline.fit(X, y)

    # Add predictions
    dataset[f"{label_col.replace('_label','')}_pred"] = pipeline.predict(X)

    if visualize or save_plots_dir:
        # --- CV Accuracy per Fold ---
        plt.figure()
        plt.plot(range(1, len(cv_scores)+1), cv_scores, marker="o")
        plt.title(f"{label_col} Accuracy per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            fname = f"{model_type}_{label_col}cv_accuracy_folds.png"
            plt.savefig(os.path.join(save_plots_dir, fname))
            plt.close()
        elif visualize:
            plt.show()
        plt.close()

        # --- CV Accuracy Distribution ---
        plt.figure()
        plt.hist(cv_scores, bins=n_splits)
        plt.title(f"{label_col} Accuracy Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            fname = f"{model_type}_{label_col}sv_accuracy_distribution.png"
            plt.savefig(os.path.join(save_plots_dir, fname))
            plt.close()
        elif visualize:
            plt.show()
        plt.close()

        # --- Confusion Matrix ---
        y_pred = cross_val_predict(pipeline, X, y, cv=kf)
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred)
        plt.title(f"{label_col} Confusion Matrix (Cross-Val)")
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            fname = f"{model_type}_{label_col}confusion.png"
            plt.savefig(os.path.join(save_plots_dir, fname))
            plt.close()
        elif visualize:
            plt.show()
        plt.close()

        # --- Feature Importance ---
        # Extract trained components
        trained_model = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["scaler"]

        if model_type == "logreg":
            feature_importance = pd.Series(trained_model.coef_[0], index=feature_cols)
            top_features = feature_importance.abs().sort_values(ascending=False).head(15)

        elif model_type == "randomforest":
            feature_importance = pd.Series(trained_model.feature_importances_, index=feature_cols)
            top_features = feature_importance.sort_values(ascending=False).head(15)

        else:
            top_features = None  # SVM with RBF kernel doesn't provide feature importance

        if top_features is not None:
            plt.figure()
            top_features.plot(kind="barh")
            plt.title(f"Top Features for {label_col} ({model_type.upper()} Importance)")
            plt.xlabel("Importance")
            if save_plots_dir:
                fname = f"{model_type}_{label_col}_feature_importance.png"
                plt.savefig(os.path.join(save_plots_dir, fname))
                plt.close()
            elif visualize:
                plt.show()
            plt.close()

    return trained_model, scaler, dataset