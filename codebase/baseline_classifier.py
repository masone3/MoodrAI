"""
baseline_classifier.py
----------------------
Baseline models for valence/arousal prediction.
Includes Logistic Regression as a starting point.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, auc

def set_backend(interactive=True):
    if interactive:
        matplotlib.use("TkAgg")  # normal GUI backend
    else:
        matplotlib.use("Agg")    # headless backend

def train_baseline_model(
    dataset: pd.DataFrame, 
    label_col: str, 
    model_type: str = "logreg",   # "logreg" or "svm" or "randomforest",
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

    set_backend(visualize)

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

        # --- Extra Metrics ---
        f1 = f1_score(y, y_pred, average="macro")
        print(f"📊 {model_type.upper()} {label_col} Metrics:")
        print(f"   Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   F1-score (macro): {f1:.3f}")

        # ROC-AUC (only valid for binary classification by default)
        y_proba = None
        roc_auc = None
        try:
            if hasattr(trained_model, "predict_proba"):
                y_proba = cross_val_predict(pipeline, X, y, cv=kf, method="predict_proba")[:, 1]
            elif hasattr(trained_model, "decision_function"):
                y_proba = cross_val_predict(pipeline, X, y, cv=kf, method="decision_function")

            if y_proba is not None:
                roc_auc = roc_auc_score(y, y_proba)
                print(f"   ROC-AUC: {roc_auc:.3f}")

                fpr, tpr, _ = roc_curve(y, y_proba)
                roc_auc_val = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
                plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve for {label_col} ({model_type.upper()})")
                plt.legend(loc="lower right")

                if save_plots_dir:
                    fname = f"{model_type}_{label_col}_roc.png"
                    plt.savefig(os.path.join(save_plots_dir, fname))
                    plt.close()
                elif visualize:
                    plt.show()
                plt.close()
        except Exception as e:
            print(f"⚠️ Could not compute ROC-AUC: {e}")
        

    return trained_model, scaler, dataset