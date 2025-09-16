"""
baseline_classifier.py
----------------------
Baseline models for valence/arousal prediction.
Includes Logistic Regression as a starting point.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_logistic_regression(df: pd.DataFrame, label_col: str, test_size=0.2, random_state=42):
    """
    Trains a logistic regression classifier on extracted features.
    
    Args:
        df: DataFrame containing features + labels
        label_col: column name of the binary target label
        test_size: proportion of dataset used for testing
        random_state: reproducibility
    
    Returns:
        model, scaler, df with predictions
    """
    # Select feature columns (adjust if you add more)
    feature_cols = ['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'mfcc_14', 'mfcc_15', 'mfcc_16', 'mfcc_17', 'mfcc_18', 'mfcc_19', 'chroma_0', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'contrast_0', 'contrast_1', 'contrast_2', 'contrast_3', 'contrast_4', 'contrast_5', 'contrast_6', 'centroid', 'bandwidth', 'rolloff', 'zcr', 'tempo', 'song_id', 'valence_mean', 'arousal_mean']

    # Expand vector features into separate columns
    expanded = []
    for idx, row in df.iterrows():
        row_feats = {}
        for col in feature_cols:
            val = row[col]
            if isinstance(val, np.ndarray):  # expand vectors
                for i, v in enumerate(val):
                    row_feats[f"{col}_{i}"] = v
            else:  # keep scalar as is
                row_feats[col] = val
        expanded.append(row_feats)

    X = pd.DataFrame(expanded)
    y = df[label_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"{label_col} Logistic Regression Accuracy: {acc:.2f}")

    # Save predictions back into DataFrame (only for test set for now)
    df.loc[X_test.index, f"{label_col}_pred"] = y_pred

    return model, scaler, df
