"""
train_regression.py
-------------------
Train models that predict DEAM's continuous valence and arousal scores (1-9)
instead of a high/low label, and report how well they do.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from codebase.data_loader import load_deam_annotations

warnings.filterwarnings("ignore")

ANNOTATIONS = "data/DEAM/processed_annotations.csv"
FEATURES = "data/DEAM/features.csv"
MODELS_DIR = Path("models")

REGRESSORS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "svr": lambda: SVR(kernel="rbf", C=3.0),
    "rf": lambda: RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
}


HARMONY = "data/DEAM/features_harmony.csv"


def load_data(with_harmony=False):
    annotations = load_deam_annotations(ANNOTATIONS)
    features = pd.read_csv(FEATURES)
    df = features.merge(
        annotations[["song_id", "valence_mean", "arousal_mean"]], on="song_id"
    )
    feature_cols = [c for c in features.columns if c != "song_id"]
    if with_harmony:
        harmony = pd.read_csv(HARMONY)
        df = df.merge(harmony, on="song_id")
        feature_cols += [c for c in harmony.columns if c != "song_id"]
    return df, feature_cols


def make_pipeline(name: str) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", REGRESSORS[name]()),
    ])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df, feature_cols = load_data(with_harmony=True)
    X = df[feature_cols]
    print(f"Training on {len(df)} songs, {len(feature_cols)} features\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    reference = {}

    for axis in ("valence", "arousal"):
        y = df[f"{axis}_mean"]
        guess_mean_mae = mean_absolute_error(y, np.full(len(y), y.mean()))

        for name in REGRESSORS:
            pipe = make_pipeline(name)
            pred = cross_val_predict(pipe, X, y, cv=cv)
            if name == "rf":
                reference[axis] = np.sort(pred)

            rows.append({
                "axis": axis,
                "model": name,
                "MAE": round(mean_absolute_error(y, pred), 3),
                "always-guess-mean MAE": round(guess_mean_mae, 3),
                "R2": round(r2_score(y, pred), 3),
                "pred range": f"{pred.min():.1f}-{pred.max():.1f}",
            })

            pipe.fit(X, y)  # final model trained on every song
            with open(MODELS_DIR / f"reg_{name}_{axis}.pkl", "wb") as f:
                pickle.dump(pipe, f)

    with open(MODELS_DIR / "reg_reference.pkl", "wb") as f:
        pickle.dump(reference, f)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTrue score range: valence {df.valence_mean.min():.1f}-{df.valence_mean.max():.1f}, "
          f"arousal {df.arousal_mean.min():.1f}-{df.arousal_mean.max():.1f}")


if __name__ == "__main__":
    main()