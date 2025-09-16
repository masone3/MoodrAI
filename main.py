"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

from codebase.data_loader import load_deam_annotations, add_binary_labels
from codebase.features import extract_features
from codebase.baseline_classifier import train_logistic_regression


def main():
    # -------------------------------------------------------------------
    # Step 1. Load annotations (valence, arousal, etc.)
    # -------------------------------------------------------------------
    annotations_csv = "data/DEAM/processed_annotations.csv"
    annotations = load_deam_annotations(annotations_csv)

    # Add binary labels (valence: high vs low, arousal: high vs low)
    annotations = add_binary_labels(annotations)

    print(f"✅ Loaded {len(annotations)} annotations with labels")
    print(annotations[["song_id", "valence_mean", "valence_label", 
                       "arousal_mean", "arousal_label"]].head())

    # -------------------------------------------------------------------
    # Step 2. Extract features for each audio file
    # -------------------------------------------------------------------
    features_out = "data/DEAM/features.csv"

    if os.path.exists(features_out):
        print("🔄 Loading cached features...")
        df_features = pd.read_csv(features_out)
    else:
        print("⏳ Extracting features (first run)...")
        features_list = []
        for _, row in annotations.iterrows():
            track_id = row["song_id"]
            audio_path = f"data/DEAM/audio/{track_id}.mp3"

            try:
                feats = extract_features(audio_path)
                feats["song_id"] = track_id
                feats["valence_mean"] = row["valence_mean"]
                feats["arousal_mean"] = row["arousal_mean"]
                features_list.append(feats)
                print(f"✅ Extracted features for track {track_id}")
            except Exception as e:
                print(f"⚠️ Skipping {track_id} due to error: {e}")

        df_features = pd.DataFrame(features_list)
        df_features.to_csv(features_out, index=False)
        print(f"✅ Saved extracted features to {features_out}")

    # -------------------------------------------------------------------
    # Step 3. Visualization
    # -------------------------------------------------------------------
    if "valence_mean" in df_features and "arousal_mean" in df_features:
        plt.scatter(df_features["valence_mean"], df_features["arousal_mean"])
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        plt.title("DEAM Dataset: Valence vs Arousal")
        plt.show()
    else:
        print("⚠️ Skipping visualization (valence/arousal not found in features DataFrame).")

    # -------------------------------------------------------------------
    # Step 4. Merge features with annotations for training
    # -------------------------------------------------------------------
    #dataset = pd.merge(df_features, annotations, on="song_id", how="inner")
    dataset = pd.merge(df_features, annotations, on="song_id")

    print(f"✅ Merged dataset shape: {dataset.shape}")

    # -------------------------------------------------------------------
    # Step 5. Train baseline models
    # -------------------------------------------------------------------
    
    # Train on valence
    valence_model, scaler, dataset = train_logistic_regression(dataset, "valence_label")

    # Train on arousal
    arousal_model, scaler, dataset = train_logistic_regression(dataset, "arousal_label")

    print(dataset[["valence_label", "valence_pred", "arousal_label", "arousal_pred"]].head())


if __name__ == "__main__":
    main()
