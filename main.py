"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from codebase.data_loader import load_deam_annotations, add_binary_labels
from codebase.features import extract_features
from codebase.baseline_classifier import train_logistic_regression


def main():
    # -------------------------------------------------------------------
    # Step 1. Load annotations + add labels
    # -------------------------------------------------------------------
    annotations_csv = "data/DEAM/processed_annotations.csv"
    annotations = load_deam_annotations(annotations_csv)

    df_labels = add_binary_labels(annotations)

    print(f"✅ Loaded {len(df_labels)} annotations with labels")
    print(df_labels[["song_id", "valence_mean", "valence_label",
                     "arousal_mean", "arousal_label"]])

    # -------------------------------------------------------------------
    # Step 2. Extract or load cached features
    # -------------------------------------------------------------------
    features_out = "data/DEAM/features.csv"

    if os.path.exists(features_out):
        print("🔄 Loading cached features...")
        df_features = pd.read_csv(features_out)
    else:
        print("⏳ Extracting features (first run)...")
        features_list = []
        for _, row in df_labels.iterrows():
            track_id = row["song_id"]
            audio_path = f"data/DEAM/audio/{track_id}.mp3"

            try:
                feats = extract_features(audio_path)
                feats["song_id"] = track_id
                features_list.append(feats)
                print(f"✅ Extracted features for track {track_id}")
            except Exception as e:
                print(f"⚠️ Skipping {track_id} due to error: {e}")

        df_features = pd.DataFrame(features_list)
        df_features.to_csv(features_out, index=False)
        print(f"✅ Saved extracted features to {features_out}")

    # -------------------------------------------------------------------
    # Step 3. Visualization (Valence vs Arousal)
    # -------------------------------------------------------------------
    plt.scatter(df_labels["valence_mean"], df_labels["arousal_mean"])
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("DEAM Dataset: Valence vs Arousal")
    plt.show()

    # -------------------------------------------------------------------
    # Step 4. Merge features with labels
    # -------------------------------------------------------------------
    dataset = pd.merge(df_features, df_labels, on="song_id")

    print("\n✅ Merged dataset with features + labels:")
    print(dataset)

    # Resolve duplicate valence/arousal columns after merge
    if "valence_mean_x" in dataset.columns and "valence_mean_y" in dataset.columns:
        # keep the labels version (_y) and drop the features version (_x)
        dataset = dataset.drop(columns=["valence_mean_x", "arousal_mean_x"])
        dataset = dataset.rename(columns={
            "valence_mean_y": "valence_mean",
            "arousal_mean_y": "arousal_mean"
        })

    # -------------------------------------------------------------------
    # Step 5. Train classifiers
    # -------------------------------------------------------------------

    print(f"Dataset shape before training: {dataset.shape}")

    plottedgraphs = "data/images"

    # Train on valence
    valence_model, valence_scaler, dataset = train_logistic_regression(dataset, "valence_label", visualize=False, save_plots_dir=plottedgraphs)

    # Train on arousal
    arousal_model, arousal_scaler, dataset = train_logistic_regression(dataset, "arousal_label",visualize=True, save_plots_dir=plottedgraphs)

    # Show predictions vs ground truth
    print(dataset[["valence_label", "valence_pred", "arousal_label", "arousal_pred"]])

    with open("valence_model.pkl", "wb") as f:
        pickle.dump(valence_model, f)
    with open("valence_scaler.pkl", "wb") as f:
        pickle.dump(valence_scaler, f)
    with open("arousal_model.pkl", "wb") as f:
        pickle.dump(arousal_model, f)
    with open("arousal_scaler.pkl", "wb") as f:
        pickle.dump(arousal_scaler, f)


if __name__ == "__main__":
    main()
