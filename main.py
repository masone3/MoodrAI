"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from codebase.data_loader import load_deam_annotations, add_binary_labels
from codebase.features import extract_features
from codebase.baseline_classifier import train_baseline_model


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
        print("Feature columns:", df_features.columns.tolist())
        print("Total:", len(df_features.columns))
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
    print(dataset.head())

    # Explicitly define feature columns (exclude IDs, means, labels)
    feature_cols = [
        col for col in df_features.columns
        if col not in ["song_id", "valence_mean", "arousal_mean",
                       "valence_label", "arousal_label"]
    ]
    print(f"Using {len(feature_cols)} features for training")

    # -------------------------------------------------------------------
    # Step 5. Train classifiers
    # -------------------------------------------------------------------
    plottedgraphs = "data/images"

    # Logistic Regression
    valence_logreg, valence_logreg_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="logreg", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_logreg, arousal_logreg_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="logreg", visualize=False, save_plots_dir=plottedgraphs
    )

    # SVM
    valence_svm, valence_svm_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="svm", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_svm, arousal_svm_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="svm", visualize=False, save_plots_dir=plottedgraphs
    )

    # Random Forest
    valence_rf, _, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="randomforest", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_rf, _, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="randomforest", visualize=False, save_plots_dir=plottedgraphs
    )

    # -------------------------------------------------------------------
    # Step 6. Save models + scalers
    # -------------------------------------------------------------------
    with open("models/valence_model.pkl", "wb") as f:
        pickle.dump(valence_logreg, f)
    with open("models/valence_scaler.pkl", "wb") as f:
        pickle.dump(valence_logreg_scaler, f)
    with open("models/arousal_model.pkl", "wb") as f:
        pickle.dump(arousal_logreg, f)
    with open("models/arousal_scaler.pkl", "wb") as f:
        pickle.dump(arousal_logreg_scaler, f)

    # Debug: Check feature counts
    valence_scaler = pickle.load(open("models/valence_scaler.pkl", "rb"))
    print("Scaler expects (valence):", valence_scaler.n_features_in_)
    arousal_scaler = pickle.load(open("models/arousal_scaler.pkl", "rb"))
    print("Scaler expects (arousal):", arousal_scaler.n_features_in_)


if __name__ == "__main__":
    main()
