"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

from codebase.data_loader import load_deam_annotations, get_audio_files
from codebase.features import extract_features


def main():
    # -------------------------------------------------------------------
    # Step 1. Load annotations (valence, arousal, etc.)
    # -------------------------------------------------------------------
    annotations_csv = "data/DEAM/processed_annotations.csv"
    annotations = load_deam_annotations(annotations_csv)

    print(f"Loaded {len(annotations)} annotations")

    # -------------------------------------------------------------------
    # Step 2. Extract features for each audio file
    # -------------------------------------------------------------------
    features_list = []

    for idx, row in annotations.iterrows():
        track_id = row["song_id"]
        audio_path = f"data/DEAM/audio/{track_id}.mp3"
        print(f"Looking for audio: {audio_path}")

        try:
            feats = extract_features(audio_path)
            feats["song_id"] = track_id
            feats["valence_mean"] = row["valence_mean"]
            feats["arousal_mean"] = row["arousal_mean"]

            features_list.append(feats)
            print(f"✅ Extracted features for track {track_id}")

        except Exception as e:
            print(f"⚠️ Skipping {track_id} due to error: {e}")

    # -------------------------------------------------------------------
    # Step 3. Build DataFrame
    # -------------------------------------------------------------------
    features_df = pd.DataFrame(features_list)

    print("\n✅ Feature DataFrame created:")
    print(features_df.head())

    # -------------------------------------------------------------------
    # Step 4. Quick Visualization
    # -------------------------------------------------------------------
    plt.scatter(features_df["valence_mean"], features_df["arousal_mean"])
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("DEAM Dataset: Valence vs Arousal")
    plt.show()

    # -------------------------------------------------------------------
    # Step 5. Save dataset
    # -------------------------------------------------------------------
    features_out = "data/DEAM/features.csv"

    if os.path.exists(features_out):
        print("🔄 Loading cached features...")
        df_features = pd.read_csv(features_out)
    else:
        print("⏳ Extracting features (first run)...")
        all_features = []
        for _, row in annotations.iterrows():
            track_id = row["song_id"]
            audio_path = f"data/DEAM/audio/{str(track_id).zfill(4)}.mp3"

            try:
                feats = extract_features(audio_path)
                feats["song_id"] = track_id
                feats["valence_mean"] = row["valence_mean"]
                feats["arousal_mean"] = row["arousal_mean"]
                all_features.append(feats)
            except Exception as e:
                print(f"⚠️ Skipping {track_id}: {e}")

        df_features = pd.DataFrame(all_features)
        df_features.to_csv(features_out, index=False)
        print(f"✅ Saved extracted features to {features_out}")

if __name__ == "__main__":
    main()
