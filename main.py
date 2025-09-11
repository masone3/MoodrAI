"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

from codebase.data_loader import load_deam_annotations, get_audio_files
from codebase.features import extract_features
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Paths
    audio_dir = "data/DEAM/audio"

    # 1. Load annotations
    print("Loading annotations...")
    annotations_df = load_deam_annotations()

    # 2. Match audio files to labels
    print("Matching audio files...")
    dataset = get_audio_files(audio_dir, annotations_df)

    # Debug: print first 3 matches
    print(f"Found {len(dataset)} songs")
    print("Sample:", dataset[:3])

    # 3. Extract features for first few tracks
    print("\nExtracting features...")
    for i, (audio_path, label) in enumerate(dataset[:7]):  # just test a few tracks
        features = extract_features(audio_path)
        print(f"Track {i+1}: {audio_path}")
        print("  Label (valence, arousal):", label)
        print("  Features shape:", features.shape)
        print("  Features sample:", features[:5], "\n")

    file1 = "data/DEAM/annotations.csv"
    df = pd.read_csv(file1, sep=",")

    plt.scatter(df["valence_mean"], df["arousal_mean"])
    plt.xlabel("Valence")
    plt.ylabel("Arousal")

    plt.show()

if __name__ == "__main__":
    main()
