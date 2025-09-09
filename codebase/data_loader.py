"""
data_loader.py
--------------
Handles loading audio datasets (DEAM, EMO-Music, etc.) and extracting
features like MFCCs or Mel-spectrograms.
"""

import os
import pandas as pd

def load_deam_annotations():
    """Load static song-level DEAM annotations into a single DataFrame."""
    file1 = "data/DEAM/annotations.csv"

    df = pd.read_csv(file1, sep=",")
    return df

def get_audio_files(audio_dir, annotations_df):
    """
    Match audio files to labels in the annotation DataFrame.
    Assumes audio files are named with IDs corresponding to annotations.
    """
    audio_map = []
    for _, row in annotations_df.iterrows():
        track_id = row["song_id"]   # <-- check DEAM column name
        valence = row["valence_mean"]
        arousal = row["arousal_mean"]

        audio_path = os.path.join(audio_dir, f"{track_id}.mp3")  # or .wav
        if os.path.exists(audio_path):
            audio_map.append((audio_path, (valence, arousal)))
    return audio_map
