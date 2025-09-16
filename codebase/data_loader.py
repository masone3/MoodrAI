"""
data_loader.py
--------------
Handles loading audio datasets (DEAM, EMO-Music, etc.) and extracting
features like MFCCs or Mel-spectrograms.
"""

import os
import pandas as pd

def load_deam_annotations(file1):
    """Load static song-level DEAM annotations into a single DataFrame."""
    
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

def add_binary_labels(df: pd.DataFrame, valence_col="valence_mean", arousal_col="arousal_mean") -> pd.DataFrame:
    """
    Adds binary labels for valence and arousal based on the mean scores.
    Args:
        df: DataFrame with valence/arousal continuous values.
        valence_col: Column containing valence values.
        arousal_col: Column containing arousal values.
    Returns:
        DataFrame with new columns: 'valence_label' and 'arousal_label'
    """
    # Use median split for balance
    valence_threshold = df[valence_col].median()
    arousal_threshold = df[arousal_col].median()

    df["valence_label"] = (df[valence_col] >= valence_threshold).astype(int)
    df["arousal_label"] = (df[arousal_col] >= arousal_threshold).astype(int)

    return df