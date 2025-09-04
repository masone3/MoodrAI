"""
data_loader.py
--------------
Handles loading audio datasets (DEAM, EMO-Music, etc.) and extracting
features like MFCCs or Mel-spectrograms.
"""

import os
import pandas as pd
import numpy as np
import librosa

from codebase.features import extract_features


def load_deam_dataset_mfcc(audio_dir, annotations_file, n_samples=None, method="mfcc", duration=30):
    """
    Load the DEAM dataset and extract features for each audio file.

    Args:
        audio_dir (str): Path to DEAM audio files.
        annotations_file (str): Path to DEAM annotations CSV.
        n_samples (int, optional): Number of samples to load (useful for testing).
        method (str): Feature extraction method ("mfcc" or "mel").
        duration (int): Duration (seconds) to load from each track.
    
    Returns:
        X (np.ndarray): Feature matrix (samples x features).
        y (np.ndarray): Labels (mood categories).
    """

    # Load annotations
    annotations = pd.read_csv(annotations_file)

    # For now, assume there's a 'filename' column and a 'label' column.
    # If DEAM labels are dimensional (valence/arousal), I can map them later.
    files = annotations["filename"].tolist()
    labels = annotations["label"].tolist()

    if n_samples:
        files = files[:n_samples]
        labels = labels[:n_samples]

    X, y = [], []

    for idx, (fname, label) in enumerate(zip(files, labels)):
        fpath = os.path.join(audio_dir, fname)

        try:
            # Load audio (first 30s by default)
            y_audio, sr = librosa.load(fpath, duration=duration)

            # Extract features
            features = extract_features(y_audio, sr, method=method)

            # Flatten features if 2D (e.g. spectrogram for CNN later)
            if features.ndim > 1 and method == "mel":
                features = np.mean(features, axis=1)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"❌ Error processing {fpath}: {e}")

    return np.array(X), np.array(y)
