"""
features.py
-----------
Feature extraction utilities for audio (MFCCs, spectrograms, etc.)
Used by data_loader.py.
"""


import librosa
import numpy as np

def extract_features(audio_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file.
    Returns a numpy array (feature vector).
    """
    y, sr = librosa.load(audio_path, duration=30)  # load first 30s
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # average across time
    return mfcc_mean
