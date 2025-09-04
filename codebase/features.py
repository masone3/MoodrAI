"""
features.py
-----------
Feature extraction utilities for audio (MFCCs, spectrograms, etc.)
Used by data_loader.py.
"""

import librosa
import numpy as np


def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract MFCC features from an audio signal.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
        n_mfcc (int): Number of MFCCs to extract.
    
    Returns:
        np.ndarray: 1D feature vector (mean of MFCCs across time).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # take mean across frames
    return mfcc_mean


def extract_melspectrogram(y, sr, n_mels=64):
    """
    Extract Mel Spectrogram features (useful for CNN later).
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
        n_mels (int): Number of Mel bands to generate.
    
    Returns:
        np.ndarray: 2D spectrogram (frames x n_mels).
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_features(y, sr, method="mfcc"):
    """
    Unified feature extraction wrapper.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        method (str): "mfcc" or "mel".
    
    Returns:
        np.ndarray: Feature vector (1D if mfcc, 2D if mel).
    """
    if method == "mfcc":
        return extract_mfcc(y, sr)
    elif method == "mel":
        return extract_melspectrogram(y, sr)
    else:
        raise ValueError("Invalid method. Choose 'mfcc' or 'mel'.")
