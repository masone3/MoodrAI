"""
features.py
-----------
Feature extraction utilities for audio (MFCCs, spectrograms, etc.)
Used by data_loader.py.
"""


import librosa
import numpy as np

def extract_features(audio_path, n_mfcc=20) -> dict:
    """
    Extract audio features from an audio file and return as a flat dict.
    Each feature is reduced to scalars so it works with pandas DataFrame.
    """
    y, sr = librosa.load(audio_path, duration=30)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048)
    centroid_mean = np.mean(centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048)
    bandwidth_mean = np.mean(bandwidth)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048)
    contrast_mean = np.mean(contrast, axis=1)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048)
    rolloff_mean = np.mean(rolloff)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    zcr_mean = np.mean(zcr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Flatten everything into scalar keys
    features = {}

    # expand MFCCs
    for i, val in enumerate(mfcc_mean):
        features[f"mfcc_{i}"] = float(val)

    # expand Chroma
    for i, val in enumerate(chroma_mean):
        features[f"chroma_{i}"] = float(val)

    # expand Contrast
    for i, val in enumerate(contrast_mean):
        features[f"contrast_{i}"] = float(val)

    # scalars
    features["centroid"] = float(centroid_mean)
    features["bandwidth"] = float(bandwidth_mean)
    features["rolloff"] = float(rolloff_mean)
    features["zcr"] = float(zcr_mean)
    features["tempo"] = float(tempo)

    return features
