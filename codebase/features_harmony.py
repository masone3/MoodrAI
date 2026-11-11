"""
features_harmony.py
-------------------
Extra features aimed at valence: major vs minor mode, plus loudness and rhythmic drive.
"""
import os

import librosa
import numpy as np

# Krumhansl-Schmuckler key profiles: how strongly listeners expect each of the
# 12 pitch classes in a major / minor key, starting from the key's tonic.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _best_key_correlation(chroma_mean: np.ndarray, profile: np.ndarray) -> float:
    """Correlate the song's note distribution with the profile in all 12 keys; keep the best."""
    best = max(np.corrcoef(chroma_mean, np.roll(profile, k))[0, 1] for k in range(12))
    return float(np.nan_to_num(best))


def extract_harmony_features(y_or_path, sr: int = 22050) -> dict:
    """Accepts a file path or an audio array. Returns six scalar features."""
    if isinstance(y_or_path, str):
        if not os.path.isfile(y_or_path):
            raise FileNotFoundError(f"Audio file not found: {y_or_path}")
        y, sr = librosa.load(y_or_path, sr=sr, duration=30)
    else:
        y = y_or_path

    chroma_mean = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    major = _best_key_correlation(chroma_mean, MAJOR_PROFILE)
    minor = _best_key_correlation(chroma_mean, MINOR_PROFILE)

    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    return {
        "key_major_corr": major,
        "key_minor_corr": minor,
        "mode_diff": major - minor,       # positive leans major, negative leans minor
        "rms_mean": float(np.mean(rms)),  # overall loudness
        "rms_std": float(np.std(rms)),    # how much the loudness varies
        "onset_mean": float(np.mean(onset)),  # how strongly notes and beats hit
    }