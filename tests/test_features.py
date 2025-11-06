from codebase.features import extract_features
from codebase.features_extra import extract_audio_stats
import numpy as np
import os

def test_extract_features_valid_file():
    feats = extract_features("data/DEAM/audio/1001.mp3")
    assert isinstance(feats, dict)
    assert "spectral_centroid" in feats

def test_extract_audio_stats_returns_array():
    arr = extract_audio_stats("data/DEAM/audio/1001.mp3")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)

def test_extract_features_handles_missing_file():
    try:
        extract_features("fake/path.mp3")
    except Exception as e:
        assert isinstance(e, FileNotFoundError)
