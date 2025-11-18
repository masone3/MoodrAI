import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from codebase.features import extract_features
from codebase.features_extra import extract_audio_stats
import numpy as np
import pytest

def test_extract_features_valid_file():
    """Ensure extract_features returns a dict with known feature keys."""
    feats = extract_features("data/DEAM/audio/1001.mp3")
    assert isinstance(feats, dict)
    assert "spectral_centroid" in feats


def test_extract_audio_stats_returns_array():
    """Ensure extract_audio_stats returns a fixed-length numeric array."""
    arr = extract_audio_stats("data/DEAM/audio/1001.mp3")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)


def test_extract_features_handles_missing_file():
    """Missing files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_features("fake/path.mp3")


def test_extract_features_short_clip(short_sine_wave_file):
    """Short audio clips (<1s) should still produce valid features."""
    feats = extract_features(short_sine_wave_file)
    assert isinstance(feats, dict)
    assert len(feats.keys()) > 0


def test_extract_audio_stats_empty_audio(empty_audio_file):
    """Silent audio should return numeric stats, no NaNs."""
    arr = extract_audio_stats(empty_audio_file)
    assert isinstance(arr, np.ndarray)
    assert np.isfinite(arr).all()
