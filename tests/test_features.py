import sys
import os
import numpy as np
import pytest
from codebase.features_extra import extract_audio_stats

# Add project root to sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from codebase.features import extract_features
from codebase.features_extra import extract_audio_stats


def test_extract_features_valid_file():
    feats = extract_features("data/DEAM/audio/1001.mp3")
    assert isinstance(feats, dict)
    assert "centroid" in feats


def test_extract_audio_stats_returns_array():
    arr = extract_audio_stats("data/DEAM/audio/1001.mp3")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)  # expected 4 stats


def test_extract_features_handles_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_features("fake/path.mp3")


def test_extract_features_short_clip(short_sine_wave_file):
    feats = extract_features(short_sine_wave_file)
    assert isinstance(feats, dict)
    assert len(feats) > 0


def test_extract_audio_stats_empty_audio(empty_audio_file):
    arr = extract_audio_stats(empty_audio_file)
    assert isinstance(arr, np.ndarray)
    assert np.isfinite(arr).all()

def test_extract_audio_stats_not_all_zeros():
    sr = 22050
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 3 s of a 440 Hz tone
    stats = extract_audio_stats(y, sr=sr)
    assert stats.shape == (4,)
    assert np.any(stats != 0)s