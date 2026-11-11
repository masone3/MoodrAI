import numpy as np

from codebase.features_harmony import extract_harmony_features

SR = 22050


def _chord(freqs, seconds=4):
    t = np.linspace(0, seconds, seconds * SR, endpoint=False)
    return sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32) / len(freqs)


def test_major_chord_leans_major_and_minor_chord_leans_minor():
    c_major = extract_harmony_features(_chord([261.63, 329.63, 392.00]), sr=SR)  # C-E-G
    c_minor = extract_harmony_features(_chord([261.63, 311.13, 392.00]), sr=SR)  # C-Eb-G
    assert c_major["mode_diff"] > c_minor["mode_diff"]


def test_returns_six_finite_values():
    feats = extract_harmony_features(_chord([440.0]), sr=SR)
    assert len(feats) == 6
    assert all(np.isfinite(v) for v in feats.values())