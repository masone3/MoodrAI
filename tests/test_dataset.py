import numpy as np
import torch
import sys
import os

# Ensure project root is in Python path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from codebase.cnn_model import SpectrogramDataset


def test_spectrogram_dataset_item():
    # Create dataset with a single sample (path, label)
    file_label_pairs = [("data/DEAM/audio/1001.mp3", 1)]
    ds = SpectrogramDataset(file_label_pairs)

    S, global_feats, label = ds[0]

    # Spectrogram shape checks
    assert isinstance(S, torch.Tensor)
    assert S.ndim == 3       # (1, mel_bins, time_steps)
    assert S.shape[0] == 1   # channel dimension

    # Global features check
    assert isinstance(global_feats, torch.Tensor)
    assert global_feats.numel() == 4  # RMS, tempo, flux, dyn_range

    # Label check
    assert label in (0, 1)


def test_spectrogram_dataset_normalization():
    file_label_pairs = [("data/DEAM/audio/1001.mp3", 1)]
    ds = SpectrogramDataset(file_label_pairs, normalize=True)

    S, _, _ = ds[0]
    arr = S.numpy().squeeze(0)

    # Mean should be near zero
    assert abs(arr.mean()) < 1e-3

    # Should not collapse to zero variance
    assert arr.std() > 1e-5
