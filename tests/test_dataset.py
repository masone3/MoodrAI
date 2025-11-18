import numpy as np
import torch
from codebase.cnn_model import SpectrogramDataset

def test_spectrogram_dataset_item():
    ds = SpectrogramDataset([("data/DEAM/audio/1001.mp3", 1)])
    X, label = ds[0]
    assert isinstance(X, torch.Tensor)
    assert X.ndim == 3  # (1, mel, time)
    assert label in (0,1)

def test_spectrogram_dataset_normalization():
    ds = SpectrogramDataset([("data/DEAM/audio/1001.mp3", 1)], normalize=True)
    X, _ = ds[0]
    arr = X.numpy().squeeze(0)
    assert abs(arr.mean()) < 1e-3
    assert arr.std() > 0.1
