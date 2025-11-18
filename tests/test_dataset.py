import numpy as np
import torch
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
