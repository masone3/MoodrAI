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

from codebase.cnn_model import MusicMoodCNN
import numpy as np

def test_cnn_forward_shape():
    model = MusicMoodCNN(num_classes=2)
    X = torch.randn(1, 1, 128, 128)
    global_feats = torch.randn(1, 4)
    output = model(X, global_feats)
    assert output.shape == (1, 2)

def test_cnn_no_nan_output():
    model = MusicMoodCNN(num_classes=2)
    X = torch.randn(1, 1, 128, 128)
    global_feats = torch.randn(1, 4)
    output = model(X, global_feats)
    assert not torch.isnan(output).any()

def test_cnn_handles_nan_inputs():
    """Model should not crash when spectrogram contains NaNs."""
    model = MusicMoodCNN(num_classes=2)
    X = torch.full((1,1,128,128), float('nan'))
    global_feats = torch.randn(1, 4)
    X = torch.nan_to_num(X, nan=0.0)
    output = model(X, global_feats)
    assert torch.isfinite(output).all()
