import torch
from codebase.cnn_model import MusicMoodCNN

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
