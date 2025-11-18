import pickle
import numpy as np
import pytest
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "logreg_valence_model.pkl"
)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_logreg_model_loads():
    model = load_model()
    assert hasattr(model, "predict_proba")


def test_logreg_prediction_shape():
    model = load_model()
    X = np.random.rand(1, model.n_features_in_)
    preds = model.predict_proba(X)
    assert preds.shape == (1, 2)


def test_logreg_prediction_probabilities():
    model = load_model()
    X = np.random.rand(1, model.n_features_in_)
    preds = model.predict_proba(X)
    assert np.all((preds >= 0) & (preds <= 1))
