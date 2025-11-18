import pickle
import numpy as np
import pytest

def test_logreg_model_loads():
    model = pickle.load(open("models/logreg_valence_model.pkl", "rb"))
    assert hasattr(model, "predict_proba")


def test_logreg_prediction_shape():
    model = pickle.load(open("models/logreg_valence_model.pkl", "rb"))
    X = np.random.rand(1, model.n_features_in_)
    preds = model.predict_proba(X)
    assert preds.shape == (1, 2)


def test_logreg_prediction_probabilities():
    """Probability outputs must lie between 0 and 1."""
    model = pickle.load(open("models/logreg_valence_model.pkl", "rb"))
    X = np.random.rand(1, model.n_features_in_)
    preds = model.predict_proba(X)
    assert (preds >= 0).all() and (preds <= 1).all()
