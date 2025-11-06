import pickle
import numpy as np

def test_logreg_model_loads():
    model = pickle.load(open("models/logreg_valence_model.pkl", "rb"))
    assert hasattr(model, "predict_proba")

def test_logreg_prediction_shape():
    model = pickle.load(open("models/logreg_valence_model.pkl", "rb"))
    X = np.random.rand(1, model.n_features_in_)
    preds = model.predict_proba(X)
    assert preds.shape == (1, 2)
