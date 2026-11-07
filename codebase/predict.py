"""
predict.py
----------
Headless prediction: audio file in, valence/arousal/mood out.
No Tkinter, so the GUI, tests, and later the Spotify/catalogue code can all use it.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from codebase.features import extract_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "data" / "DEAM" / "features.csv"

SKLEARN_MODELS = ("logreg", "svm", "rf")
MODEL_TYPES = SKLEARN_MODELS + ("cnn",)
AXES = ("valence", "arousal")


@dataclass(frozen=True)
class Prediction:
    valence: float   # probability of "high valence", 0..1
    arousal: float   # probability of "high arousal", 0..1
    mood: str
    model_type: str


def mood_from_scores(valence: float, arousal: float, threshold: float = 0.5) -> str:
    """Map a (valence, arousal) point to one of four quadrant moods."""
    if valence >= threshold and arousal >= threshold:
        return "happy"
    if valence < threshold and arousal >= threshold:
        return "angry"
    if valence < threshold:
        return "sad"
    return "calm"


@lru_cache(maxsize=None)
def load_sklearn_models(model_type: str):
    """Load the valence/arousal models and scalers once, then reuse them."""
    def _load(name):
        with open(MODELS_DIR / name, "rb") as f:
            return pickle.load(f)

    bundle = {
        axis: (_load(f"{model_type}_{axis}_model.pkl"),
               _load(f"{model_type}_{axis}_scaler.pkl"))
        for axis in AXES
    }
    # The models expect features in the same column order used in training.
    scaler = bundle["valence"][1]
    columns = getattr(scaler, "feature_names_in_", None)
    if columns is None:
        header = pd.read_csv(FEATURES_CSV, nrows=0).columns
        columns = [c for c in header if c != "song_id"]
    return bundle, list(columns)


@lru_cache(maxsize=None)
def load_cnn_models():
    import torch  # imported here so scikit-learn-only use doesn't pay torch's startup cost
    from codebase.cnn_model import MusicMoodCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for axis in AXES:
        model = MusicMoodCNN(num_classes=2)
        model.load_state_dict(
            torch.load(MODELS_DIR / f"cnn_{axis}_model.pth", map_location=device)
        )
        models[axis] = model.to(device).eval()
    return models, device


def _predict_sklearn(path: str, model_type: str):
    bundle, columns = load_sklearn_models(model_type)
    feats = extract_features(path)  # the slow step, so do it once for both axes
    X = pd.DataFrame([feats]).reindex(columns=columns, fill_value=0)

    probs = {}
    for axis, (model, scaler) in bundle.items():
        probs[axis] = float(model.predict_proba(scaler.transform(X.values))[0][1])
    return probs["valence"], probs["arousal"]


def _predict_cnn(path: str):
    import torch
    from codebase.cnn_model import extract_mel_spectrogram
    from codebase.features_extra import extract_audio_stats

    models, device = load_cnn_models()

    spec = np.nan_to_num(extract_mel_spectrogram(path))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    stats = np.nan_to_num(extract_audio_stats(path))
    stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0).to(device)

    probs = {}
    with torch.no_grad():
        for axis, model in models.items():
            probs[axis] = torch.softmax(model(spec, stats), dim=1)[0, 1].item()
    return probs["valence"], probs["arousal"]


def predict_track(path: str, model_type: str = "rf") -> Prediction:
    """Classify one audio file. This is the function the rest of the project calls."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from {MODEL_TYPES}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if model_type == "cnn":
        valence, arousal = _predict_cnn(path)
    else:
        valence, arousal = _predict_sklearn(path, model_type)

    return Prediction(valence, arousal, mood_from_scores(valence, arousal), model_type)