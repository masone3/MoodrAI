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
from codebase.features_harmony import extract_harmony_features
from codebase.moods import BORDERLINE_MARGIN, classify_mood, mood_margin

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "data" / "DEAM" / "features.csv"

SKLEARN_MODELS = ("logreg", "svm", "rf")
MODEL_TYPES = SKLEARN_MODELS + ("cnn",)
REG = "reg"
ENSEMBLE = "ensemble"
ALL_TYPES = MODEL_TYPES + (REG, ENSEMBLE)
AXES = ("valence", "arousal")


@dataclass(frozen=True)
class Prediction:
    valence: float      # 0..1 (probability of "high", or percentile for the reg model)
    arousal: float      # 0..1
    mood: str           # one of the nine names in moods.py
    margin: float       # distance to the nearest mood boundary
    model_type: str

    @property
    def borderline(self) -> bool:
        return self.margin < BORDERLINE_MARGIN


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
    scaler = bundle["valence"][1]
    columns = getattr(scaler, "feature_names_in_", None)
    if columns is None:
        header = pd.read_csv(FEATURES_CSV, nrows=0).columns
        columns = [c for c in header if c != "song_id"]
    return bundle, list(columns)


@lru_cache(maxsize=None)
def load_cnn_models():
    import torch
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


@lru_cache(maxsize=None)
def load_regression_models():
    def _load(name):
        with open(MODELS_DIR / name, "rb") as f:
            return pickle.load(f)

    models = {axis: _load(f"reg_rf_{axis}.pkl") for axis in AXES}
    return models, _load("reg_reference.pkl")


def _predict_sklearn(path: str, model_type: str, feats: dict | None = None):
    bundle, columns = load_sklearn_models(model_type)
    if feats is None:
        feats = extract_features(path)
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


def _predict_regression(path: str):
    """Predict 1-9 scores, then convert each to a 0..1 percentile among DEAM songs."""
    models, reference = load_regression_models()
    feats = {**extract_features(path), **extract_harmony_features(path)}
    out = {}
    for axis, pipe in models.items():
        X = pd.DataFrame([feats]).reindex(columns=list(pipe.feature_names_in_), fill_value=0)
        score = float(pipe.predict(X)[0])
        out[axis] = float(np.searchsorted(reference[axis], score) / len(reference[axis]))
    return out["valence"], out["arousal"]


def _predict_ensemble(path: str):
    """Average the four classifiers' scores into one valence and one arousal."""
    feats = extract_features(path)  # computed once, shared by the three sklearn models
    scores = [_predict_sklearn(path, m, feats) for m in SKLEARN_MODELS]
    scores.append(_predict_cnn(path))
    valence = float(np.mean([s[0] for s in scores]))
    arousal = float(np.mean([s[1] for s in scores]))
    return valence, arousal


def predict_track(path: str, model_type: str = ENSEMBLE) -> Prediction:
    """Classify one audio file. This is the function the rest of the project calls."""
    if model_type not in ALL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from {ALL_TYPES}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if model_type == ENSEMBLE:
        valence, arousal = _predict_ensemble(path)
    elif model_type == REG:
        valence, arousal = _predict_regression(path)
    elif model_type == "cnn":
        valence, arousal = _predict_cnn(path)
    else:
        valence, arousal = _predict_sklearn(path, model_type)

    return Prediction(
        valence=valence,
        arousal=arousal,
        mood=classify_mood(valence, arousal),
        margin=mood_margin(valence, arousal),
        model_type=model_type,
    )