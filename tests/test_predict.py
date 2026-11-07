import pytest
from pathlib import Path

from codebase.predict import MODEL_TYPES, ROOT, mood_from_scores, predict_track

SAMPLE = ROOT / "data" / "DEAM" / "audio" / "1001.mp3"


@pytest.mark.parametrize("v,a,expected", [
    (0.9, 0.9, "happy"), (0.1, 0.9, "angry"),
    (0.1, 0.1, "sad"),   (0.9, 0.1, "calm"),
])
def test_mood_from_scores_quadrants(v, a, expected):
    assert mood_from_scores(v, a) == expected


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        predict_track(str(SAMPLE), "banana")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_track("fake/path.mp3", "rf")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_track_returns_valid_scores(model_type):
    result = predict_track(str(SAMPLE), model_type)
    assert 0.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert result.mood in {"happy", "angry", "sad", "calm"}