import pytest

from codebase.predict import ENSEMBLE, MODEL_TYPES, ROOT, predict_track
from codebase.moods import MOODS

SAMPLE = ROOT / "data" / "DEAM" / "audio" / "1001.mp3"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        predict_track(str(SAMPLE), "banana")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_track("fake/path.mp3", "rf")


@pytest.mark.parametrize("model_type", MODEL_TYPES + (ENSEMBLE,))
def test_predict_track_returns_valid_scores(model_type):
    result = predict_track(str(SAMPLE), model_type)
    assert 0.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert result.mood in MOODS
    assert result.margin >= 0.0
    assert result.borderline == (result.margin < 0.05)