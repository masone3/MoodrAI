import pytest

from codebase.moods import MOODS
from codebase.predict import ALL_TYPES, feature_columns, predict_track


def test_unknown_model_raises(sample_audio):
    with pytest.raises(ValueError):
        predict_track(sample_audio, "banana")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_track("fake/path.mp3", "rf")


def test_feature_columns_file_lists_the_training_features():
    cols = feature_columns()
    assert len(cols) == 44 and "tempo" in cols and "mfcc_0" in cols and "song_id" not in cols


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_predict_track_returns_valid_scores(model_type, sample_audio):
    result = predict_track(sample_audio, model_type)
    assert 0.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert result.mood in MOODS
    assert result.margin >= 0.0
    assert result.borderline == (result.margin < 0.05)