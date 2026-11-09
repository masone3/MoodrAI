import pytest
from codebase.moods import GRID, MOODS, classify_mood, mood_margin


@pytest.mark.parametrize("v,a,expected", [
    (0.9, 0.9, "euphoric"), (0.1, 0.9, "intense"),
    (0.1, 0.1, "melancholy"), (0.9, 0.1, "serene"),
    (0.5, 0.5, "neutral"),
])
def test_corners_and_center(v, a, expected):
    assert classify_mood(v, a) == expected


def test_every_cell_has_a_distinct_mood():
    assert len(MOODS) == len(GRID) == 9


def test_margin_is_small_near_a_boundary():
    assert mood_margin(0.692, 0.82) < 0.03   # Scream's CNN score
    assert mood_margin(0.5, 0.5) > 0.15      # dead center