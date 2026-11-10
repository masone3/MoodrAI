"""
moods.py
--------
Maps a (valence, arousal) point to a mood name. Pure logic, no models or audio,
so it is easy to test and easy to change when we tune the boundaries.
"""

# Boundaries between "low", "mid" and "high" on both axes (0..1 scale).
CUTS = (1 / 3, 2 / 3)

# A song closer that this to a boundary line is flagged "borderline".
BORDERLINE_MARGIN = 0.05

# (arousal band, valence band) -> mood name
GRID = {
    ("high", "low"): "intense",   ("high", "mid"): "energetic", ("high", "high"): "euphoric",
    ("mid", "low"): "brooding",   ("mid", "mid"): "neutral",    ("mid", "high"): "cheerful",
    ("low", "low"): "melancholy", ("low", "mid"): "mellow",     ("low", "high"): "serene",
}

MOODS = sorted(set(GRID.values()))


def band(x: float, cuts=CUTS) -> str:
    """Turn a 0..1 score into 'low', 'mid' or 'high'."""
    if x < cuts[0]:
        return "low"
    if x < cuts[1]:
        return "mid"
    return "high"


def classify_mood(valence: float, arousal: float, cuts=CUTS) -> str:
    return GRID[(band(arousal, cuts), band(valence, cuts))]


def mood_margin(valence: float, arousal: float, cuts=CUTS) -> float:
    """Distance to the nearest boundary line. Small means the song is
    near a border and could easily have landed in a neighboring mood."""
    return min(abs(s - c) for s in (valence, arousal) for c in cuts)