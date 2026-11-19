import sqlite3

import pytest

from codebase.catalogue import (add_track, by_mood, has_file, mood_counts, open_db, recent,
                                refresh_moods, set_override, update_match)


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def test_add_and_fetch(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf", title="A", artist="X")
    assert has_file(conn, "a.mp3") and not has_file(conn, "b.mp3")
    row = by_mood(conn, "euphoric")[0]
    assert row["title"] == "A" and row["margin"] > 0


def test_duplicate_file_is_rejected(conn):
    add_track(conn, file="a.mp3", valence=0.5, arousal=0.5, model_type="rf")
    with pytest.raises(sqlite3.IntegrityError):
        add_track(conn, file="a.mp3", valence=0.1, arousal=0.1, model_type="rf")


def test_mood_counts(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf")
    add_track(conn, file="b.mp3", valence=0.95, arousal=0.85, model_type="rf")
    add_track(conn, file="c.mp3", valence=0.1, arousal=0.1, model_type="rf")
    assert mood_counts(conn) == {"euphoric": 2, "melancholy": 1}


def test_recent_lists_newest_first(conn):
    add_track(conn, file="old.mp3", valence=0.5, arousal=0.5, model_type="rf",
              added_at="2026-01-01T00:00:00+00:00")
    add_track(conn, file="new.mp3", valence=0.5, arousal=0.5, model_type="rf",
              added_at="2026-06-01T00:00:00+00:00")
    assert [r["file"] for r in recent(conn)] == ["new.mp3", "old.mp3"]


def test_refresh_moods_repairs_stale_moods(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf")
    conn.execute("UPDATE tracks SET mood = 'wrong'")
    assert refresh_moods(conn) == 1
    assert by_mood(conn, "euphoric")


def test_override_wins_and_survives_refresh(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf")
    assert set_override(conn, "a.mp3", "serene")
    refresh_moods(conn)
    assert by_mood(conn, "serene") and not by_mood(conn, "euphoric")
    set_override(conn, "a.mp3", None)  # clearing returns the song to the model's opinion
    assert by_mood(conn, "euphoric")


def test_override_rejects_unknown_mood_and_missing_file(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf")
    with pytest.raises(ValueError):
        set_override(conn, "a.mp3", "banana")
    assert set_override(conn, "zzz.mp3", "serene") is False


def test_update_match_keeps_scores_and_mood(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="rf",
              match_status="review", match_score=0.57)
    update_match(conn, "a.mp3", {"spotify_id": "id1", "uri": "spotify:track:id1", "url": "u",
                                 "status": "matched", "score": 1.0, "title": "T", "artist": "A"})
    row = by_mood(conn, "euphoric")[0]
    assert row["match_status"] == "matched" and row["title"] == "T" and row["valence"] == 0.9