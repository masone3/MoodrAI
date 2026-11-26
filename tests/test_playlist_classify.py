import pytest

from codebase.catalogue import add_track, open_db, set_override
from codebase.playlist_classify import classify_playlist


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def track(i, title="T", artist="A"):
    return {"id": i, "uri": f"spotify:track:{i}", "title": title, "artist": artist, "url": ""}


def test_known_and_unknown_tracks(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="reg",
              spotify_id="sp1", match_status="matched")
    out = classify_playlist(conn, [track("sp1"), track("sp2")])
    assert out["classified"] == 1 and out["total"] == 2
    assert out["counts"] == {"euphoric": 1}
    assert [i["status"] for i in out["items"]] == ["classified", "no_audio"]


def test_pinned_mood_is_used_and_marked(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="reg",
              spotify_id="sp1", match_status="matched")
    set_override(conn, "a.mp3", "serene")
    item = classify_playlist(conn, [track("sp1")])["items"][0]
    assert item["mood"] == "serene" and item["pinned"] and not item["borderline"]


def test_unmatched_catalogue_songs_are_not_used(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="reg",
              spotify_id="sp1", match_status="no_match")
    assert classify_playlist(conn, [track("sp1")])["classified"] == 0