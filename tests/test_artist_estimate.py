import pytest

from codebase.artist_estimate import artist_key, build_index, estimate
from codebase.catalogue import add_track, open_db, set_album, set_override
from codebase.playlist_classify import classify_playlist


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def song(conn, name, v=0.9, a=0.9, artist="Band", album="", sid=None, status="matched"):
    add_track(conn, file=name, valence=v, arousal=a, model_type="reg", artist=artist, album=album,
              spotify_id=sid or "sp-" + name, match_status=status)


def playlist_track(i, artist="Band"):
    return {"id": i, "uri": "", "title": "T", "artist": artist, "url": ""}


def test_artist_key_uses_the_primary_artist_and_drops_accents():
    assert artist_key("Daft Punk, Pharrell Williams, Nile Rodgers") == "daft punk"
    assert artist_key("Sigur Rós") == "sigur ros"
    assert artist_key("") == "" and artist_key(None) == ""


def test_needs_at_least_three_songs():
    two = [{"mood": "euphoric", "pinned": False, "album": ""}] * 2
    three = two + [{"mood": "euphoric", "pinned": False, "album": ""}]
    assert estimate(two) is None
    assert estimate(three)["mood"] == "euphoric"


def test_pinned_songs_count_double():
    songs = [{"mood": "euphoric", "pinned": False, "album": ""},
             {"mood": "euphoric", "pinned": False, "album": ""},
             {"mood": "serene", "pinned": True, "album": ""},
             {"mood": "serene", "pinned": True, "album": ""}]
    assert estimate(songs)["mood"] == "serene"  # weight 4 against 2


def test_artists_whose_songs_scatter_get_no_estimate():
    moods = ["intense", "serene", "melancholy", "cheerful"]
    songs = [{"mood": m, "pinned": False, "album": ""} for m in moods]
    est = estimate(songs)
    assert est["mood"] is None and est["share"] == 0.25


def test_albums_are_reported_only_when_every_song_has_one():
    full = [{"mood": "euphoric", "pinned": False, "album": a} for a in ("A", "B", "B")]
    part = full[:2] + [{"mood": "euphoric", "pinned": False, "album": ""}]
    assert estimate(full)["albums"] == 2
    assert estimate(part)["albums"] is None


def test_playlist_track_gets_a_labelled_estimate(conn):
    for i in range(3):
        song(conn, f"{i}.mp3", album=f"Album {i}")
    out = classify_playlist(conn, [playlist_track("new", "Band, Guest Singer")])
    item = out["items"][0]
    assert item["status"] == "estimated" and item["mood"] == "euphoric"
    assert item["n"] == 3 and item["albums"] == 3
    assert out["estimated"] == 1 and out["classified"] == 0
    assert out["estimated_counts"] == {"euphoric": 1} and out["counts"] == {}


def test_two_songs_are_not_enough_and_unmatched_songs_are_ignored(conn):
    song(conn, "1.mp3")
    song(conn, "2.mp3")
    song(conn, "3.mp3", status="no_match")  # wrong-artist risk: never used for estimates
    assert classify_playlist(conn, [playlist_track("x")])["items"][0]["status"] == "no_audio"


def test_real_classification_wins_over_an_estimate(conn):
    for i in range(3):
        song(conn, f"{i}.mp3", sid=f"s{i}")
    item = classify_playlist(conn, [playlist_track("s1")])["items"][0]
    assert item["status"] == "classified"


def test_varied_artist_is_labelled_not_guessed(conn):
    for name, (v, a) in {"1.mp3": (0.9, 0.9), "2.mp3": (0.1, 0.1), "3.mp3": (0.9, 0.1)}.items():
        song(conn, name, v=v, a=a)
    assert classify_playlist(conn, [playlist_track("x")])["items"][0]["status"] == "varied"


def test_set_album_and_pins_feed_the_index(conn):
    song(conn, "1.mp3")
    set_album(conn, "1.mp3", "Debut")
    set_override(conn, "1.mp3", "serene")
    entry = build_index(conn)["band"][0]
    assert entry["album"] == "Debut" and entry["pinned"] and entry["mood"] == "serene"