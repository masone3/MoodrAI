import pytest

from codebase.catalogue import add_track, open_db, refresh_moods
from codebase.sync import apply, plan


class FakeSpotify:
    """Records what would be sent to Spotify, so tests need no login or network."""
    def __init__(self, existing=None):
        self.existing = existing or []
        self.created, self.added, self.removed = [], [], []

    def my_playlists(self):
        return self.existing

    def create_playlist(self, name, description=""):
        self.created.append(name)
        return f"id-{name}"

    def add_items(self, playlist_id, uris):
        self.added.append((playlist_id, list(uris)))

    def remove_items(self, playlist_id, uris):
        self.removed.append((playlist_id, list(uris)))
        return True


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def song(conn, file, v, a, status="matched", uri=None):
    add_track(conn, file=file, valence=v, arousal=a, model_type="rf", match_status=status,
              spotify_uri=uri if uri is not None else f"spotify:track:{file}")


def run(conn, client, **kw):
    adds, removes = plan(conn, **kw)
    return apply(conn, client, adds, removes)


def test_song_goes_to_its_mood_playlist(conn):
    song(conn, "a.mp3", 0.9, 0.9)
    client = FakeSpotify()
    run(conn, client)
    assert client.created == ["Moodr: euphoric"]
    assert client.added == [("id-Moodr: euphoric", ["spotify:track:a.mp3"])]


def test_second_sync_does_nothing(conn):
    song(conn, "a.mp3", 0.9, 0.9)
    run(conn, FakeSpotify())
    again = FakeSpotify()
    assert run(conn, again)["added"] == 0 and not again.added and not again.created


def test_unmatched_and_review_songs_are_skipped_by_default(conn):
    song(conn, "r.mp3", 0.9, 0.9, status="review")
    song(conn, "n.mp3", 0.9, 0.9, status="no_match", uri="")
    client = FakeSpotify()
    run(conn, client)
    assert not client.added
    run(conn, client, include_review=True)
    assert client.added[0][1] == ["spotify:track:r.mp3"]


def test_song_that_changes_mood_is_moved(conn):
    song(conn, "a.mp3", 0.9, 0.9)
    run(conn, FakeSpotify())
    conn.execute("UPDATE tracks SET valence = 0.1, arousal = 0.1")
    refresh_moods(conn)
    client = FakeSpotify()
    run(conn, client)
    assert client.removed == [("id-Moodr: euphoric", ["spotify:track:a.mp3"])]
    assert client.added[0][0] == "id-Moodr: melancholy"


def test_existing_playlist_is_reused_not_duplicated(conn):
    song(conn, "a.mp3", 0.9, 0.9)
    client = FakeSpotify(existing=[{"id": "old", "name": "Moodr: euphoric"}])
    run(conn, client)
    assert not client.created and client.added[0][0] == "old"