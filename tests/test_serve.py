import http.client
import json
import threading

import pytest

from codebase.catalogue import add_track, by_mood, open_db
from serve import make_server

from codebase.spotify_client import SpotifyClient


class FakeResponse:
    def __init__(self, items):
        self._items = items

    def raise_for_status(self):
        pass

    def json(self):
        return {"items": self._items}


def client_returning(items):
    c = object.__new__(SpotifyClient)  # skips __init__, so no login is attempted
    c.request = lambda method, path, **kw: FakeResponse(items)
    return c


def test_playlist_tracks_carries_the_id_and_skips_local_files():
    items = [
        {"item": {"id": "a1", "uri": "spotify:track:a1", "name": "Scream",
                  "artists": [{"name": "Avenged Sevenfold"}],
                  "external_urls": {"spotify": "https://open.spotify.com/track/a1"}}},
        {"track": {"id": "b2", "uri": "spotify:track:b2", "name": "Old Shape",
                   "artists": [{"name": "X"}, {"name": "Y"}]}},  # the pre-2026 field name
        {"track": {"id": None, "name": "A local file"}},
        None,
    ]
    tracks = client_returning(items).playlist_tracks("pl1")
    assert [t["id"] for t in tracks] == ["a1", "b2"]
    assert tracks[1]["artist"] == "X, Y" and tracks[0]["url"].startswith("https://open.spotify.com/")


class FakeSpotify:
    def my_playlists(self):
        return [{"id": "pl1", "name": "Road trip", "tracks": {"total": 2}}]

    def playlist_tracks(self, playlist_id):
        return [{"id": "sp1", "uri": "spotify:track:sp1", "title": "Scream",
                 "artist": "Avenged Sevenfold", "url": ""},
                {"id": "sp2", "uri": "spotify:track:sp2", "title": "Unknown",
                 "artist": "Nobody", "url": ""}]


@pytest.fixture
def app(tmp_path):
    db = tmp_path / "c.db"
    conn = open_db(db)
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="reg",
              spotify_id="sp1", match_status="matched")
    conn.close()
    server = make_server(db_path=db, spotify_factory=FakeSpotify, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, db
    server.shutdown()
    server.server_close()


def call(server, method, path, body=None, token="right", host=None):
    headers = {"Content-Type": "application/json"}
    if token == "right":
        headers["X-Moodr-Token"] = server.token
    elif token:
        headers["X-Moodr-Token"] = token
    if host:
        headers["Host"] = host
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    c.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def test_page_carries_the_token_and_the_panel(app):
    server, _ = app
    status, data = call(server, "GET", "/", token=None)
    assert status == 200
    assert server.token.encode() in data and b"Classify a playlist" in data


def test_api_needs_the_right_token(app):
    server, _ = app
    assert call(server, "GET", "/api/playlists", token=None)[0] == 403
    assert call(server, "GET", "/api/playlists", token="wrong")[0] == 403
    assert call(server, "POST", "/api/pin", {"file": "a.mp3", "mood": "serene"}, token=None)[0] == 403


def test_wrong_host_is_rejected(app):
    server, _ = app
    assert call(server, "GET", "/", token=None, host="evil.example")[0] == 403


def test_playlists_and_classify(app):
    server, _ = app
    status, data = call(server, "GET", "/api/playlists")
    assert status == 200 and json.loads(data)[0]["name"] == "Road trip"
    status, data = call(server, "POST", "/api/classify", {"playlist_id": "pl1"})
    out = json.loads(data)
    assert status == 200 and out["classified"] == 1 and out["total"] == 2
    assert out["counts"] == {"euphoric": 1}


def test_classify_rejects_odd_playlist_ids(app):
    server, _ = app
    assert call(server, "POST", "/api/classify", {"playlist_id": "../me"})[0] == 400


def test_pin_and_clear(app):
    server, db = app
    assert call(server, "POST", "/api/pin", {"file": "a.mp3", "mood": "serene"})[0] == 200
    conn = open_db(db)
    assert by_mood(conn, "serene")
    conn.close()
    assert call(server, "POST", "/api/pin", {"file": "a.mp3", "mood": None})[0] == 200
    conn = open_db(db)
    assert by_mood(conn, "euphoric")
    conn.close()


def test_pin_rejects_bad_requests(app):
    server, _ = app
    assert call(server, "POST", "/api/pin", {"file": "a.mp3", "mood": "banana"})[0] == 400
    assert call(server, "POST", "/api/pin", {"file": "zzz.mp3", "mood": "serene"})[0] == 400