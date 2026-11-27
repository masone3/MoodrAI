import http.client
import json
import threading
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from codebase.catalogue import has_file, open_db
from serve import make_server


def fake_predict(path):
    if "bad" in path:
        raise RuntimeError("cannot decode")
    return SimpleNamespace(valence=0.9, arousal=0.9, model_type="reg")


@pytest.fixture
def app(tmp_path):
    db = tmp_path / "c.db"
    open_db(db).close()
    server = make_server(db_path=db, predict_fn=fake_predict, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, db
    server.shutdown()
    server.server_close()


def post(server, path, body, content_type, token=True):
    headers = {"Content-Type": content_type}
    if token:
        headers["X-Moodr-Token"] = server.token
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    c.request("POST", path, body=body, headers=headers)
    r = c.getresponse()
    data = json.loads(r.read() or b"{}")
    c.close()
    return r.status, data


def upload(server, name, data=b"audio bytes", token=True):
    return post(server, f"/api/upload?name={quote(name)}", data, "application/octet-stream", token)


def test_uploaded_song_is_classified_and_stored(app):
    server, db = app
    status, out = upload(server, "Artist - Song.mp3")
    assert status == 200 and out["added"] == 1 and out["failed"] == 0
    conn = open_db(db)
    assert has_file(conn, "Artist - Song.mp3")
    conn.close()


def test_uploading_the_same_song_twice_adds_it_once(app):
    server, _ = app
    upload(server, "a.mp3")
    status, out = upload(server, "a.mp3")
    assert status == 200 and out["added"] == 0 and out["skipped_existing"] == 1


def test_a_song_that_cannot_be_read_is_reported_not_fatal(app):
    server, db = app
    status, out = upload(server, "bad.mp3")
    assert status == 200 and out["added"] == 0 and out["failed"] == 1
    assert "cannot decode" in out["errors"][0]
    conn = open_db(db)
    assert not has_file(conn, "bad.mp3")
    conn.close()


def test_unsafe_or_wrong_file_names_are_rejected(app):
    server, _ = app
    for name in ("../evil.mp3", "sub/dir.mp3", ".hidden.mp3", "notes.txt", "back\\slash.mp3", ""):
        assert upload(server, name)[0] == 400, name


def test_empty_upload_is_rejected(app):
    server, _ = app
    assert upload(server, "a.mp3", data=b"")[0] == 400


def test_upload_and_have_need_the_token(app):
    server, _ = app
    assert upload(server, "a.mp3", token=False)[0] == 403
    assert post(server, "/api/have", json.dumps({"names": []}), "application/json", token=False)[0] == 403


def test_have_lists_the_songs_already_in_the_catalogue(app):
    server, _ = app
    upload(server, "a.mp3")
    status, out = post(server, "/api/have", json.dumps({"names": ["a.mp3", "b.mp3"]}),
                       "application/json")
    assert status == 200 and out["have"] == ["a.mp3"]
    assert post(server, "/api/have", json.dumps({"names": "a.mp3"}), "application/json")[0] == 400


def test_page_has_the_drop_zone(app):
    server, _ = app
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    c.request("GET", "/")
    page = c.getresponse().read()
    c.close()
    assert b"Drop .mp3 or .wav files" in page and b"Choose folder" in page
    assert server.token.encode() in page