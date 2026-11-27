from types import SimpleNamespace

import pytest

from codebase.catalogue import has_file, open_db
from codebase.ingest import ingest, list_audio


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def touch(folder, *names):
    for n in names:
        p = folder / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def ok(path):
    return SimpleNamespace(valence=0.9, arousal=0.9, model_type="reg")


def test_list_audio_filters_and_recurses(tmp_path):
    touch(tmp_path, "a.mp3", "B.WAV", "notes.txt", "._junk.mp3", "sub/c.mp3")
    assert {p.name for p in list_audio(tmp_path)} == {"a.mp3", "B.WAV"}
    assert {p.name for p in list_audio(tmp_path, recursive=True)} == {"a.mp3", "B.WAV", "c.mp3"}


def test_adds_new_songs_and_skips_existing_ones(conn, tmp_path):
    touch(tmp_path, "a.mp3", "b.mp3")
    files = list_audio(tmp_path)
    first = ingest(conn, files, ok)
    assert first["added"] == 2 and has_file(conn, "a.mp3")
    again = ingest(conn, files, ok)
    assert again["added"] == 0 and again["skipped_existing"] == 2


def test_one_failure_does_not_stop_the_rest(conn, tmp_path):
    touch(tmp_path, "a.mp3", "bad.mp3", "c.mp3")

    def predict(path):
        if "bad" in path:
            raise RuntimeError("cannot decode")
        return ok(path)

    s = ingest(conn, list_audio(tmp_path), predict)
    assert s["added"] == 2 and s["failed"] == 1 and "bad.mp3" in s["errors"][0]
    assert not has_file(conn, "bad.mp3")


def test_match_is_stored_and_a_match_failure_keeps_the_song(conn, tmp_path):
    touch(tmp_path, "a.mp3", "b.mp3")
    good = {"status": "matched", "score": 1.0, "spotify_id": "sp1", "uri": "u", "url": "l",
            "title": "Scream", "artist": "Avenged Sevenfold"}

    def match(name):
        if name == "b.mp3":
            raise ConnectionError("offline")
        return good

    s = ingest(conn, list_audio(tmp_path), ok, match)
    assert s["added"] == 2 and s["match_failed"] == 1
    rows = {r["file"]: r for r in conn.execute("SELECT * FROM tracks")}
    assert rows["a.mp3"]["title"] == "Scream" and rows["a.mp3"]["match_status"] == "matched"
    assert rows["b.mp3"]["match_status"] == "no_match"


def test_same_filename_in_two_folders_is_added_once(conn, tmp_path):
    touch(tmp_path, "x/song.mp3", "y/song.mp3")
    s = ingest(conn, list_audio(tmp_path, recursive=True), ok)
    assert s["added"] == 1 and s["skipped_existing"] == 1


def test_stop_request_ends_the_scan_early(conn, tmp_path):
    touch(tmp_path, "a.mp3", "b.mp3", "c.mp3")
    calls = []
    s = ingest(conn, list_audio(tmp_path), lambda p: (calls.append(p), ok(p))[1],
               should_stop=lambda: len(calls) >= 1)
    assert s["cancelled"] and s["added"] == 1