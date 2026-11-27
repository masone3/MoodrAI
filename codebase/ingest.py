"""
ingest.py
---------
Turn audio files into catalogue rows: classify, optionally match to Spotify, store.
Shared by build_catalogue.py and the local app, so both behave the same.
"""
from pathlib import Path

from codebase.catalogue import add_track, has_file

AUDIO = {".mp3", ".wav"}
NO_MATCH = {"status": "no_match", "score": 0.0, "spotify_id": "", "uri": "", "url": "",
            "title": "", "artist": ""}


def list_audio(folder, recursive=False):
    """Audio files in a folder (sorted), ignoring hidden files like macOS '._song.mp3' leftovers."""
    folder = Path(folder)
    entries = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in entries
                  if p.is_file() and p.suffix.lower() in AUDIO and not p.name.startswith("."))


def ingest(conn, files, predict, match=None, on_progress=None, should_stop=None):
    """Classify and store every file not already in the catalogue. Returns a summary dict.

    predict(path_str) -> object with valence, arousal, model_type
    match(filename)   -> a Spotify match dict (see NO_MATCH), or None to skip matching
    """
    seen, new, skipped = set(), [], 0
    for p in files:
        if p.name in seen or has_file(conn, p.name):  # the catalogue identifies songs by filename
            skipped += 1
        else:
            seen.add(p.name)
            new.append(p)

    summary = {"total": len(new), "done": 0, "added": 0, "failed": 0, "match_failed": 0,
               "skipped_existing": skipped, "errors": [], "cancelled": False}
    for p in new:
        if should_stop and should_stop():
            summary["cancelled"] = True
            break
        if on_progress:
            on_progress(dict(summary, current=p.name))
        try:
            pred = predict(str(p))
            m = NO_MATCH
            if match:
                try:
                    m = match(p.name)
                except Exception:  # keep the song; `rematch_catalogue.py` can retry the match later
                    summary["match_failed"] += 1
            add_track(conn, file=p.name, valence=pred.valence, arousal=pred.arousal,
                      model_type=pred.model_type, title=m["title"] or p.stem, artist=m["artist"],
                      spotify_id=m["spotify_id"], spotify_uri=m["uri"], spotify_url=m["url"],
                      match_status=m["status"], match_score=m["score"])
            summary["added"] += 1
        except Exception as e:
            summary["failed"] += 1
            if len(summary["errors"]) < 5:
                summary["errors"].append(f"{p.name}: {e}")
        summary["done"] += 1
    if on_progress:
        on_progress(dict(summary, current=""))
    return summary