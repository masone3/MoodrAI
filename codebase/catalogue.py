"""
catalogue.py
------------
The song catalogue: one SQLite table holding each song's Spotify match, scores and mood.
Mood and margin are always computed from the stored scores, so they can never disagree.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from codebase.moods import classify_mood, mood_margin

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "catalogue.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file         TEXT NOT NULL UNIQUE,
    title        TEXT,
    artist       TEXT,
    spotify_id   TEXT,
    spotify_uri  TEXT,
    spotify_url  TEXT,
    match_status TEXT,
    match_score  REAL,
    valence      REAL NOT NULL,
    arousal      REAL NOT NULL,
    mood         TEXT NOT NULL,
    margin       REAL NOT NULL,
    model_type   TEXT NOT NULL,
    added_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_mood ON tracks(mood);
"""


def open_db(path=DB_PATH):
    """Open (and create, if needed) the catalogue. Pass ':memory:' for a throwaway test copy."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # rows can be read by column name
    conn.executescript(SCHEMA)
    return conn


def has_file(conn, file):
    return conn.execute("SELECT 1 FROM tracks WHERE file = ?", (file,)).fetchone() is not None


def add_track(conn, *, file, valence, arousal, model_type, title="", artist="",
              spotify_id="", spotify_uri="", spotify_url="",
              match_status="no_match", match_score=0.0, added_at=None):
    """Insert one song. Raises sqlite3.IntegrityError if the file is already in the catalogue."""
    conn.execute(
        """INSERT INTO tracks (file, title, artist, spotify_id, spotify_uri, spotify_url,
                               match_status, match_score, valence, arousal, mood, margin,
                               model_type, added_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (file, title, artist, spotify_id, spotify_uri, spotify_url,
         match_status, match_score, valence, arousal,
         classify_mood(valence, arousal), mood_margin(valence, arousal),
         model_type, added_at or datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()


def recent(conn, limit=20):
    return conn.execute("SELECT * FROM tracks ORDER BY added_at DESC, id DESC LIMIT ?",
                        (limit,)).fetchall()


def by_mood(conn, mood):
    return conn.execute("SELECT * FROM tracks WHERE mood = ? ORDER BY artist, title",
                        (mood,)).fetchall()


def mood_counts(conn):
    rows = conn.execute("SELECT mood, COUNT(*) AS n FROM tracks GROUP BY mood ORDER BY n DESC")
    return {r["mood"]: r["n"] for r in rows}


def refresh_moods(conn):
    """Recompute every song's mood from its stored scores. Run after changing the cuts in
    moods.py, so no audio needs reprocessing. Returns how many songs changed mood."""
    changed = 0
    for row in conn.execute("SELECT id, valence, arousal, mood FROM tracks").fetchall():
        mood = classify_mood(row["valence"], row["arousal"])
        conn.execute("UPDATE tracks SET mood = ?, margin = ? WHERE id = ?",
                     (mood, mood_margin(row["valence"], row["arousal"]), row["id"]))
        changed += mood != row["mood"]
    conn.commit()
    return changed