"""
catalogue.py
------------
The song catalogue: one SQLite table holding each song's Spotify match, scores and mood.
Mood and margin are computed from the stored scores unless you pinned a mood by hand.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from codebase.moods import MOODS, classify_mood, mood_margin

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "catalogue.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file          TEXT NOT NULL UNIQUE,
    title         TEXT,
    artist        TEXT,
    spotify_id    TEXT,
    spotify_uri   TEXT,
    spotify_url   TEXT,
    match_status  TEXT,
    match_score   REAL,
    valence       REAL NOT NULL,
    arousal       REAL NOT NULL,
    mood          TEXT NOT NULL,
    margin        REAL NOT NULL,
    model_type    TEXT NOT NULL,
    added_at      TEXT NOT NULL,
    synced_mood   TEXT,
    mood_override TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_mood ON tracks(mood);
CREATE TABLE IF NOT EXISTS playlists (
    mood        TEXT PRIMARY KEY,
    playlist_id TEXT NOT NULL
);
"""


def open_db(path=DB_PATH):
    """Open (and create or upgrade) the catalogue. Pass ':memory:' for a throwaway test copy."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # rows can be read by column name
    conn.executescript(SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tracks)")}
    for col in ("synced_mood", "mood_override"):  # upgrade older catalogues in place
        if col not in cols:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} TEXT")
    conn.commit()
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
    """Recompute every song's mood from its stored scores, keeping any mood you pinned by hand.
    Run after changing the cuts in moods.py. Returns how many songs changed mood."""
    changed = 0
    for row in conn.execute(
            "SELECT id, valence, arousal, mood, mood_override FROM tracks").fetchall():
        mood = row["mood_override"] or classify_mood(row["valence"], row["arousal"])
        conn.execute("UPDATE tracks SET mood = ?, margin = ? WHERE id = ?",
                     (mood, mood_margin(row["valence"], row["arousal"]), row["id"]))
        changed += mood != row["mood"]
    conn.commit()
    return changed


def set_override(conn, file, mood):
    """Pin a song to a mood you chose (mood=None clears it). False if the song isn't in the catalogue."""
    if mood is not None and mood not in MOODS:
        raise ValueError(f"Unknown mood {mood!r}; choose from {MOODS}")
    row = conn.execute("SELECT valence, arousal FROM tracks WHERE file = ?", (file,)).fetchone()
    if row is None:
        return False
    effective = mood or classify_mood(row["valence"], row["arousal"])
    conn.execute("UPDATE tracks SET mood_override = ?, mood = ? WHERE file = ?",
                 (mood, effective, file))
    conn.commit()
    return True


def update_match(conn, file, m):
    """Replace a song's Spotify match. Leaves its scores, mood and sync state alone."""
    conn.execute(
        """UPDATE tracks SET spotify_id=?, spotify_uri=?, spotify_url=?,
                             match_status=?, match_score=?,
                             title  = CASE WHEN ? != '' THEN ? ELSE title  END,
                             artist = CASE WHEN ? != '' THEN ? ELSE artist END
           WHERE file = ?""",
        (m["spotify_id"], m["uri"], m["url"], m["status"], m["score"],
         m["title"], m["title"], m["artist"], m["artist"], file))
    conn.commit()


def get_playlist_id(conn, mood):
    row = conn.execute("SELECT playlist_id FROM playlists WHERE mood = ?", (mood,)).fetchone()
    return row["playlist_id"] if row else None


def set_playlist_id(conn, mood, playlist_id):
    conn.execute("INSERT OR REPLACE INTO playlists (mood, playlist_id) VALUES (?, ?)",
                 (mood, playlist_id))
    conn.commit()


def unsynced(conn, include_review=False):
    """Songs with a usable Spotify match whose playlist doesn't yet match their mood."""
    statuses = ("matched", "review") if include_review else ("matched",)
    marks = ",".join("?" * len(statuses))
    return conn.execute(
        f"""SELECT * FROM tracks
            WHERE spotify_uri != '' AND match_status IN ({marks})
              AND (synced_mood IS NULL OR synced_mood != mood)""", statuses).fetchall()


def mark_synced(conn, track_id, mood):
    conn.execute("UPDATE tracks SET synced_mood = ? WHERE id = ?", (mood, track_id))
    conn.commit()


def by_spotify_ids(conn, ids):
    """{spotify_id: row} for catalogue songs matched to any of these Spotify track ids."""
    ids = [i for i in dict.fromkeys(ids) if i]
    found = {}
    for start in range(0, len(ids), 500):  # SQLite limits how many ? placeholders one query may hold
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT * FROM tracks WHERE spotify_id IN ({marks}) "
                "AND match_status IN ('matched', 'review')", chunk):
            found.setdefault(row["spotify_id"], row)
    return found