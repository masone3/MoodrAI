"""
artist_estimate.py
------------------
A mood estimate for songs you have no audio for, from the songs by the same artist that you
HAVE added. It is a guess about an artist's usual sound, not a measurement of the song, so
the app always labels it as an estimate and says how many songs it rests on.
"""
import re
import unicodedata
from collections import Counter

MIN_SONGS = 3      # an artist needs at least this many songs in your catalogue
MIN_SHARE = 0.5    # and at least this share of the (weighted) votes must agree
PIN_WEIGHT = 2     # a mood you pinned counts double a model-assigned one


def artist_key(artist):
    """Primary artist, lower-case and accent-free. Spotify joins credited artists with ', '."""
    first = (artist or "").split(", ")[0]
    text = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def build_index(conn):
    """{artist key: [song, ...]} from catalogue songs with a confident Spotify match."""
    index = {}
    for r in conn.execute("SELECT file, mood, mood_override, artist, album FROM tracks "
                          "WHERE match_status = 'matched' AND artist != ''"):
        key = artist_key(r["artist"])
        if key:
            index.setdefault(key, []).append({
                "file": r["file"], "mood": r["mood"],
                "pinned": bool(r["mood_override"]), "album": r["album"] or ""})
    return index


def estimate(songs, min_songs=MIN_SONGS, min_share=MIN_SHARE):
    """None when there are too few songs. Otherwise a dict whose `mood` is None if they disagree."""
    n = len(songs)
    if n < min_songs:
        return None
    votes = Counter()
    for s in songs:
        votes[s["mood"]] += PIN_WEIGHT if s["pinned"] else 1
    top_mood, top = votes.most_common(1)[0]
    share = top / sum(votes.values())
    albums = {s["album"] for s in songs}
    return {
        "mood": top_mood if share >= min_share else None,
        "top_mood": top_mood,
        "share": round(share, 2),
        "n": n,
        "albums": len(albums) if all(albums) else None,  # only when every song's album is known
        "pinned": sum(s["pinned"] for s in songs),
    }