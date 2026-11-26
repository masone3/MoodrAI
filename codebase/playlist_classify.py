"""
playlist_classify.py
--------------------
Look up a Spotify playlist's tracks in the catalogue and summarise their moods.
Tracks with no audio file on this machine can't be classified, and are listed as such.
"""
from collections import Counter

from codebase.catalogue import by_spotify_ids
from codebase.moods import BORDERLINE_MARGIN


def classify_playlist(conn, tracks):
    known = by_spotify_ids(conn, [t["id"] for t in tracks])
    items, counts = [], Counter()
    for t in tracks:
        row = known.get(t["id"])
        if row is None:
            items.append({"title": t["title"], "artist": t["artist"], "url": t["url"],
                          "status": "no_audio"})
            continue
        pinned = bool(row["mood_override"])
        items.append({
            "title": t["title"], "artist": t["artist"], "url": t["url"],
            "status": "classified", "file": row["file"], "mood": row["mood"],
            "pinned": pinned, "borderline": (not pinned) and row["margin"] < BORDERLINE_MARGIN,
            "match": row["match_status"], "added": row["added_at"],
        })
        counts[row["mood"]] += 1
    return {"items": items, "counts": dict(counts.most_common()),
            "classified": sum(counts.values()), "total": len(tracks)}