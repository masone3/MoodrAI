"""
playlist_classify.py
--------------------
Look up a Spotify playlist's tracks in the catalogue and summarise their moods.
Tracks with no audio file on this machine can't be classified, but if you've added enough songs
by the same artist they get a clearly labelled estimate.
"""
from collections import Counter

from codebase.artist_estimate import artist_key, build_index, estimate
from codebase.catalogue import by_spotify_ids
from codebase.moods import BORDERLINE_MARGIN


def classify_playlist(conn, tracks):
    known = by_spotify_ids(conn, [t["id"] for t in tracks])
    index = build_index(conn)
    items, counts, est_counts = [], Counter(), Counter()
    for t in tracks:
        base = {"title": t["title"], "artist": t["artist"], "url": t["url"]}
        row = known.get(t["id"])
        if row is not None:
            pinned = bool(row["mood_override"])
            items.append({
                **base, "status": "classified", "file": row["file"], "mood": row["mood"],
                "pinned": pinned, "borderline": (not pinned) and row["margin"] < BORDERLINE_MARGIN,
                "match": row["match_status"], "added": row["added_at"],
            })
            counts[row["mood"]] += 1
            continue
        est = estimate(index.get(artist_key(t["artist"]), []))
        if est is None:
            items.append({**base, "status": "no_audio"})
        elif est["mood"] is None:
            items.append({**base, "status": "varied", "n": est["n"]})
        else:
            items.append({**base, "status": "estimated", "mood": est["mood"],
                          "share": est["share"], "n": est["n"], "albums": est["albums"],
                          "pinned": est["pinned"]})
            est_counts[est["mood"]] += 1
    return {"items": items, "counts": dict(counts.most_common()),
            "classified": sum(counts.values()),
            "estimated": sum(est_counts.values()),
            "estimated_counts": dict(est_counts.most_common()),
            "total": len(tracks)}