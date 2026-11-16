"""
sync.py
-------
Put each matched song into the Spotify playlist for its mood: "Moodr: <mood>".
"""
from codebase.catalogue import (get_playlist_id, mark_synced, set_playlist_id, unsynced)

PREFIX = "Moodr: "
DESCRIPTION = "Sorted by mood by Moodr AI"


def plan(conn, include_review=False):
    """Work out what to do without doing it. Returns (adds, removes), each {mood: [rows]}."""
    adds, removes = {}, {}
    for row in unsynced(conn, include_review):
        adds.setdefault(row["mood"], []).append(row)
        if row["synced_mood"]:  # the song is sitting in another mood's playlist
            removes.setdefault(row["synced_mood"], []).append(row)
    return adds, removes


def ensure_playlist(conn, client, mood):
    """Return this mood's playlist id: remembered, found by name, or newly created."""
    playlist_id = get_playlist_id(conn, mood)
    if playlist_id:
        return playlist_id
    name = PREFIX + mood
    playlist_id = next((p["id"] for p in client.my_playlists() if p.get("name") == name), None)
    if playlist_id is None:
        playlist_id = client.create_playlist(name, DESCRIPTION)
    set_playlist_id(conn, mood, playlist_id)
    return playlist_id


def apply(conn, client, adds, removes):
    """Carry out a plan. Returns a summary of what happened."""
    summary = {"added": 0, "removed": 0, "remove_failed": 0}

    for mood, rows in removes.items():
        playlist_id = get_playlist_id(conn, mood)
        if not playlist_id:
            continue
        uris = list(dict.fromkeys(r["spotify_uri"] for r in rows))
        if client.remove_items(playlist_id, uris):
            summary["removed"] += len(uris)
        else:
            summary["remove_failed"] += len(uris)

    for mood, rows in adds.items():
        playlist_id = ensure_playlist(conn, client, mood)
        uris = list(dict.fromkeys(r["spotify_uri"] for r in rows))  # one copy of each track
        client.add_items(playlist_id, uris)
        for row in rows:
            mark_synced(conn, row["id"], mood)
        summary["added"] += len(uris)

    return summary