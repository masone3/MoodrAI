"""
Re-run Spotify matching.

    python3 -W ignore rematch_catalogue.py            # retry songs that aren't a confident match yet
    python3 -W ignore rematch_catalogue.py --albums   # also fill in the album for songs matched earlier
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from codebase.catalogue import open_db, set_album, update_match
from codebase.matching import match_track
from codebase.spotify_client import SpotifyClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--albums", action="store_true")
    args = ap.parse_args()

    conn = open_db()
    weak = conn.execute("SELECT file FROM tracks WHERE match_status != 'matched'").fetchall()
    missing = conn.execute(
        "SELECT file, spotify_id FROM tracks "
        "WHERE match_status = 'matched' AND (album IS NULL OR album = '')").fetchall() \
        if args.albums else []
    if not weak and not missing:
        print("Nothing to do.")
        return

    client = SpotifyClient()
    for row in weak:
        m = match_track(client, row["file"])
        update_match(conn, row["file"], m)
        print(f"{m['status']:9s} {m['score']:.2f}  {row['file']}  ->  {m['title']} - {m['artist']}")
    for row in missing:
        m = match_track(client, row["file"])
        if m["spotify_id"] == row["spotify_id"] and m.get("album"):
            set_album(conn, row["file"], m["album"])
            print(f"album     {m['album']}  <-  {row['file']}")
        else:  # never swap a song's Spotify track while only filling in an album
            print(f"left alone (the search now returns a different track): {row['file']}")


if __name__ == "__main__":
    main()