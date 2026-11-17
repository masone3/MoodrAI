"""Re-run Spotify matching for catalogue songs that aren't a confident match yet.
Use after improving matching.py. Only touches match columns."""
import warnings

warnings.filterwarnings("ignore")

from codebase.catalogue import open_db, update_match
from codebase.matching import match_track
from codebase.spotify_client import SpotifyClient


def main():
    conn = open_db()
    rows = conn.execute("SELECT file FROM tracks WHERE match_status != 'matched'").fetchall()
    if not rows:
        print("Nothing to rematch.")
        return
    client = SpotifyClient()
    for row in rows:
        m = match_track(client, row["file"])
        update_match(conn, row["file"], m)
        print(f"{m['status']:9s} {m['score']:.2f}  {row['file']}  ->  {m['title']} - {m['artist']}")


if __name__ == "__main__":
    main()