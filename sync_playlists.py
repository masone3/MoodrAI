"""
Sync the catalogue to Spotify: one private playlist per mood.

    python3 -W ignore sync_playlists.py --dry-run          # show the plan, change nothing
    python3 -W ignore sync_playlists.py                    # do it
    python3 -W ignore sync_playlists.py --include-review   # also add medium-confidence matches
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from codebase.catalogue import open_db
from codebase.spotify_client import SpotifyClient
from codebase.sync import apply, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-review", action="store_true")
    args = ap.parse_args()

    conn = open_db()
    adds, removes = plan(conn, args.include_review)
    if not adds and not removes:
        print("Nothing to sync. Everything matched is already in its mood playlist.")
        return

    for mood, rows in sorted(adds.items()):
        print(f"+ Moodr: {mood}: adding {len(rows)}")
        for r in rows:
            print(f"    {r['artist']} - {r['title']}")
    for mood, rows in sorted(removes.items()):
        print(f"- Moodr: {mood}: removing {len(rows)} moved songs")

    if args.dry_run:
        print("\nDry run only. Nothing was changed.")
        return
    print("\nDone:", apply(conn, SpotifyClient(), adds, removes))


if __name__ == "__main__":
    main()