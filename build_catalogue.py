"""
Classify and match every NEW song in a folder, then store it in the catalogue.
Songs already in the catalogue are skipped, so this is safe to rerun as the folder grows.

    python3 -W ignore build_catalogue.py ~/Music/Metal
    python3 -W ignore build_catalogue.py ~/Music --recursive
    python3 -W ignore build_catalogue.py ~/Music --no-spotify      # classify only, no login
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from codebase.catalogue import mood_counts, open_db
from codebase.ingest import ingest, list_audio
from codebase.matching import match_track
from codebase.predict import ALL_TYPES, REG, predict_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/testsongs")
    ap.add_argument("--model", default=REG, choices=ALL_TYPES)
    ap.add_argument("--no-spotify", action="store_true")
    ap.add_argument("--recursive", action="store_true", help="include subfolders")
    args = ap.parse_args()

    match = None
    if not args.no_spotify:
        from codebase.spotify_client import SpotifyClient
        client = SpotifyClient()
        match = lambda name: match_track(client, name)

    def progress(s):
        if s["current"]:
            print(f"[{s['done'] + 1}/{s['total']}] {s['current']}")

    conn = open_db()
    files = list_audio(args.folder, args.recursive)
    print(f"{len(files)} audio files found\n")
    s = ingest(conn, files, lambda path: predict_track(path, args.model), match, progress)

    print(f"\nAdded {s['added']}, already in catalogue {s['skipped_existing']}, "
          f"failed {s['failed']}, not matched to Spotify {s['match_failed']}")
    for e in s["errors"]:
        print("  ", e)
    print("Catalogue now holds:", mood_counts(conn))


if __name__ == "__main__":
    main()