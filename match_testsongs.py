"""Match every song in data/testsongs to a Spotify track and save the results for review."""
from pathlib import Path

import pandas as pd

from codebase.matching import match_track
from codebase.spotify_client import SpotifyClient

SONG_DIR = Path("data/testsongs")
OUT = Path("data/validation/spotify_matches.csv")
AUDIO = {".mp3", ".wav"}


def main():
    client = SpotifyClient()
    rows = []
    for path in sorted(SONG_DIR.iterdir()):
        if path.suffix.lower() not in AUDIO:
            continue
        r = match_track(client, path.name)
        rows.append(r)
        found = f"{r['title']} - {r['artist']}" if r["spotify_id"] else "(nothing found)"
        print(f"{r['status']:9s} {r['score']:.2f}  {path.name}  ->  {found}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()