"""
Classify and match every NEW song in a folder, then store it in the catalogue.
Songs already in the catalogue are skipped, so this is safe to rerun as the folder grows.

    python3 -W ignore build_catalogue.py                      # data/testsongs, ensemble model
    python3 -W ignore build_catalogue.py ~/Music/Metal        # any folder
    python3 -W ignore build_catalogue.py --no-spotify         # classify only, no login needed
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from codebase.catalogue import add_track, has_file, mood_counts, open_db
from codebase.matching import match_track
from codebase.predict import ALL_TYPES, ENSEMBLE, predict_track

AUDIO = {".mp3", ".wav"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/testsongs")
    ap.add_argument("--model", default=ENSEMBLE, choices=ALL_TYPES)
    ap.add_argument("--no-spotify", action="store_true")
    args = ap.parse_args()

    client = None
    if not args.no_spotify:
        from codebase.spotify_client import SpotifyClient
        client = SpotifyClient()

    conn = open_db()
    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in AUDIO)
    new = [p for p in files if not has_file(conn, p.name)]
    print(f"{len(files)} songs in folder, {len(new)} new\n")

    for n, path in enumerate(new, 1):
        try:
            pred = predict_track(str(path), args.model)
            match = (match_track(client, path.name) if client else
                     {"status": "no_match", "score": 0.0, "spotify_id": "", "uri": "",
                      "url": "", "title": "", "artist": ""})
            add_track(
                conn, file=path.name, valence=pred.valence, arousal=pred.arousal,
                model_type=pred.model_type,
                title=match["title"] or path.stem, artist=match["artist"],
                spotify_id=match["spotify_id"], spotify_uri=match["uri"],
                spotify_url=match["url"], match_status=match["status"],
                match_score=match["score"],
            )
            flag = " (borderline)" if pred.borderline else ""
            print(f"[{n}/{len(new)}] {pred.mood:10s}{flag:13s} {match['status']:9s} {path.name}")
        except Exception as e:
            print(f"[{n}/{len(new)}] SKIPPED {path.name}: {e}")

    print("\nCatalogue now holds:", mood_counts(conn))


if __name__ == "__main__":
    main()