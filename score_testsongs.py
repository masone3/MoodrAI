"""Score every song in data/testsongs with all four models and write a ratings sheet.
Re-running only scores NEW songs, so ratings you've already typed are never overwritten."""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # hides the scikit-learn version warnings

from codebase.predict import MODEL_TYPES, predict_track

SONG_DIR = Path("data/testsongs")
OUT = Path("data/validation/ratings.csv")
AUDIO = {".mp3", ".wav"}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done = set(old["file"]) if not old.empty else set()

    new_rows = []
    for path in sorted(SONG_DIR.iterdir()):
        if path.suffix.lower() not in AUDIO or path.name in done:
            continue
        print(f"Scoring {path.name} ...")
        row = {"file": path.name}
        for m in MODEL_TYPES:
            try:
                r = predict_track(str(path), m)
                row[f"{m}_valence"], row[f"{m}_arousal"] = round(r.valence, 3), round(r.arousal, 3)
            except Exception as e:
                print(f"  {m} failed: {e}")
                row[f"{m}_valence"] = row[f"{m}_arousal"] = None
        row["my_valence"] = row["my_arousal"] = row["my_mood"] = None  # you fill these in
        new_rows.append(row)

    out = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} songs to {OUT} ({len(new_rows)} new)")


if __name__ == "__main__":
    main()