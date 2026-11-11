"""Compute harmony features for every song in features.csv. Safe to stop and restart."""
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from codebase.features_harmony import extract_harmony_features

warnings.filterwarnings("ignore")

IDS_FROM = "data/DEAM/features.csv"
AUDIO_DIR = Path("data/DEAM/audio")
OUT = Path("data/DEAM/features_harmony.csv")
SAVE_EVERY = 50


def main():
    ids = pd.read_csv(IDS_FROM, usecols=["song_id"])["song_id"].tolist()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done_ids = set(done["song_id"]) if not done.empty else set()
    rows = done.to_dict("records")
    todo = [i for i in ids if i not in done_ids]
    print(f"{len(done_ids)} already done, {len(todo)} to go")

    for n, song_id in enumerate(tqdm(todo), 1):
        try:
            feats = extract_harmony_features(str(AUDIO_DIR / f"{song_id}.mp3"))
            feats["song_id"] = song_id
            rows.append(feats)
        except Exception as e:
            tqdm.write(f"skipping {song_id}: {e}")
        if n % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)  # checkpoint

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()