"""Create a blind rating sheet: song names only, with no model scores to bias you."""
from pathlib import Path

import pandas as pd

SRC = Path("data/validation/ratings.csv")
OUT = Path("data/validation/rate_me.csv")
COLS = ["file", "my_valence", "my_arousal", "my_mood"]


def main():
    files = pd.read_csv(SRC)["file"].tolist()
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    new = pd.DataFrame({"file": [f for f in files if f not in set(old["file"])]})
    out = pd.concat([old, new], ignore_index=True)[COLS]  # your earlier ratings are kept
    out.to_csv(OUT, index=False)
    print(f"{len(out)} songs in {OUT}, {out['my_valence'].isna().sum()} still to rate")


if __name__ == "__main__":
    main()