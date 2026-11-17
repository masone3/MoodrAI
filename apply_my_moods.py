"""Pin the catalogue songs you rated to the moods you chose."""
import pandas as pd

from codebase.catalogue import open_db, set_override
from codebase.moods import MOODS


def main():
    mine = pd.read_csv("data/validation/my_moods.csv")
    conn = open_db()
    applied, skipped = 0, []
    for _, r in mine.iterrows():
        mood = str(r["my_mood"]).strip().lower()
        if mood not in MOODS:
            skipped.append((r["file"], f"'{r['my_mood']}' is not a single mood"))
        elif set_override(conn, r["file"], mood):
            applied += 1
        else:
            skipped.append((r["file"], "not in the catalogue"))
    print(f"Pinned {applied} songs to your moods.")
    for file, why in skipped:
        print(f"  skipped {file}: {why}")


if __name__ == "__main__":
    main()