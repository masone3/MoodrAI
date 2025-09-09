# codebase/prepare_deam_annotations.py

import os
import pandas as pd

# Paths to the DEAM annotation CSVs
csv_paths = [
    "data/DEAM/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv",
    "data/DEAM/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv",
]

out_csv = "data/DEAM/annotations.csv"


def normalize_columns(df):
    """Make columns lowercase, strip spaces, replace spaces with underscores."""
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
    return df


def load_and_clean(paths, out_csv):
    dfs = []
    for path in paths:
        print(f"[INFO] Reading: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}")

        # Try to read flexibly
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")

        df = normalize_columns(df)
        print(f"[DEBUG] Columns in {path}: {df.columns.tolist()}")
        dfs.append(df)

    # Merge into one DataFrame
    df_all = pd.concat(dfs, ignore_index=True)

    # Guess column names
    song_col = next((c for c in df_all.columns if "song" in c and "id" in c), None)
    val_col = next((c for c in df_all.columns if "valence" in c), None)
    aro_col = next((c for c in df_all.columns if "arousal" in c), None)

    if not song_col:
        raise KeyError(f"Could not detect song ID column. Found: {df_all.columns.tolist()}")
    if not val_col:
        raise KeyError(f"Could not detect valence column. Found: {df_all.columns.tolist()}")
    if not aro_col:
        print("[WARNING] Could not detect arousal column, continuing without it.")
        aro_col = None

    # Rename to consistent names
    rename_map = {song_col: "song_id", val_col: "valence_mean"}
    if aro_col:
        rename_map[aro_col] = "arousal_mean"
    df_all = df_all.rename(columns=rename_map)

    # Ensure song_id is string and create a filename column
    df_all["song_id"] = df_all["song_id"].astype(str).str.strip()
    df_all["filename"] = df_all["song_id"].apply(lambda x: f"{x}.wav")

    # Save cleaned file
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"[INFO] Cleaned annotations saved to: {out_csv}")
    print("[PREVIEW]")
    print(df_all.head())

    return df_all


if __name__ == "__main__":
    load_and_clean(csv_paths, out_csv)
