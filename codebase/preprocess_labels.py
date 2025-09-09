"""
preprocess_labels.py - Convert DEAM static annotations into a simple annotations.csv
"""

import pandas as pd
import os

def create_annotation_csv(raw_annotations: str, output_csv: str):
    """
    Reads the DEAM static annotations and converts them into a (filename, label) CSV.
    Args:
        raw_annotations (str): Path to DEAM static annotations CSV.
        output_csv (str): Output path for cleaned annotations.
    """
    print("[INFO] Loading raw annotations from:", raw_annotations)

    # Load DEAM static annotations (comma-separated)
    df = pd.read_csv(raw_annotations)

    # Debug: show column names so you can confirm
    print("[DEBUG] Columns in raw file:", df.columns.tolist())
    print("[DEBUG] First few rows:")
    print(df.head())

    # --- Labeling Logic ---
    # DEAM provides valence_mean ∈ [0,1], where 0 = negative, 1 = positive
    # We can turn this into simple mood classes
    def label_from_valence(val: float) -> str:
        if val >= 0.66:
            return "happy"
        elif val >= 0.33:
            return "neutral"
        else:
            return "sad"

    # Apply mood labels
    df["label"] = df["valence_mean"].apply(label_from_valence)

    # Convert song_id → filename (DEAM audio files are typically named like "1.wav", "2.wav", etc.)
    df["filename"] = df["song_id"].astype(str) + ".wav"

    # Keep only needed columns
    df_out = df[["filename", "label"]]
    df_out.to_csv(output_csv, index=False)

    print(f"[INFO] Saved cleaned annotations to: {output_csv}")
    print("[DEBUG] Sample of processed annotations:")
    print(df_out.head())

if __name__ == "__main__":
    # Adjust if your DEAM dataset is stored differently
    raw_path = "data/DEAM/annotations/static_annotations.csv"
    out_path = "data/DEAM/annotations.csv"

    # Ensure output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    create_annotation_csv(raw_annotations=raw_path, output_csv=out_path)
