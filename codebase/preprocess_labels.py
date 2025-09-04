# codebase/preprocess_labels.py

import pandas as pd

def map_valence_arousal_to_mood(valence, arousal):
    """
    Map continuous valence/arousal values to discrete moods.
    """
    if valence >= 0 and arousal >= 0:
        return "happy"       # High Valence, High Arousal
    elif valence >= 0 and arousal < 0:
        return "calm"        # High Valence, Low Arousal
    elif valence < 0 and arousal >= 0:
        return "angry"       # Low Valence, High Arousal
    else:
        return "sad"         # Low Valence, Low Arousal


def create_annotation_csv(raw_annotations, output_csv):
    """
    Process raw DEAM annotations and create a CSV with filename + label.
    """
    df = pd.read_csv(raw_annotations)

    # Assume columns are: filename, valence, arousal
    df["label"] = df.apply(
        lambda row: map_valence_arousal_to_mood(row["valence"], row["arousal"]),
        axis=1
    )

    # Keep only needed columns
    df = df[["filename", "label"]]

    # Save for use with data_loader.py
    df.to_csv(output_csv, index=False)

    print(f"✅ Processed annotations saved to {output_csv}")
