# codebase/features_extra.py

import librosa
import numpy as np

def extract_audio_stats(y_or_path, sr=22050):
    try:
        if isinstance(y_or_path, str):
            y, sr = librosa.load(y_or_path, sr=sr)
        else:
            y = y_or_path

        # Ensure y is not empty
        if y.size == 0:
            raise ValueError("Empty audio input")

        rms_vals = librosa.feature.rms(y=y)
        rms = float(np.mean(rms_vals))
        dyn_range = float(np.std(rms_vals))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        flux = float(np.mean(np.diff(onset_env))) if onset_env.size > 1 else 0.0

        arr = np.array([rms, float(tempo), flux, dyn_range], dtype=np.float32)
        return np.nan_to_num(arr)

    except Exception as e:
        print(f"⚠️ Error extracting audio stats: {e}")
        return np.zeros(4, dtype=np.float32)
