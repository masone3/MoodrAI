# codebase/features_extra.py

import librosa
import numpy as np

def extract_audio_stats(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        # Root mean square (energy)
        rms = float(librosa.feature.rms(y=y).mean())

        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        # Spectral flux
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        flux = float(np.mean(np.diff(onset_env))) if len(onset_env) > 1 else 0.0

        # Dynamic range
        dyn_range = float(np.std(librosa.feature.rms(y=y)))

        arr = np.array([rms, tempo, flux, dyn_range], dtype=np.float32)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr
    except Exception as e:
        print(f"⚠️ Error extracting global features for {audio_path}: {e}")
        return np.zeros(4, dtype=np.float32)
