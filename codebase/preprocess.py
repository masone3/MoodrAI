import librosa
import numpy as np

def extract_features(file_path, sr=22050, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    Returns numpy array of shape (n_mfcc, time_frames).
    """
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # average over time
    return mfccs_mean
