import pytest
import numpy as np
import soundfile as sf
import os

@pytest.fixture
def short_sine_wave_file(tmp_path):
    """Generate a 0.5 second sine wave WAV file."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    file = tmp_path / "short.wav"
    sf.write(file, y, sr)
    return str(file)

@pytest.fixture
def empty_audio_file(tmp_path):
    """Generate a silent audio file."""
    sr = 22050
    y = np.zeros(sr)
    file = tmp_path / "silent.wav"
    sf.write(file, y, sr)
    return str(file)
