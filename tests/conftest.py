import pytest
import numpy as np
import soundfile as sf

# -------------------------------------------------------
# 0.5-second sine wave
# -------------------------------------------------------
@pytest.fixture
def short_sine_wave_file(tmp_path):
    """Generate a 0.5 second sine wave WAV file."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    file = tmp_path / "short.wav"
    sf.write(file, y, sr)
    return str(file)


# -------------------------------------------------------
# 1-second silent audio
# -------------------------------------------------------
@pytest.fixture
def empty_audio_file(tmp_path):
    """Generate a silent audio file."""
    sr = 22050
    y = np.zeros(sr)
    file = tmp_path / "silent.wav"
    sf.write(file, y, sr)
    return str(file)


# -------------------------------------------------------
# CORRUPTED AUDIO FIXTURE
# -------------------------------------------------------
@pytest.fixture
def corrupted_audio_file(tmp_path):
    """
    Create a corrupted audio file which should fail decoding.
    Useful for testing error handling.
    """
    file = tmp_path / "corrupted.wav"
    # Write random bytes that are NOT a valid WAV header
    with open(file, "wb") as f:
        f.write(b"THIS_IS_NOT_A_VALID_WAV_FILE")
    return str(file)


# -------------------------------------------------------
# EXTREMELY SHORT CLIP FIXTURE (<5ms)
# -------------------------------------------------------
@pytest.fixture
def tiny_audio_file(tmp_path):
    """
    Generate a very short audio clip (less than 5 milliseconds).
    Many feature extractors struggle with this.
    """
    sr = 22050
    n_samples = max(1, int(sr * 0.005))  # ~5 ms
    y = np.random.randn(n_samples).astype(np.float32)
    file = tmp_path / "tiny.wav"
    sf.write(file, y, sr)
    return str(file)
