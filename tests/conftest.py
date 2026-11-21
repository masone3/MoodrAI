import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

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

@pytest.fixture(scope="session")
def sample_audio(tmp_path_factory):
    """A 30-second synthetic clip, so tests need no dataset or audio files."""
    sr = 22050
    t = np.linspace(0, 30, 30 * sr, endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t) * (t > 15)
    y += 0.02 * np.random.default_rng(0).standard_normal(t.size)
    path = tmp_path_factory.mktemp("audio") / "sample.wav"
    sf.write(path, y.astype(np.float32), sr)
    return str(path)


@pytest.fixture
def deam_dir():
    """For tests of the training code only. Skips cleanly when the dataset isn't downloaded."""
    path = Path(__file__).resolve().parent.parent / "data" / "DEAM"
    if not (path / "features.csv").exists():
        pytest.skip("DEAM dataset not downloaded (only needed for training)")
    return path