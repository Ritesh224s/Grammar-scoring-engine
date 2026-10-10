"""Automated unit tests for audio loading, preprocessing, resampling, and health checks."""

import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    AudioSignal,
    compute_silence_stats,
    inspect_audio_pipeline,
    load_audio,
)


def _generate_synthetic_wav(
    path: Path,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> Path:
    """Helper to synthesize uncompressed WAV files with known properties."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)

    if channels == 1:
        data = amplitude * np.sin(2 * np.pi * frequency * t)
    else:
        # Multi-channel: different frequencies/amplitudes per channel
        ch1 = amplitude * np.sin(2 * np.pi * frequency * t)
        ch2 = (amplitude * 0.5) * np.sin(2 * np.pi * (frequency * 1.5) * t)
        data = np.stack([ch1, ch2], axis=1)

    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


# -----------------------------------------------------------------------------
# 1. Mono & Stereo Loading Tests
# -----------------------------------------------------------------------------
def test_load_mono_audio(tmp_path):
    wav_path = _generate_synthetic_wav(tmp_path / "mono.wav", duration_s=1.0, sample_rate=16000, channels=1)
    sig = load_audio(wav_path, target_sr=16000, mono=True)

    assert isinstance(sig, AudioSignal)
    assert sig.samples.ndim == 1
    assert sig.samples.dtype == np.float32
    assert sig.sample_rate == 16000
    assert sig.orig_sample_rate == 16000
    assert sig.orig_channels == 1
    assert sig.num_samples == 16000
    assert abs(sig.duration_s - 1.0) < 1e-4
    assert sig.is_silent is False
    assert sig.peak_amplitude > 0.4


def test_stereo_to_mono_conversion(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    num_samples = 16000
    # Left channel: +0.6, Right channel: -0.6 -> Mean should be 0.0
    left = np.full((num_samples, 1), 0.6, dtype=np.float32)
    right = np.full((num_samples, 1), -0.6, dtype=np.float32)
    stereo_data = np.hstack([left, right])
    sf.write(str(wav_path), stereo_data, 16000, subtype="PCM_16")

    sig = load_audio(wav_path, target_sr=16000, mono=True)
    assert sig.orig_channels == 2
    assert sig.samples.ndim == 1
    assert sig.num_samples == num_samples
    assert np.allclose(sig.samples, 0.0, atol=1e-3)


# -----------------------------------------------------------------------------
# 2. Resampling & Duration Tests
# -----------------------------------------------------------------------------
def test_correct_resampling(tmp_path):
    # Original: 44,100 Hz, 1.0 second -> 44,100 samples
    wav_path = _generate_synthetic_wav(tmp_path / "resample_44k.wav", duration_s=1.0, sample_rate=44100, channels=1)
    sig = load_audio(wav_path, target_sr=16000, mono=True)

    assert sig.orig_sample_rate == 44100
    assert sig.sample_rate == 16000
    # 1.0s resampled to 16,000 Hz must yield exactly ~16,000 samples
    assert abs(sig.num_samples - 16000) <= 2
    assert abs(sig.duration_s - 1.0) < 0.01


def test_duration_calculation(tmp_path):
    wav_path = _generate_synthetic_wav(tmp_path / "dur_2p5.wav", duration_s=2.5, sample_rate=16000)
    sig = load_audio(wav_path, target_sr=16000)

    assert abs(sig.duration_s - 2.5) < 1e-3
    assert sig.num_samples == 40000


# -----------------------------------------------------------------------------
# 3. Silence, Empty, and Non-Finite Tests
# -----------------------------------------------------------------------------
def test_silent_audio_flag(tmp_path):
    wav_path = tmp_path / "silent.wav"
    silence = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav_path), silence, 16000, subtype="PCM_16")

    sig = load_audio(wav_path, target_sr=16000)
    assert sig.is_silent is True
    assert sig.peak_amplitude == 0.0
    assert sig.rms_amplitude == 0.0


def test_empty_audio_raises(tmp_path):
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        load_audio(empty_path)


def test_invalid_target_sample_rate(tmp_path):
    wav_path = _generate_synthetic_wav(tmp_path / "normal.wav")

    with pytest.raises(ValueError, match="positive integer"):
        load_audio(wav_path, target_sr=-16000)

    with pytest.raises(ValueError, match="positive integer"):
        load_audio(wav_path, target_sr=0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "non_existent_clip.wav")


def test_corrupt_file_raises(tmp_path):
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt corrupted garbage data")

    with pytest.raises((RuntimeError, ValueError)):
        load_audio(corrupt_path)


# -----------------------------------------------------------------------------
# 4. Silence Stats & Diagnostic Pipeline Tests
# -----------------------------------------------------------------------------
def test_compute_silence_stats():
    sr = 16000
    # 1 second of speech followed by 1 second of silence
    speech = 0.8 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32))
    silence = np.zeros(sr, dtype=np.float32)
    composite = np.concatenate([speech, silence])

    stats = compute_silence_stats(composite, sr, silence_top_db=30.0)
    assert 0.4 <= stats["silence_ratio"] <= 0.6
    assert 0.8 <= stats["speech_duration_s"] <= 1.2


def test_inspect_audio_pipeline(tmp_path):
    wav_path = _generate_synthetic_wav(tmp_path / "pipeline.wav", duration_s=1.2, sample_rate=16000)
    diag = inspect_audio_pipeline(wav_path, target_sr=16000)

    assert diag["success"] is True
    assert diag["error"] is None
    assert diag["array_shape"] == [19200]
    assert diag["orig_sample_rate"] == 16000
    assert abs(diag["duration_s"] - 1.2) < 0.05
    assert diag["is_silent"] is False
