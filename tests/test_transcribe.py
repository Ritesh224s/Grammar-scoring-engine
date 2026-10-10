"""Automated unit tests for speech transcription pipeline, caching, and error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.audio import AudioSignal
from src.transcribe import (
    CACHE_COLUMNS,
    generate_kwargs_for_model,
    load_asr_pipeline,
    load_transcripts_cache,
    save_transcripts_cache_atomic,
    transcribe_audio_signal,
    transcribe_single_file,
    upsert_cache_record,
)


# -----------------------------------------------------------------------------
# 1. Generation Kwargs Tests
# -----------------------------------------------------------------------------
def test_generate_kwargs_for_model():
    # English-only model should have empty kwargs
    kwargs_en = generate_kwargs_for_model("openai/whisper-base.en")
    assert kwargs_en == {}

    kwargs_small_en = generate_kwargs_for_model("openai/whisper-small.en")
    assert kwargs_small_en == {}

    # Multilingual model should have language and task specified
    kwargs_multi = generate_kwargs_for_model("openai/whisper-base")
    assert kwargs_multi == {"language": "english", "task": "transcribe"}


# -----------------------------------------------------------------------------
# 2. Collision-Free Cache & Split Disambiguation Tests
# -----------------------------------------------------------------------------
def test_split_aware_cache_disambiguation(tmp_path):
    cache_path = tmp_path / "transcripts.csv"

    # Simulate two files with the exact same base name ('audio_0.wav') in train and test
    train_record = {
        "sample_id": "train/audio_0.wav",
        "split": "train",
        "filename": "audio_0.wav",
        "relative_path": "audio_0.wav",
        "transcript": "Training speech transcript.",
        "status": "ok",
        "model_name": "openai/whisper-base.en",
        "device": "cuda:0",
        "sample_rate": 16000,
        "duration_s": 55.0,
        "elapsed_time_s": 1.2,
        "error": "",
    }
    test_record = {
        "sample_id": "test/audio_0.wav",
        "split": "test",
        "filename": "audio_0.wav",
        "relative_path": "audio_0.wav",
        "transcript": "Test speech transcript.",
        "status": "ok",
        "model_name": "openai/whisper-base.en",
        "device": "cuda:0",
        "sample_rate": 16000,
        "duration_s": 45.0,
        "elapsed_time_s": 1.0,
        "error": "",
    }

    df = pd.DataFrame(columns=CACHE_COLUMNS)
    df = upsert_cache_record(df, train_record, cache_path)
    df = upsert_cache_record(df, test_record, cache_path)

    loaded = load_transcripts_cache(cache_path)
    assert len(loaded) == 2
    assert set(loaded["sample_id"]) == {"train/audio_0.wav", "test/audio_0.wav"}

    train_row = loaded[loaded["sample_id"] == "train/audio_0.wav"].iloc[0]
    test_row = loaded[loaded["sample_id"] == "test/audio_0.wav"].iloc[0]

    assert train_row["transcript"] == "Training speech transcript."
    assert test_row["transcript"] == "Test speech transcript."
    assert train_row["duration_s"] == 55.0
    assert test_row["duration_s"] == 45.0


def test_cache_preserves_valid_record(tmp_path):
    cache_path = tmp_path / "cache.csv"
    record = {
        "sample_id": "train/audio_10.wav",
        "split": "train",
        "filename": "audio_10.wav",
        "relative_path": "audio_10.wav",
        "transcript": "Original valid transcript.",
        "status": "ok",
        "model_name": "openai/whisper-base.en",
        "device": "cuda:0",
        "sample_rate": 16000,
        "duration_s": 50.0,
        "elapsed_time_s": 1.1,
        "error": "",
    }

    df = pd.DataFrame(columns=CACHE_COLUMNS)
    df = upsert_cache_record(df, record, cache_path)

    # Attempt to overwrite with a different transcript without force_refresh
    new_attempt = dict(record)
    new_attempt["transcript"] = "Overwritten transcript."
    df = upsert_cache_record(df, new_attempt, cache_path, force_refresh=False)

    loaded = load_transcripts_cache(cache_path)
    assert loaded.iloc[0]["transcript"] == "Original valid transcript."

    # When force_refresh=True, it should overwrite
    df = upsert_cache_record(df, new_attempt, cache_path, force_refresh=True)
    loaded_refreshed = load_transcripts_cache(cache_path)
    assert loaded_refreshed.iloc[0]["transcript"] == "Overwritten transcript."


def test_cache_updates_failed_record(tmp_path):
    cache_path = tmp_path / "cache.csv"
    failed_record = {
        "sample_id": "train/audio_20.wav",
        "split": "train",
        "filename": "audio_20.wav",
        "relative_path": "audio_20.wav",
        "transcript": "",
        "status": "failed",
        "model_name": "openai/whisper-base.en",
        "device": "cuda:0",
        "sample_rate": 16000,
        "duration_s": 0.0,
        "elapsed_time_s": 0.0,
        "error": "load: Corrupt audio file",
    }

    df = pd.DataFrame(columns=CACHE_COLUMNS)
    df = upsert_cache_record(df, failed_record, cache_path)

    # Now retry and succeed
    retry_record = dict(failed_record)
    retry_record["status"] = "ok"
    retry_record["transcript"] = "Recovered transcript."
    retry_record["error"] = ""

    df = upsert_cache_record(df, retry_record, cache_path, force_refresh=False)
    loaded = load_transcripts_cache(cache_path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["status"] == "ok"
    assert loaded.iloc[0]["transcript"] == "Recovered transcript."


def test_save_transcripts_cache_atomic(tmp_path):
    cache_path = tmp_path / "atomic_cache.csv"
    df = pd.DataFrame([{"sample_id": "test/audio_1.wav", "status": "ok", "transcript": "Hello"}])
    save_transcripts_cache_atomic(df, cache_path)

    assert cache_path.exists()
    assert not cache_path.with_suffix(".tmp.csv").exists()

    loaded = load_transcripts_cache(cache_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["sample_id"] == "test/audio_1.wav"


# -----------------------------------------------------------------------------
# 3. Audio Handling & Silence Logic Tests
# -----------------------------------------------------------------------------
def test_transcribe_silent_audio():
    # Silent signal should return status='silent' without invoking model inference
    mock_pipe = MagicMock()
    silent_samples = np.zeros(16000, dtype=np.float32)
    sig = AudioSignal(
        samples=silent_samples,
        sample_rate=16000,
        orig_sample_rate=16000,
        orig_channels=1,
        duration_s=1.0,
        peak_amplitude=0.0,
        rms_amplitude=0.0,
        is_silent=True,
    )

    res = transcribe_audio_signal(mock_pipe, sig)
    assert res["status"] == "silent"
    assert res["transcript"] == ""
    assert res["error"] == ""
    # Model pipeline should NEVER be called on silence
    mock_pipe.assert_not_called()


def test_transcribe_audio_signal_mocked_success():
    mock_pipe = MagicMock(return_value={"text": "This is a clean speech transcript."})
    samples = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, dtype=np.float32))
    sig = AudioSignal(
        samples=samples,
        sample_rate=16000,
        orig_sample_rate=16000,
        orig_channels=1,
        duration_s=1.0,
        peak_amplitude=0.5,
        rms_amplitude=0.35,
        is_silent=False,
    )

    res = transcribe_audio_signal(mock_pipe, sig, model_name="openai/whisper-base.en")
    assert res["status"] == "ok"
    assert res["transcript"] == "This is a clean speech transcript."
    assert res["error"] == ""
    mock_pipe.assert_called_once()


def test_transcribe_audio_signal_mocked_failure():
    mock_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory error"))
    samples = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, dtype=np.float32))
    sig = AudioSignal(
        samples=samples,
        sample_rate=16000,
        orig_sample_rate=16000,
        orig_channels=1,
        duration_s=1.0,
        peak_amplitude=0.5,
        rms_amplitude=0.35,
        is_silent=False,
    )

    res = transcribe_audio_signal(mock_pipe, sig)
    assert res["status"] == "failed"
    assert res["transcript"] == ""
    assert "CUDA out of memory" in res["error"]


def test_transcribe_single_file_unreadable(tmp_path):
    mock_pipe = MagicMock()
    res = transcribe_single_file(
        audio_path=tmp_path / "non_existent.wav",
        pipe=mock_pipe,
        sample_id="train/non_existent.wav",
        split="train",
    )
    assert res["status"] == "failed"
    assert "load:" in res["error"]
    assert res["transcript"] == ""
    mock_pipe.assert_not_called()


# -----------------------------------------------------------------------------
# 4. Pipeline Configuration & Fallback Tests
# -----------------------------------------------------------------------------
@patch("torch.cuda.is_available", return_value=True)
@patch("src.transcribe.pipeline")
def test_pipeline_long_audio_chunk_length(mock_pipeline, mock_cuda):
    # Verify that pipeline is initialized with chunk_length_s=30 to handle arbitrary long audio
    load_asr_pipeline("openai/whisper-base.en", requested_device="cuda")
    mock_pipeline.assert_called_once()
    kwargs = mock_pipeline.call_args[1]
    assert kwargs.get("chunk_length_s") == 30
    assert kwargs.get("device") == "cuda:0"


@patch("torch.cuda.is_available", return_value=True)
@patch("src.transcribe.pipeline")
def test_pipeline_gpu_failure_cpu_fallback(mock_pipeline, mock_cuda):
    # First call (CUDA) raises exception, second call (CPU) succeeds
    mock_pipeline.side_effect = [RuntimeError("CUDA driver failure"), MagicMock()]

    pipe, device_used = load_asr_pipeline("openai/whisper-base.en", requested_device="cuda")
    assert device_used == "cpu"
    assert mock_pipeline.call_count == 2
    # Second call should target CPU
    assert mock_pipeline.call_args_list[1][1]["device"] == "cpu"
