"""Automated unit and integration tests for dataset ingestion, validation, and audio matching."""

import os
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig, get_config
from src.dataset import (
    analyze_submission_mismatch,
    inspect_audio_file,
    load_and_prepare_dataset,
    load_raw_csvs,
    match_audio_records,
    validate_csv_schema,
    validate_full_dataset,
    validate_labels,
)


def _create_dummy_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> Path:
    """Helper to generate a minimal valid WAV file for testing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_s)
    # 16-bit mono silence
    data = b"\x00\x00" * num_frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return path


# -----------------------------------------------------------------------------
# 1. CSV Loading & Schema Tests
# -----------------------------------------------------------------------------
def test_load_raw_csvs_success(tmp_path):
    train = pd.DataFrame({"filename": ["audio_1.wav"], "label": [3.5]})
    test = pd.DataFrame({"filename": ["audio_2.wav"], "label": [-1]})
    sub = pd.DataFrame({"filename": ["audio_2.wav"], "label": [-1]})

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sub.to_csv(tmp_path / "sample_submission.csv", index=False)

    csvs = load_raw_csvs(tmp_path)
    assert len(csvs["train"]) == 1
    assert len(csvs["test"]) == 1
    assert len(csvs["sample_sub"]) == 1


def test_load_raw_csvs_missing_file(tmp_path):
    train = pd.DataFrame({"filename": ["audio_1.wav"], "label": [3.5]})
    train.to_csv(tmp_path / "train.csv", index=False)

    with pytest.raises(FileNotFoundError):
        load_raw_csvs(tmp_path)


def test_validate_csv_schema():
    df = pd.DataFrame({"filename": ["a.wav"], "other": [1]})
    errors = validate_csv_schema(df, ["filename", "label"], "train.csv")
    assert len(errors) == 1
    assert "Missing required columns: ['label']" in errors[0]


# -----------------------------------------------------------------------------
# 2. Label Validation Tests
# -----------------------------------------------------------------------------
def test_validate_labels_valid():
    df = pd.DataFrame({"label": [0.0, 2.5, 3.5, 5.0]})
    res = validate_labels(df, "label", 0.0, 5.0)
    assert len(res["errors"]) == 0
    assert res["missing_count"] == 0
    assert res["out_of_bounds_count"] == 0
    assert res["stats"]["min"] == 0.0
    assert res["stats"]["max"] == 5.0
    assert res["stats"]["mean"] == 2.75


def test_validate_labels_missing_and_invalid():
    df = pd.DataFrame({"label": [3.0, None, "invalid", 4.0]})
    res = validate_labels(df, "label", 0.0, 5.0)
    assert res["missing_count"] == 1
    assert res["non_numeric_count"] == 1
    assert len(res["errors"]) == 2


def test_validate_labels_out_of_bounds():
    df = pd.DataFrame({"label": [-0.5, 2.0, 5.5]})
    res = validate_labels(df, "label", 0.0, 5.0)
    assert res["out_of_bounds_count"] == 2
    assert any("outside [0.0, 5.0]" in e for e in res["errors"])


# -----------------------------------------------------------------------------
# 3. Audio Integrity & Matching Tests
# -----------------------------------------------------------------------------
def test_inspect_audio_file(tmp_path):
    wav_path = _create_dummy_wav(tmp_path / "test.wav", duration_s=1.5, sample_rate=16000)
    info = inspect_audio_file(wav_path)
    assert info["exists"] is True
    assert info["valid_wav"] is True
    assert info["samplerate"] == 16000
    assert info["channels"] == 1
    assert abs(info["duration_s"] - 1.5) < 0.05


def test_inspect_audio_file_missing_and_corrupt(tmp_path):
    missing_info = inspect_audio_file(tmp_path / "nonexistent.wav")
    assert missing_info["exists"] is False
    assert missing_info["valid_wav"] is False

    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"not a real wav file header")
    corrupt_info = inspect_audio_file(corrupt_path)
    assert corrupt_info["exists"] is True
    assert corrupt_info["valid_wav"] is False


def test_match_audio_records_and_missing(tmp_path):
    audio_dir = tmp_path / "train_audio"
    _create_dummy_wav(audio_dir / "audio_1.wav")
    _create_dummy_wav(audio_dir / "audio_3_unexpected.wav")

    df = pd.DataFrame({"filename": ["audio_1.wav", "audio_2_missing.wav"]})
    matched_df, metrics = match_audio_records(df, audio_dir, "train", "filename")

    assert metrics["matched_count"] == 1
    assert metrics["missing_count"] == 1
    assert metrics["missing_files"] == ["audio_2_missing.wav"]
    assert metrics["unexpected_count"] == 1
    assert "audio_3_unexpected.wav" in metrics["unexpected_files"]
    assert bool(matched_df.loc[0, "audio_exists"]) is True
    assert bool(matched_df.loc[1, "audio_exists"]) is False


# -----------------------------------------------------------------------------
# 4. Train / Test Overlap & Disambiguation Test
# -----------------------------------------------------------------------------
def test_train_test_filename_overlap_disambiguation(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _create_dummy_wav(train_dir / "audio_0.wav", duration_s=2.0)
    _create_dummy_wav(test_dir / "audio_0.wav", duration_s=1.0)

    train_df = pd.DataFrame({"filename": ["audio_0.wav"]})
    test_df = pd.DataFrame({"filename": ["audio_0.wav"]})

    matched_train, _ = match_audio_records(train_df, train_dir, "train")
    matched_test, _ = match_audio_records(test_df, test_dir, "test")

    # sample_id must be completely distinct
    assert matched_train.loc[0, "sample_id"] == "train/audio_0.wav"
    assert matched_test.loc[0, "sample_id"] == "test/audio_0.wav"
    assert matched_train.loc[0, "sample_id"] != matched_test.loc[0, "sample_id"]

    # audio paths must point to respective folders
    assert "train" in str(matched_train.loc[0, "audio_path"])
    assert "test" in str(matched_test.loc[0, "audio_path"])


# -----------------------------------------------------------------------------
# 5. Submission Mismatch Investigation Test
# -----------------------------------------------------------------------------
def test_submission_mismatch_detection():
    test_df = pd.DataFrame({"filename": [f"audio_{i}.wav" for i in range(216)]})
    sub_df = pd.DataFrame({"filename": [f"audio_{i}.wav" for i in range(100, 304)]})  # 204 rows
    train_df = pd.DataFrame({"filename": [f"audio_{i}.wav" for i in range(100, 200)]})

    analysis = analyze_submission_mismatch(test_df, sub_df, train_df)

    assert analysis["test_rows"] == 216
    assert analysis["sample_sub_rows"] == 204
    assert analysis["row_count_equal"] is False
    assert analysis["is_exact_match"] is False
    assert analysis["overlap_count"] == 116
    assert analysis["sample_sub_found_in_train_count"] == 100


# -----------------------------------------------------------------------------
# 6. Test Set Dummy Label Removal Test
# -----------------------------------------------------------------------------
def test_prepare_dataset_drops_test_dummy_label(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _create_dummy_wav(train_dir / "audio_1.wav")
    _create_dummy_wav(test_dir / "audio_2.wav")

    train_df = pd.DataFrame({"filename": ["audio_1.wav"], "label": [3.0]})
    test_df = pd.DataFrame({"filename": ["audio_2.wav"], "label": [-1]})
    sub_df = pd.DataFrame({"filename": ["audio_2.wav"], "label": [-1]})

    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)
    sub_df.to_csv(tmp_path / "sample_submission.csv", index=False)

    cfg = get_config(data_dir=tmp_path)
    train_out, test_out, sub_out, report = load_and_prepare_dataset(cfg)

    assert "label" in train_out.columns
    assert "label" not in test_out.columns  # Dummy label stripped!
    assert "sample_id" in train_out.columns
    assert "sample_id" in test_out.columns


# -----------------------------------------------------------------------------
# 7. Live Dataset Integration Test (when dataset path exists)
# -----------------------------------------------------------------------------
def test_live_dataset_full_validation():
    data_dir = os.environ.get("SHL_DATA_DIR", r"C:\Users\aacha\Downloads\shl-hiring-assessment-2026\Dataset_Final")
    if not Path(data_dir).exists():
        pytest.skip(f"Live dataset directory {data_dir} not available.")

    cfg = get_config(data_dir=Path(data_dir))
    report = validate_full_dataset(cfg, inspect_all_audio=True)

    assert report["status"] in {"PASSED", "PASSED_WITH_WARNINGS"}
    assert len(report["errors"]) == 0

    # Verify counts
    train_metrics = report["audio_matching"]["train"]
    test_metrics = report["audio_matching"]["test"]

    assert train_metrics["csv_rows"] == 769
    assert train_metrics["matched_count"] == 769
    assert train_metrics["missing_count"] == 0
    assert train_metrics["corrupt_count"] == 0

    assert test_metrics["csv_rows"] == 216
    assert test_metrics["matched_count"] == 216
    assert test_metrics["missing_count"] == 0
    assert test_metrics["corrupt_count"] == 0

    # Verify score stats
    stats = report["train_labels"]["stats"]
    assert stats["count"] == 769
    assert stats["min"] >= 0.0
    assert stats["max"] <= 5.0
