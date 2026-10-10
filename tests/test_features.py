"""Automated unit tests for grammar scoring feature extraction pipeline."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.audio import AudioSignal
from src.features import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    compute_speaking_rate_features,
    extract_audio_features,
    extract_features_for_dataframe,
    extract_single_sample_features,
    extract_text_features,
    get_feature_columns,
    get_feature_descriptions,
    separate_features_and_metadata,
    tokenize_sentences,
    tokenize_words,
)


# -----------------------------------------------------------------------------
# 1. Linguistic & Text Features Tests
# -----------------------------------------------------------------------------
def test_empty_and_whitespace_transcripts():
    for raw in ["", "   ", "\n\t  ", None]:
        feat = extract_text_features(raw)
        assert feat["word_count"] == 0
        assert feat["sentence_count"] == 0
        assert feat["char_count"] == 0
        assert feat["avg_word_length"] == 0.0
        assert feat["avg_sentence_length_words"] == 0.0
        assert feat["ttr"] == 0.0
        assert feat["root_ttr"] == 0.0
        assert feat["log_ttr"] == 0.0
        assert feat["stop_word_count"] == 0
        assert feat["repeated_word_count"] == 0
        assert feat["total_filler_count"] == 0
        # Ensure every numerical feature is non-NaN and finite
        for k, v in feat.items():
            assert not np.isnan(v), f"Feature {k} returned NaN for input: {repr(raw)}"
            assert not np.isinf(v), f"Feature {k} returned Inf for input: {repr(raw)}"


def test_punctuation_only_transcripts():
    raw = "... ??? !!! ,,, ;;; :::"
    feat = extract_text_features(raw)
    assert feat["word_count"] == 0
    assert feat["sentence_count"] == 0
    assert feat["punctuation_count"] > 0
    assert feat["comma_count"] == 3
    assert feat["period_count"] == 3
    assert feat["question_count"] == 3
    assert feat["exclamation_count"] == 3
    assert feat["ttr"] == 0.0
    for k, v in feat.items():
        assert not np.isnan(v)
        assert not np.isinf(v)


def test_word_and_sentence_counts():
    text = "Hello world! This is a simple test. Are you ready?"
    feat = extract_text_features(text)
    assert feat["word_count"] == 10
    assert feat["sentence_count"] == 3
    # Sentence lengths: 2 ("Hello world"), 5 ("This is a simple test"), 3 ("Are you ready")
    assert feat["min_sentence_length_words"] == 2
    assert feat["max_sentence_length_words"] == 5
    assert feat["avg_sentence_length_words"] == round(10 / 3, 4)
    assert feat["question_count"] == 1
    assert feat["exclamation_count"] == 1
    assert feat["period_count"] == 1


def test_lexical_diversity_metrics():
    # 6 words, exactly 3 unique ("the", "cat", "saw")
    text = "The cat saw the cat saw."
    feat = extract_text_features(text)
    assert feat["word_count"] == 6
    assert feat["unique_word_count"] == 3
    assert feat["ttr"] == round(3 / 6, 5)
    assert feat["root_ttr"] == round(3 / math.sqrt(6), 5)
    assert feat["log_ttr"] == round(math.log(3) / math.log(6), 5)


def test_repeated_word_and_filler_features():
    # Immediate repetition: "that that"
    # Overt disfluency: "um"
    # Discourse fillers: "like", "you know"
    text = "Um, I think that that is like, you know, very good."
    feat = extract_text_features(text)
    assert feat["repeated_word_count"] == 1
    assert feat["filler_disfluency_count"] == 1  # "um"
    assert feat["filler_discourse_count"] >= 2   # "like", "you know"
    assert feat["total_filler_count"] >= 3
    assert feat["total_filler_rate"] > 0.0


def test_grammar_and_syntactic_signals():
    text = "Because she was studying, she succeeded although the exam was very difficult."
    feat = extract_text_features(text)
    # Subordinating conjunctions: "because", "although"
    assert feat["subordinating_conjunction_count"] == 2
    assert feat["coordinating_conjunction_count"] == 0
    assert feat["subordination_index"] == 1.0
    # Pronouns: "she" (personal pronoun)
    assert feat["personal_pronoun_count"] >= 2
    # Verb auxiliaries: "was" (appears twice)
    assert feat["auxiliary_verb_count"] >= 2
    # Past tense regular: "succeeded"
    assert feat["past_tense_marker_count"] >= 1
    # Prepositions: "the" is stop word, but conjunctions are detected
    assert feat["subordinating_conjunction_ratio"] > 0.0


# -----------------------------------------------------------------------------
# 2. Audio & Multimodal Delivery Features Tests
# -----------------------------------------------------------------------------
def test_audio_features_zero_duration_and_silence():
    # Synthetic silent signal
    sig = AudioSignal(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        orig_sample_rate=16000,
        orig_channels=1,
        duration_s=0.0,
        peak_amplitude=0.0,
        rms_amplitude=0.0,
        is_silent=True,
    )
    feat = extract_audio_features(sig)
    assert feat["audio_duration_s"] == 0.0
    assert feat["audio_silence_ratio"] == 1.0
    assert feat["audio_speech_duration_s"] == 0.0
    for k, v in feat.items():
        assert not np.isnan(v)
        assert not np.isinf(v)


def test_audio_features_valid_signal():
    # 1 second 440Hz sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig = AudioSignal(
        samples=samples,
        sample_rate=16000,
        orig_sample_rate=16000,
        orig_channels=1,
        duration_s=1.0,
        peak_amplitude=0.5,
        rms_amplitude=float(np.sqrt(np.mean(samples ** 2))),
        is_silent=False,
    )
    feat = extract_audio_features(sig)
    assert feat["audio_duration_s"] == 1.0
    assert feat["audio_rms_peak"] == 0.5
    assert feat["audio_rms_mean"] > 0.0
    assert feat["audio_zcr_mean"] > 0.0
    assert feat["audio_speech_ratio"] > 0.0


def test_speaking_rate_multimodal_alignment():
    text_feat = {"word_count": 10}
    audio_feat = {"audio_duration_s": 5.0, "audio_speech_duration_s": 4.0}
    rates = compute_speaking_rate_features(text_feat, audio_feat)
    assert rates["speaking_rate_wps"] == 2.0
    assert rates["speaking_rate_wpm"] == 120.0
    assert rates["phonation_rate_wps"] == 2.5
    assert rates["articulation_pace"] == 0.5


# -----------------------------------------------------------------------------
# 3. Extraction Health, Error Handling & Schema Consistency Tests
# -----------------------------------------------------------------------------
def test_missing_or_failed_transcript_handling(tmp_path):
    # Dummy audio file
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"RIFFdummydata")

    # When transcript status is 'failed' or transcript is None
    rec = extract_single_sample_features(
        sample_id="train/dummy.wav",
        split="train",
        filename="dummy.wav",
        audio_path=audio_path,
        transcript=None,
        transcript_status="failed",
        transcript_error="Corrupt audio file",
    )
    assert rec["extraction_status"].startswith("missing_transcript")
    assert "Corrupt audio" in rec["extraction_error"]
    # All numerical feature columns should be NaN (NOT silently converted to 0.0!)
    for f in FEATURE_COLUMNS:
        assert np.isnan(rec[f]), f"Feature {f} should be NaN on failed transcript"


def test_unreadable_audio_error_handling(tmp_path):
    # Non-existent audio file
    non_existent = tmp_path / "does_not_exist.wav"
    rec = extract_single_sample_features(
        sample_id="test/missing.wav",
        split="test",
        filename="missing.wav",
        audio_path=non_existent,
        transcript="This is valid text",
        transcript_status="ok",
    )
    assert rec["extraction_status"] == "failed_audio_extraction"
    assert rec["extraction_error"] != ""
    for f in FEATURE_COLUMNS:
        assert np.isnan(rec[f])


def test_train_test_filename_collision_handling():
    # Simulate train and test having identical filenames (audio_0.wav)
    train_data = pd.DataFrame([{
        "sample_id": "train/audio_0.wav",
        "split": "train",
        "filename": "audio_0.wav",
        "audio_path": "fake/path/audio_0.wav",
        "label": 4.25,
    }])
    test_data = pd.DataFrame([{
        "sample_id": "test/audio_0.wav",
        "split": "test",
        "filename": "audio_0.wav",
        "audio_path": "fake/path/audio_0.wav",
    }])

    cache_df = pd.DataFrame([
        {
            "sample_id": "train/audio_0.wav",
            "transcript": "Train speech content here.",
            "status": "ok",
        },
        {
            "sample_id": "test/audio_0.wav",
            "transcript": "Different test speech recording.",
            "status": "ok",
        },
    ])

    with patch("src.features.extract_audio_features") as mock_audio:
        mock_audio.return_value = {
            "audio_duration_s": 5.0,
            "audio_rms_mean": 0.1,
            "audio_rms_std": 0.02,
            "audio_rms_peak": 0.3,
            "audio_zcr_mean": 0.05,
            "audio_zcr_std": 0.01,
            "audio_silence_ratio": 0.2,
            "audio_speech_ratio": 0.8,
            "audio_speech_duration_s": 4.0,
        }

        train_feat = extract_features_for_dataframe(train_data, cache_df, "train")
        test_feat = extract_features_for_dataframe(test_data, cache_df, "test")

        # Confirm distinct IDs
        assert train_feat.iloc[0]["sample_id"] == "train/audio_0.wav"
        assert test_feat.iloc[0]["sample_id"] == "test/audio_0.wav"

        # Confirm train has label and test does NOT
        assert "label" in train_feat.columns
        assert "label" not in test_feat.columns
        assert train_feat.iloc[0]["label"] == 4.25

        # Confirm feature schemas match perfectly
        train_features = [c for c in train_feat.columns if c in FEATURE_COLUMNS]
        test_features = [c for c in test_feat.columns if c in FEATURE_COLUMNS]
        assert train_features == test_features == FEATURE_COLUMNS


def test_separation_of_features_and_metadata():
    # Build sample DataFrame with metadata, label, and features
    row = {
        "sample_id": "train/audio_1.wav",
        "split": "train",
        "filename": "audio_1.wav",
        "extraction_status": "ok",
        "extraction_error": "",
        "label": 3.75,
    }
    for col in FEATURE_COLUMNS:
        row[col] = 1.0

    df = pd.DataFrame([row])
    X, y, meta = separate_features_and_metadata(df)

    # Guarantees
    assert list(X.columns) == FEATURE_COLUMNS
    assert "sample_id" not in X.columns
    assert "filename" not in X.columns
    assert "split" not in X.columns
    assert "label" not in X.columns
    assert y is not None and y.iloc[0] == 3.75
    assert meta.iloc[0]["sample_id"] == "train/audio_1.wav"
