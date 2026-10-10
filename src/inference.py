"""Production inference engine for the Grammar Scoring Engine.

Loads the final trained model (models/final_grammar_scorer.joblib),
manages Whisper ASR speech-to-text pipeline, extracts the exact 72-feature
multimodal representation, and predicts continuous grammar scores [0.0, 5.0].
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import AppConfig, get_config
from .dataset import inspect_audio_file
from .features import FEATURE_COLUMNS, extract_single_sample_features
from .transcribe import load_asr_pipeline, transcribe_single_file
from .utils import get_logger

logger = get_logger("inference")


class GrammarScoringEngine:
    """Singleton inference engine managing model loading and audio evaluation."""

    _instance: GrammarScoringEngine | None = None

    def __init__(self, config: AppConfig | None = None) -> None:
        self.cfg = config or get_config()
        self.model_pipeline: Any | None = None
        self.model_metadata: dict[str, Any] = {}
        self.feature_columns: list[str] = FEATURE_COLUMNS
        self.asr_pipe: Any | None = None
        self.asr_device: str = "cpu"
        self._is_initialized: bool = False

    @classmethod
    def get_instance(cls, config: AppConfig | None = None) -> GrammarScoringEngine:
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = GrammarScoringEngine(config)
            cls._instance.initialize()
        return cls._instance

    def initialize(self) -> None:
        """Load the trained model and initialize ASR pipeline."""
        if self._is_initialized:
            return

        self._load_trained_model()
        self._load_asr_pipeline()
        self._is_initialized = True
        logger.info("GrammarScoringEngine fully initialized and ready for inference.")

    def _load_trained_model(self) -> None:
        """Load the final trained model artifact."""
        model_path = self.cfg.final_model_path
        if not model_path.exists():
            # Fallback to Phase 6 best_model if final_grammar_scorer not found
            fallback_path = self.cfg.models_dir / "best_model.joblib"
            if fallback_path.exists():
                logger.warning("Final model not found at %s. Using fallback %s.", model_path, fallback_path)
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"No trained model artifact found at {model_path} or {fallback_path}.")

        logger.info("Loading model artifact from %s...", model_path)
        artifact = joblib.load(model_path)

        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.model_pipeline = artifact["pipeline"]
            self.model_metadata = artifact.get("metadata", {})
            self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
            self.model_name = artifact.get("model_name", "Extra Trees")
        else:
            self.model_pipeline = artifact
            self.model_name = "Trained Pipeline"

        logger.info("Loaded model '%s' with %d features.", self.model_name, len(self.feature_columns))

    def _load_asr_pipeline(self) -> None:
        """Load or connect to Whisper ASR pipeline."""
        try:
            pipe, device = load_asr_pipeline(model_name=self.cfg.whisper_model_name)
            self.asr_pipe = pipe
            self.asr_device = device
            logger.info("Whisper ASR pipeline loaded on device: %s", device)
        except Exception as exc:
            logger.error("Failed to initialize Whisper ASR: %s", exc)
            raise exc

    def score_audio(
        self,
        audio_path: Path | str,
        sample_id: str = "upload/audio.wav",
    ) -> dict[str, Any]:
        """Perform end-to-end evaluation of an input WAV audio recording.

        Steps:
        1. Validates audio format, existence, and readability.
        2. Transcribes spoken English with Whisper ASR.
        3. Extracts the identical 72 numerical features used in training.
        4. Evaluates the Scikit-learn regression pipeline.
        5. Clips prediction to [0.0, 5.0] and computes rich analysis metrics.

        Returns:
            Dictionary containing score, transcript, audio metadata, and linguistic metrics.
        """
        self.initialize()
        path_obj = Path(audio_path).resolve()

        # 1. Audio validation
        if not path_obj.exists():
            return {
                "success": False,
                "error": f"Audio file not found at: {path_obj}",
                "error_code": "FILE_NOT_FOUND",
            }

        audio_info = inspect_audio_file(path_obj)
        if not audio_info["valid_wav"]:
            return {
                "success": False,
                "error": f"Invalid audio file: {audio_info.get('error', 'Not a valid WAV recording.')}",
                "error_code": "INVALID_WAV",
            }

        duration = audio_info["duration_s"]
        if duration is None or duration < 0.5:
            return {
                "success": False,
                "error": "Audio recording is too short (minimum duration is 0.5 seconds).",
                "error_code": "AUDIO_TOO_SHORT",
            }

        start_time = time.perf_counter()

        # 2. Whisper Speech-to-Text Transcription
        logger.info("Transcribing '%s' (%.1fs)...", path_obj.name, duration)
        transcription_rec = transcribe_single_file(
            audio_path=path_obj,
            pipe=self.asr_pipe,
            sample_id=sample_id,
            split="inference",
            target_sr=self.cfg.target_sample_rate,
            model_name=self.cfg.whisper_model_name,
            device_name=self.asr_device,
        )

        transcript = transcription_rec.get("transcript", "").strip()
        t_status = transcription_rec.get("status", "failed")
        t_error = transcription_rec.get("error", "")

        # 3. 72-Feature Extraction
        logger.info("Extracting 72 multimodal features for '%s'...", path_obj.name)
        feature_record = extract_single_sample_features(
            sample_id=sample_id,
            split="inference",
            filename=path_obj.name,
            audio_path=path_obj,
            transcript=transcript if transcript else "(No speech detected)",
            transcript_status="ok" if t_status in ("ok", "silent") else t_status,
            transcript_error=t_error,
            target_sr=self.cfg.target_sample_rate,
        )

        if feature_record["extraction_status"] != "ok":
            logger.warning("Feature extraction flagged status: %s", feature_record["extraction_status"])

        # 4. Construct Feature Vector matching training schema exactly
        X_df = pd.DataFrame(
            [[feature_record.get(col, np.nan) for col in self.feature_columns]],
            columns=self.feature_columns,
        )

        # 5. Model Inference
        raw_pred = self.model_pipeline.predict(X_df)[0]
        score = float(np.clip(raw_pred, self.cfg.min_score, self.cfg.max_score))
        elapsed_total = round(time.perf_counter() - start_time, 2)

        # 6. Interpretive Scoring Band & Feedback
        score_band, band_desc, percentile = self._categorize_score(score)

        return {
            "success": True,
            "filename": path_obj.name,
            "grammar_score": round(score, 2),
            "score_precise": round(score, 4),
            "score_band": score_band,
            "score_description": band_desc,
            "percentile_estimate": percentile,
            "scale": {"min": self.cfg.min_score, "max": self.cfg.max_score},
            "transcript": transcript if transcript else "(No recognizable speech detected)",
            "transcription_status": t_status,
            "elapsed_time_s": elapsed_total,
            "model_used": self.model_name,
            "device": self.asr_device,
            "audio_metrics": {
                "duration_s": round(duration, 2),
                "sample_rate": audio_info.get("sample_rate", self.cfg.target_sample_rate),
                "channels": audio_info.get("channels", 1),
                "silence_ratio": round(float(feature_record.get("audio_silence_ratio", 0.0)), 3),
                "rms_mean": round(float(feature_record.get("audio_rms_mean", 0.0)), 4),
            },
            "linguistic_metrics": {
                "word_count": int(feature_record.get("word_count", 0)),
                "sentence_count": int(feature_record.get("sentence_count", 0)),
                "speaking_rate_wpm": round(float(feature_record.get("speaking_rate_wpm", 0.0)), 1),
                "ttr_lexical_diversity": round(float(feature_record.get("ttr", 0.0)), 3),
                "filler_count": int(feature_record.get("overt_filler_count", 0) + feature_record.get("discourse_markers_count", 0)),
                "subordination_index": round(float(feature_record.get("subordination_index", 0.0)), 3),
                "avg_sentence_length_words": round(float(feature_record.get("avg_sentence_length_words", 0.0)), 1),
                "repeated_words_ratio": round(float(feature_record.get("repeated_words_ratio", 0.0)), 3),
                "modal_verbs_count": int(feature_record.get("modal_verbs_count", 0)),
                "conjunction_count": int(feature_record.get("conjunction_count", 0)),
            },
        }

    @staticmethod
    def _categorize_score(score: float) -> tuple[str, str, str]:
        """Map continuous score [0, 5] to intuitive proficiency band and percentile."""
        if score >= 4.2:
            return (
                "Expert / Native-like",
                "Demonstrates exceptional syntactic control, sophisticated clause structure, and expansive lexical diversity with fluent, natural delivery.",
                "Top 10% (90th+ percentile)",
            )
        elif score >= 3.5:
            return (
                "Proficient / Advanced",
                "Demonstrates strong grammatical accuracy and varied sentence patterns. Expresses ideas coherently with minimal noticeable grammatical hesitation.",
                "Top 30% (70th - 89th percentile)",
            )
        elif score >= 2.8:
            return (
                "Competent / Intermediate",
                "Good functional control of English grammar. Able to form complex ideas, with occasional minor syntactic inconsistencies or repetitive phrasing.",
                "Average (40th - 69th percentile)",
            )
        elif score >= 2.0:
            return (
                "Developing / Modest",
                "Basic grammatical command. Relies primarily on simple coordinate sentences, frequent hesitation pauses, or limited lexical diversity.",
                "Developing (15th - 39th percentile)",
            )
        else:
            return (
                "Foundational / Novice",
                "Limited grammatical cohesion. Frequent structural errors, extensive hesitations, or very brief spoken response.",
                "Foundational (< 15th percentile)",
            )
