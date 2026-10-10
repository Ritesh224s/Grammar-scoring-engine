"""Flask Web Application for the Grammar Scoring Engine.

Allows users to upload English WAV speech recordings, view Whisper ASR transcripts,
extract the 72 multimodal features, and obtain continuous grammar ratings [0.0, 5.0].
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from src.config import get_config
from src.inference import GrammarScoringEngine
from src.utils import get_logger

logger = get_logger("webapp")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max audio upload

# Setup upload directory
cfg = get_config()
UPLOAD_DIR = cfg.project_root / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Pre-warm or get singleton inference engine
engine = GrammarScoringEngine.get_instance(cfg)

# Curated benchmark samples from the dataset for 1-click evaluation
BENCHMARK_SAMPLES = [
    {
        "filename": "audio_192.wav",
        "split": "train",
        "title": "Benchmark Sample 1 (Train #192)",
        "description": "Standard spontaneous monologue (~60.5s). Measured grammar score: ~3.0 - 3.5.",
    },
    {
        "filename": "audio_436.wav",
        "split": "train",
        "title": "Benchmark Sample 2 (Train #436)",
        "description": "Extended speech sample (~60.1s). Measured grammar score: ~3.5 - 4.0.",
    },
    {
        "filename": "audio_128.wav",
        "split": "test",
        "title": "Benchmark Sample 3 (Test #128)",
        "description": "Evaluated competition test sample (~45.1s). High fluency profile.",
    },
    {
        "filename": "audio_107.wav",
        "split": "test",
        "title": "Benchmark Sample 4 (Test #107)",
        "description": "Brief speech excerpt (~6.5s) for evaluating short response dynamics.",
    },
]


def _find_sample_path(filename: str, split: str = "train") -> Path | None:
    """Locate sample WAV file in dataset directory."""
    ds_files = cfg.get_dataset_files()
    audio_dir = ds_files["train_audio_dir"] if split == "train" else ds_files["test_audio_dir"]

    if audio_dir and (audio_dir / filename).exists():
        return audio_dir / filename

    # Secondary check in opposite folder
    alt_dir = ds_files["test_audio_dir"] if split == "train" else ds_files["train_audio_dir"]
    if alt_dir and (alt_dir / filename).exists():
        return alt_dir / filename

    return None


@app.route("/")
def index():
    """Render main web dashboard."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Healthcheck endpoint reporting model and device readiness."""
    return jsonify({
        "status": "healthy",
        "model_name": engine.model_name,
        "feature_count": len(engine.feature_columns),
        "asr_model": cfg.whisper_model_name,
        "asr_device": engine.asr_device,
    })


@app.route("/api/samples")
def get_samples():
    """Return list of available benchmark sample audio recordings."""
    available_samples = []
    for s in BENCHMARK_SAMPLES:
        p = _find_sample_path(s["filename"], s["split"])
        if p and p.exists():
            available_samples.append({
                "filename": s["filename"],
                "title": s["title"],
                "description": s["description"],
            })
    return jsonify({"samples": available_samples})


@app.route("/api/samples/<filename>")
def stream_sample(filename: str):
    """Stream sample audio file for in-browser playback."""
    clean_name = secure_filename(filename)
    path = _find_sample_path(clean_name, "train") or _find_sample_path(clean_name, "test")
    if not path or not path.exists():
        return jsonify({"error": "Sample file not found"}), 404
    return send_file(path, mimetype="audio/wav")


@app.route("/api/score", methods=["POST"])
def score_audio_endpoint():
    """Score uploaded WAV audio or curated sample clip."""
    # Case A: Curated sample requested
    sample_name = request.form.get("sample_name")
    if sample_name:
        clean_name = secure_filename(sample_name)
        sample_path = _find_sample_path(clean_name, "train") or _find_sample_path(clean_name, "test")
        if not sample_path or not sample_path.exists():
            return jsonify({
                "success": False,
                "error": f"Sample file '{clean_name}' could not be located in dataset.",
                "error_code": "SAMPLE_NOT_FOUND",
            }), 404

        try:
            result = engine.score_audio(sample_path, sample_id=f"sample/{clean_name}")
            return jsonify(result)
        except Exception as exc:
            logger.error("Error scoring sample '%s': %s", clean_name, exc)
            return jsonify({
                "success": False,
                "error": str(exc),
                "error_code": "SCORING_ERROR",
            }), 500

    # Case B: User file upload
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided in request.",
            "error_code": "NO_FILE",
        }), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file was selected.",
            "error_code": "EMPTY_FILENAME",
        }), 400

    if not file.filename.lower().endswith(".wav"):
        return jsonify({
            "success": False,
            "error": "Only WAV audio files (.wav) are accepted.",
            "error_code": "INVALID_EXTENSION",
        }), 400

    # Save to unique temporary path
    safe_name = secure_filename(file.filename) or "upload.wav"
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    temp_file_path = Path(temp_dir) / safe_name

    try:
        file.save(str(temp_file_path))

        # Check file size
        if temp_file_path.stat().st_size < 1000:
            return jsonify({
                "success": False,
                "error": "Audio file is empty or corrupted (< 1 KB).",
                "error_code": "EMPTY_FILE",
            }), 400

        # Run inference
        result = engine.score_audio(temp_file_path, sample_id=f"upload/{safe_name}")
        return jsonify(result)

    except Exception as exc:
        logger.error("Inference failed on uploaded file '%s': %s", safe_name, exc)
        return jsonify({
            "success": False,
            "error": f"Evaluation error: {exc}",
            "error_code": "INFERENCE_FAILED",
        }), 500

    finally:
        # Clean up temporary upload directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Could not clean up temporary dir %s: %e", temp_dir, e)


def main():
    """Start Flask development server."""
    import argparse
    parser = argparse.ArgumentParser(description="Start Grammar Scoring Engine Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(">>> SHL GRAMMAR SCORING ENGINE - WEB APPLICATION")
    print("=" * 70)
    print(f"URL            : http://{args.host}:{args.port}")
    print(f"Model Pipeline : {engine.model_name} (72 Features)")
    print(f"ASR Pipeline   : {cfg.whisper_model_name} on {engine.asr_device}")
    print("=" * 70 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
