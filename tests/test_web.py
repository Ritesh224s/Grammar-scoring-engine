"""Automated tests for the Flask web application and inference endpoints."""

import io
from pathlib import Path
import pytest
from app import app
from src.config import get_config


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page_loads(client):
    """Verify that the home page renders correctly."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "SHL Grammar Scoring Engine" in html
    assert "dropZone" in html
    assert "analyzeBtn" in html


def test_health_endpoint(client):
    """Verify the /api/health endpoint."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["feature_count"] == 72
    assert "model_name" in data


def test_samples_endpoint(client):
    """Verify that the /api/samples endpoint lists benchmark files."""
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "samples" in data
    assert isinstance(data["samples"], list)
    if len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "filename" in sample
        assert "title" in sample


def test_score_no_file(client):
    """Verify error when no file is uploaded."""
    resp = client.post("/api/score", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["error_code"] == "NO_FILE"


def test_score_invalid_extension(client):
    """Verify rejection of non-WAV audio files."""
    data = {
        "audio": (io.BytesIO(b"dummy text content"), "speech.mp3")
    }
    resp = client.post("/api/score", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_json = resp.get_json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "INVALID_EXTENSION"


def test_score_empty_wav(client):
    """Verify rejection of empty or zero-byte files."""
    data = {
        "audio": (io.BytesIO(b""), "empty.wav")
    }
    resp = client.post("/api/score", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_json = resp.get_json()
    assert res_json["success"] is False


def test_score_sample_audio_benchmark(client):
    """Test full inference end-to-end on the short benchmark clip (audio_107.wav)."""
    cfg = get_config()
    ds = cfg.get_dataset_files()
    test_dir = ds["test_audio_dir"]

    if not test_dir or not (test_dir / "audio_107.wav").exists():
        pytest.skip("audio_107.wav not available in dataset path.")

    resp = client.post("/api/score", data={"sample_name": "audio_107.wav"})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert 0.0 <= data["grammar_score"] <= 5.0
    assert "transcript" in data
    assert len(data["transcript"]) > 0
    assert "audio_metrics" in data
    assert "linguistic_metrics" in data
    assert data["linguistic_metrics"]["word_count"] > 0
    assert data["linguistic_metrics"]["speaking_rate_wpm"] > 0
