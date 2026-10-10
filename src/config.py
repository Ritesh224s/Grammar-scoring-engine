"""Configuration module for the SHL Grammar Scoring Engine.

Defines project directory paths, audio specifications, ML parameters,
target bounds, and environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Base project root (resolved dynamically relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_env_path(name: str, default: Path) -> Path:
    """Read an environment variable as a resolved Path or return default."""
    val = os.environ.get(name)
    if val:
        return Path(val).expanduser().resolve()
    # If default data/raw doesn't contain train.csv, check standard competition dataset path
    if name == "SHL_DATA_DIR" and not (default / "train.csv").exists():
        standard_path = Path("C:/Users/aacha/Downloads/shl-hiring-assessment-2026/Dataset_Final")
        if (standard_path / "train.csv").exists():
            return standard_path
    return default


@dataclass
class AppConfig:
    """Project-wide configuration settings."""

    # -------------------------------------------------------------------------
    # Directory paths
    # -------------------------------------------------------------------------
    project_root: Path = PROJECT_ROOT

    # Data directory: reads SHL_DATA_DIR if set, else defaults to data/raw
    data_dir: Path = field(
        default_factory=lambda: _get_env_path("SHL_DATA_DIR", PROJECT_ROOT / "data" / "raw")
    )
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    models_dir: Path = PROJECT_ROOT / "models"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"
    predictions_dir: Path = PROJECT_ROOT / "outputs" / "predictions"
    provisional_predictions_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "test_predictions.csv"
    final_model_path: Path = PROJECT_ROOT / "models" / "final_grammar_scorer.joblib"
    submission_path: Path = field(
        default_factory=lambda: _get_env_path(
            "SHL_SUBMISSION_PATH", PROJECT_ROOT / "submission.csv"
        )
    )

    # -------------------------------------------------------------------------
    # Target score bounds and evaluation
    # -------------------------------------------------------------------------
    min_score: float = 0.0
    max_score: float = 5.0
    target_column: str = "label"
    id_column: str = "filename"

    # -------------------------------------------------------------------------
    # Audio preprocessing parameters
    # -------------------------------------------------------------------------
    target_sample_rate: int = 16000  # Whisper & audio feature standard
    silence_top_db: float = 30.0     # Split threshold for silence analysis
    min_pause_duration_s: float = 0.3  # Pauses >= 0.3s considered hesitation pauses

    # -------------------------------------------------------------------------
    # Speech-to-Text (ASR) settings
    # -------------------------------------------------------------------------
    whisper_model_name: str = field(
        default_factory=lambda: os.environ.get("SHL_WHISPER_MODEL", "openai/whisper-base.en")
    )
    asr_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("SHL_ASR_BATCH_SIZE", "1"))
    )

    # -------------------------------------------------------------------------
    # Machine Learning & Cross-Validation settings
    # -------------------------------------------------------------------------
    seed: int = 42
    n_splits: int = 5              # 5-fold cross-validation
    holdout_fraction: float = 0.2  # 20% untouched holdout validation

    @property
    def train_features_path(self) -> Path:
        """Path to processed training feature matrix."""
        return self.processed_data_dir / "train_features.csv"

    @property
    def test_features_path(self) -> Path:
        """Path to processed test feature matrix."""
        return self.processed_data_dir / "test_features.csv"

    @property
    def feature_report_path(self) -> Path:
        """Path to JSON feature extraction report."""
        return self.metrics_dir / "feature_extraction_report.json"

    @property
    def feature_summary_path(self) -> Path:
        """Path to Markdown feature extraction summary."""
        return self.metrics_dir / "feature_extraction_summary.md"

    @property
    def feature_distributions_plot(self) -> Path:
        """Path to feature distributions plot."""
        return self.plots_dir / "feature_distributions.png"

    @property
    def feature_correlation_plot(self) -> Path:
        """Path to feature correlation heatmap plot."""
        return self.plots_dir / "feature_correlation.png"

    def get_transcripts_cache_path(self, model_name: str | None = None) -> Path:
        """Return the CSV cache path for a specific Whisper model's transcripts."""
        m_name = model_name or self.whisper_model_name
        slug = "".join(c if c.isalnum() else "_" for c in m_name).strip("_").lower()
        return self.cache_dir / f"transcripts_{slug}.csv"

    def ensure_directories(self) -> None:
        """Create all required working directories if they do not exist."""
        for d in (
            self.data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.models_dir,
            self.metrics_dir,
            self.plots_dir,
            self.notebooks_dir,
            self.predictions_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def get_dataset_files(self) -> dict[str, Path | None]:
        """Locate raw dataset CSV files and audio directories.

        Returns dictionary of paths for:
          - 'train_csv'
          - 'test_csv'
          - 'sample_sub_csv'
          - 'train_audio_dir'
          - 'test_audio_dir'
        """
        paths: dict[str, Path | None] = {
            "train_csv": None,
            "test_csv": None,
            "sample_sub_csv": None,
            "train_audio_dir": None,
            "test_audio_dir": None,
        }

        # Look in data_dir
        if not self.data_dir.exists():
            return paths

        # Find CSVs
        for p in self.data_dir.glob("*.csv"):
            name = p.name.lower()
            if "train" in name:
                paths["train_csv"] = p
            elif "sample" in name or "sub" in name:
                paths["sample_sub_csv"] = p
            elif "test" in name:
                paths["test_csv"] = p

        # Find audio directories
        for p in self.data_dir.iterdir():
            if p.is_dir():
                name = p.name.lower()
                if "train" in name:
                    paths["train_audio_dir"] = p
                elif "test" in name:
                    paths["test_audio_dir"] = p

        return paths


def get_config(**overrides) -> AppConfig:
    """Build and return an AppConfig instance with optional overrides."""
    cfg = AppConfig(**overrides)
    cfg.ensure_directories()
    return cfg
