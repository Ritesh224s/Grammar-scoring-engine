"""SHL Grammar Scoring Engine 2026.

Predicting continuous English grammar scores (0 to 5) from audio recordings.
"""

from .audio import AudioSignal, compute_silence_stats, inspect_audio_pipeline, load_audio
from .config import AppConfig, get_config
from .dataset import load_and_prepare_dataset, load_raw_csvs, validate_full_dataset
from .utils import get_device_info, get_logger, set_seed

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "get_config",
    "get_logger",
    "set_seed",
    "get_device_info",
    "load_raw_csvs",
    "validate_full_dataset",
    "load_and_prepare_dataset",
    "AudioSignal",
    "load_audio",
    "compute_silence_stats",
    "inspect_audio_pipeline",
]
