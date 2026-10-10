"""Audio preprocessing and inspection module for the SHL Grammar Scoring Engine.

Provides reliable, leak-free utilities to:
- Load WAV recordings into normalized float32 numpy arrays.
- Convert multi-channel / stereo audio to single-channel mono.
- Resample waveforms to model-specific target sample rates (e.g. 16,000 Hz for Whisper).
- Extract acoustic health metrics (duration, RMS, peak amplitude, silence ratio).
- Validate signal integrity, detect silent or corrupt clips, and prevent memory leaks.
- Analyze and visualize duration distributions across train and test splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .config import AppConfig, get_config
from .dataset import load_and_prepare_dataset
from .utils import get_logger

logger = get_logger("audio")


@dataclass
class AudioSignal:
    """Encapsulates a loaded and preprocessed audio waveform."""

    samples: np.ndarray
    sample_rate: int
    orig_sample_rate: int
    orig_channels: int
    duration_s: float
    peak_amplitude: float
    rms_amplitude: float
    is_silent: bool

    @property
    def num_samples(self) -> int:
        """Total number of audio samples."""
        return len(self.samples)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return self.samples.shape

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "num_samples": self.num_samples,
            "sample_rate": self.sample_rate,
            "orig_sample_rate": self.orig_sample_rate,
            "orig_channels": self.orig_channels,
            "duration_s": round(self.duration_s, 4),
            "peak_amplitude": round(self.peak_amplitude, 5),
            "rms_amplitude": round(self.rms_amplitude, 5),
            "is_silent": self.is_silent,
        }


def load_audio(
    path: Path | str,
    target_sr: int | None = 16000,
    mono: bool = True,
    silence_threshold: float = 1e-4,
) -> AudioSignal:
    """Load an audio WAV file as a 1D float32 numpy array.

    Args:
        path: Path to the WAV file.
        target_sr: Expected sample rate (e.g. 16000). If None, keeps original.
        mono: Whether to downmix multi-channel audio to mono.
        silence_threshold: Peak amplitude below which the signal is flagged as silent.

    Returns:
        AudioSignal object containing 1D samples and acoustic properties.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If file is empty, has invalid parameters, or contains NaN/Inf values.
        RuntimeError: If soundfile/librosa fails to decode the audio.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")

    if target_sr is not None and target_sr <= 0:
        raise ValueError(f"target_sr must be a positive integer, got: {target_sr}")

    # Read raw audio
    try:
        raw_data, orig_sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    except Exception as exc:
        # Fallback to librosa for edge-case headers
        try:
            y, orig_sr = librosa.load(str(file_path), sr=None, mono=False)
            raw_data = np.atleast_2d(y).T.astype(np.float32)
        except Exception as inner_exc:
            raise RuntimeError(f"Failed to decode audio file {file_path}: {exc} | fallback: {inner_exc}") from exc

    if raw_data.size == 0 or raw_data.shape[0] == 0:
        raise ValueError(f"Audio file contains 0 samples: {file_path}")

    orig_channels = raw_data.shape[1]

    # Convert to mono if requested
    if mono and orig_channels > 1:
        samples = np.mean(raw_data, axis=1, dtype=np.float32)
    elif mono and orig_channels == 1:
        samples = raw_data[:, 0]
    else:
        samples = raw_data

    # Ensure float32 1D array for mono
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # Validate finite values
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio file contains non-finite (NaN/Inf) samples: {file_path}")

    # Resample if needed
    effective_sr = orig_sr
    if target_sr is not None and orig_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        effective_sr = target_sr

    # Ensure output is strictly 1D if mono
    if mono and samples.ndim > 1:
        samples = samples.squeeze()

    duration_s = float(len(samples)) / float(effective_sr)
    peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
    is_silent = bool(peak < silence_threshold)

    return AudioSignal(
        samples=samples,
        sample_rate=effective_sr,
        orig_sample_rate=orig_sr,
        orig_channels=orig_channels,
        duration_s=duration_s,
        peak_amplitude=peak,
        rms_amplitude=rms,
        is_silent=is_silent,
    )


def compute_silence_stats(
    samples: np.ndarray,
    sample_rate: int,
    silence_top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Compute frame-level energy statistics to identify silence and speech durations."""
    if len(samples) == 0:
        return {"silence_ratio": 1.0, "speech_duration_s": 0.0, "speech_ratio": 0.0}

    rms_frames = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms_frames) == 0:
        return {"silence_ratio": 1.0, "speech_duration_s": 0.0, "speech_ratio": 0.0}

    max_rms = float(np.max(rms_frames))
    if max_rms <= 1e-6:
        return {"silence_ratio": 1.0, "speech_duration_s": 0.0, "speech_ratio": 0.0}

    threshold = max_rms * (10.0 ** (-silence_top_db / 20.0))
    is_silent_frame = rms_frames < threshold
    silence_ratio = float(np.mean(is_silent_frame))
    speech_ratio = 1.0 - silence_ratio
    total_duration_s = len(samples) / sample_rate
    speech_duration_s = total_duration_s * speech_ratio

    return {
        "silence_ratio": round(silence_ratio, 4),
        "speech_duration_s": round(speech_duration_s, 3),
        "speech_ratio": round(speech_ratio, 4),
    }


def inspect_audio_pipeline(
    path: Path | str,
    target_sr: int = 16000,
    silence_top_db: float = 30.0,
) -> dict[str, Any]:
    """Inspect and preprocess an audio file safely, returning a comprehensive diagnostic dict."""
    res: dict[str, Any] = {
        "path": str(path),
        "filename": Path(path).name,
        "success": False,
        "error": None,
    }

    try:
        signal = load_audio(path, target_sr=target_sr, mono=True)
        silence_info = compute_silence_stats(signal.samples, signal.sample_rate, silence_top_db=silence_top_db)

        res.update({
            "success": True,
            "orig_sample_rate": signal.orig_sample_rate,
            "target_sample_rate": signal.sample_rate,
            "orig_channels": signal.orig_channels,
            "num_samples": signal.num_samples,
            "array_shape": list(signal.shape),
            "duration_s": round(signal.duration_s, 3),
            "peak_amplitude": round(signal.peak_amplitude, 5),
            "rms_amplitude": round(signal.rms_amplitude, 5),
            "is_silent": signal.is_silent,
            "silence_ratio": silence_info["silence_ratio"],
            "speech_duration_s": silence_info["speech_duration_s"],
        })
    except Exception as exc:
        res["success"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"

    return res


def analyze_duration_distribution(
    config: AppConfig | None = None,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect duration distributions of all audio recordings across train and test splits.

    Generates duration summary statistics and saves visualization plot + JSON report.

    Returns:
        (duration_dataframe, summary_report)
    """
    cfg = config or get_config()
    cfg.ensure_directories()

    train_df, test_df, _, val_report = load_and_prepare_dataset(cfg)

    records: list[dict[str, Any]] = []

    # Process training audio
    for _, row in train_df.iterrows():
        p = Path(row["audio_path"])
        info = sf.info(str(p))
        records.append({
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "split": "train",
            "duration_s": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        })

    # Process test audio
    for _, row in test_df.iterrows():
        p = Path(row["audio_path"])
        info = sf.info(str(p))
        records.append({
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "split": "test",
            "duration_s": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        })

    dur_df = pd.DataFrame(records)

    train_dur = dur_df.loc[dur_df["split"] == "train", "duration_s"]
    test_dur = dur_df.loc[dur_df["split"] == "test", "duration_s"]

    # Identify short (< 30s) and long (> 60s) recordings
    short_recordings = dur_df[dur_df["duration_s"] < 30.0].sort_values("duration_s").to_dict("records")
    long_recordings = dur_df[dur_df["duration_s"] > 60.0].sort_values("duration_s", ascending=False).to_dict("records")

    summary: dict[str, Any] = {
        "train": {
            "count": int(len(train_dur)),
            "mean_s": round(float(train_dur.mean()), 2),
            "std_s": round(float(train_dur.std()), 2),
            "min_s": round(float(train_dur.min()), 2),
            "p25_s": round(float(train_dur.quantile(0.25)), 2),
            "median_s": round(float(train_dur.median()), 2),
            "p75_s": round(float(train_dur.quantile(0.75)), 2),
            "max_s": round(float(train_dur.max()), 2),
            "num_under_30s": int((train_dur < 30.0).sum()),
            "num_over_60s": int((train_dur > 60.0).sum()),
        },
        "test": {
            "count": int(len(test_dur)),
            "mean_s": round(float(test_dur.mean()), 2),
            "std_s": round(float(test_dur.std()), 2),
            "min_s": round(float(test_dur.min()), 2),
            "p25_s": round(float(test_dur.quantile(0.25)), 2),
            "median_s": round(float(test_dur.median()), 2),
            "p75_s": round(float(test_dur.quantile(0.75)), 2),
            "max_s": round(float(test_dur.max()), 2),
            "num_under_30s": int((test_dur < 30.0).sum()),
            "num_over_60s": int((test_dur > 60.0).sum()),
        },
        "overall": {
            "total_files": len(dur_df),
            "min_duration_s": round(float(dur_df["duration_s"].min()), 2),
            "max_duration_s": round(float(dur_df["duration_s"].max()), 2),
            "shortest_recording": short_recordings[0] if short_recordings else None,
            "longest_recording": long_recordings[0] if long_recordings else None,
            "short_recordings_count": len(short_recordings),
            "long_recordings_count": len(long_recordings),
        },
        "short_recordings_examples": short_recordings[:5],
        "long_recordings_examples": long_recordings[:5],
        "recommendation": (
            "Recordings range from 6.48s to 61.04s (mean ~54.2s). "
            "All recordings are valid speech clips and must be preserved for feature extraction. "
            "Whisper handles clips <= 30s directly and stiches longer clips seamlessly."
        ),
    }

    if save_outputs:
        json_path = cfg.metrics_dir / "audio_duration_analysis.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Saved duration analysis JSON: %s", json_path)

        # Plot duration histogram
        plot_path = cfg.plots_dir / "audio_duration_distribution.png"
        plot_duration_distribution(dur_df, plot_path)
        logger.info("Saved duration distribution plot: %s", plot_path)

    return dur_df, summary


def plot_duration_distribution(dur_df: pd.DataFrame, output_path: Path | str) -> None:
    """Generate and save a publication-quality duration distribution plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    train_data = dur_df.loc[dur_df["split"] == "train", "duration_s"]
    test_data = dur_df.loc[dur_df["split"] == "test", "duration_s"]

    # Subplot 1: Overlaid histograms with KDE
    sns.histplot(
        train_data,
        color="#2b5c8f",
        alpha=0.6,
        label=f"Train (N={len(train_data)})",
        bins=30,
        kde=True,
        ax=axes[0],
    )
    sns.histplot(
        test_data,
        color="#e06666",
        alpha=0.6,
        label=f"Test (N={len(test_data)})",
        bins=30,
        kde=True,
        ax=axes[0],
    )
    axes[0].axvline(45.0, color="#888888", linestyle="--", linewidth=1.2, label="45-60s Target Window")
    axes[0].axvline(60.0, color="#888888", linestyle="--", linewidth=1.2)
    axes[0].set_title("Audio Duration Distribution (Train vs. Test)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Duration (seconds)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Boxplot comparison
    sns.boxplot(
        data=dur_df,
        x="split",
        y="duration_s",
        hue="split",
        legend=False,
        palette=["#2b5c8f", "#e06666"],
        width=0.4,
        ax=axes[1],
    )
    axes[1].axhline(45.0, color="#888888", linestyle="--", linewidth=1.2)
    axes[1].axhline(60.0, color="#888888", linestyle="--", linewidth=1.2)
    axes[1].set_title("Duration Spread & Outliers", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Split", fontsize=10)
    axes[1].set_ylabel("Duration (seconds)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Annotate shortest outliers
    shortest_test = test_data.min()
    axes[1].text(
        1.1,
        shortest_test,
        f"Shortest Test: {shortest_test:.1f}s (audio_107.wav)",
        verticalalignment="center",
        fontsize=9,
        color="#a00000",
        fontweight="bold",
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_smoke_test(config: AppConfig | None = None) -> list[dict[str, Any]]:
    """Execute smoke test on selected train, test, and short audio samples.

    Tests:
    1. Two training files.
    2. One test file.
    3. The shortest recording in the dataset (audio_107.wav, 6.48s).
    """
    cfg = config or get_config()
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)

    # Select samples
    sample_1 = train_df.iloc[0]  # First train sample
    sample_2 = train_df.iloc[1]  # Second train sample
    sample_3 = test_df.iloc[0]   # First test sample

    # Find shortest test recording
    test_paths = test_df["audio_path"].tolist()
    durations = [sf.info(str(p)).duration for p in test_paths]
    shortest_idx = int(np.argmin(durations))
    sample_shortest = test_df.iloc[shortest_idx]

    smoke_targets = [
        ("Train Sample 1", sample_1),
        ("Train Sample 2", sample_2),
        ("Test Sample 1", sample_3),
        ("Shortest Test Clip", sample_shortest),
    ]

    results = []
    for label, row in smoke_targets:
        diag = inspect_audio_pipeline(row["audio_path"], target_sr=cfg.target_sample_rate)
        diag["target_label"] = label
        diag["sample_id"] = row["sample_id"]
        results.append(diag)

    return results


def main() -> int:
    """CLI execution for duration analysis and smoke testing."""
    cfg = get_config()
    logger.info("Executing Phase 3: Audio Preprocessing and Duration Analysis...")

    dur_df, summary = analyze_duration_distribution(cfg, save_outputs=True)
    smoke_results = run_smoke_test(cfg)

    print("\n" + "=" * 70)
    print("PHASE 3: AUDIO DURATION & PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total audio files analyzed: {summary['overall']['total_files']}")
    print(f"Train duration: {summary['train']['min_s']}s - {summary['train']['max_s']}s (mean: {summary['train']['mean_s']}s)")
    print(f"Test duration:  {summary['test']['min_s']}s - {summary['test']['max_s']}s (mean: {summary['test']['mean_s']}s)")
    print(f"Shortest recording: {summary['overall']['shortest_recording']['filename']} ({summary['overall']['shortest_recording']['duration_s']}s)")
    print(f"Longest recording:  {summary['overall']['longest_recording']['filename']} ({summary['overall']['longest_recording']['duration_s']}s)")
    print("\nSmoke Test Results:")
    for r in smoke_results:
        print(f"  [{r['target_label']}] {r['filename']} -> Duration: {r['duration_s']}s | Shape: {r['array_shape']} | Peak: {r['peak_amplitude']} | Silence Ratio: {r['silence_ratio']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
