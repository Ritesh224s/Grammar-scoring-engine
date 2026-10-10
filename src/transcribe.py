"""Speech-to-text transcription module using pretrained Whisper models (Hugging Face).

Provides reliable, leak-free, and collision-safe transcription:
- Loads pretrained Whisper models (default: openai/whisper-base.en).
- Processes short and long clips without manual 30s truncation.
- Saves raw English transcripts without grammar correction.
- Uses inference mode and FP16 precision on CUDA with automatic CPU fallback.
- Implements collision-free disk caching indexed by unique sample_id (train/ vs test/).
- Supports atomic cache writes, resume capability, and focused smoke tests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from .audio import AudioSignal, load_audio
from .config import AppConfig, get_config
from .dataset import load_and_prepare_dataset
from .utils import get_logger

logger = get_logger("transcribe")

CACHE_COLUMNS = [
    "sample_id",
    "split",
    "filename",
    "relative_path",
    "transcript",
    "status",
    "model_name",
    "device",
    "sample_rate",
    "duration_s",
    "elapsed_time_s",
    "error",
]


def generate_kwargs_for_model(model_name: str) -> dict[str, Any]:
    """Return model-specific generation kwargs.

    English-only Whisper checkpoints (*.en) do not take 'language' or 'task' args.
    Multilingual checkpoints require language='english' and task='transcribe'.
    """
    if model_name.endswith(".en"):
        return {}
    return {"language": "english", "task": "transcribe"}


def load_asr_pipeline(
    model_name: str = "openai/whisper-base.en",
    requested_device: str | None = None,
) -> tuple[Any, str]:
    """Initialize the Hugging Face ASR pipeline with GPU acceleration and CPU fallback.

    Args:
        model_name: Hugging Face model repository or local path.
        requested_device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Tuple of (pipeline_instance, device_name_used).
    """
    if torch is None or pipeline is None:
        raise ImportError(
            "Transcription requires `torch` and `transformers`. "
            "Please ensure requirements are installed."
        )

    cuda_available = torch.cuda.is_available()
    if requested_device is not None:
        use_cuda = requested_device.lower().startswith("cuda") and cuda_available
        if requested_device.lower().startswith("cuda") and not cuda_available:
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
    else:
        use_cuda = cuda_available

    target_device = "cuda:0" if use_cuda else "cpu"
    target_dtype = torch.float16 if use_cuda else torch.float32

    logger.info("Initializing Whisper ASR pipeline ('%s') on %s...", model_name, target_device)

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=target_device,
            torch_dtype=target_dtype,
            chunk_length_s=30,  # Handles audio of any length >30s seamlessly
        )
        return pipe, target_device
    except Exception as exc:
        if use_cuda:
            logger.warning(
                "Failed to initialize Whisper on CUDA (%s). Retrying fallback on CPU...",
                exc,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device="cpu",
                torch_dtype=torch.float32,
                chunk_length_s=30,
            )
            return pipe, "cpu"
        raise exc


def transcribe_audio_signal(
    pipe: Any,
    signal: AudioSignal,
    model_name: str = "openai/whisper-base.en",
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Transcribe an in-memory AudioSignal using the provided ASR pipeline.

    Returns:
        Dict with status ('ok', 'silent', 'failed'), transcript, elapsed_time_s, error.
    """
    import torch

    # Handle silent clips safely
    if signal.is_silent or signal.peak_amplitude < 1e-4:
        return {
            "status": "silent",
            "transcript": "",
            "elapsed_time_s": 0.0,
            "error": "",
        }

    gen_kwargs = generate_kwargs_for_model(model_name)
    input_data = {"raw": signal.samples, "sampling_rate": signal.sample_rate}

    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            out = pipe(input_data, generate_kwargs=gen_kwargs)

        elapsed = round(time.perf_counter() - start_time, 3)

        if isinstance(out, dict):
            text = out.get("text", "").strip()
        elif isinstance(out, list) and len(out) > 0:
            text = out[0].get("text", "").strip()
        else:
            text = str(out).strip()

        return {
            "status": "ok",
            "transcript": text,
            "elapsed_time_s": elapsed,
            "error": "",
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.warning("Inference error on audio (%s): %s", signal.duration_s, exc)
        return {
            "status": "failed",
            "transcript": "",
            "elapsed_time_s": elapsed,
            "error": f"asr: {type(exc).__name__}: {exc}",
        }


def transcribe_single_file(
    audio_path: Path | str,
    pipe: Any,
    sample_id: str,
    split: str,
    target_sr: int = 16000,
    model_name: str = "openai/whisper-base.en",
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Load, inspect, and transcribe a single audio file, returning a complete record."""
    path_obj = Path(audio_path)
    base_record: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "filename": path_obj.name,
        "relative_path": str(path_obj.name),
        "transcript": "",
        "status": "failed",
        "model_name": model_name,
        "device": device_name,
        "sample_rate": target_sr,
        "duration_s": 0.0,
        "elapsed_time_s": 0.0,
        "error": "",
    }

    try:
        sig = load_audio(path_obj, target_sr=target_sr, mono=True)
        base_record["duration_s"] = round(sig.duration_s, 3)
        base_record["sample_rate"] = sig.sample_rate
    except Exception as exc:
        base_record["status"] = "failed"
        base_record["error"] = f"load: {type(exc).__name__}: {exc}"
        return base_record

    res = transcribe_audio_signal(
        pipe=pipe,
        signal=sig,
        model_name=model_name,
        device_name=device_name,
    )
    base_record["status"] = res["status"]
    base_record["transcript"] = res["transcript"]
    base_record["elapsed_time_s"] = res["elapsed_time_s"]
    base_record["error"] = res["error"]

    return base_record


# -----------------------------------------------------------------------------
# Collision-Free Caching Operations
# -----------------------------------------------------------------------------
def load_transcripts_cache(cache_path: Path) -> pd.DataFrame:
    """Load existing transcripts cache if present, otherwise return empty DataFrame."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, keep_default_na=False)
            # Validate required columns
            if "sample_id" in df.columns:
                return df
        except Exception as exc:
            logger.warning("Could not read existing cache at %s: %s. Starting fresh.", cache_path, exc)

    return pd.DataFrame(columns=CACHE_COLUMNS)


def save_transcripts_cache_atomic(df: pd.DataFrame, cache_path: Path) -> None:
    """Atomically save the cache DataFrame to disk to prevent corruptions."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.csv")

    # Ensure consistent column ordering
    cols = [c for c in CACHE_COLUMNS if c in df.columns]
    for c in df.columns:
        if c not in cols:
            cols.append(c)

    df[cols].to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(str(tmp_path), str(cache_path))


def upsert_cache_record(
    cache_df: pd.DataFrame,
    new_record: dict[str, Any],
    cache_path: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Insert or update a single record in the cache DataFrame and save atomically."""
    sample_id = new_record["sample_id"]

    if "sample_id" not in cache_df.columns:
        cache_df = pd.DataFrame(columns=CACHE_COLUMNS)

    existing_mask = cache_df["sample_id"] == sample_id
    if existing_mask.any():
        existing_status = cache_df.loc[existing_mask, "status"].iloc[0]
        # Preserve valid existing cache unless force_refresh is requested
        if existing_status in {"ok", "silent"} and not force_refresh:
            return cache_df
        # Update existing record
        idx = cache_df[existing_mask].index[0]
        for k, v in new_record.items():
            cache_df.loc[idx, k] = v
    else:
        # Append new record cleanly without empty DataFrame concatenation deprecation warnings
        if cache_df.empty:
            cache_df = pd.DataFrame([new_record], columns=CACHE_COLUMNS)
        else:
            new_row = pd.DataFrame([new_record])
            cache_df = pd.concat([cache_df, new_row], ignore_index=True)

    save_transcripts_cache_atomic(cache_df, cache_path)
    return cache_df


# -----------------------------------------------------------------------------
# Smoke Test Implementation
# -----------------------------------------------------------------------------
SMOKE_TEST_TARGETS = [
    ("train", "audio_192.wav"),
    ("train", "audio_436.wav"),
    ("test", "audio_128.wav"),
    ("test", "audio_107.wav"),  # Known shortest test clip (6.48s)
]


def run_smoke_test(
    config: AppConfig | None = None,
    force_refresh: bool = False,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Execute real-model transcription on exactly 4 specific smoke-test clips.

    Target clips:
    - train/audio_192.wav
    - train/audio_436.wav
    - test/audio_128.wav
    - test/audio_107.wav (shortest test recording, 6.48s)

    Does NOT process any other files in the dataset.
    """
    cfg = config or get_config()
    model_name = cfg.whisper_model_name
    cache_path = cfg.get_transcripts_cache_path(model_name)

    logger.info("Executing Phase 4 Smoke Test with Whisper model: %s", model_name)
    logger.info("Cache location: %s", cache_path)

    # 1. Locate files using dataset loader
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)

    targets_to_run: list[dict[str, Any]] = []
    for split, fname in SMOKE_TEST_TARGETS:
        source_df = train_df if split == "train" else test_df
        match = source_df[source_df["filename"] == fname]
        if match.empty:
            logger.warning("Target sample %s/%s not found in %s set.", split, fname, split)
            continue
        row = match.iloc[0]
        targets_to_run.append({
            "sample_id": row["sample_id"],
            "split": split,
            "filename": fname,
            "audio_path": Path(row["audio_path"]),
        })

    logger.info("Identified %d target smoke test files.", len(targets_to_run))
    for t in targets_to_run:
        logger.info("  Target: %s -> %s", t["sample_id"], t["audio_path"])

    # 2. Load existing cache
    cache_df = load_transcripts_cache(cache_path)

    # 3. Load model pipeline
    pipe, device_used = load_asr_pipeline(model_name=model_name, requested_device=device)

    # 4. Process each target
    results: list[dict[str, Any]] = []
    print("\n" + "=" * 76)
    print("WHISPER TRANSCRIPTION SMOKE TEST EXECUTION")
    print(f"Model: {model_name} | Device: {device_used} | Cache: {cache_path.name}")
    print("=" * 76)

    for item in targets_to_run:
        sample_id = item["sample_id"]
        split = item["split"]
        fname = item["filename"]
        audio_path = item["audio_path"]

        # Check existing valid cache
        if not force_refresh and "sample_id" in cache_df.columns:
            existing = cache_df[cache_df["sample_id"] == sample_id]
            if not existing.empty and existing.iloc[0]["status"] in {"ok", "silent"}:
                cached_row = existing.iloc[0].to_dict()
                logger.info("Using cached transcription for %s", sample_id)
                results.append(cached_row)
                _print_sample_result(cached_row, cached=True)
                continue

        # Transcribe fresh
        logger.info("Transcribing %s (%s)...", sample_id, audio_path.name)
        rec = transcribe_single_file(
            audio_path=audio_path,
            pipe=pipe,
            sample_id=sample_id,
            split=split,
            target_sr=cfg.target_sample_rate,
            model_name=model_name,
            device_name=device_used,
        )

        # Update cache atomically
        cache_df = upsert_cache_record(cache_df, rec, cache_path, force_refresh=force_refresh)
        results.append(rec)
        _print_sample_result(rec, cached=False)

    print("=" * 76 + "\n")
    return results


def check_transcript_coverage(config: AppConfig | None = None) -> dict[str, Any]:
    """Inspect transcript cache coverage across all train (769) and test (216) samples."""
    cfg = config or get_config()
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)
    cache_path = cfg.get_transcripts_cache_path(cfg.whisper_model_name)
    cache_df = load_transcripts_cache(cache_path)

    cached_map: dict[str, str] = {}
    if not cache_df.empty and "sample_id" in cache_df.columns:
        for _, r in cache_df.iterrows():
            cached_map[str(r["sample_id"])] = str(r["status"])

    train_total = len(train_df)
    test_total = len(test_df)

    train_ok = sum(1 for sid in train_df["sample_id"] if cached_map.get(sid) in {"ok", "silent"})
    train_failed = sum(1 for sid in train_df["sample_id"] if cached_map.get(sid) == "failed")
    train_missing = train_total - train_ok - train_failed

    test_ok = sum(1 for sid in test_df["sample_id"] if cached_map.get(sid) in {"ok", "silent"})
    test_failed = sum(1 for sid in test_df["sample_id"] if cached_map.get(sid) == "failed")
    test_missing = test_total - test_ok - test_failed

    coverage: dict[str, Any] = {
        "train": {
            "total": train_total,
            "cached_valid": train_ok,
            "cached_failed": train_failed,
            "missing": train_missing,
        },
        "test": {
            "total": test_total,
            "cached_valid": test_ok,
            "cached_failed": test_failed,
            "missing": test_missing,
        },
        "total": {
            "total": train_total + test_total,
            "cached_valid": train_ok + test_ok,
            "cached_failed": train_failed + test_failed,
            "missing": train_missing + test_missing,
        },
        "cache_path": str(cache_path),
        "model_name": cfg.whisper_model_name,
    }
    return coverage


def transcribe_dataset(
    config: AppConfig | None = None,
    split_filter: str | None = None,
    force_refresh: bool = False,
    device: str | None = None,
    max_samples: int | None = None,
    progress_interval: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transcribe all missing or invalid audio recordings across train and test splits.

    Guarantees:
    - Reuses valid cached records without re-running Whisper.
    - Saves cache progress atomically after every sample so interrupted runs can resume instantly.
    - Preserves train/test logical separation using composite sample_id.
    - Gracefully handles silent or unreadable clips.
    """
    cfg = config or get_config()
    model_name = cfg.whisper_model_name
    cache_path = cfg.get_transcripts_cache_path(model_name)
    train_df, test_df, _, _ = load_and_prepare_dataset(cfg)
    cache_df = load_transcripts_cache(cache_path)

    items_to_process: list[dict[str, Any]] = []

    # Prepare candidate items based on split_filter
    if split_filter in (None, "train"):
        for _, row in train_df.iterrows():
            items_to_process.append({
                "sample_id": row["sample_id"],
                "split": "train",
                "filename": row["filename"],
                "audio_path": Path(row["audio_path"]),
            })
    if split_filter in (None, "test"):
        for _, row in test_df.iterrows():
            items_to_process.append({
                "sample_id": row["sample_id"],
                "split": "test",
                "filename": row["filename"],
                "audio_path": Path(row["audio_path"]),
            })

    # Filter to pending items
    cached_ids: set[str] = set()
    if not cache_df.empty and "sample_id" in cache_df.columns:
        valid_rows = cache_df[cache_df["status"].isin(["ok", "silent"])]
        cached_ids = set(valid_rows["sample_id"].astype(str))

    pending: list[dict[str, Any]] = []
    for item in items_to_process:
        if force_refresh or item["sample_id"] not in cached_ids:
            pending.append(item)

    total_candidates = len(items_to_process)
    already_cached = total_candidates - len(pending)
    logger.info(
        "Transcription coverage: %d/%d already cached. Pending to transcribe: %d",
        already_cached, total_candidates, len(pending),
    )

    if max_samples is not None and max_samples > 0:
        pending = pending[:max_samples]
        logger.info("Applying max_samples limit: processing %d records.", len(pending))

    if not pending:
        logger.info("All %d requested records are already validly cached. Zero inference needed.", total_candidates)
        return cache_df, {
            "processed": 0,
            "already_cached": already_cached,
            "failed": 0,
            "total": total_candidates,
            "elapsed_total_s": 0.0,
        }

    # Load ASR pipeline
    pipe, device_used = load_asr_pipeline(model_name=model_name, requested_device=device)

    start_batch_time = time.time()
    n_processed = 0
    n_failed = 0

    print("\n" + "=" * 76)
    print(f"BATCH TRANSCRIPTION: {len(pending)} PENDING SAMPLES")
    print(f"Model: {model_name} | Device: {device_used} | Cache: {cache_path.name}")
    print("=" * 76)

    for idx, item in enumerate(pending, 1):
        sid = item["sample_id"]
        split = item["split"]
        fname = item["filename"]
        apath = item["audio_path"]

        t0 = time.time()
        rec = transcribe_single_file(
            audio_path=apath,
            pipe=pipe,
            sample_id=sid,
            split=split,
            target_sr=cfg.target_sample_rate,
            model_name=model_name,
            device_name=device_used,
        )
        elapsed = time.time() - t0

        if rec["status"] == "failed":
            n_failed += 1
            logger.warning("[%d/%d] FAILED %s: %s", idx, len(pending), sid, rec.get("error", "Unknown error"))
        else:
            n_processed += 1
            if idx % progress_interval == 0 or idx == len(pending) or idx <= 5:
                rate = (time.time() - start_batch_time) / idx
                eta_s = rate * (len(pending) - idx)
                logger.info(
                    "[%d/%d] %s (%s, %.1fs audio) -> %s in %.2fs (ETA: %.1fm)",
                    idx, len(pending), sid, fname, rec.get("duration_s", 0.0), rec["status"], elapsed, eta_s / 60.0,
                )

        # Upsert atomically after every sample to guarantee resilience against interruptions
        cache_df = upsert_cache_record(cache_df, rec, cache_path, force_refresh=force_refresh)

    total_time = round(time.time() - start_batch_time, 2)
    print("=" * 76)
    print(f"BATCH TRANSCRIPTION COMPLETE: {n_processed} processed, {n_failed} failed in {total_time}s ({round(total_time/60.0, 1)}m)")
    print("=" * 76 + "\n")

    summary = {
        "processed": n_processed,
        "already_cached": already_cached,
        "failed": n_failed,
        "total": total_candidates,
        "elapsed_total_s": total_time,
    }
    return cache_df, summary


def _print_sample_result(rec: dict[str, Any], cached: bool = False) -> None:
    """Helper to format smoke test output cleanly to the console."""
    tag = "CACHED" if cached else "FRESH "
    print(f"\n[{tag}] Sample ID : {rec['sample_id']}")
    print(f"       Split     : {rec['split']} | Filename: {rec['filename']}")
    print(f"       Status    : {rec['status']} | Device: {rec['device']} | Time: {rec['elapsed_time_s']}s | Duration: {rec['duration_s']}s")
    if rec.get("error"):
        print(f"       Error     : {rec['error']}")
    print(f"       Transcript: \"{rec['transcript']}\"")


def main() -> int:
    """CLI entrypoint for transcription operations."""
    parser = argparse.ArgumentParser(description="Whisper speech transcription pipeline.")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on 4 specific clips only")
    parser.add_argument("--all", action="store_true", help="Transcribe all pending dataset samples (train and test)")
    parser.add_argument("--check-coverage", action="store_true", help="Report transcript cache coverage")
    parser.add_argument("--split", type=str, choices=["train", "test"], default=None, help="Process only train or test")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to process")
    parser.add_argument("--model", type=str, default=None, help="Whisper model override")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda' or 'cpu')")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--force", action="store_true", help="Force re-transcription ignoring cache")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["whisper_model_name"] = args.model
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)

    cfg = get_config(**overrides)

    if args.check_coverage:
        cov = check_transcript_coverage(cfg)
        print("\n" + "=" * 60)
        print("TRANSCRIPT CACHE COVERAGE REPORT")
        print("=" * 60)
        print(f"Model: {cov['model_name']} | Cache: {cov['cache_path']}")
        print(f"Train: {cov['train']['cached_valid']}/{cov['train']['total']} valid, {cov['train']['cached_failed']} failed, {cov['train']['missing']} missing")
        print(f"Test : {cov['test']['cached_valid']}/{cov['test']['total']} valid, {cov['test']['cached_failed']} failed, {cov['test']['missing']} missing")
        print(f"Total: {cov['total']['cached_valid']}/{cov['total']['total']} valid, {cov['total']['cached_failed']} failed, {cov['total']['missing']} missing")
        print("=" * 60 + "\n")
        return 0

    if args.all or args.split is not None or args.max_samples is not None:
        _, summary = transcribe_dataset(
            cfg,
            split_filter=args.split,
            force_refresh=args.force,
            device=args.device,
            max_samples=args.max_samples,
        )
        return 0 if summary["failed"] == 0 else 1

    if args.smoke_test or len(sys.argv) == 1:
        # Default behavior is to execute the smoke test safely
        results = run_smoke_test(cfg, force_refresh=args.force, device=args.device)
        success = all(r.get("status") in {"ok", "silent"} for r in results)
        return 0 if success else 1

    logger.info("Use --check-coverage, --all, or --smoke-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
