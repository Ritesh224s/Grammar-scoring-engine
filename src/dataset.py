"""Dataset loading, validation, and audio matching module for the SHL Grammar Scoring Engine.

Provides reusable, leak-free functions to:
- Load raw CSVs (train.csv, test.csv, sample_submission.csv).
- Validate schema, labels, and value ranges.
- Check audio file existence, header integrity, and duration.
- Disambiguate overlapping filenames using unique sample identifiers (e.g. 'train/audio_0.wav').
- Thoroughly analyze sample_submission.csv vs test.csv discrepancies.
- Export machine-readable and human-readable validation reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from .config import AppConfig, get_config
from .utils import get_logger

logger = get_logger("dataset")


def load_raw_csvs(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load train.csv, test.csv, and sample_submission.csv from the data directory.

    Args:
        data_dir: Directory containing the CSVs. If None, uses AppConfig.data_dir.

    Returns:
        Dictionary mapping 'train', 'test', and 'sample_sub' to pandas DataFrames.

    Raises:
        FileNotFoundError: If any required CSV cannot be located.
    """
    if data_dir is None:
        cfg = get_config()
        files = cfg.get_dataset_files()
        train_path = files["train_csv"]
        test_path = files["test_csv"]
        sub_path = files["sample_sub_csv"]
    else:
        d = Path(data_dir)
        train_path = d / "train.csv"
        test_path = d / "test.csv"
        sub_path = d / "sample_submission.csv"

    if not train_path or not Path(train_path).exists():
        raise FileNotFoundError(f"train.csv not found at {train_path}")
    if not test_path or not Path(test_path).exists():
        raise FileNotFoundError(f"test.csv not found at {test_path}")
    if not sub_path or not Path(sub_path).exists():
        raise FileNotFoundError(f"sample_submission.csv not found at {sub_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path)

    return {
        "train": train_df,
        "test": test_df,
        "sample_sub": sub_df,
    }


def validate_csv_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "dataset",
) -> list[str]:
    """Verify that expected columns exist in the DataFrame."""
    errors = []
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"[{name}] Missing required columns: {missing_cols}. Found: {df.columns.tolist()}")
    return errors


def validate_labels(
    df: pd.DataFrame,
    target_col: str = "label",
    min_score: float = 0.0,
    max_score: float = 5.0,
) -> dict[str, Any]:
    """Validate continuous grammar score labels in the training set.

    Checks:
    - Missing / null values
    - Non-numeric entries
    - Bounds violations [min_score, max_score]
    - Summary distribution stats
    """
    results: dict[str, Any] = {
        "total_rows": len(df),
        "missing_count": 0,
        "non_numeric_count": 0,
        "out_of_bounds_count": 0,
        "unique_values_count": 0,
        "stats": {},
        "errors": [],
        "warnings": [],
    }

    if target_col not in df.columns:
        results["errors"].append(f"Target column '{target_col}' not found in DataFrame.")
        return results

    raw_series = df[target_col]
    results["missing_count"] = int(raw_series.isna().sum())
    if results["missing_count"] > 0:
        results["errors"].append(f"Found {results['missing_count']} missing label values.")

    # Convert to numeric
    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    results["non_numeric_count"] = int(numeric_series.isna().sum() - results["missing_count"])
    if results["non_numeric_count"] > 0:
        results["errors"].append(f"Found {results['non_numeric_count']} non-numeric label entries.")

    valid_labels = numeric_series.dropna()
    if not valid_labels.empty:
        results["unique_values_count"] = int(valid_labels.nunique())
        min_val = float(valid_labels.min())
        max_val = float(valid_labels.max())

        oob = (valid_labels < min_score) | (valid_labels > max_score)
        results["out_of_bounds_count"] = int(oob.sum())
        if results["out_of_bounds_count"] > 0:
            results["errors"].append(
                f"Found {results['out_of_bounds_count']} labels outside [{min_score}, {max_score}] (min={min_val}, max={max_val})."
            )

        results["stats"] = {
            "count": int(valid_labels.count()),
            "mean": round(float(valid_labels.mean()), 4),
            "std": round(float(valid_labels.std()), 4),
            "min": round(min_val, 4),
            "25%": round(float(valid_labels.quantile(0.25)), 4),
            "50%": round(float(valid_labels.quantile(0.50)), 4),
            "75%": round(float(valid_labels.quantile(0.75)), 4),
            "max": round(max_val, 4),
        }
    else:
        results["errors"].append("No valid numeric labels found.")

    return results


def inspect_audio_file(path: Path) -> dict[str, Any]:
    """Inspect a WAV audio file's header without loading the entire waveform into memory.

    Returns:
        Dict with exists, size_bytes, valid_wav, samplerate, channels, duration_s, and error.
    """
    path = Path(path)
    res: dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": 0,
        "valid_wav": False,
        "samplerate": None,
        "channels": None,
        "duration_s": None,
        "error": None,
    }

    if not res["exists"]:
        res["error"] = "File does not exist"
        return res

    try:
        res["size_bytes"] = path.stat().st_size
        if res["size_bytes"] == 0:
            res["error"] = "File is empty (0 bytes)"
            return res

        info = sf.info(str(path))
        res["valid_wav"] = True
        res["samplerate"] = int(info.samplerate)
        res["channels"] = int(info.channels)
        res["duration_s"] = round(float(info.duration), 3)
    except Exception as exc:
        res["valid_wav"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"

    return res


def match_audio_records(
    df: pd.DataFrame,
    audio_dir: Path | str | None,
    split: str,
    id_col: str = "filename",
    inspect_audio: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match CSV rows to audio files and build unique sample identifiers.

    To resolve the train/test filename overlap (e.g. 'audio_0.wav' existing in both train/ and test/),
    this function assigns a unique `sample_id` formatted as `{split}/{filename}`.

    Args:
        df: Input DataFrame (train or test).
        audio_dir: Path to directory containing audio recordings.
        split: 'train' or 'test'.
        id_col: Column containing audio file names.
        inspect_audio: If True, reads WAV headers for sample rate, duration, and integrity.

    Returns:
        Tuple of (augmented_dataframe, validation_metrics).
    """
    df = df.copy()
    audio_path_obj = Path(audio_dir) if audio_dir else None

    # Attach split and collision-free unique sample_id
    df["split"] = split
    df["sample_id"] = [f"{split}/{fname}" for fname in df[id_col]]

    resolved_paths: list[Path | None] = []
    exists_list: list[bool] = []
    durations: list[float | None] = []
    sample_rates: list[int | None] = []
    corrupt_files: list[str] = []

    for fname in df[id_col]:
        p = (audio_path_obj / fname) if audio_path_obj else None
        if p and p.exists():
            resolved_paths.append(p)
            exists_list.append(True)
            if inspect_audio:
                info = inspect_audio_file(p)
                if info["valid_wav"]:
                    durations.append(info["duration_s"])
                    sample_rates.append(info["samplerate"])
                else:
                    durations.append(None)
                    sample_rates.append(None)
                    corrupt_files.append(fname)
            else:
                durations.append(None)
                sample_rates.append(None)
        else:
            resolved_paths.append(p)
            exists_list.append(False)
            durations.append(None)
            sample_rates.append(None)

    df["audio_path"] = resolved_paths
    df["audio_exists"] = exists_list
    df["duration_s"] = durations
    df["sample_rate"] = sample_rates

    # Scan directory for unexpected WAV files not in CSV
    unexpected_files: list[str] = []
    total_audio_in_dir = 0
    if audio_path_obj and audio_path_obj.exists():
        dir_wavs = set(p.name for p in audio_path_obj.glob("*.wav"))
        total_audio_in_dir = len(dir_wavs)
        csv_wavs = set(df[id_col].astype(str))
        unexpected_files = sorted(list(dir_wavs - csv_wavs))

    missing_files = df.loc[~df["audio_exists"], id_col].tolist()

    duration_valid = [d for d in durations if d is not None]
    metrics: dict[str, Any] = {
        "split": split,
        "csv_rows": len(df),
        "audio_dir": str(audio_path_obj) if audio_path_obj else None,
        "total_audio_in_dir": total_audio_in_dir,
        "matched_count": int(df["audio_exists"].sum()),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "unexpected_count": len(unexpected_files),
        "unexpected_files": unexpected_files[:10],
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "duration_min_s": round(min(duration_valid), 2) if duration_valid else None,
        "duration_mean_s": round(float(np.mean(duration_valid)), 2) if duration_valid else None,
        "duration_max_s": round(max(duration_valid), 2) if duration_valid else None,
    }

    return df, metrics


def analyze_submission_mismatch(
    test_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    id_col: str = "filename",
) -> dict[str, Any]:
    """Examine differences between test.csv and sample_submission.csv.

    Identifies row count mismatches, ID set differences, and overlaps with train.csv.
    """
    test_ids = set(test_df[id_col].astype(str))
    sub_ids = set(sub_df[id_col].astype(str))
    train_ids = set(train_df[id_col].astype(str)) if train_df is not None else set()

    intersection_test_sub = test_ids & sub_ids
    test_not_in_sub = test_ids - sub_ids
    sub_not_in_test = sub_ids - test_ids
    sub_in_train = sub_ids & train_ids

    is_exact_match = (test_ids == sub_ids) and (len(test_df) == len(sub_df))

    return {
        "test_rows": len(test_df),
        "sample_sub_rows": len(sub_df),
        "row_count_equal": len(test_df) == len(sub_df),
        "id_sets_equal": test_ids == sub_ids,
        "overlap_count": len(intersection_test_sub),
        "test_missing_from_sample_sub_count": len(test_not_in_sub),
        "test_missing_from_sample_sub_examples": sorted(list(test_not_in_sub))[:5],
        "sample_sub_not_in_test_count": len(sub_not_in_test),
        "sample_sub_not_in_test_examples": sorted(list(sub_not_in_test))[:5],
        "sample_sub_found_in_train_count": len(sub_in_train),
        "is_exact_match": is_exact_match,
        "conclusion": (
            "Mismatch detected: sample_submission.csv (204 rows) appears to be from a prior/draft split. "
            "test.csv (216 rows, audio_0.wav to audio_215.wav) represents the true evaluation set. "
            "Safe resolution: Generate predictions for all 216 test.csv rows in ['filename', 'label'] schema."
            if not is_exact_match
            else "sample_submission.csv aligns perfectly with test.csv."
        ),
    }


def validate_full_dataset(
    config: AppConfig | None = None,
    inspect_all_audio: bool = True,
) -> dict[str, Any]:
    """Run full dataset validation protocol and produce a comprehensive report.

    Args:
        config: AppConfig instance. If None, loaded via get_config().
        inspect_all_audio: If True, checks WAV headers and audio durations for all files.

    Returns:
        Structured dictionary containing full validation metrics and status.
    """
    cfg = config or get_config()
    data_files = cfg.get_dataset_files()
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(cfg.data_dir),
        "status": "PASSED",
        "errors": [],
        "warnings": [],
        "paths": {k: str(v) if v else None for k, v in data_files.items()},
    }

    errors = report["errors"]
    warnings = report["warnings"]

    # 1. Load CSVs
    try:
        csvs = load_raw_csvs(cfg.data_dir)
        train_df = csvs["train"]
        test_df = csvs["test"]
        sub_df = csvs["sample_sub"]
    except Exception as exc:
        errors.append(f"Failed to load raw CSVs: {exc}")
        report["status"] = "FAILED"
        return report

    report["schemas"] = {
        "train_columns": train_df.columns.tolist(),
        "test_columns": test_df.columns.tolist(),
        "sample_sub_columns": sub_df.columns.tolist(),
    }

    # 2. Check schemas
    errors.extend(validate_csv_schema(train_df, [cfg.id_column, cfg.target_column], "train.csv"))
    errors.extend(validate_csv_schema(test_df, [cfg.id_column], "test.csv"))
    errors.extend(validate_csv_schema(sub_df, [cfg.id_column, cfg.target_column], "sample_submission.csv"))

    # 3. Duplicate checks
    train_dup = int(train_df[cfg.id_column].duplicated().sum())
    test_dup = int(test_df[cfg.id_column].duplicated().sum())
    sub_dup = int(sub_df[cfg.id_column].duplicated().sum())

    report["duplicates"] = {
        "train_duplicate_filenames": train_dup,
        "test_duplicate_filenames": test_dup,
        "sample_sub_duplicate_filenames": sub_dup,
    }
    if train_dup > 0:
        errors.append(f"train.csv contains {train_dup} duplicate filenames.")
    if test_dup > 0:
        errors.append(f"test.csv contains {test_dup} duplicate filenames.")

    # 4. Label validation
    train_label_res = validate_labels(train_df, cfg.target_column, cfg.min_score, cfg.max_score)
    report["train_labels"] = train_label_res
    errors.extend(train_label_res["errors"])
    warnings.extend(train_label_res["warnings"])

    # Test dummy label verification
    if cfg.target_column in test_df.columns:
        test_labels_unique = test_df[cfg.target_column].unique().tolist()
        report["test_label_check"] = {
            "has_label_column": True,
            "unique_values": test_labels_unique,
            "treated_as_placeholder": True,
        }
        warnings.append(
            f"test.csv contains '{cfg.target_column}' column with values {test_labels_unique}. "
            "These are dummy placeholders and must never be treated as ground truth."
        )
    else:
        report["test_label_check"] = {"has_label_column": False}

    # 5. Audio matching
    train_matched_df, train_audio_metrics = match_audio_records(
        train_df, data_files["train_audio_dir"], "train", cfg.id_column, inspect_all_audio
    )
    test_matched_df, test_audio_metrics = match_audio_records(
        test_df, data_files["test_audio_dir"], "test", cfg.id_column, inspect_all_audio
    )

    report["audio_matching"] = {
        "train": train_audio_metrics,
        "test": test_audio_metrics,
    }

    if train_audio_metrics["missing_count"] > 0:
        errors.append(f"{train_audio_metrics['missing_count']} training audio files missing on disk.")
    if test_audio_metrics["missing_count"] > 0:
        errors.append(f"{test_audio_metrics['missing_count']} test audio files missing on disk.")
    if train_audio_metrics["corrupt_count"] > 0:
        errors.append(f"{train_audio_metrics['corrupt_count']} corrupted training WAV files.")
    if test_audio_metrics["corrupt_count"] > 0:
        errors.append(f"{test_audio_metrics['corrupt_count']} corrupted test WAV files.")

    # 6. Filename overlap check (Train vs Test)
    overlap = set(train_df[cfg.id_column]) & set(test_df[cfg.id_column])
    report["train_test_overlap"] = {
        "overlapping_count": len(overlap),
        "resolution": "Resolved by namespacing audio with unique sample_id ('train/...' vs 'test/...').",
        "sample_overlapping_names": sorted(list(overlap))[:5],
    }
    if overlap:
        warnings.append(
            f"{len(overlap)} audio filenames appear in BOTH train.csv and test.csv (e.g. 'audio_0.wav'). "
            "Handled safely: features and paths are indexed by unique sample_id ('train/...' vs 'test/...')."
        )

    # 7. Submission mismatch investigation
    sub_mismatch = analyze_submission_mismatch(test_df, sub_df, train_df, cfg.id_column)
    report["submission_mismatch"] = sub_mismatch
    if not sub_mismatch["is_exact_match"]:
        warnings.append(
            f"Submission mismatch: test.csv has {len(test_df)} rows while sample_submission.csv has "
            f"{len(sub_df)} rows. Prediction pipeline will output all {len(test_df)} test samples."
        )

    # 8. Set final status
    if errors:
        report["status"] = "FAILED"
    elif warnings:
        report["status"] = "PASSED_WITH_WARNINGS"
    else:
        report["status"] = "PASSED"

    return report


def load_and_prepare_dataset(
    config: AppConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean loading function for downstream training and feature extraction.

    Returns:
        (train_df, test_df, sample_sub_df, validation_report)

    Key guarantees:
    - Unique `sample_id` column present in both train_df and test_df.
    - Verified `audio_path` column pointing to real WAV files on disk.
    - Test set dummy labels are dropped so they can never contaminate evaluation.
    - Training rows without valid audio or numeric label are filtered out.
    """
    cfg = config or get_config()
    report = validate_full_dataset(cfg, inspect_all_audio=False)

    if report["status"] == "FAILED":
        raise RuntimeError(f"Dataset validation failed: {report['errors']}")

    csvs = load_raw_csvs(cfg.data_dir)
    data_files = cfg.get_dataset_files()

    train_df, _ = match_audio_records(
        csvs["train"], data_files["train_audio_dir"], "train", cfg.id_column, inspect_audio=False
    )
    test_df, _ = match_audio_records(
        csvs["test"], data_files["test_audio_dir"], "test", cfg.id_column, inspect_audio=False
    )

    # Ensure label is numeric in train
    train_df["label"] = pd.to_numeric(train_df[cfg.target_column], errors="coerce")
    train_df = train_df.dropna(subset=["label", "audio_path"]).reset_index(drop=True)

    # Strip dummy label column from test to prevent leakage
    if cfg.target_column in test_df.columns:
        test_df = test_df.drop(columns=[cfg.target_column])

    return train_df, test_df, csvs["sample_sub"], report


def generate_validation_summary_markdown(report: dict[str, Any]) -> str:
    """Format the validation report as clean, readable Markdown."""
    train_stats = report.get("train_labels", {}).get("stats", {})
    train_audio = report.get("audio_matching", {}).get("train", {})
    test_audio = report.get("audio_matching", {}).get("test", {})
    sub_info = report.get("submission_mismatch", {})

    md = f"""# Dataset Ingestion & Validation Summary

**Execution Time**: `{report.get('timestamp')}`  
**Data Directory**: `{report.get('data_dir')}`  
**Validation Status**: **{report.get('status')}**

---

## 1. Dataset Counts & Matching

| Split | CSV Rows | Audio Dir WAVs | Matched Audio | Missing Audio | Corrupt WAVs | Duration (min/mean/max) |
|---|---|---|---|---|---|---|
| **Train** | {train_audio.get('csv_rows')} | {train_audio.get('total_audio_in_dir')} | {train_audio.get('matched_count')} | {train_audio.get('missing_count')} | {train_audio.get('corrupt_count')} | {train_audio.get('duration_min_s')}s / {train_audio.get('duration_mean_s')}s / {train_audio.get('duration_max_s')}s |
| **Test** | {test_audio.get('csv_rows')} | {test_audio.get('total_audio_in_dir')} | {test_audio.get('matched_count')} | {test_audio.get('missing_count')} | {test_audio.get('corrupt_count')} | {test_audio.get('duration_min_s')}s / {test_audio.get('duration_mean_s')}s / {test_audio.get('duration_max_s')}s |
| **Sample Sub** | {sub_info.get('sample_sub_rows')} | N/A | N/A | N/A | N/A | N/A |

---

## 2. Training Target Label Statistics

- **Total Labels**: {train_stats.get('count')}
- **Mean Score**: {train_stats.get('mean')}
- **Standard Deviation**: {train_stats.get('std')}
- **Score Range**: [{train_stats.get('min')}, {train_stats.get('max')}]
- **Quartiles (25% / 50% / 75%)**: {train_stats.get('25%')} / {train_stats.get('50%')} / {train_stats.get('75%')}
- **Missing / Invalid Labels**: {report.get('train_labels', {}).get('missing_count', 0)} missing, {report.get('train_labels', {}).get('out_of_bounds_count', 0)} out of bounds

---

## 3. Filename Overlap Analysis

- **Train/Test Shared Filenames**: {report.get('train_test_overlap', {}).get('overlapping_count')} files share identical names (e.g. `audio_0.wav` ... `audio_215.wav`).
- **Collision Safeguard**: Every sample is assigned a globally unique `sample_id` (`train/<filename>` and `test/<filename>`). Audio paths point to separate directories.

---

## 4. Submission Mismatch Investigation

- `test.csv` rows: **{sub_info.get('test_rows')}** (`audio_0.wav` to `audio_215.wav`)
- `sample_submission.csv` rows: **{sub_info.get('sample_sub_rows')}**
- **Overlapping IDs between Test and Sample Submission**: {sub_info.get('overlap_count')}
- **IDs in Sample Submission that belong to Train**: {sub_info.get('sample_sub_found_in_train_count')}
- **Diagnosis**: {sub_info.get('conclusion')}

---

## 5. Audit Warnings & Errors

### Warnings ({len(report.get('warnings', []))})
"""
    for w in report.get("warnings", []):
        md += f"- [WARNING] {w}\n"

    if not report.get("warnings"):
        md += "- None\n"

    md += f"\n### Errors ({len(report.get('errors', []))})\n"
    for e in report.get("errors", []):
        md += f"- [ERROR] {e}\n"

    if not report.get("errors"):
        md += "- None\n"

    return md


def main() -> int:
    """CLI execution for dataset validation."""
    cfg = get_config()
    logger.info("Executing comprehensive dataset validation on %s...", cfg.data_dir)

    report = validate_full_dataset(cfg, inspect_all_audio=True)

    json_path = cfg.metrics_dir / "dataset_validation_report.json"
    md_path = cfg.metrics_dir / "dataset_validation_summary.md"

    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_summary = generate_validation_summary_markdown(report)
    md_path.write_text(md_summary, encoding="utf-8")

    logger.info("Validation Status: %s", report["status"])
    logger.info("Saved JSON report: %s", json_path)
    logger.info("Saved Markdown summary: %s", md_path)

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        print("\n" + md_summary)
    except Exception:
        safe_summary = md_summary.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_summary)

    return 0 if report["status"] != "FAILED" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
