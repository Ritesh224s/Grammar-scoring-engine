"""Automated unit tests for machine learning models and validation pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.models import (
    compute_regression_metrics,
    create_train_val_split,
    evaluate_models_holdout,
    get_candidate_models,
    run_kfold_cross_validation,
)


def test_compute_regression_metrics_exact():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["pearson_r"] == 1.0


def test_compute_regression_metrics_clipping():
    # Out of bounds predictions should be clipped to [0, 5]
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-2.0, 10.0])
    metrics = compute_regression_metrics(y_true, y_pred, clip_bounds=(0.0, 5.0))

    # After clipping, y_pred is [0.0, 5.0], so error is 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_compute_regression_metrics_constant_prediction():
    # When predictions are constant, Pearson r is mathematically undefined
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.5, 2.5, 2.5, 2.5])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert not np.isnan(metrics["rmse"])
    assert np.isnan(metrics["pearson_r"])


def test_create_train_val_split():
    # Create synthetic DataFrame
    rows = []
    for i in range(100):
        r = {
            "sample_id": f"train/audio_{i}.wav",
            "split": "train",
            "filename": f"audio_{i}.wav",
            "extraction_status": "ok",
            "extraction_error": "",
            "label": float(i % 5) + 1.0,
        }
        for f in FEATURE_COLUMNS:
            r[f] = float(i) * 0.1
        rows.append(r)

    df = pd.DataFrame(rows)
    X_train, y_train, meta_train, X_val, y_val, meta_val = create_train_val_split(
        df, test_size=0.2, seed=42
    )

    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert "label" not in X_train.columns
    assert "sample_id" not in X_train.columns

    # Verify no overlap in sample_ids
    train_ids = set(meta_train["sample_id"])
    val_ids = set(meta_val["sample_id"])
    assert len(train_ids.intersection(val_ids)) == 0


def test_evaluate_candidate_models_synthetic():
    # Synthetic dataset with 50 samples
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    # Target linearly correlated with first feature
    y = pd.Series(np.clip(2.5 + 0.8 * X[FEATURE_COLUMNS[0]] + np.random.randn(50) * 0.2, 0.0, 5.0))

    X_train, X_val = X.iloc[:40], X.iloc[40:]
    y_train, y_val = y.iloc[:40], y.iloc[40:]

    comparison_df, val_preds, fitted = evaluate_models_holdout(
        X_train, y_train, X_val, y_val, seed=42
    )

    assert len(comparison_df) == 5  # 5 models
    assert "Dummy (Mean Baseline)" in comparison_df["Model"].values
    assert "Ridge Regression" in comparison_df["Model"].values
    assert "Random Forest" in comparison_df["Model"].values
    assert "Extra Trees" in comparison_df["Model"].values
    assert "HistGradientBoosting" in comparison_df["Model"].values

    # Every model produced predictions within [0, 5]
    for name, preds in val_preds.items():
        assert len(preds) == 10
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 5.0)


def test_kfold_cross_validation_synthetic():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(30, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y = pd.Series(np.random.uniform(1.0, 4.0, size=30))

    cv_results = run_kfold_cross_validation(X, y, n_splits=3, seed=42)

    assert "Ridge Regression" in cv_results
    assert "cv_val_rmse_mean" in cv_results["Ridge Regression"]
    assert cv_results["Ridge Regression"]["cv_val_rmse_mean"] > 0.0


# -----------------------------------------------------------------------------
# Phase 7 Tests: Final Model, Predictions, and Submission Handling
# -----------------------------------------------------------------------------
def test_fit_final_model_in_sample_metrics():
    from src.models import fit_final_model

    rows = []
    for i in range(50):
        r = {
            "sample_id": f"train/audio_{i}.wav",
            "split": "train",
            "filename": f"audio_{i}.wav",
            "extraction_status": "ok",
            "extraction_error": "",
            "label": 2.0 + (i % 4) * 0.5,
        }
        for f in FEATURE_COLUMNS:
            r[f] = float(i) * 0.05
        rows.append(r)

    df = pd.DataFrame(rows)
    pipeline, meta = fit_final_model(df, model_name="Extra Trees", save_artifact=False)

    assert meta["model_name"] == "Extra Trees"
    assert meta["training_samples_count"] == 50
    assert "in_sample_train_rmse" in meta["in_sample_training_metrics"]
    assert meta["in_sample_training_metrics"]["in_sample_train_rmse"] >= 0.0
    assert "not generalization" in meta["in_sample_training_metrics"]["metric_note"].lower()


def test_generate_provisional_test_predictions_synthetic(tmp_path):
    from src.config import get_config
    from src.models import generate_provisional_test_predictions, get_candidate_models

    pipeline = get_candidate_models(seed=42)["Extra Trees"]
    # Fit on synthetic data first
    X_synth = pd.DataFrame(np.random.randn(20, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y_synth = pd.Series(np.random.uniform(1.0, 4.0, size=20))
    pipeline.fit(X_synth, y_synth)

    # Build synthetic test DataFrame with exactly 216 rows
    test_rows = []
    for i in range(216):
        r = {
            "sample_id": f"test/audio_{i}.wav",
            "split": "test",
            "filename": f"audio_{i}.wav",
            "extraction_status": "ok",
            "extraction_error": "",
        }
        for f in FEATURE_COLUMNS:
            r[f] = float(i) * 0.02
        test_rows.append(r)
    test_df = pd.DataFrame(test_rows)

    cfg = get_config()
    provisional_df, summary = generate_provisional_test_predictions(
        pipeline, test_df, config=cfg, save_files=False
    )

    # Exactly 216 predictions
    assert len(provisional_df) == 216
    assert summary["total_test_predictions"] == 216

    # Ordering & filenames preserved
    assert list(provisional_df["filename"]) == [f"audio_{i}.wav" for i in range(216)]

    # Scores within [0, 5] and finite
    assert provisional_df["predicted_score"].isna().sum() == 0
    assert np.all(provisional_df["predicted_score"] >= 0.0)
    assert np.all(provisional_df["predicted_score"] <= 5.0)

    # No label column
    assert "label" not in provisional_df.columns


def test_final_saved_model_artifact_loading():
    import joblib
    from src.config import get_config

    cfg = get_config()
    final_model_path = cfg.final_model_path
    assert final_model_path.exists(), f"Final model not found at {final_model_path}"

    artifact = joblib.load(final_model_path)
    assert "model_name" in artifact
    assert "pipeline" in artifact
    assert "metadata" in artifact
    assert "feature_columns" in artifact

    # Test single-sample prediction
    pipeline = artifact["pipeline"]
    sample_X = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS)
    pred = pipeline.predict(sample_X)
    assert len(pred) == 1
    assert np.isfinite(pred[0])


def test_provisional_predictions_file_integrity():
    from src.config import get_config

    cfg = get_config()
    pred_path = cfg.provisional_predictions_path
    sub_216_path = cfg.predictions_dir / "submission_test_216.csv"

    assert pred_path.exists(), f"Provisional predictions file not found at {pred_path}"
    assert sub_216_path.exists(), f"Aligned submission file not found at {sub_216_path}"

    pred_df = pd.read_csv(pred_path)
    sub_216_df = pd.read_csv(sub_216_path)

    # Exactly 216 rows
    assert len(pred_df) == 216
    assert len(sub_216_df) == 216

    # Check columns
    assert "filename" in pred_df.columns
    assert "predicted_score" in pred_df.columns
    assert list(sub_216_df.columns) == ["filename", "label"]

    # Bounds [0, 5]
    assert (pred_df["predicted_score"] >= 0.0).all()
    assert (pred_df["predicted_score"] <= 5.0).all()
    assert (sub_216_df["label"] >= 0.0).all()
    assert (sub_216_df["label"] <= 5.0).all()

    # Zero NaNs
    assert pred_df["predicted_score"].isna().sum() == 0
    assert sub_216_df["label"].isna().sum() == 0

    # Test order alignment with raw test.csv
    ds_files = cfg.get_dataset_files()
    if ds_files["test_csv"] and ds_files["test_csv"].exists():
        raw_test = pd.read_csv(ds_files["test_csv"])
        assert list(pred_df["filename"]) == list(raw_test["filename"])
        assert list(sub_216_df["filename"]) == list(raw_test["filename"])


def test_submission_template_mismatch_handling():
    from src.config import get_config

    cfg = get_config()
    diag_path = cfg.predictions_dir / "submission_format_diagnosis.md"
    summary_path = cfg.predictions_dir / "predictions_summary.json"

    assert diag_path.exists(), f"Diagnosis markdown not found at {diag_path}"
    assert summary_path.exists(), f"Summary JSON not found at {summary_path}"

    diag_text = diag_path.read_text(encoding="utf-8")
    assert "216" in diag_text
    assert "204" in diag_text
    assert "Mismatch" in diag_text

