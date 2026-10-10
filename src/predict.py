"""Prediction and submission generation CLI for the Grammar Scoring Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from .config import get_config
from .models import fit_final_model, generate_provisional_test_predictions
from .utils import get_logger

logger = get_logger("predict")


def main() -> int:
    """CLI entrypoint for generating test set predictions."""
    parser = argparse.ArgumentParser(description="Generate predictions for test audio recordings.")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Re-fit the final model on all 769 training samples before predicting.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Extra Trees",
        help="Candidate model architecture to use (default: 'Extra Trees').",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg.ensure_directories()

    try:
        final_model_path = cfg.final_model_path

        if args.refit or not final_model_path.exists():
            logger.info("Fitting final model (%s) on all 769 training records...", args.model)
            train_df = pd.read_csv(cfg.train_features_path)
            pipeline, meta = fit_final_model(
                train_df, model_name=args.model, config=cfg, save_artifact=True
            )
            print(f"Final Model Fitted: {args.model}")
            print(f"In-Sample Training RMSE: {meta['in_sample_training_metrics']['in_sample_train_rmse']:.4f}")
            print(f"In-Sample Training Pearson r: {meta['in_sample_training_metrics']['in_sample_train_pearson_r']:.4f}")
        else:
            logger.info("Loading existing final model artifact from %s", final_model_path)
            artifact = joblib.load(final_model_path)
            pipeline = artifact["pipeline"]
            print(f"Loaded Final Model: {artifact.get('model_name', 'Unknown')}")

        logger.info("Generating predictions for 216 test records...")
        test_df = pd.read_csv(cfg.test_features_path)
        provisional_df, summary = generate_provisional_test_predictions(
            model_pipeline=pipeline,
            test_features_df=test_df,
            config=cfg,
            save_files=True,
        )

        print("\n" + "=" * 76)
        print("PREDICTION GENERATION COMPLETE")
        print("=" * 76)
        print(f"Total Test Predictions : {summary['total_test_predictions']} records")
        print(f"Score Range            : [{summary['prediction_range'][0]:.4f}, {summary['prediction_range'][1]:.4f}]")
        print(f"Score Mean +/- Std     : {summary['prediction_mean']:.4f} +/- {summary['prediction_std']:.4f}")
        print(f"Score Median           : {summary['prediction_median']:.4f}")
        print(f"Provisional CSV        : {cfg.provisional_predictions_path.name}")
        print(f"Aligned Submission CSV : submission_test_216.csv")
        print("=" * 76 + "\n")
        return 0

    except Exception as exc:
        logger.error("Prediction generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
