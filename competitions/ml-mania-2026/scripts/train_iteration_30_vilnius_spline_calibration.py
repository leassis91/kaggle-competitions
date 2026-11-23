from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from scipy.interpolate import UnivariateSpline

import train_iteration_26_pooled_selection_sunday_bracket_calibration as calibration_base
import train_iteration_28_mike_filtered_quality as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_30_vilnius_spline_calibration")
SPLINE_CLIP = 25.0


def fit_spline_calibrator(predictions: np.ndarray, targets: np.ndarray) -> UnivariateSpline | None:
    calibration_frame = (
        pl.DataFrame(
            {
                "prediction": np.clip(predictions, -SPLINE_CLIP, SPLINE_CLIP),
                "target": targets.astype(np.float64),
            }
        )
        .group_by("prediction")
        .agg(pl.col("target").mean().alias("win_rate"))
        .sort("prediction")
    )

    if calibration_frame.height < 2:
        return None

    spline_degree = min(5, calibration_frame.height - 1)
    return UnivariateSpline(
        calibration_frame["prediction"].to_numpy(),
        calibration_frame["win_rate"].to_numpy(),
        k=spline_degree,
    )


def fit_margin_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
) -> UnivariateSpline | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_predictions: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets, _validation_rows = calibration_base.raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
        )
        oof_predictions.append(predictions)
        oof_targets.append(targets)

    if not oof_predictions:
        return None

    return fit_spline_calibrator(np.concatenate(oof_predictions), np.concatenate(oof_targets))


def apply_margin_calibration(
    predictions: np.ndarray,
    calibrator: UnivariateSpline | None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(1.0 / (1.0 + np.exp(-predictions / 10.0)), 0.0, 1.0)

    clipped_predictions = np.clip(predictions, -SPLINE_CLIP, SPLINE_CLIP)
    return np.clip(calibrator(clipped_predictions), 0.01, 0.99)


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = calibration_base.baseline.baseline.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = calibration_base.raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=validation_season,
        )
        calibrator = fit_margin_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
        )
        probabilities = apply_margin_calibration(raw_predictions, calibrator)
        men_mask = validation_rows["men_women"].to_numpy() == 1
        women_mask = ~men_mask

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": calibration_base.baseline.baseline.brier_score(probabilities, targets),
                "men_brier": calibration_base.baseline.baseline.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": calibration_base.baseline.baseline.brier_score(probabilities[women_mask], targets[women_mask]),
            }
        )

    return {
        "validation_seasons": validation_seasons,
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": calibration_base.baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": calibration_base.baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_vilnius_spline_from_predicted_margin",
        "feature_family": "iteration_28_with_vilnius_spline_calibration",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = benchmark.build_pooled_margin_matchups()
    validation_summary = run_validation(matchups, feature_columns)
    write_validation_summary(validation_summary, feature_columns)

    print(f"Mean validation Brier: {validation_summary['mean_brier']:.6f}")
    fold_summaries = cast(list[dict[str, float | int]], validation_summary["folds"])
    for fold_summary in fold_summaries:
        print(
            f"Season {fold_summary['season']}: "
            f"combined={fold_summary['combined_brier']:.6f}, "
            f"men={fold_summary['men_brier']:.6f}, "
            f"women={fold_summary['women_brier']:.6f}"
        )


if __name__ == "__main__":
    main()
