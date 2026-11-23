from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from sklearn.isotonic import IsotonicRegression

import train_iteration_16_pooled_selection_sunday_baseline as baseline
import train_iteration_20_pooled_selection_sunday_smooth as smooth


ARTIFACT_DIR = Path("artifacts/iteration_22_pooled_selection_sunday_smooth_isotonic")


def raw_probabilities(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray]:
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)

    model = smooth.build_model()
    model.fit(training_rows.select(feature_columns).to_numpy(), training_rows["Target"].to_numpy())
    probabilities = model.predict_proba(validation_rows.select(feature_columns).to_numpy())[:, 1]
    return probabilities, validation_rows["Target"].to_numpy()


def fit_oof_isotonic_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
) -> IsotonicRegression | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_probabilities: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        probabilities, targets = raw_probabilities(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
        )
        oof_probabilities.append(probabilities)
        oof_targets.append(targets)

    if not oof_probabilities:
        return None

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.concatenate(oof_probabilities), np.concatenate(oof_targets))
    return calibrator


def apply_calibration(probabilities: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    if calibrator is None:
        return probabilities
    return np.clip(calibrator.transform(probabilities), 0.0, 1.0)


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = baseline.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_fold_probabilities, targets = raw_probabilities(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=validation_season,
        )
        calibrator = fit_oof_isotonic_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
        )
        probabilities = apply_calibration(raw_fold_probabilities, calibrator)
        validation_rows = matchups.filter(pl.col("Season") == validation_season)
        men_mask = validation_rows["men_women"].to_numpy() == 1
        women_mask = ~men_mask

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": baseline.brier_score(probabilities, targets),
                "men_brier": baseline.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": baseline.brier_score(probabilities[women_mask], targets[women_mask]),
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
        "lightgbm_params": smooth.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_calibration": "nested_oof_isotonic",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = baseline.build_pooled_matchups()
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
