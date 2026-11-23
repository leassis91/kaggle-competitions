from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import orjson
import polars as pl
from sklearn.linear_model import LogisticRegression

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_21_pooled_selection_sunday_margin_regression")
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 5.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
}


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LIGHTGBM_PARAMS)


def build_margin_targets() -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_csv(config.tournament_results_path)
            .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
            .select(
                pl.col("Season").cast(pl.Int64),
                pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
                pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
                pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
                .then((pl.col("WScore") - pl.col("LScore")).cast(pl.Float64))
                .otherwise((pl.col("LScore") - pl.col("WScore")).cast(pl.Float64))
                .alias("TargetMargin"),
            )
            for config in baseline.DIVISION_CONFIGS
        ],
        how="vertical_relaxed",
    )


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = baseline.build_pooled_matchups()
    return matchups.join(
        build_margin_targets(),
        on=["Season", "LowerTeamID", "HigherTeamID"],
        how="inner",
    ), feature_columns


def raw_margin_predictions(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray]:
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)

    model = build_model()
    model.fit(training_rows.select(feature_columns).to_numpy(), training_rows["TargetMargin"].to_numpy())
    predictions = model.predict(validation_rows.select(feature_columns).to_numpy())
    return predictions, validation_rows["Target"].to_numpy()


def fit_margin_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
) -> LogisticRegression | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_predictions: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets = raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
        )
        oof_predictions.append(predictions)
        oof_targets.append(targets)

    if not oof_predictions:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.concatenate(oof_predictions).reshape(-1, 1), np.concatenate(oof_targets))
    return calibrator


def apply_margin_calibration(predictions: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return np.clip(1.0 / (1.0 + np.exp(-predictions / 10.0)), 0.0, 1.0)
    return np.clip(calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1], 0.0, 1.0)


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = baseline.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets = raw_margin_predictions(
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
        "lightgbm_params": LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_margin_matchups()
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
