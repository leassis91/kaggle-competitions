from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline
import train_iteration_19_pooled_selection_sunday_team_quality as team_quality
import train_iteration_20_pooled_selection_sunday_smooth as smooth


ARTIFACT_DIR = Path("artifacts/iteration_23_pooled_selection_sunday_smooth_blend")
ENSEMBLE_WEIGHTS = {
    "baseline_smooth": 0.75,
    "team_quality_smooth": 0.25,
}


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


def run_validation() -> dict[str, object]:
    validation_seasons = baseline.latest_completed_validation_seasons()
    baseline_matchups, baseline_feature_columns = baseline.build_pooled_matchups()
    team_quality_matchups, team_quality_feature_columns = team_quality.build_pooled_matchups()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        baseline_probabilities, targets = raw_probabilities(
            matchups=baseline_matchups,
            feature_columns=baseline_feature_columns,
            validation_season=validation_season,
        )
        team_quality_probabilities, _ = raw_probabilities(
            matchups=team_quality_matchups,
            feature_columns=team_quality_feature_columns,
            validation_season=validation_season,
        )
        probabilities = (
            ENSEMBLE_WEIGHTS["baseline_smooth"] * baseline_probabilities
            + ENSEMBLE_WEIGHTS["team_quality_smooth"] * team_quality_probabilities
        )
        validation_rows = baseline_matchups.filter(pl.col("Season") == validation_season)
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


def write_validation_summary(validation_summary: dict[str, object]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": smooth.LIGHTGBM_PARAMS,
        "minimum_pooled_season": baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "ensemble_components": [
            "pooled_selection_sunday_smooth_baseline_features",
            "pooled_selection_sunday_smooth_team_quality_features",
        ],
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    validation_summary = run_validation()
    write_validation_summary(validation_summary)

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
