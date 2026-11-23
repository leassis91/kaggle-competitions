from __future__ import annotations

import warnings
from typing import Callable, cast

import polars as pl

import official_feature_family_current_season as current_season
import official_feature_family_driver as family_driver


add_team_metric_features = cast(
    Callable[[pl.DataFrame, list[str], dict[str, float]], tuple[pl.DataFrame, list[str]]],
    getattr(current_season, "add_team_metric_features"),
)


NEUTRAL_METRICS = [
    "neutral_game_share",
    "neutral_win_pct",
    "neutral_avg_margin",
    "neutral_off_rating",
    "neutral_def_rating",
]


def neutral_fill_defaults() -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for metric_name in NEUTRAL_METRICS:
        default_value = 0.5 if metric_name.endswith("win_pct") else 0.0
        fill_defaults[f"low_{metric_name}"] = default_value
        fill_defaults[f"high_{metric_name}"] = default_value
        fill_defaults[f"delta_{metric_name}"] = 0.0
    return fill_defaults


def neutral_site_strength_builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
    _ = feature_columns
    return add_team_metric_features(matchups, NEUTRAL_METRICS, neutral_fill_defaults())


FAMILY_SPEC = family_driver.FamilySpec(
    name="current_season_neutral_site_only_strength_features",
    artifact_slug="current_season_neutral_site_only_strength_features",
    hypothesis="Neutral-only current-season strength views may add portable signal beyond the fixed official benchmark.",
    extend_iteration_38=neutral_site_strength_builder,
)


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    evaluations, final_benchmark_state = family_driver.run_family_sequence([FAMILY_SPEC])
    evaluation = evaluations[0]

    print(f"Accepted: {evaluation.accepted}")
    print(f"Mean official-score validation Brier: {cast(float, evaluation.validation_summary['mean_brier']):.6f}")
    print(f"Improved seasons: {list(evaluation.improved_seasons)}")
    print(f"Artifact dir: {evaluation.artifact_dir}")
    print(f"Active benchmark after run: {final_benchmark_state.name}")


if __name__ == "__main__":
    main()
