from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Callable, cast

import orjson
import polars as pl

import official_feature_family_current_season as current_season
import official_feature_family_driver as family_driver


ARTIFACT_DIR = Path("artifacts/iteration_118_current_season_split_context_family_batch")

add_team_metric_features = cast(
    Callable[[pl.DataFrame, list[str], dict[str, float]], tuple[pl.DataFrame, list[str]]],
    getattr(current_season, "add_team_metric_features"),
)
add_pair_metric_features = cast(
    Callable[[pl.DataFrame, pl.DataFrame, list[str], dict[str, float]], tuple[pl.DataFrame, list[str]]],
    getattr(current_season, "add_pair_metric_features"),
)
build_pooled_common_opponent_features = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_common_opponent_features"),
)
build_pooled_regular_season_pair_features = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_regular_season_pair_features"),
)
build_pooled_team_city_features = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_team_city_features"),
)
build_pooled_team_feature_table = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_team_feature_table"),
)
team_feature_output_columns = cast(
    Callable[[list[str]], list[str]],
    getattr(current_season, "team_feature_output_columns"),
)


def metric_fill_defaults(metric_names: list[str]) -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for metric_name in metric_names:
        default_value = 0.5 if "win_pct" in metric_name else 0.0
        fill_defaults[f"low_{metric_name}"] = default_value
        fill_defaults[f"high_{metric_name}"] = default_value
        fill_defaults[f"delta_{metric_name}"] = 0.0
    return fill_defaults


def pair_fill_defaults(metric_names: list[str]) -> dict[str, float]:
    return {metric_name: 0.5 if "win_pct" in metric_name else 0.0 for metric_name in metric_names}


def direct_team_family(
    name: str,
    artifact_slug: str,
    hypothesis: str,
    metric_names: list[str],
) -> family_driver.FamilySpec:
    def builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns
        return add_team_metric_features(matchups, metric_names, metric_fill_defaults(metric_names))

    return family_driver.FamilySpec(
        name=name,
        artifact_slug=artifact_slug,
        hypothesis=hypothesis,
        extend_iteration_38=builder,
    )


def pair_family(
    name: str,
    artifact_slug: str,
    hypothesis: str,
    pair_features_builder: Callable[[], pl.DataFrame],
    metric_names: list[str],
) -> family_driver.FamilySpec:
    def builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns
        return add_pair_metric_features(matchups, pair_features_builder(), metric_names, pair_fill_defaults(metric_names))

    return family_driver.FamilySpec(
        name=name,
        artifact_slug=artifact_slug,
        hypothesis=hypothesis,
        extend_iteration_38=builder,
    )


def conference_relative_family() -> family_driver.FamilySpec:
    metric_names = [
        "conference_margin_percentile",
        "conference_margin_zscore",
        "conference_offense_zscore",
        "conference_strength_composite",
        "conference_margin_percentile_centered",
    ]

    def builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns
        team_features = build_pooled_team_feature_table().with_columns(
            (pl.col("conference_margin_zscore") + pl.col("conference_offense_zscore")).alias(
                "conference_strength_composite"
            ),
            (pl.col("conference_margin_percentile") - 0.5).alias("conference_margin_percentile_centered"),
        )
        low_features = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("LowerTeamID"),
            pl.col("men_women"),
            *[pl.col(metric_name).alias(f"low_{metric_name}") for metric_name in metric_names],
        )
        high_features = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("HigherTeamID"),
            pl.col("men_women"),
            *[pl.col(metric_name).alias(f"high_{metric_name}") for metric_name in metric_names],
        )
        fill_defaults = metric_fill_defaults(metric_names)
        matchup_frame = (
            matchups.join(low_features, on=["Season", "LowerTeamID", "men_women"], how="left")
            .join(high_features, on=["Season", "HigherTeamID", "men_women"], how="left")
            .with_columns(
                *[
                    pl.col(column_name).fill_null(default_value)
                    for column_name, default_value in fill_defaults.items()
                    if column_name.startswith("low_") or column_name.startswith("high_")
                ]
            )
            .with_columns(
                *[
                    (pl.col(f"low_{metric_name}") - pl.col(f"high_{metric_name}")).alias(f"delta_{metric_name}")
                    for metric_name in metric_names
                ]
            )
        )
        return matchup_frame, team_feature_output_columns(metric_names)

    return family_driver.FamilySpec(
        name="current_season_conference_relative_percentile_and_zscore_features",
        artifact_slug="current_season_conference_relative_percentile_and_zscore_features",
        hypothesis="Conference-relative percentile and z-score views may separate strong teams from weak conference inflation.",
        extend_iteration_38=builder,
    )


FAMILY_SPECS = [
    direct_team_family(
        name="current_season_home_road_split_gap_features",
        artifact_slug="current_season_home_road_split_gap_features",
        hypothesis="Home-road split gaps may expose teams whose true neutral profile differs from raw season strength.",
        metric_names=[
            "home_road_win_pct_gap",
            "home_road_margin_gap",
            "home_road_off_rating_gap",
            "home_road_defense_gap",
        ],
    ),
    direct_team_family(
        name="current_season_conference_vs_nonconference_split_features",
        artifact_slug="current_season_conference_vs_nonconference_split_features",
        hypothesis="Conference versus nonconference split gaps may reveal schedule-context inflation that the official benchmark misses.",
        metric_names=[
            "conference_nonconference_win_pct_gap",
            "conference_nonconference_margin_gap",
            "conference_nonconference_off_rating_gap",
            "conference_nonconference_defense_gap",
        ],
    ),
    direct_team_family(
        name="current_season_preconference_vs_conference_play_trajectory_features",
        artifact_slug="current_season_preconference_vs_conference_play_trajectory_features",
        hypothesis="Preconference to conference-phase deltas may capture adaptation and late-season drift better than plain season averages.",
        metric_names=[
            "conference_phase_win_pct_delta",
            "conference_phase_margin_delta",
            "conference_phase_off_rating_delta",
            "conference_phase_defense_delta",
        ],
    ),
    direct_team_family(
        name="current_season_rest_day_gap_features",
        artifact_slug="current_season_rest_day_gap_features",
        hypothesis="Rest-pattern gaps may expose teams whose performance is unusually sensitive to cadence changes.",
        metric_names=[
            "avg_rest_gap",
            "max_day_gap",
            "short_rest_win_pct",
            "short_rest_avg_margin",
            "long_rest_win_pct",
            "long_rest_avg_margin",
        ],
    ),
    direct_team_family(
        name="current_season_schedule_density_and_fatigue_features",
        artifact_slug="current_season_schedule_density_and_fatigue_features",
        hypothesis="Schedule density and fatigue metrics may add context that raw strength misses in compressed tournament settings.",
        metric_names=[
            "games_per_week",
            "avg_day_gap",
            "max_day_gap",
            "back_to_back_share",
            "short_rest_share",
            "long_rest_share",
        ],
    ),
    direct_team_family(
        name="current_season_overtime_propensity_and_performance_features",
        artifact_slug="current_season_overtime_propensity_and_performance_features",
        hypothesis="Overtime frequency and overtime outcomes may capture clutch depth and fatigue exposure.",
        metric_names=[
            "overtime_game_share",
            "overtime_win_pct",
            "overtime_avg_margin",
            "avg_overtime_periods",
        ],
    ),
    direct_team_family(
        name="current_season_close_game_execution_features",
        artifact_slug="current_season_close_game_execution_features",
        hypothesis="Close-game execution may reveal win-probability conversion skill not reflected in average margin alone.",
        metric_names=[
            "close_game_share",
            "close_game_win_pct",
            "close_game_avg_margin",
            "close_game_net_rating",
        ],
    ),
    direct_team_family(
        name="current_season_blowout_tendency_features",
        artifact_slug="current_season_blowout_tendency_features",
        hypothesis="Blowout win and loss tendencies may help separate ceiling teams from fragile teams at similar average strength.",
        metric_names=[
            "blowout_win_share",
            "blowout_loss_share",
            "blowout_game_share",
            "blowout_avg_margin",
        ],
    ),
    direct_team_family(
        name="current_season_margin_volatility_and_consistency_features",
        artifact_slug="current_season_margin_volatility_and_consistency_features",
        hypothesis="Margin volatility may improve tournament predictions by distinguishing stable teams from noisy profiles.",
        metric_names=[
            "avg_margin",
            "margin_std",
            "avg_abs_margin",
            "abs_margin_std",
        ],
    ),
    pair_family(
        name="current_season_common_opponent_comparison_features",
        artifact_slug="current_season_common_opponent_comparison_features",
        hypothesis="Common-opponent comparisons may add matchup-specific signal beyond independent team ratings.",
        pair_features_builder=build_pooled_common_opponent_features,
        metric_names=[
            "common_opponent_count",
            "lower_common_win_pct",
            "higher_common_win_pct",
            "lower_common_avg_margin",
            "higher_common_avg_margin",
            "lower_common_off_rating",
            "higher_common_off_rating",
            "lower_common_def_rating",
            "higher_common_def_rating",
            "delta_common_win_pct",
            "delta_common_avg_margin",
            "delta_common_off_rating",
            "delta_common_defense_edge",
        ],
    ),
    direct_team_family(
        name="current_season_opponent_tier_record_features",
        artifact_slug="current_season_opponent_tier_record_features",
        hypothesis="Top-tier and bottom-tier opponent records may sharpen schedule-strength and consistency context.",
        metric_names=[
            "top_tier_schedule_share",
            "top_tier_win_pct",
            "top_tier_avg_margin",
            "bottom_tier_schedule_share",
            "bottom_tier_win_pct",
            "bottom_tier_avg_margin",
            "high_quality_opponent_share",
        ],
    ),
    pair_family(
        name="current_season_rematch_familiarity_features",
        artifact_slug="current_season_rematch_familiarity_features",
        hypothesis="Regular-season rematch context may help when teams have already adjusted to each other.",
        pair_features_builder=build_pooled_regular_season_pair_features,
        metric_names=[
            "regular_season_meetings",
            "lower_regular_season_win_pct",
            "lower_regular_season_avg_margin",
            "latest_lower_regular_season_margin",
            "last_regular_season_meeting_daynum",
            "days_since_last_meeting",
            "regular_season_neutral_meeting_share",
            "regular_season_overtime_meeting_share",
            "regular_season_close_meeting_share",
            "split_regular_season_series_flag",
        ],
    ),
    direct_team_family(
        name="venue_familiarity_from_repeated_game_city_exposure",
        artifact_slug="venue_familiarity_from_repeated_game_city_exposure",
        hypothesis="Repeated city exposure may proxy tournament travel comfort and neutral-site familiarity.",
        metric_names=[
            "unique_city_count",
            "max_city_game_count",
            "avg_city_game_count",
            "avg_city_win_pct",
            "avg_city_neutral_share",
            "repeat_city_game_count",
            "repeat_city_exposure_score",
        ],
    ),
    conference_relative_family(),
]


def write_batch_summary(
    evaluations: list[family_driver.FamilyEvaluation],
    initial_benchmark_state: family_driver.BenchmarkState,
    final_benchmark_state: family_driver.BenchmarkState,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family_batch": "current_season_split_and_context_families",
        "initial_benchmark_name": initial_benchmark_state.name,
        "initial_benchmark_mean_brier": initial_benchmark_state.mean_brier,
        "final_benchmark_name": final_benchmark_state.name,
        "final_benchmark_mean_brier": final_benchmark_state.mean_brier,
        "accepted_families": [
            evaluation.spec.name
            for evaluation in evaluations
            if evaluation.accepted
        ],
        "results": [
            {
                "name": evaluation.spec.name,
                "accepted": evaluation.accepted,
                "artifact_dir": str(evaluation.artifact_dir),
                "mean_brier": cast(float, evaluation.validation_summary["mean_brier"]),
                "improved_seasons": list(evaluation.improved_seasons),
            }
            for evaluation in evaluations
        ],
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    initial_benchmark_state = family_driver.build_base_benchmark_state()
    evaluations, final_benchmark_state = family_driver.run_family_sequence(
        FAMILY_SPECS,
        initial_benchmark_state=initial_benchmark_state,
    )
    write_batch_summary(
        evaluations=evaluations,
        initial_benchmark_state=initial_benchmark_state,
        final_benchmark_state=final_benchmark_state,
    )

    print(f"Initial benchmark: {initial_benchmark_state.name} ({initial_benchmark_state.mean_brier:.6f})")
    print(f"Final benchmark: {final_benchmark_state.name} ({final_benchmark_state.mean_brier:.6f})")
    print(
        "Accepted families:",
        [evaluation.spec.name for evaluation in evaluations if evaluation.accepted],
    )


if __name__ == "__main__":
    main()
