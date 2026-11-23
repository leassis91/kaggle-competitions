from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Callable, cast

import orjson
import polars as pl

import official_feature_family_current_season as current_season
import official_feature_family_driver as family_driver


ARTIFACT_DIR = Path("artifacts/iteration_119_current_season_stat_family_batch")

add_team_metric_features = cast(
    Callable[[pl.DataFrame, list[str], dict[str, float]], tuple[pl.DataFrame, list[str]]],
    getattr(current_season, "add_team_metric_features"),
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


def derived_team_family(
    name: str,
    artifact_slug: str,
    hypothesis: str,
    metric_names: list[str],
    derived_expressions: list[pl.Expr],
) -> family_driver.FamilySpec:
    def builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns
        team_features = build_pooled_team_feature_table().with_columns(*derived_expressions)
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
        name=name,
        artifact_slug=artifact_slug,
        hypothesis=hypothesis,
        extend_iteration_38=builder,
    )


FAMILY_SPECS = [
    direct_team_family(
        name="current_season_pace_estimate_features",
        artifact_slug="current_season_pace_estimate_features",
        hypothesis="Pace level and volatility may help the official benchmark distinguish style-driven upset surfaces.",
        metric_names=["pace", "pace_std", "above_average_pace_flag", "games_per_week"],
    ),
    direct_team_family(
        name="current_season_offensive_rating_features",
        artifact_slug="current_season_offensive_rating_features",
        hypothesis="Offensive rating features may improve matchup strength estimation beyond the blended benchmark.",
        metric_names=["off_rating", "late_off_rating", "above_average_offense_flag", "conference_offense_zscore"],
    ),
    direct_team_family(
        name="current_season_defensive_rating_features",
        artifact_slug="current_season_defensive_rating_features",
        hypothesis="Defensive rating features may add stability against overconfident offensive teams.",
        metric_names=["def_rating", "late_def_rating", "above_average_defense_flag", "conference_margin_zscore"],
    ),
    direct_team_family(
        name="current_season_effective_field_goal_features",
        artifact_slug="current_season_effective_field_goal_features",
        hypothesis="Effective field goal features may capture shot-quality execution better than margin alone.",
        metric_names=["effective_fg_pct", "opp_effective_fg_pct", "late_effective_fg_pct", "effective_fg_trend_slope"],
    ),
    direct_team_family(
        name="current_season_true_shooting_features",
        artifact_slug="current_season_true_shooting_features",
        hypothesis="True shooting features may help when free-throw pressure and shot mix matter in tight tournament games.",
        metric_names=["true_shooting_pct", "opp_true_shooting_pct", "late_true_shooting_pct", "off_rating_trend_slope"],
    ),
    direct_team_family(
        name="current_season_two_point_efficiency_features",
        artifact_slug="current_season_two_point_efficiency_features",
        hypothesis="Two-point efficiency features may separate interior-oriented teams more cleanly.",
        metric_names=["two_point_pct", "opp_two_point_pct", "late_two_point_pct", "off_rating"],
    ),
    direct_team_family(
        name="current_season_three_point_efficiency_features",
        artifact_slug="current_season_three_point_efficiency_features",
        hypothesis="Three-point efficiency features may expose volatility and ceiling effects the benchmark misses.",
        metric_names=["three_point_pct", "opp_three_point_pct", "late_three_point_pct", "three_point_attempt_rate"],
    ),
    direct_team_family(
        name="current_season_free_throw_accuracy_features",
        artifact_slug="current_season_free_throw_accuracy_features",
        hypothesis="Free-throw accuracy may matter disproportionately in end-game tournament scenarios.",
        metric_names=["free_throw_pct", "opp_free_throw_pct", "late_free_throw_pct", "free_throw_rate"],
    ),
    direct_team_family(
        name="current_season_shot_mix_composition_features",
        artifact_slug="current_season_shot_mix_composition_features",
        hypothesis="Shot-mix composition may sharpen style separation beyond raw efficiency rates.",
        metric_names=["three_point_attempt_rate", "two_point_attempt_rate", "free_throw_rate", "pace"],
    ),
    direct_team_family(
        name="current_season_assist_rate_features",
        artifact_slug="current_season_assist_rate_features",
        hypothesis="Assist rate may capture offensive connectivity that the benchmark underweights.",
        metric_names=["assist_rate", "opp_assist_rate", "late_assist_rate", "off_rating"],
    ),
    direct_team_family(
        name="current_season_assist_to_turnover_ratio_features",
        artifact_slug="current_season_assist_to_turnover_ratio_features",
        hypothesis="Assist-to-turnover features may help identify stable guards and decision-making quality.",
        metric_names=["assist_to_turnover_ratio", "late_assist_to_turnover_ratio", "assist_rate", "turnover_rate"],
    ),
    direct_team_family(
        name="current_season_turnover_rate_features",
        artifact_slug="current_season_turnover_rate_features",
        hypothesis="Turnover-rate features may improve upset calibration in turnover-sensitive matchups.",
        metric_names=["turnover_rate", "late_turnover_rate", "turnover_rate_trend_slope", "assist_to_turnover_ratio"],
    ),
    direct_team_family(
        name="current_season_steal_rate_features",
        artifact_slug="current_season_steal_rate_features",
        hypothesis="Steal-rate features may capture disruptive defenses and transition creation.",
        metric_names=["steal_rate", "late_steal_rate", "turnover_rate", "def_rating"],
    ),
    direct_team_family(
        name="current_season_block_rate_features",
        artifact_slug="current_season_block_rate_features",
        hypothesis="Block-rate features may capture interior deterrence that seed and margin features miss.",
        metric_names=["block_rate", "late_block_rate", "def_rating", "opp_two_point_pct"],
    ),
    direct_team_family(
        name="current_season_foul_differential_features",
        artifact_slug="current_season_foul_differential_features",
        hypothesis="Foul-differential features may help separate disciplined teams from fragile defenses.",
        metric_names=["foul_diff_rate", "free_throw_rate", "opp_free_throw_pct", "true_shooting_pct"],
    ),
    direct_team_family(
        name="current_season_offensive_rebound_share_features",
        artifact_slug="current_season_offensive_rebound_share_features",
        hypothesis="Offensive rebound share may improve possession extension context in tournament games.",
        metric_names=["offensive_rebound_share", "late_offensive_rebound_share", "off_rating", "avg_margin"],
    ),
    direct_team_family(
        name="current_season_defensive_rebound_share_features",
        artifact_slug="current_season_defensive_rebound_share_features",
        hypothesis="Defensive rebound share may help close out possessions and stabilize favorites.",
        metric_names=["defensive_rebound_share", "late_defensive_rebound_share", "def_rating", "avg_margin"],
    ),
    direct_team_family(
        name="current_season_detailed_stat_trend_slope_features",
        artifact_slug="current_season_detailed_stat_trend_slope_features",
        hypothesis="Late-season trend slopes may capture directional improvement better than static averages.",
        metric_names=[
            "off_rating_trend_slope",
            "def_rating_trend_slope",
            "effective_fg_trend_slope",
            "turnover_rate_trend_slope",
        ],
    ),
    derived_team_family(
        name="record_versus_seeded_opponent_features",
        artifact_slug="record_versus_seeded_opponent_features",
        hypothesis="Seeded-opponent record features may better distinguish tested teams from inflated resumes.",
        metric_names=[
            "seeded_opponent_game_count",
            "seeded_opponent_schedule_share",
            "seeded_opponent_win_pct",
            "seeded_opponent_avg_margin",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("seeded_opponent_schedule_share")).alias("seeded_opponent_game_count"),
        ],
    ),
    derived_team_family(
        name="margin_versus_seeded_opponent_features",
        artifact_slug="margin_versus_seeded_opponent_features",
        hypothesis="Seeded-opponent margin features may sharpen separation among already-good teams.",
        metric_names=[
            "seeded_opponent_game_count",
            "seeded_opponent_avg_margin",
            "seeded_margin_delta_vs_overall",
            "seeded_opponent_schedule_share",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("seeded_opponent_schedule_share")).alias("seeded_opponent_game_count"),
            (pl.col("seeded_opponent_avg_margin") - pl.col("avg_margin")).alias("seeded_margin_delta_vs_overall"),
        ],
    ),
    derived_team_family(
        name="seeded_opponent_schedule_share_features",
        artifact_slug="seeded_opponent_schedule_share_features",
        hypothesis="Seeded-opponent schedule share may better capture résumé difficulty than generic schedule strength.",
        metric_names=[
            "seeded_opponent_game_count",
            "seeded_opponent_schedule_share",
            "high_quality_opponent_share",
            "games_played",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("seeded_opponent_schedule_share")).alias("seeded_opponent_game_count"),
        ],
    ),
    derived_team_family(
        name="record_versus_top_massey_opponent_features",
        artifact_slug="record_versus_top_massey_opponent_features",
        hypothesis="Top-Massey opponent records may be a cleaner quality-of-competition lens than seeded-opponent filters alone.",
        metric_names=[
            "top_massey_game_count",
            "top_massey_schedule_share",
            "top_massey_win_pct",
            "top_massey_avg_margin",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("top_massey_schedule_share")).alias("top_massey_game_count"),
        ],
    ),
    derived_team_family(
        name="margin_versus_top_massey_opponent_features",
        artifact_slug="margin_versus_top_massey_opponent_features",
        hypothesis="Top-Massey margin features may sharpen the benchmark on elite-opponent context.",
        metric_names=[
            "top_massey_game_count",
            "top_massey_avg_margin",
            "top_massey_margin_delta_vs_overall",
            "top_massey_schedule_share",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("top_massey_schedule_share")).alias("top_massey_game_count"),
            (pl.col("top_massey_avg_margin") - pl.col("avg_margin")).alias("top_massey_margin_delta_vs_overall"),
        ],
    ),
    derived_team_family(
        name="top_massey_opponent_schedule_share_features",
        artifact_slug="top_massey_opponent_schedule_share_features",
        hypothesis="Top-Massey schedule share may add a ranking-based strength-of-schedule lens.",
        metric_names=[
            "top_massey_game_count",
            "top_massey_schedule_share",
            "high_quality_opponent_share",
            "games_played",
        ],
        derived_expressions=[
            (pl.col("games_played") * pl.col("top_massey_schedule_share")).alias("top_massey_game_count"),
        ],
    ),
    direct_team_family(
        name="quality_win_count_and_bad_loss_count_features",
        artifact_slug="quality_win_count_and_bad_loss_count_features",
        hypothesis="Quality-win and bad-loss counts may expose résumé asymmetry that average margin smooths away.",
        metric_names=["quality_win_count", "bad_loss_count", "quality_balance", "high_quality_opponent_share"],
    ),
    direct_team_family(
        name="short_rest_and_compressed_turnaround_features",
        artifact_slug="short_rest_and_compressed_turnaround_features",
        hypothesis="Short-rest and compressed-turnaround features may matter when tournament cadence compresses recovery.",
        metric_names=[
            "short_rest_game_count",
            "short_rest_share",
            "short_rest_win_pct",
            "short_rest_avg_margin",
            "back_to_back_share",
        ],
    ),
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
        "feature_family_batch": "current_season_stat_and_quality_families",
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
