from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Callable, cast

import orjson
import polars as pl

import official_feature_family_current_season as current_season
import official_feature_family_driver as family_driver
import official_feature_family_history as history_features


ARTIFACT_DIR = Path("artifacts/iteration_120_historical_prior_family_batch")

build_pooled_team_feature_table = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_team_feature_table"),
)
build_pooled_program_history_features = cast(
    Callable[[], pl.DataFrame],
    getattr(history_features, "build_pooled_program_history_features"),
)
build_men_coach_history_features = cast(
    Callable[[], pl.DataFrame],
    getattr(history_features, "build_men_coach_history_features"),
)
build_pooled_conference_history_features = cast(
    Callable[[], pl.DataFrame],
    getattr(history_features, "build_pooled_conference_history_features"),
)
build_pooled_seed_pair_history_features = cast(
    Callable[[], pl.DataFrame],
    getattr(history_features, "build_pooled_seed_pair_history_features"),
)
build_pooled_massey_history_features = cast(
    Callable[[], pl.DataFrame],
    getattr(history_features, "build_pooled_massey_history_features"),
)
add_team_feature_views = cast(
    Callable[[pl.DataFrame, pl.DataFrame, list[str], dict[str, float] | None], tuple[pl.DataFrame, list[str]]],
    getattr(history_features, "add_team_feature_views"),
)
add_pair_feature_views = cast(
    Callable[[pl.DataFrame, pl.DataFrame, list[str], dict[str, float] | None], tuple[pl.DataFrame, list[str]]],
    getattr(history_features, "add_pair_feature_views"),
)
team_feature_view_columns = cast(
    Callable[[list[str]], list[str]],
    getattr(history_features, "team_feature_view_columns"),
)


def team_fill_defaults(feature_columns: list[str]) -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for feature_column in feature_columns:
        fill_defaults[feature_column] = 0.0
    return fill_defaults


def pair_fill_defaults(feature_columns: list[str]) -> dict[str, float]:
    return {feature_column: 0.0 for feature_column in feature_columns}


def historical_team_family(
    name: str,
    artifact_slug: str,
    hypothesis: str,
    team_features_builder: Callable[[], pl.DataFrame],
    feature_columns: list[str],
) -> family_driver.FamilySpec:
    def builder(matchups: pl.DataFrame, feature_columns_unused: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns_unused
        return add_team_feature_views(
            matchups,
            team_features_builder(),
            feature_columns,
            team_fill_defaults(feature_columns),
        )

    return family_driver.FamilySpec(
        name=name,
        artifact_slug=artifact_slug,
        hypothesis=hypothesis,
        extend_iteration_38=builder,
    )


def historical_pair_family(
    name: str,
    artifact_slug: str,
    hypothesis: str,
    pair_features_builder: Callable[[], pl.DataFrame],
    feature_columns: list[str],
) -> family_driver.FamilySpec:
    def builder(matchups: pl.DataFrame, feature_columns_unused: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns_unused
        return add_pair_feature_views(
            matchups,
            pair_features_builder(),
            feature_columns,
            pair_fill_defaults(feature_columns),
        )

    return family_driver.FamilySpec(
        name=name,
        artifact_slug=artifact_slug,
        hypothesis=hypothesis,
        extend_iteration_38=builder,
    )


def seed_vs_massey_family() -> family_driver.FamilySpec:
    feature_columns = [
        "seed_massey_mean_rank_gap",
        "seed_massey_best_rank_gap",
        "seed_massey_rank_std",
        "seed_massey_rank_density",
    ]

    def builder(matchups: pl.DataFrame, feature_columns_unused: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns_unused
        team_features = build_pooled_team_feature_table().with_columns(
            (pl.col("seed_number") - pl.col("massey_mean_rank")).alias("seed_massey_mean_rank_gap"),
            (pl.col("seed_number") - pl.col("massey_best_rank")).alias("seed_massey_best_rank_gap"),
            pl.col("massey_rank_std").alias("seed_massey_rank_std"),
            (pl.col("massey_system_count") / (1.0 + pl.col("massey_mean_rank"))).alias("seed_massey_rank_density"),
        ).select("Season", "TeamID", "men_women", *feature_columns)
        return add_team_feature_views(
            matchups,
            team_features,
            feature_columns,
            team_fill_defaults(feature_columns),
        )

    return family_driver.FamilySpec(
        name="seed_vs_massey_discrepancy_features",
        artifact_slug="seed_vs_massey_discrepancy_features",
        hypothesis="Seed-versus-Massey discrepancies may expose committee disagreement and hidden strength.",
        extend_iteration_38=builder,
    )


PROGRAM_FEATURES = build_pooled_program_history_features
COACH_FEATURES = build_men_coach_history_features
CONFERENCE_FEATURES = build_pooled_conference_history_features
SEED_PAIR_FEATURES = build_pooled_seed_pair_history_features
MASSEY_FEATURES = build_pooled_massey_history_features

FAMILY_SPECS = [
    historical_team_family(
        name="men_coach_tenure_and_continuity_features",
        artifact_slug="men_coach_tenure_and_continuity_features",
        hypothesis="Coach tenure and continuity may matter once bracket pressure moves decisions out of the regular-season regime.",
        team_features_builder=COACH_FEATURES,
        feature_columns=[
            "coach_stints_in_season",
            "coach_coverage_days",
            "coach_prior_unique_team_count",
            "coach_prior_team_seasons",
        ],
    ),
    historical_team_family(
        name="men_midseason_coaching_change_features",
        artifact_slug="men_midseason_coaching_change_features",
        hypothesis="Midseason coaching changes may leave residual instability or rebound effects by tournament time.",
        team_features_builder=COACH_FEATURES,
        feature_columns=[
            "midseason_change_flag",
            "coach_stints_in_season",
            "coach_coverage_days",
            "coach_prior_team_seasons",
        ],
    ),
    historical_team_family(
        name="men_coach_tournament_history_prior_features",
        artifact_slug="men_coach_tournament_history_prior_features",
        hypothesis="Coach tournament history may help once teams enter unusual bracket contexts.",
        team_features_builder=COACH_FEATURES,
        feature_columns=[
            "coach_prior_tournament_appearances",
            "coach_prior_tournament_games",
            "coach_prior_tournament_wins",
            "coach_prior_sweet16_appearances",
            "coach_prior_final_four_appearances",
            "coach_prior_championships",
        ],
    ),
    historical_team_family(
        name="men_program_division_i_tenure_and_transition_recency_features",
        artifact_slug="men_program_division_i_tenure_and_transition_recency_features",
        hypothesis="Program Division-I tenure and recent transition status may matter for tournament readiness.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "program_d1_tenure_seasons",
            "recent_division_transition_flag",
            "prior_ncaa_appearances",
            "recent_3y_ncaa_appearances",
        ],
    ),
    historical_team_family(
        name="previous_season_secondary_tournament_participation_features",
        artifact_slug="previous_season_secondary_tournament_participation_features",
        hypothesis="Secondary-tournament participation history may capture near-miss programs with recent postseason reps.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_secondary_participations",
            "secondary_participation_streak",
            "recent_3y_secondary_participations",
            "prior_secondary_games",
        ],
    ),
    historical_team_family(
        name="previous_season_secondary_tournament_performance_features",
        artifact_slug="previous_season_secondary_tournament_performance_features",
        hypothesis="Secondary-tournament performance may proxy recent postseason competence outside the NCAA field.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_secondary_games",
            "prior_secondary_wins",
            "prior_secondary_championships",
            "recent_tournament_depth_score",
        ],
    ),
    historical_team_family(
        name="men_massey_rank_trend_over_time_features",
        artifact_slug="men_massey_rank_trend_over_time_features",
        hypothesis="Massey rank trends may add directionality that static snapshots miss.",
        team_features_builder=MASSEY_FEATURES,
        feature_columns=[
            "massey_latest_mean_rank",
            "massey_rank_slope",
            "massey_rank_improvement",
            "massey_late_mean_rank",
        ],
    ),
    historical_team_family(
        name="men_massey_late_season_rank_volatility_features",
        artifact_slug="men_massey_late_season_rank_volatility_features",
        hypothesis="Late-season Massey volatility may identify unstable teams that the benchmark still overrates.",
        team_features_builder=MASSEY_FEATURES,
        feature_columns=[
            "massey_latest_rank_std",
            "massey_late_rank_std",
            "massey_elite_rank_std",
            "massey_latest_mean_rank",
        ],
    ),
    seed_vs_massey_family(),
    historical_pair_family(
        name="seed_pair_upset_rate_prior_features",
        artifact_slug="seed_pair_upset_rate_prior_features",
        hypothesis="Historical seed-pair upset rates may inform bracket priors where the current benchmark is overconfident.",
        pair_features_builder=SEED_PAIR_FEATURES,
        feature_columns=[
            "seed_pair_prior_games",
            "seed_pair_prior_lower_team_win_rate",
            "seed_pair_upset_rate_prior",
            "seed_gap_prior_lower_team_win_rate",
        ],
    ),
    historical_pair_family(
        name="slot_specific_historical_upset_prior_features",
        artifact_slug="slot_specific_historical_upset_prior_features",
        hypothesis="Slot-specific upset priors may encode bracket micro-structure beyond simple seed gaps.",
        pair_features_builder=SEED_PAIR_FEATURES,
        feature_columns=[
            "slot_prior_games",
            "slot_prior_lower_team_win_rate",
            "slot_upset_rate_prior",
            "path_chalkiness_prior",
        ],
    ),
    historical_pair_family(
        name="path_chalkiness_prior_features",
        artifact_slug="path_chalkiness_prior_features",
        hypothesis="Path chalkiness priors may help in areas of the bracket that historically collapse or hold form.",
        pair_features_builder=SEED_PAIR_FEATURES,
        feature_columns=[
            "path_chalkiness_prior",
            "seed_pair_upset_rate_prior",
            "slot_upset_rate_prior",
            "seed_gap_prior_lower_team_win_rate",
        ],
    ),
    historical_team_family(
        name="program_ncaa_appearance_streak_features",
        artifact_slug="program_ncaa_appearance_streak_features",
        hypothesis="NCAA appearance streaks may proxy sustainable program quality and institutional readiness.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_ncaa_appearances",
            "prior_ncaa_appearance_streak",
            "recent_3y_ncaa_appearances",
            "prior_tournament_games",
        ],
    ),
    historical_team_family(
        name="program_seed_stability_features",
        artifact_slug="program_seed_stability_features",
        hypothesis="Program seed stability may help differentiate sustainable powers from noisy one-off seasons.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_mean_seed",
            "prior_best_seed",
            "prior_seed_std",
            "prior_seed_stability",
        ],
    ),
    historical_team_family(
        name="program_historical_seed_outperformance_features",
        artifact_slug="program_historical_seed_outperformance_features",
        hypothesis="Seed outperformance priors may capture programs that consistently beat or miss their seed line.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_seed_outperformance",
            "prior_mean_seed",
            "prior_best_seed",
            "recent_tournament_depth_score",
        ],
    ),
    historical_team_family(
        name="program_underdog_upset_propensity_features",
        artifact_slug="program_underdog_upset_propensity_features",
        hypothesis="Underdog upset propensity may help on upset-prone teams where static ratings understate variance.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_underdog_upset_rate",
            "prior_tournament_wins",
            "recent_tournament_depth_score",
            "prior_ncaa_appearances",
        ],
    ),
    historical_team_family(
        name="program_favorite_hold_serve_propensity_features",
        artifact_slug="program_favorite_hold_serve_propensity_features",
        hypothesis="Favorite hold-serve propensity may improve calibration on teams expected to avoid early chaos.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_favorite_hold_rate",
            "prior_tournament_wins",
            "prior_mean_seed",
            "prior_ncaa_appearances",
        ],
    ),
    historical_team_family(
        name="program_recent_tournament_depth_features",
        artifact_slug="program_recent_tournament_depth_features",
        hypothesis="Recent tournament depth may proxy current institutional tournament readiness.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "recent_tournament_depth_score",
            "prior_sweet16_appearances",
            "prior_elite8_appearances",
            "prior_final_four_appearances",
        ],
    ),
    historical_team_family(
        name="recent_ncaa_conference_performance_prior_features",
        artifact_slug="recent_ncaa_conference_performance_prior_features",
        hypothesis="Conference NCAA priors may help after the benchmark already captures team-level strength.",
        team_features_builder=CONFERENCE_FEATURES,
        feature_columns=[
            "prior_conference_ncaa_games",
            "prior_conference_ncaa_wins",
            "prior_conference_ncaa_win_rate",
            "prior_conference_sweet16_teams",
            "prior_conference_elite8_teams",
            "prior_conference_final_four_teams",
            "prior_conference_championships",
        ],
    ),
    historical_team_family(
        name="conference_auto_bid_versus_at_large_proxy_features",
        artifact_slug="conference_auto_bid_versus_at_large_proxy_features",
        hypothesis="Conference auto-bid versus at-large priors may capture structural selection-quality differences.",
        team_features_builder=CONFERENCE_FEATURES,
        feature_columns=[
            "prior_conference_bid_count_mean",
            "prior_conference_one_bid_rate",
            "prior_conference_top4_seed_share_mean",
            "prior_conference_top8_seed_share_mean",
        ],
    ),
    historical_team_family(
        name="conference_seed_distribution_strength_features",
        artifact_slug="conference_seed_distribution_strength_features",
        hypothesis="Conference seed-distribution strength may distinguish deep leagues from top-heavy or shallow ones.",
        team_features_builder=CONFERENCE_FEATURES,
        feature_columns=[
            "prior_conference_bid_count_mean",
            "prior_conference_seed_std_mean",
            "prior_conference_top4_seed_share_mean",
            "prior_conference_top8_seed_share_mean",
        ],
    ),
    historical_team_family(
        name="conference_parity_and_top_heaviness_features",
        artifact_slug="conference_parity_and_top_heaviness_features",
        hypothesis="Conference parity and top-heaviness priors may help contextualize résumé difficulty and team robustness.",
        team_features_builder=CONFERENCE_FEATURES,
        feature_columns=[
            "prior_conference_seed_std_mean",
            "prior_conference_top4_seed_share_mean",
            "prior_conference_top8_seed_share_mean",
            "prior_conference_bid_count_mean",
        ],
    ),
    historical_team_family(
        name="secondary_tournament_championship_depth_features",
        artifact_slug="secondary_tournament_championship_depth_features",
        hypothesis="Secondary-tournament championship depth may proxy recent postseason competence outside the NCAA field.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_secondary_championships",
            "prior_secondary_wins",
            "prior_secondary_games",
            "recent_3y_secondary_participations",
        ],
    ),
    historical_team_family(
        name="secondary_tournament_participation_streak_features",
        artifact_slug="secondary_tournament_participation_streak_features",
        hypothesis="Secondary-tournament participation streaks may flag consistent near-tournament quality.",
        team_features_builder=PROGRAM_FEATURES,
        feature_columns=[
            "prior_secondary_participations",
            "secondary_participation_streak",
            "recent_3y_secondary_participations",
            "prior_secondary_games",
        ],
    ),
    historical_team_family(
        name="men_massey_elite_system_subset_consensus_features",
        artifact_slug="men_massey_elite_system_subset_consensus_features",
        hypothesis="Elite-system Massey consensus may help the benchmark lean on the most reliable ranking subset.",
        team_features_builder=MASSEY_FEATURES,
        feature_columns=[
            "massey_elite_mean_rank",
            "massey_elite_rank_std",
            "massey_elite_top10_share",
            "massey_elite_top25_share",
            "massey_elite_top50_share",
        ],
    ),
    historical_team_family(
        name="men_massey_top_n_bucket_share_features",
        artifact_slug="men_massey_top_n_bucket_share_features",
        hypothesis="Top-N bucket shares may capture rank-consensus shape better than mean rank alone.",
        team_features_builder=MASSEY_FEATURES,
        feature_columns=[
            "massey_all_system_top10_share",
            "massey_all_system_top25_share",
            "massey_all_system_top50_share",
            "massey_elite_top10_share",
            "massey_elite_top25_share",
            "massey_elite_top50_share",
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
        "feature_family_batch": "historical_prior_families",
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
