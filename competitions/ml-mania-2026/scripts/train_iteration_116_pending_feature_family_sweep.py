from __future__ import annotations

import datetime
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable

import orjson
import polars as pl

import official_feature_family_current_season as current
import official_feature_family_driver as family_driver
import official_feature_family_history as history
import pooled_bracket_features as bracket_features
import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_116_pending_feature_family_sweep")
SUMMARY_PATH = ARTIFACT_DIR / "family_sweep_summary.json"

MatchupBuilder = Callable[[pl.DataFrame, list[str]], tuple[pl.DataFrame, list[str]]]


def inferred_fill_defaults(feature_columns: list[str]) -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for feature_column in feature_columns:
        if "win_pct" in feature_column or feature_column.endswith("_pct"):
            fill_defaults[feature_column] = 0.5
        else:
            fill_defaults[feature_column] = 0.0
    return fill_defaults


def compose_builders(*builders: MatchupBuilder) -> MatchupBuilder:
    def composed_builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        updated_matchups = matchups
        updated_feature_columns = feature_columns
        for builder in builders:
            updated_matchups, updated_feature_columns = builder(updated_matchups, updated_feature_columns)
        return updated_matchups, updated_feature_columns

    return composed_builder


def build_team_alias_builder(
    source_frame_builder: Callable[[], pl.DataFrame],
    alias_map: dict[str, str],
    fill_defaults: dict[str, float] | None = None,
) -> MatchupBuilder:
    feature_columns = list(alias_map.values())
    fill_defaults = fill_defaults or inferred_fill_defaults(feature_columns)

    def builder(matchups: pl.DataFrame, feature_columns_in_model: list[str]) -> tuple[pl.DataFrame, list[str]]:
        source_frame = source_frame_builder().select(
            "Season",
            "TeamID",
            "men_women",
            *[pl.col(source_column).alias(target_column) for source_column, target_column in alias_map.items()],
        )
        updated_matchups, added_columns = history.add_team_feature_views(
            matchup_frame=matchups,
            team_features=source_frame,
            feature_columns=feature_columns,
            fill_defaults=fill_defaults,
        )
        return updated_matchups, feature_columns_in_model + added_columns

    return builder


def build_pair_alias_builder(
    source_frame_builder: Callable[[], pl.DataFrame],
    alias_map: dict[str, str],
    fill_defaults: dict[str, float] | None = None,
) -> MatchupBuilder:
    feature_columns = list(alias_map.values())
    fill_defaults = fill_defaults or inferred_fill_defaults(feature_columns)

    def builder(matchups: pl.DataFrame, feature_columns_in_model: list[str]) -> tuple[pl.DataFrame, list[str]]:
        source_frame = source_frame_builder().select(
            *current.MATCHUP_KEY_COLUMNS,
            *[pl.col(source_column).alias(target_column) for source_column, target_column in alias_map.items()],
        )
        updated_matchups, added_columns = history.add_pair_feature_views(
            matchup_frame=matchups,
            pair_features=source_frame,
            feature_columns=feature_columns,
            fill_defaults=fill_defaults,
        )
        return updated_matchups, feature_columns_in_model + added_columns

    return builder


@lru_cache(maxsize=1)
def build_conference_membership_features() -> pl.DataFrame:
    return (
        current.build_pooled_team_games()
        .select("Season", "TeamID", "men_women", "team_conf_abbrev")
        .unique()
        .filter(pl.col("team_conf_abbrev").is_not_null())
    )


@lru_cache(maxsize=None)
def build_metric_split_features(metric_name: str, prefix: str) -> pl.DataFrame:
    team_games = current.build_pooled_team_games()
    return team_games.group_by(current.TEAM_KEY_COLUMNS).agg(
        pl.col(metric_name).mean().alias(f"{prefix}_overall"),
        pl.col(metric_name).filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias(f"{prefix}_late"),
        pl.col(metric_name).filter(pl.col("IsNeutral") == 1).mean().fill_null(0.0).alias(f"{prefix}_neutral"),
        (
            pl.col(metric_name).filter(pl.col("IsHome") == 1).mean().fill_null(0.0)
            - pl.col(metric_name).filter(pl.col("IsAway") == 1).mean().fill_null(0.0)
        ).alias(f"{prefix}_home_road_gap"),
    )


@lru_cache(maxsize=1)
def build_shot_mix_features() -> pl.DataFrame:
    team_features = current.build_pooled_team_feature_table()
    return team_features.select(
        "Season",
        "TeamID",
        "men_women",
        pl.col("three_point_attempt_rate").alias("shot_mix_three_point_rate"),
        pl.col("two_point_attempt_rate").alias("shot_mix_two_point_rate"),
        pl.col("free_throw_rate").alias("shot_mix_free_throw_rate"),
        (pl.col("three_point_attempt_rate") - pl.col("free_throw_rate")).alias("shot_mix_perimeter_bias"),
    )


@lru_cache(maxsize=1)
def build_seed_massey_discrepancy_features() -> pl.DataFrame:
    team_features = current.build_pooled_team_feature_table()
    seed_anchor = 4.0 * pl.col("seed_number") - 2.0
    return team_features.select(
        "Season",
        "TeamID",
        "men_women",
        (pl.col("seed_number").fill_null(17.0)).alias("seed_massey_seed_number"),
        (pl.col("massey_mean_rank") - seed_anchor).alias("seed_massey_mean_gap"),
        (pl.col("massey_best_rank") - seed_anchor).alias("seed_massey_best_gap"),
        pl.col("massey_rank_std").alias("seed_massey_rank_dispersion"),
        (pl.col("massey_mean_rank") > seed_anchor + 12.0).cast(pl.Float64).alias("seed_massey_overseed_flag"),
        (pl.col("massey_mean_rank") < seed_anchor - 12.0).cast(pl.Float64).alias("seed_massey_underseed_flag"),
    )


@lru_cache(maxsize=1)
def build_seeded_opponent_bucket_features() -> pl.DataFrame:
    team_games = current.build_pooled_team_games()
    return team_games.group_by(current.TEAM_KEY_COLUMNS).agg(
        pl.col("faced_seeded_opponent").mean().alias("seeded_bucket_any_share"),
        pl.col("Win").filter(pl.col("faced_seeded_opponent") == 1).mean().fill_null(0.5).alias("seeded_bucket_any_win_pct"),
        pl.col("Margin").filter(pl.col("faced_seeded_opponent") == 1).mean().fill_null(0.0).alias("seeded_bucket_any_margin"),
        (pl.col("opp_seed_number") <= 4).cast(pl.Float64).mean().alias("seeded_bucket_top4_share"),
        (pl.col("opp_seed_number") <= 8).cast(pl.Float64).mean().alias("seeded_bucket_top8_share"),
        (pl.col("opp_seed_number") <= 12).cast(pl.Float64).mean().alias("seeded_bucket_top12_share"),
        pl.col("Win").filter(pl.col("opp_seed_number") <= 8).mean().fill_null(0.5).alias("seeded_bucket_top8_win_pct"),
        pl.col("Margin").filter(pl.col("opp_seed_number") <= 8).mean().fill_null(0.0).alias("seeded_bucket_top8_margin"),
    )


@lru_cache(maxsize=1)
def build_top_massey_bucket_features() -> pl.DataFrame:
    team_games = current.build_pooled_team_games()
    return team_games.group_by(current.TEAM_KEY_COLUMNS).agg(
        (pl.col("opp_massey_mean_rank") <= 10).cast(pl.Float64).mean().alias("massey_bucket_top10_share"),
        (pl.col("opp_massey_mean_rank") <= 25).cast(pl.Float64).mean().alias("massey_bucket_top25_share"),
        (pl.col("opp_massey_mean_rank") <= 50).cast(pl.Float64).mean().alias("massey_bucket_top50_share"),
        pl.col("Win").filter(pl.col("opp_massey_mean_rank") <= 25).mean().fill_null(0.5).alias("massey_bucket_top25_win_pct"),
        pl.col("Margin").filter(pl.col("opp_massey_mean_rank") <= 25).mean().fill_null(0.0).alias("massey_bucket_top25_margin"),
        pl.col("Win").filter(pl.col("opp_massey_mean_rank") <= 10).mean().fill_null(0.5).alias("massey_bucket_top10_win_pct"),
        pl.col("Margin").filter(pl.col("opp_massey_mean_rank") <= 10).mean().fill_null(0.0).alias("massey_bucket_top10_margin"),
    )


@lru_cache(maxsize=1)
def build_possession_regime_features() -> pl.DataFrame:
    team_games = current.build_pooled_team_games().join(
        current.build_pooled_team_feature_table().select("Season", "TeamID", "men_women", "season_avg_pace"),
        on=current.TEAM_KEY_COLUMNS,
        how="left",
    ).with_columns(
        (pl.col("pace") >= pl.col("season_avg_pace")).cast(pl.Int8).alias("is_fast_pace_game"),
        (pl.col("pace") < pl.col("season_avg_pace")).cast(pl.Int8).alias("is_slow_pace_game"),
    )
    return team_games.group_by(current.TEAM_KEY_COLUMNS).agg(
        pl.col("Win").filter(pl.col("is_fast_pace_game") == 1).mean().fill_null(0.5).alias("pace_regime_fast_win_pct"),
        pl.col("Margin").filter(pl.col("is_fast_pace_game") == 1).mean().fill_null(0.0).alias("pace_regime_fast_margin"),
        pl.col("off_rating").filter(pl.col("is_fast_pace_game") == 1).mean().fill_null(0.0).alias("pace_regime_fast_off_rating"),
        pl.col("Win").filter(pl.col("is_slow_pace_game") == 1).mean().fill_null(0.5).alias("pace_regime_slow_win_pct"),
        pl.col("Margin").filter(pl.col("is_slow_pace_game") == 1).mean().fill_null(0.0).alias("pace_regime_slow_margin"),
        pl.col("def_rating").filter(pl.col("is_slow_pace_game") == 1).mean().fill_null(0.0).alias("pace_regime_slow_def_rating"),
    )


def build_division_conference_tourney_team_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    conference_games = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}ConferenceTourneyGames.csv").filter(
        pl.col("Season") >= baseline.MIN_POOLED_SEASON
    )
    if conference_games.is_empty():
        return pl.DataFrame(schema={"Season": pl.Int64, "TeamID": pl.Int64, "men_women": pl.Int8})

    pre_tourney_quality = current.build_pooled_team_feature_table().filter(pl.col("men_women") == config.men_women_flag).select(
        "Season",
        "TeamID",
        pl.col("avg_margin").alias("team_quality"),
    )
    final_day_lookup = conference_games.group_by(["Season", "ConfAbbrev"]).agg(pl.col("DayNum").max().alias("conf_final_daynum"))
    winner_rows = conference_games.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("ConfAbbrev"),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64).alias("TeamID"),
        pl.col("LTeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.lit(1.0).alias("Win"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )
    loser_rows = conference_games.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("ConfAbbrev"),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64).alias("TeamID"),
        pl.col("WTeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.lit(0.0).alias("Win"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )
    team_rows = pl.concat([winner_rows, loser_rows], how="vertical_relaxed").join(
        final_day_lookup,
        on=["Season", "ConfAbbrev"],
        how="left",
    ).join(
        pre_tourney_quality.rename({"TeamID": "OppTeamID", "team_quality": "opp_team_quality"}),
        on=["Season", "OppTeamID"],
        how="left",
    )
    champion_rows = conference_games.join(final_day_lookup, on=["Season", "ConfAbbrev"], how="left").filter(
        pl.col("DayNum") == pl.col("conf_final_daynum")
    )
    champion_flags = champion_rows.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64).alias("TeamID"),
        pl.lit(1.0).alias("conf_tourney_champion_flag"),
    )
    finalist_flags = pl.concat(
        [
            champion_rows.select(
                pl.col("Season").cast(pl.Int64),
                pl.col("WTeamID").cast(pl.Int64).alias("TeamID"),
                pl.lit(1.0).alias("conf_tourney_finalist_flag"),
            ),
            champion_rows.select(
                pl.col("Season").cast(pl.Int64),
                pl.col("LTeamID").cast(pl.Int64).alias("TeamID"),
                pl.lit(1.0).alias("conf_tourney_finalist_flag"),
            ),
        ],
        how="vertical_relaxed",
    ).group_by(["Season", "TeamID"]).agg(pl.col("conf_tourney_finalist_flag").max())
    return (
        team_rows.group_by(current.TEAM_KEY_COLUMNS)
        .agg(
            pl.len().cast(pl.Float64).alias("conf_tourney_games"),
            pl.col("Win").sum().cast(pl.Float64).alias("conf_tourney_wins"),
            pl.col("DayNum").max().cast(pl.Float64).alias("conf_tourney_last_daynum"),
            pl.col("opp_team_quality").mean().fill_null(0.0).alias("conf_tourney_avg_opp_quality"),
            pl.col("opp_team_quality").max().fill_null(0.0).alias("conf_tourney_max_opp_quality"),
        )
        .join(champion_flags, on=["Season", "TeamID"], how="left")
        .join(finalist_flags, on=["Season", "TeamID"], how="left")
        .with_columns(
            pl.col("conf_tourney_champion_flag").fill_null(0.0),
            pl.col("conf_tourney_finalist_flag").fill_null(0.0),
            (current.SELECTION_SUNDAY_DAYNUM - pl.col("conf_tourney_last_daynum")).alias(
                "conf_tourney_days_before_selection"
            ),
        )
    )


@lru_cache(maxsize=1)
def build_conference_tourney_team_features() -> pl.DataFrame:
    return pl.concat(
        [build_division_conference_tourney_team_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def build_division_conference_tourney_conference_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    conference_games = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}ConferenceTourneyGames.csv").filter(
        pl.col("Season") >= baseline.MIN_POOLED_SEASON
    )
    if conference_games.is_empty():
        return pl.DataFrame(schema={"Season": pl.Int64, "ConfAbbrev": pl.String, "men_women": pl.Int8})
    quality_frame = current.build_pooled_team_feature_table().filter(pl.col("men_women") == config.men_women_flag).select(
        "Season",
        "TeamID",
        pl.col("avg_margin").alias("team_quality"),
    )
    upset_rows = conference_games.join(
        quality_frame.rename({"TeamID": "WTeamID", "team_quality": "winner_quality"}),
        on=["Season", "WTeamID"],
        how="left",
    ).join(
        quality_frame.rename({"TeamID": "LTeamID", "team_quality": "loser_quality"}),
        on=["Season", "LTeamID"],
        how="left",
    ).with_columns(
        (pl.col("winner_quality") - pl.col("loser_quality")).abs().fill_null(0.0).alias("conf_tourney_quality_gap"),
    ).with_columns(
        (pl.col("winner_quality") < pl.col("loser_quality")).cast(pl.Float64).alias("conf_tourney_upset"),
        (pl.col("conf_tourney_quality_gap") <= current.CLOSE_GAME_MARGIN).cast(pl.Float64).alias(
            "conf_tourney_close_game"
        ),
        pl.col("conf_tourney_quality_gap").cast(pl.Float64).alias("conf_tourney_abs_margin"),
        ((pl.col("winner_quality") + pl.col("loser_quality")) / 2.0).fill_null(0.0).alias("conf_tourney_path_difficulty"),
    )
    conference_summary = upset_rows.group_by(["Season", "ConfAbbrev"]).agg(
        pl.len().cast(pl.Float64).alias("conf_tourney_game_count"),
        pl.col("conf_tourney_upset").mean().alias("conf_tourney_upset_rate"),
        pl.col("conf_tourney_close_game").mean().alias("conf_tourney_close_game_share"),
        pl.col("conf_tourney_abs_margin").mean().alias("conf_tourney_avg_abs_margin"),
        pl.col("conf_tourney_path_difficulty").mean().alias("conf_tourney_avg_path_difficulty"),
    )
    membership = (
        build_conference_membership_features()
        .filter(pl.col("men_women") == config.men_women_flag)
        .rename({"team_conf_abbrev": "ConfAbbrev"})
    )
    return membership.join(conference_summary, on=["Season", "ConfAbbrev"], how="left").with_columns(
        pl.col("conf_tourney_game_count").fill_null(0.0),
        pl.col("conf_tourney_upset_rate").fill_null(0.0),
        pl.col("conf_tourney_close_game_share").fill_null(0.0),
        pl.col("conf_tourney_avg_abs_margin").fill_null(0.0),
        pl.col("conf_tourney_avg_path_difficulty").fill_null(0.0),
    )


@lru_cache(maxsize=1)
def build_conference_tourney_conference_features() -> pl.DataFrame:
    return pl.concat(
        [build_division_conference_tourney_conference_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def build_seed_line_history_team_features() -> pl.DataFrame:
    output_frames: list[pl.DataFrame] = []
    for config in baseline.DIVISION_CONFIGS:
        seed_features = bracket_features.build_seed_team_features(config).select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
            pl.col("seed_number").cast(pl.Int64),
        )
        tournament_results = pl.read_csv(config.tournament_results_path).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        winner_rows = tournament_results.select(
            pl.col("Season").cast(pl.Int64),
            pl.col("WTeamID").cast(pl.Int64).alias("TeamID"),
            pl.lit(1.0).alias("win"),
        )
        loser_rows = tournament_results.select(
            pl.col("Season").cast(pl.Int64),
            pl.col("LTeamID").cast(pl.Int64).alias("TeamID"),
            pl.lit(0.0).alias("win"),
        )
        team_wins = pl.concat([winner_rows, loser_rows], how="vertical_relaxed").group_by(["Season", "TeamID"]).agg(
            pl.col("win").sum().alias("tournament_wins")
        )
        seed_summary = seed_features.join(team_wins, on=["Season", "TeamID"], how="left").with_columns(
            pl.col("tournament_wins").fill_null(0.0),
            (pl.col("tournament_wins") >= 2.0).cast(pl.Float64).alias("sweet16_flag"),
            (pl.col("tournament_wins") >= 4.0).cast(pl.Float64).alias("final_four_flag"),
            (pl.col("tournament_wins") >= 6.0).cast(pl.Float64).alias("championship_flag"),
        )
        prior_rows: list[dict[str, float | int]] = []
        for season in sorted(seed_summary["Season"].unique().to_list()):
            prior = seed_summary.filter(pl.col("Season") < season)
            current_season = seed_summary.filter(pl.col("Season") == season)
            prior_by_seed = {
                int(seed_number): (
                    float(mean_wins),
                    float(sweet16_rate),
                    float(final_four_rate),
                    float(championship_rate),
                )
                for seed_number, mean_wins, sweet16_rate, final_four_rate, championship_rate in prior.group_by("seed_number").agg(
                    pl.col("tournament_wins").mean().alias("mean_wins"),
                    pl.col("sweet16_flag").mean().alias("sweet16_rate"),
                    pl.col("final_four_flag").mean().alias("final_four_rate"),
                    pl.col("championship_flag").mean().alias("championship_rate"),
                ).iter_rows()
            }
            for row in current_season.iter_rows(named=True):
                mean_wins, sweet16_rate, final_four_rate, championship_rate = prior_by_seed.get(
                    int(row["seed_number"]),
                    (0.0, 0.0, 0.0, 0.0),
                )
                prior_rows.append(
                    {
                        "Season": int(row["Season"]),
                        "TeamID": int(row["TeamID"]),
                        "men_women": config.men_women_flag,
                        "seed_line_prior_mean_wins": mean_wins,
                        "seed_line_prior_sweet16_rate": sweet16_rate,
                        "seed_line_prior_final_four_rate": final_four_rate,
                        "seed_line_prior_championship_rate": championship_rate,
                    }
                )
        output_frames.append(pl.DataFrame(prior_rows))
    return pl.concat(output_frames, how="vertical_relaxed")


@lru_cache(maxsize=1)
def build_same_conference_pair_features() -> pl.DataFrame:
    pair_features = bracket_features.build_pooled_pair_bracket_features()
    conference_membership = build_conference_membership_features()
    regular_pairs = current.build_pooled_regular_season_pair_features()
    conference_low = conference_membership.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col("team_conf_abbrev").alias("low_conf_abbrev"),
    )
    conference_high = conference_membership.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col("team_conf_abbrev").alias("high_conf_abbrev"),
    )
    return (
        pair_features.join(conference_low, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(conference_high, on=["Season", "HigherTeamID", "men_women"], how="left")
        .join(regular_pairs, on=current.MATCHUP_KEY_COLUMNS, how="left")
        .with_columns(
            (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Float64).alias("pair_same_conference_flag"),
            (pl.col("same_first_weekend_pod_flag") * (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Int8))
            .cast(pl.Float64)
            .alias("pair_same_conference_pod_flag"),
            (pl.col("same_regional_quadrant_flag") * (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Int8))
            .cast(pl.Float64)
            .alias("pair_same_conference_quadrant_flag"),
            (pl.col("round64_pair_flag") * (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Int8))
            .cast(pl.Float64)
            .alias("pair_same_conference_round64_flag"),
            (pl.col("round32_pair_flag") * (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Int8))
            .cast(pl.Float64)
            .alias("pair_same_conference_round32_flag"),
            pl.col("opening_slot_gap").cast(pl.Float64).alias("pair_opening_slot_gap"),
            pl.col("regular_season_meetings").fill_null(0.0).alias("pair_regular_season_meetings"),
            pl.col("split_regular_season_series_flag").fill_null(0.0).alias("pair_split_series_flag"),
            pl.col("days_since_last_meeting").fill_null(float(current.SELECTION_SUNDAY_DAYNUM)).alias(
                "pair_days_since_last_meeting"
            ),
            pl.col("lower_regular_season_win_pct").fill_null(0.5).alias("pair_lower_regular_season_win_pct"),
        )
        .select(
            *current.MATCHUP_KEY_COLUMNS,
            "pair_same_conference_flag",
            "pair_same_conference_pod_flag",
            "pair_same_conference_quadrant_flag",
            "pair_same_conference_round64_flag",
            "pair_same_conference_round32_flag",
            "pair_opening_slot_gap",
            "pair_regular_season_meetings",
            "pair_split_series_flag",
            "pair_days_since_last_meeting",
            "pair_lower_regular_season_win_pct",
        )
    )


@lru_cache(maxsize=1)
def build_conference_pair_interaction_features() -> pl.DataFrame:
    pair_features = bracket_features.build_pooled_pair_bracket_features()
    conference_membership = build_conference_membership_features()
    conference_history = history.build_pooled_conference_history_features()
    regular_pairs = current.build_pooled_regular_season_pair_features()
    low_team = conference_membership.join(
        conference_history.select(
            "Season",
            "TeamID",
            "men_women",
            pl.col("prior_conference_ncaa_win_rate").alias("low_conf_prior_ncaa_win_rate"),
            pl.col("prior_conference_bid_count_mean").alias("low_conf_prior_bid_count_mean"),
        ),
        on=["Season", "TeamID", "men_women"],
        how="left",
    ).select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col("team_conf_abbrev").alias("low_conf_abbrev"),
        "low_conf_prior_ncaa_win_rate",
        "low_conf_prior_bid_count_mean",
    )
    high_team = conference_membership.join(
        conference_history.select(
            "Season",
            "TeamID",
            "men_women",
            pl.col("prior_conference_ncaa_win_rate").alias("high_conf_prior_ncaa_win_rate"),
            pl.col("prior_conference_bid_count_mean").alias("high_conf_prior_bid_count_mean"),
        ),
        on=["Season", "TeamID", "men_women"],
        how="left",
    ).select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col("team_conf_abbrev").alias("high_conf_abbrev"),
        "high_conf_prior_ncaa_win_rate",
        "high_conf_prior_bid_count_mean",
    )
    return (
        pair_features.join(low_team, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_team, on=["Season", "HigherTeamID", "men_women"], how="left")
        .join(regular_pairs, on=current.MATCHUP_KEY_COLUMNS, how="left")
        .with_columns(
            (pl.col("low_conf_abbrev") == pl.col("high_conf_abbrev")).cast(pl.Float64).alias("conference_pair_same_conf_flag"),
            (pl.col("low_conf_abbrev") != pl.col("high_conf_abbrev")).cast(pl.Float64).alias("conference_pair_interconf_flag"),
            pl.col("regular_season_meetings").fill_null(0.0).alias("conference_pair_regular_season_meetings"),
            pl.col("split_regular_season_series_flag").fill_null(0.0).alias("conference_pair_split_series_flag"),
            pl.col("low_conf_prior_ncaa_win_rate").fill_null(0.0),
            pl.col("high_conf_prior_ncaa_win_rate").fill_null(0.0),
            pl.col("low_conf_prior_bid_count_mean").fill_null(0.0),
            pl.col("high_conf_prior_bid_count_mean").fill_null(0.0),
        )
        .with_columns(
            (pl.col("low_conf_prior_ncaa_win_rate") - pl.col("high_conf_prior_ncaa_win_rate")).alias(
                "conference_pair_delta_prior_ncaa_win_rate"
            ),
            (pl.col("low_conf_prior_bid_count_mean") - pl.col("high_conf_prior_bid_count_mean")).alias(
                "conference_pair_delta_prior_bid_count_mean"
            ),
        )
        .select(
            *current.MATCHUP_KEY_COLUMNS,
            "conference_pair_same_conf_flag",
            "conference_pair_interconf_flag",
            "conference_pair_regular_season_meetings",
            "conference_pair_split_series_flag",
            "low_conf_prior_ncaa_win_rate",
            "high_conf_prior_ncaa_win_rate",
            "conference_pair_delta_prior_ncaa_win_rate",
            "low_conf_prior_bid_count_mean",
            "high_conf_prior_bid_count_mean",
            "conference_pair_delta_prior_bid_count_mean",
        )
    )


@lru_cache(maxsize=1)
def build_seed_pair_derived_features() -> pl.DataFrame:
    seed_pair_history = history.build_pooled_seed_pair_history_features()
    return seed_pair_history.with_columns(
        (1.0 - pl.col("seed_pair_prior_lower_team_win_rate")).alias("seed_pair_prior_upset_rate"),
        (1.0 - pl.col("slot_prior_lower_team_win_rate")).alias("slot_prior_favorite_loss_rate"),
    )


def build_style_compatibility_builder() -> MatchupBuilder:
    def builder(matchups: pl.DataFrame, feature_columns_in_model: list[str]) -> tuple[pl.DataFrame, list[str]]:
        team_features = current.build_pooled_team_feature_table()
        low_team = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("LowerTeamID"),
            pl.col("men_women"),
            pl.col("pace").alias("low_style_pace"),
            pl.col("effective_fg_pct").alias("low_style_efg"),
            pl.col("opp_effective_fg_pct").alias("low_style_opp_efg"),
            pl.col("turnover_rate").alias("low_style_turnover"),
            pl.col("steal_rate").alias("low_style_steal"),
            pl.col("offensive_rebound_share").alias("low_style_orb"),
            pl.col("defensive_rebound_share").alias("low_style_drb"),
            pl.col("free_throw_rate").alias("low_style_ft_rate"),
            pl.col("foul_diff_rate").alias("low_style_foul_diff"),
            pl.col("three_point_attempt_rate").alias("low_style_three_rate"),
        )
        high_team = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("HigherTeamID"),
            pl.col("men_women"),
            pl.col("pace").alias("high_style_pace"),
            pl.col("effective_fg_pct").alias("high_style_efg"),
            pl.col("opp_effective_fg_pct").alias("high_style_opp_efg"),
            pl.col("turnover_rate").alias("high_style_turnover"),
            pl.col("steal_rate").alias("high_style_steal"),
            pl.col("offensive_rebound_share").alias("high_style_orb"),
            pl.col("defensive_rebound_share").alias("high_style_drb"),
            pl.col("free_throw_rate").alias("high_style_ft_rate"),
            pl.col("foul_diff_rate").alias("high_style_foul_diff"),
            pl.col("three_point_attempt_rate").alias("high_style_three_rate"),
        )
        updated_matchups = (
            matchups.join(low_team, on=["Season", "LowerTeamID", "men_women"], how="left")
            .join(high_team, on=["Season", "HigherTeamID", "men_women"], how="left")
            .with_columns(
                (pl.col("low_style_pace") - pl.col("high_style_pace")).abs().alias("style_pair_pace_gap"),
                (pl.col("low_style_efg") - pl.col("high_style_opp_efg")).alias("style_pair_low_shooting_edge"),
                (pl.col("high_style_efg") - pl.col("low_style_opp_efg")).alias("style_pair_high_shooting_edge"),
                (pl.col("low_style_steal") - pl.col("high_style_turnover")).alias("style_pair_low_turnover_pressure"),
                (pl.col("high_style_steal") - pl.col("low_style_turnover")).alias("style_pair_high_turnover_pressure"),
                (pl.col("low_style_orb") - pl.col("high_style_drb")).alias("style_pair_low_glass_edge"),
                (pl.col("high_style_orb") - pl.col("low_style_drb")).alias("style_pair_high_glass_edge"),
                (pl.col("low_style_ft_rate") + pl.col("low_style_foul_diff")).alias("style_pair_low_foul_pressure"),
                (pl.col("high_style_ft_rate") + pl.col("high_style_foul_diff")).alias("style_pair_high_foul_pressure"),
                (pl.col("low_style_three_rate") - pl.col("high_style_three_rate")).abs().alias("style_pair_three_point_gap"),
            )
        )
        added_columns = [
            "style_pair_pace_gap",
            "style_pair_low_shooting_edge",
            "style_pair_high_shooting_edge",
            "style_pair_low_turnover_pressure",
            "style_pair_high_turnover_pressure",
            "style_pair_low_glass_edge",
            "style_pair_high_glass_edge",
            "style_pair_low_foul_pressure",
            "style_pair_high_foul_pressure",
            "style_pair_three_point_gap",
        ]
        return updated_matchups, feature_columns_in_model + added_columns

    return builder


@lru_cache(maxsize=1)
def build_auto_bid_proxy_features() -> pl.DataFrame:
    conference_history = history.build_pooled_conference_history_features()
    conf_tourney_features = build_conference_tourney_team_features()
    return conference_history.join(
        conf_tourney_features.select(
            "Season",
            "TeamID",
            "men_women",
            "conf_tourney_champion_flag",
            "conf_tourney_finalist_flag",
        ),
        on=["Season", "TeamID", "men_women"],
        how="left",
    ).with_columns(
        pl.col("conf_tourney_champion_flag").fill_null(0.0).alias("auto_bid_proxy_champion_flag"),
        pl.col("conf_tourney_finalist_flag").fill_null(0.0).alias("auto_bid_proxy_finalist_flag"),
        pl.col("prior_conference_bid_count_mean").alias("auto_bid_proxy_prior_bid_count_mean"),
        pl.col("prior_conference_one_bid_rate").alias("auto_bid_proxy_prior_one_bid_rate"),
        (pl.col("prior_conference_bid_count_mean") - 1.0).clip(lower_bound=0.0).alias("auto_bid_proxy_at_large_support"),
    ).select(
        "Season",
        "TeamID",
        "men_women",
        "auto_bid_proxy_champion_flag",
        "auto_bid_proxy_finalist_flag",
        "auto_bid_proxy_prior_bid_count_mean",
        "auto_bid_proxy_prior_one_bid_rate",
        "auto_bid_proxy_at_large_support",
    )


def family_specifications() -> list[family_driver.FamilySpec]:
    team_source = current.build_pooled_team_feature_table
    specs = [
        family_driver.FamilySpec(
            name="Current-season neutral-site-only strength features",
            artifact_slug="current_season_neutral_site_only_strength_features",
            hypothesis="Neutral-only current-season strength splits can add portable tournament signal.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "neutral_game_share": "neutral_strength_game_share",
                    "neutral_win_pct": "neutral_strength_win_pct",
                    "neutral_avg_margin": "neutral_strength_margin",
                    "neutral_off_rating": "neutral_strength_off_rating",
                    "neutral_def_rating": "neutral_strength_def_rating",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season home-road split gap features",
            artifact_slug="current_season_home_road_split_gap_features",
            hypothesis="Home-road split gaps may reveal teams whose strength travels better to the tournament.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "home_road_win_pct_gap": "site_gap_win_pct",
                    "home_road_margin_gap": "site_gap_margin",
                    "home_road_off_rating_gap": "site_gap_off_rating",
                    "home_road_defense_gap": "site_gap_defense",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season conference-vs-nonconference split features",
            artifact_slug="current_season_conference_vs_nonconference_split_features",
            hypothesis="Conference versus nonconference splits can expose schedule-context inflation.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "conference_nonconference_win_pct_gap": "conference_split_win_gap",
                    "conference_nonconference_margin_gap": "conference_split_margin_gap",
                    "conference_nonconference_off_rating_gap": "conference_split_off_gap",
                    "conference_nonconference_defense_gap": "conference_split_def_gap",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season preconference-vs-conference-play trajectory features",
            artifact_slug="current_season_preconference_vs_conference_play_trajectory_features",
            hypothesis="Preconference-to-conference changes may capture how teams scale once schedules harden.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "conference_phase_win_pct_delta": "phase_trajectory_win_delta",
                    "conference_phase_margin_delta": "phase_trajectory_margin_delta",
                    "conference_phase_off_rating_delta": "phase_trajectory_off_delta",
                    "conference_phase_defense_delta": "phase_trajectory_def_delta",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season rest-day gap features",
            artifact_slug="current_season_rest_day_gap_features",
            hypothesis="Rest-context performance may matter once the tournament compresses turnaround windows.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "avg_rest_gap": "rest_context_avg_gap",
                    "short_rest_share": "rest_context_short_share",
                    "short_rest_win_pct": "rest_context_short_win_pct",
                    "short_rest_avg_margin": "rest_context_short_margin",
                    "long_rest_win_pct": "rest_context_long_win_pct",
                    "long_rest_avg_margin": "rest_context_long_margin",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season schedule-density and fatigue features",
            artifact_slug="current_season_schedule_density_and_fatigue_features",
            hypothesis="Schedule density may separate teams that sustain quality under repeated short turns.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "games_per_week": "fatigue_games_per_week",
                    "back_to_back_share": "fatigue_back_to_back_share",
                    "short_rest_share": "fatigue_short_rest_share",
                    "avg_day_gap": "fatigue_avg_gap",
                    "short_rest_avg_margin": "fatigue_short_rest_margin",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season overtime propensity and performance features",
            artifact_slug="current_season_overtime_propensity_and_performance_features",
            hypothesis="Overtime frequency and results may capture late-game resilience and variance appetite.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "overtime_game_share": "overtime_share",
                    "overtime_win_pct": "overtime_win_pct",
                    "overtime_avg_margin": "overtime_margin",
                    "avg_overtime_periods": "overtime_avg_periods",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season close-game execution features",
            artifact_slug="current_season_close_game_execution_features",
            hypothesis="Close-game execution can matter when tournament spreads compress late in the bracket.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "close_game_share": "close_game_share_current",
                    "close_game_win_pct": "close_game_win_pct_current",
                    "close_game_avg_margin": "close_game_margin_current",
                    "close_game_net_rating": "close_game_net_rating_current",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season blowout tendency features",
            artifact_slug="current_season_blowout_tendency_features",
            hypothesis="Blowout creation and avoidance may capture team ceiling and floor better than average margin alone.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "blowout_win_share": "blowout_win_share_current",
                    "blowout_loss_share": "blowout_loss_share_current",
                    "blowout_game_share": "blowout_game_share_current",
                    "blowout_avg_margin": "blowout_margin_current",
                },
            ),
        ),
        family_driver.FamilySpec(
            name="Current-season margin volatility and consistency features",
            artifact_slug="current_season_margin_volatility_and_consistency_features",
            hypothesis="Margin volatility may help distinguish stable favorites from high-variance teams.",
            extend_iteration_38=build_team_alias_builder(
                team_source,
                {
                    "margin_std": "margin_volatility_std",
                    "avg_abs_margin": "margin_volatility_abs_mean",
                    "abs_margin_std": "margin_volatility_abs_std",
                    "close_game_share": "margin_volatility_close_share",
                    "blowout_game_share": "margin_volatility_blowout_share",
                },
            ),
        ),
    ]
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Current-season common-opponent comparison features",
                artifact_slug="current_season_common_opponent_comparison_features",
                hypothesis="Common-opponent comparisons can anchor cross-schedule matchup translation.",
                extend_iteration_38=build_pair_alias_builder(
                    current.build_pooled_common_opponent_features,
                    {
                        "common_opponent_count": "common_opponent_count_current",
                        "lower_common_win_pct": "common_opponent_low_win_pct",
                        "higher_common_win_pct": "common_opponent_high_win_pct",
                        "delta_common_win_pct": "common_opponent_delta_win_pct",
                        "lower_common_avg_margin": "common_opponent_low_margin",
                        "higher_common_avg_margin": "common_opponent_high_margin",
                        "delta_common_avg_margin": "common_opponent_delta_margin",
                        "lower_common_off_rating": "common_opponent_low_off_rating",
                        "higher_common_off_rating": "common_opponent_high_off_rating",
                        "delta_common_off_rating": "common_opponent_delta_off_rating",
                        "lower_common_def_rating": "common_opponent_low_def_rating",
                        "higher_common_def_rating": "common_opponent_high_def_rating",
                        "delta_common_defense_edge": "common_opponent_delta_defense_edge",
                    },
                    fill_defaults={"common_opponent_count_current": 0.0},
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season opponent-tier record features",
                artifact_slug="current_season_opponent_tier_record_features",
                hypothesis="Opponent-tier splits may separate teams that beat good competition from teams that feast on weak schedules.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "top_tier_schedule_share": "opponent_tier_top_share",
                        "top_tier_win_pct": "opponent_tier_top_win_pct",
                        "top_tier_avg_margin": "opponent_tier_top_margin",
                        "bottom_tier_schedule_share": "opponent_tier_bottom_share",
                        "bottom_tier_win_pct": "opponent_tier_bottom_win_pct",
                        "bottom_tier_avg_margin": "opponent_tier_bottom_margin",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season rematch familiarity features",
                artifact_slug="current_season_rematch_familiarity_features",
                hypothesis="Regular-season familiarity can reshape tournament rematch outcomes.",
                extend_iteration_38=build_pair_alias_builder(
                    current.build_pooled_regular_season_pair_features,
                    {
                        "regular_season_meetings": "rematch_meeting_count",
                        "lower_regular_season_win_pct": "rematch_lower_win_pct",
                        "lower_regular_season_avg_margin": "rematch_lower_avg_margin",
                        "latest_lower_regular_season_margin": "rematch_latest_lower_margin",
                        "last_regular_season_meeting_daynum": "rematch_last_meeting_daynum",
                        "regular_season_neutral_meeting_share": "rematch_neutral_share",
                        "regular_season_overtime_meeting_share": "rematch_overtime_share",
                        "regular_season_close_meeting_share": "rematch_close_share",
                        "days_since_last_meeting": "rematch_days_since_last_meeting",
                        "split_regular_season_series_flag": "rematch_split_series_flag",
                    },
                    fill_defaults={"rematch_days_since_last_meeting": float(current.SELECTION_SUNDAY_DAYNUM)},
                ),
            ),
            family_driver.FamilySpec(
                name="Venue familiarity from repeated game-city exposure",
                artifact_slug="venue_familiarity_from_repeated_game_city_exposure",
                hypothesis="Repeated city exposure may proxy travel and venue comfort effects entering March.",
                extend_iteration_38=build_team_alias_builder(
                    current.build_pooled_team_city_features,
                    {
                        "unique_city_count": "venue_unique_city_count",
                        "max_city_game_count": "venue_max_city_game_count",
                        "avg_city_game_count": "venue_avg_city_game_count",
                        "avg_city_win_pct": "venue_avg_city_win_pct",
                        "avg_city_neutral_share": "venue_avg_city_neutral_share",
                        "repeat_city_exposure_score": "venue_repeat_city_exposure_score",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament participation and win-count features",
                artifact_slug="conference_tournament_participation_and_win_count_features",
                hypothesis="Conference tournament participation and wins may capture bid-pressure form entering Selection Sunday.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_team_features,
                    {
                        "conf_tourney_games": "conf_tourney_participation_games",
                        "conf_tourney_wins": "conf_tourney_participation_wins",
                        "conf_tourney_last_daynum": "conf_tourney_participation_last_day",
                        "conf_tourney_days_before_selection": "conf_tourney_participation_days_before_selection",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament champion and finalist flags",
                artifact_slug="conference_tournament_champion_and_finalist_flags",
                hypothesis="Conference tournament title and finalist signals may carry fresh information into the bracket.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_team_features,
                    {
                        "conf_tourney_champion_flag": "conf_tourney_champion_flag_current",
                        "conf_tourney_finalist_flag": "conf_tourney_finalist_flag_current",
                        "conf_tourney_games": "conf_tourney_games_current",
                        "conf_tourney_days_before_selection": "conf_tourney_days_before_selection_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament margin and dominance features",
                artifact_slug="conference_tournament_margin_and_dominance_features",
                hypothesis="Conference tournament dominance may matter more than raw participation counts.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_team_features,
                    {
                        "conf_tourney_wins": "conf_tourney_dominance_wins",
                        "conf_tourney_avg_opp_quality": "conf_tourney_dominance_avg_opp_quality",
                        "conf_tourney_max_opp_quality": "conf_tourney_dominance_max_opp_quality",
                        "conf_tourney_champion_flag": "conf_tourney_dominance_champion_flag",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference-relative percentile and z-score features",
                artifact_slug="conference_relative_percentile_and_z_score_features",
                hypothesis="Conference-relative ranks may normalize across different league environments.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "conference_margin_percentile": "conference_relative_margin_percentile",
                        "conference_margin_zscore": "conference_relative_margin_zscore",
                        "conference_offense_zscore": "conference_relative_offense_zscore",
                        "above_average_margin_flag": "conference_relative_above_average_flag",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Interconference regular-season strength features",
                artifact_slug="interconference_regular_season_strength_features",
                hypothesis="Interconference results may calibrate cross-league strength better than overall schedule summaries.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "nonconference_win_pct": "interconference_win_pct",
                        "nonconference_avg_margin": "interconference_margin",
                        "nonconference_off_rating": "interconference_off_rating",
                        "nonconference_def_rating": "interconference_def_rating",
                        "high_quality_opponent_share": "interconference_high_quality_share",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Same-conference matchup and conference-pair interaction features",
                artifact_slug="same_conference_matchup_and_conference_pair_interaction_features",
                hypothesis="Conference-pair context may capture recurring stylistic edges and same-league familiarity.",
                extend_iteration_38=build_pair_alias_builder(
                    build_conference_pair_interaction_features,
                    {
                        "conference_pair_same_conf_flag": "conference_pair_same_conf_flag_current",
                        "conference_pair_interconf_flag": "conference_pair_interconf_flag_current",
                        "conference_pair_regular_season_meetings": "conference_pair_regular_season_meetings_current",
                        "conference_pair_split_series_flag": "conference_pair_split_series_flag_current",
                        "low_conf_prior_ncaa_win_rate": "conference_pair_low_prior_ncaa_win_rate",
                        "high_conf_prior_ncaa_win_rate": "conference_pair_high_prior_ncaa_win_rate",
                        "conference_pair_delta_prior_ncaa_win_rate": "conference_pair_delta_prior_ncaa_win_rate",
                        "low_conf_prior_bid_count_mean": "conference_pair_low_prior_bid_count_mean",
                        "high_conf_prior_bid_count_mean": "conference_pair_high_prior_bid_count_mean",
                        "conference_pair_delta_prior_bid_count_mean": "conference_pair_delta_prior_bid_count_mean",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Recent NCAA conference-performance prior features",
                artifact_slug="recent_ncaa_conference_performance_prior_features",
                hypothesis="Recent conference tournament performance in the NCAA field may carry across seasons.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_conference_history_features,
                    {
                        "prior_conference_ncaa_games": "conference_prior_ncaa_games",
                        "prior_conference_ncaa_wins": "conference_prior_ncaa_wins",
                        "prior_conference_ncaa_win_rate": "conference_prior_ncaa_win_rate",
                        "prior_conference_sweet16_teams": "conference_prior_sweet16_teams",
                        "prior_conference_final_four_teams": "conference_prior_final_four_teams",
                        "prior_conference_championships": "conference_prior_championships",
                    },
                ),
            ),
        ]
    )
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Men coach tenure and continuity features",
                artifact_slug="men_coach_tenure_and_continuity_features",
                hypothesis="Men coach continuity may proxy system stability not captured by team results alone.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_men_coach_history_features,
                    {
                        "coach_prior_team_seasons": "coach_continuity_prior_team_seasons",
                        "coach_prior_unique_team_count": "coach_continuity_prior_unique_team_count",
                        "coach_prior_team_tournament_appearances": "coach_continuity_prior_team_tournament_appearances",
                        "coach_coverage_days": "coach_continuity_coverage_days",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men midseason coaching-change features",
                artifact_slug="men_midseason_coaching_change_features",
                hypothesis="Men midseason coaching changes may create structural breaks in season-long summaries.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_men_coach_history_features,
                    {
                        "midseason_change_flag": "coach_change_midseason_flag",
                        "coach_stints_in_season": "coach_change_stints_in_season",
                        "coach_coverage_days": "coach_change_coverage_days",
                        "coach_prior_unique_team_count": "coach_change_prior_unique_team_count",
                    },
                    fill_defaults={
                        "coach_change_midseason_flag": 0.0,
                        "coach_change_stints_in_season": 1.0,
                        "coach_change_coverage_days": 0.0,
                        "coach_change_prior_unique_team_count": 0.0,
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men coach tournament-history prior features",
                artifact_slug="men_coach_tournament_history_prior_features",
                hypothesis="Men coaches bring reusable tournament priors that team-only models miss.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_men_coach_history_features,
                    {
                        "coach_prior_tournament_appearances": "coach_tourney_prior_appearances",
                        "coach_prior_tournament_games": "coach_tourney_prior_games",
                        "coach_prior_tournament_wins": "coach_tourney_prior_wins",
                        "coach_prior_sweet16_appearances": "coach_tourney_prior_sweet16",
                        "coach_prior_final_four_appearances": "coach_tourney_prior_final_four",
                        "coach_prior_championships": "coach_tourney_prior_championships",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men program Division-I tenure and transition-recency features",
                artifact_slug="men_program_division_i_tenure_and_transition_recency_features",
                hypothesis="Program D-I tenure may matter for younger transitional programs on the men's side.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "program_d1_tenure_seasons": "program_d1_tenure_seasons_current",
                        "recent_division_transition_flag": "program_recent_transition_flag",
                        "prior_ncaa_appearances": "program_transition_prior_ncaa_appearances",
                        "prior_tournament_games": "program_transition_prior_tournament_games",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Previous-season secondary-tournament participation features",
                artifact_slug="previous_season_secondary_tournament_participation_features",
                hypothesis="Prior secondary-tournament participation may differentiate fringe postseason programs.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_secondary_participations": "secondary_prior_participations",
                        "secondary_participation_streak": "secondary_prior_participation_streak",
                        "recent_3y_secondary_participations": "secondary_recent_3y_participations",
                        "prior_secondary_games": "secondary_prior_games",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Previous-season secondary-tournament performance features",
                artifact_slug="previous_season_secondary_tournament_performance_features",
                hypothesis="Secondary-tournament performance may capture latent team quality for fringe bids.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_secondary_games": "secondary_performance_prior_games",
                        "prior_secondary_wins": "secondary_performance_prior_wins",
                        "prior_secondary_championships": "secondary_performance_prior_championships",
                        "prior_secondary_participations": "secondary_performance_prior_participations",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men Massey rank-trend-over-time features",
                artifact_slug="men_massey_rank_trend_over_time_features",
                hypothesis="Men Massey trend features may capture late-season improvement the current blend misses.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_massey_history_features,
                    {
                        "massey_latest_mean_rank": "massey_trend_latest_mean_rank",
                        "massey_rank_slope": "massey_trend_rank_slope",
                        "massey_rank_improvement": "massey_trend_rank_improvement",
                        "massey_late_mean_rank": "massey_trend_late_mean_rank",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men Massey late-season rank-volatility features",
                artifact_slug="men_massey_late_season_rank_volatility_features",
                hypothesis="Men late-season Massey volatility may identify unstable profiles entering the field.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_massey_history_features,
                    {
                        "massey_latest_rank_std": "massey_volatility_latest_rank_std",
                        "massey_late_rank_std": "massey_volatility_late_rank_std",
                        "massey_elite_rank_std": "massey_volatility_elite_rank_std",
                        "massey_latest_mean_rank": "massey_volatility_latest_mean_rank",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Seed-vs-Massey discrepancy features",
                artifact_slug="seed_vs_massey_discrepancy_features",
                hypothesis="Seed versus Massey disagreement may expose bracket mispricing after committee seeding.",
                extend_iteration_38=build_team_alias_builder(
                    build_seed_massey_discrepancy_features,
                    {
                        "seed_massey_seed_number": "seed_massey_seed_number_current",
                        "seed_massey_mean_gap": "seed_massey_mean_gap_current",
                        "seed_massey_best_gap": "seed_massey_best_gap_current",
                        "seed_massey_rank_dispersion": "seed_massey_rank_dispersion_current",
                        "seed_massey_overseed_flag": "seed_massey_overseed_flag_current",
                        "seed_massey_underseed_flag": "seed_massey_underseed_flag_current",
                    },
                ),
            ),
        ]
    )
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Current-season pace-estimate features",
                artifact_slug="current_season_pace_estimate_features",
                hypothesis="Current-season pace splits may sharpen matchup translation for tempo-sensitive teams.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("pace", "pace_feature"),
                    {
                        "pace_feature_overall": "pace_feature_overall",
                        "pace_feature_late": "pace_feature_late",
                        "pace_feature_neutral": "pace_feature_neutral",
                        "pace_feature_home_road_gap": "pace_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season offensive-rating features",
                artifact_slug="current_season_offensive_rating_features",
                hypothesis="Current offensive rating splits may improve absolute scoring-strength estimates.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("off_rating", "off_rating_feature"),
                    {
                        "off_rating_feature_overall": "off_rating_feature_overall",
                        "off_rating_feature_late": "off_rating_feature_late",
                        "off_rating_feature_neutral": "off_rating_feature_neutral",
                        "off_rating_feature_home_road_gap": "off_rating_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season defensive-rating features",
                artifact_slug="current_season_defensive_rating_features",
                hypothesis="Current defensive rating splits may carry more stable tournament signal than raw points allowed.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("def_rating", "def_rating_feature"),
                    {
                        "def_rating_feature_overall": "def_rating_feature_overall",
                        "def_rating_feature_late": "def_rating_feature_late",
                        "def_rating_feature_neutral": "def_rating_feature_neutral",
                        "def_rating_feature_home_road_gap": "def_rating_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season effective-field-goal features",
                artifact_slug="current_season_effective_field_goal_features",
                hypothesis="Effective field-goal splits may capture shot-quality and spacing advantages in portable form.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("effective_fg_pct", "effective_fg_feature"),
                    {
                        "effective_fg_feature_overall": "effective_fg_feature_overall",
                        "effective_fg_feature_late": "effective_fg_feature_late",
                        "effective_fg_feature_neutral": "effective_fg_feature_neutral",
                        "effective_fg_feature_home_road_gap": "effective_fg_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season true-shooting features",
                artifact_slug="current_season_true_shooting_features",
                hypothesis="True-shooting splits may encode shot-profile efficiency more robustly than raw percentages.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("true_shooting_pct", "true_shooting_feature"),
                    {
                        "true_shooting_feature_overall": "true_shooting_feature_overall",
                        "true_shooting_feature_late": "true_shooting_feature_late",
                        "true_shooting_feature_neutral": "true_shooting_feature_neutral",
                        "true_shooting_feature_home_road_gap": "true_shooting_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season two-point efficiency features",
                artifact_slug="current_season_two_point_efficiency_features",
                hypothesis="Two-point efficiency splits may help on teams that win through interior scoring.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("two_point_pct", "two_point_feature"),
                    {
                        "two_point_feature_overall": "two_point_feature_overall",
                        "two_point_feature_late": "two_point_feature_late",
                        "two_point_feature_neutral": "two_point_feature_neutral",
                        "two_point_feature_home_road_gap": "two_point_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season three-point efficiency features",
                artifact_slug="current_season_three_point_efficiency_features",
                hypothesis="Three-point efficiency splits may matter disproportionately in upset-prone games.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("three_point_pct", "three_point_feature"),
                    {
                        "three_point_feature_overall": "three_point_feature_overall",
                        "three_point_feature_late": "three_point_feature_late",
                        "three_point_feature_neutral": "three_point_feature_neutral",
                        "three_point_feature_home_road_gap": "three_point_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season free-throw accuracy features",
                artifact_slug="current_season_free_throw_accuracy_features",
                hypothesis="Free-throw accuracy splits may help in late-game tournament contexts.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("free_throw_pct", "free_throw_feature"),
                    {
                        "free_throw_feature_overall": "free_throw_feature_overall",
                        "free_throw_feature_late": "free_throw_feature_late",
                        "free_throw_feature_neutral": "free_throw_feature_neutral",
                        "free_throw_feature_home_road_gap": "free_throw_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season shot-mix composition features",
                artifact_slug="current_season_shot_mix_composition_features",
                hypothesis="Shot-mix composition may describe stylistic fit beyond pure efficiency numbers.",
                extend_iteration_38=build_team_alias_builder(
                    build_shot_mix_features,
                    {
                        "shot_mix_three_point_rate": "shot_mix_three_point_rate_current",
                        "shot_mix_two_point_rate": "shot_mix_two_point_rate_current",
                        "shot_mix_free_throw_rate": "shot_mix_free_throw_rate_current",
                        "shot_mix_perimeter_bias": "shot_mix_perimeter_bias_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season assist-rate features",
                artifact_slug="current_season_assist_rate_features",
                hypothesis="Assist-rate splits may capture ball movement quality and shot creation style.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("assist_rate", "assist_rate_feature"),
                    {
                        "assist_rate_feature_overall": "assist_rate_feature_overall",
                        "assist_rate_feature_late": "assist_rate_feature_late",
                        "assist_rate_feature_neutral": "assist_rate_feature_neutral",
                        "assist_rate_feature_home_road_gap": "assist_rate_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season assist-to-turnover ratio features",
                artifact_slug="current_season_assist_to_turnover_ratio_features",
                hypothesis="Assist-to-turnover splits may capture offensive control that persists in neutral settings.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("assist_to_turnover_ratio", "assist_to_feature"),
                    {
                        "assist_to_feature_overall": "assist_to_feature_overall",
                        "assist_to_feature_late": "assist_to_feature_late",
                        "assist_to_feature_neutral": "assist_to_feature_neutral",
                        "assist_to_feature_home_road_gap": "assist_to_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season turnover-rate features",
                artifact_slug="current_season_turnover_rate_features",
                hypothesis="Turnover-rate splits may matter more against postseason pressure defenses.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("turnover_rate", "turnover_feature"),
                    {
                        "turnover_feature_overall": "turnover_feature_overall",
                        "turnover_feature_late": "turnover_feature_late",
                        "turnover_feature_neutral": "turnover_feature_neutral",
                        "turnover_feature_home_road_gap": "turnover_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season steal-rate features",
                artifact_slug="current_season_steal_rate_features",
                hypothesis="Steal-rate splits may help identify defenses that create extra possessions in March.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("steal_rate", "steal_feature"),
                    {
                        "steal_feature_overall": "steal_feature_overall",
                        "steal_feature_late": "steal_feature_late",
                        "steal_feature_neutral": "steal_feature_neutral",
                        "steal_feature_home_road_gap": "steal_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season block-rate features",
                artifact_slug="current_season_block_rate_features",
                hypothesis="Block-rate splits may matter for interior defenses with tournament portability.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("block_rate", "block_feature"),
                    {
                        "block_feature_overall": "block_feature_overall",
                        "block_feature_late": "block_feature_late",
                        "block_feature_neutral": "block_feature_neutral",
                        "block_feature_home_road_gap": "block_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season foul-differential features",
                artifact_slug="current_season_foul_differential_features",
                hypothesis="Foul differential may proxy whistle leverage in slower postseason games.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("foul_diff_rate", "foul_feature"),
                    {
                        "foul_feature_overall": "foul_feature_overall",
                        "foul_feature_late": "foul_feature_late",
                        "foul_feature_neutral": "foul_feature_neutral",
                        "foul_feature_home_road_gap": "foul_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season offensive-rebound-share features",
                artifact_slug="current_season_offensive_rebound_share_features",
                hypothesis="Offensive rebounding splits may identify second-chance strength that survives bad shooting nights.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("offensive_rebound_share", "orb_feature"),
                    {
                        "orb_feature_overall": "orb_feature_overall",
                        "orb_feature_late": "orb_feature_late",
                        "orb_feature_neutral": "orb_feature_neutral",
                        "orb_feature_home_road_gap": "orb_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season defensive-rebound-share features",
                artifact_slug="current_season_defensive_rebound_share_features",
                hypothesis="Defensive rebounding splits may help on teams that close possessions reliably.",
                extend_iteration_38=build_team_alias_builder(
                    lambda: build_metric_split_features("defensive_rebound_share", "drb_feature"),
                    {
                        "drb_feature_overall": "drb_feature_overall",
                        "drb_feature_late": "drb_feature_late",
                        "drb_feature_neutral": "drb_feature_neutral",
                        "drb_feature_home_road_gap": "drb_feature_home_road_gap",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season possession-regime split features",
                artifact_slug="current_season_possession_regime_split_features",
                hypothesis="Fast-versus-slow performance splits may describe portability across tempo regimes.",
                extend_iteration_38=build_team_alias_builder(
                    build_possession_regime_features,
                    {
                        "pace_regime_fast_win_pct": "pace_regime_fast_win_pct_current",
                        "pace_regime_fast_margin": "pace_regime_fast_margin_current",
                        "pace_regime_fast_off_rating": "pace_regime_fast_off_rating_current",
                        "pace_regime_slow_win_pct": "pace_regime_slow_win_pct_current",
                        "pace_regime_slow_margin": "pace_regime_slow_margin_current",
                        "pace_regime_slow_def_rating": "pace_regime_slow_def_rating_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season detailed-stat trend-slope features",
                artifact_slug="current_season_detailed_stat_trend_slope_features",
                hypothesis="Detailed-stat trend slopes may capture momentum that simple late-season averages blur.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "off_rating_trend_slope": "trend_slope_off_rating",
                        "def_rating_trend_slope": "trend_slope_def_rating",
                        "effective_fg_trend_slope": "trend_slope_effective_fg",
                        "turnover_rate_trend_slope": "trend_slope_turnover_rate",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Current-season style-compatibility matchup features",
                artifact_slug="current_season_style_compatibility_matchup_features",
                hypothesis="Style-compatibility matchup features may capture interaction effects the base delta features miss.",
                extend_iteration_38=build_style_compatibility_builder(),
            ),
        ]
    )
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Record-versus-seeded-opponent features",
                artifact_slug="record_versus_seeded_opponent_features",
                hypothesis="Results versus seeded opponents may be a more tournament-relevant schedule split.",
                extend_iteration_38=build_team_alias_builder(
                    build_seeded_opponent_bucket_features,
                    {
                        "seeded_bucket_any_share": "seeded_record_any_share",
                        "seeded_bucket_any_win_pct": "seeded_record_any_win_pct",
                        "seeded_bucket_top8_share": "seeded_record_top8_share",
                        "seeded_bucket_top8_win_pct": "seeded_record_top8_win_pct",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Margin-versus-seeded-opponent features",
                artifact_slug="margin_versus_seeded_opponent_features",
                hypothesis="Margin versus seeded opponents may carry more resolution than seeded-opponent record alone.",
                extend_iteration_38=build_team_alias_builder(
                    build_seeded_opponent_bucket_features,
                    {
                        "seeded_bucket_any_margin": "seeded_margin_any_margin",
                        "seeded_bucket_top8_margin": "seeded_margin_top8_margin",
                        "seeded_bucket_any_share": "seeded_margin_any_share",
                        "seeded_bucket_top8_share": "seeded_margin_top8_share",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Seeded-opponent schedule-share features",
                artifact_slug="seeded_opponent_schedule_share_features",
                hypothesis="The mix of seeded opponents faced may improve schedule-quality calibration.",
                extend_iteration_38=build_team_alias_builder(
                    build_seeded_opponent_bucket_features,
                    {
                        "seeded_bucket_any_share": "seeded_share_any",
                        "seeded_bucket_top4_share": "seeded_share_top4",
                        "seeded_bucket_top8_share": "seeded_share_top8",
                        "seeded_bucket_top12_share": "seeded_share_top12",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Record-versus-top-Massey-opponent features",
                artifact_slug="record_versus_top_massey_opponent_features",
                hypothesis="Record versus top-Massey opponents may distinguish inflated records from tested contenders.",
                extend_iteration_38=build_team_alias_builder(
                    build_top_massey_bucket_features,
                    {
                        "massey_bucket_top10_share": "top_massey_record_top10_share",
                        "massey_bucket_top10_win_pct": "top_massey_record_top10_win_pct",
                        "massey_bucket_top25_share": "top_massey_record_top25_share",
                        "massey_bucket_top25_win_pct": "top_massey_record_top25_win_pct",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Margin-versus-top-Massey-opponent features",
                artifact_slug="margin_versus_top_massey_opponent_features",
                hypothesis="Margins versus top-Massey opponents may provide a stronger strength proxy than win-loss splits alone.",
                extend_iteration_38=build_team_alias_builder(
                    build_top_massey_bucket_features,
                    {
                        "massey_bucket_top10_margin": "top_massey_margin_top10_margin",
                        "massey_bucket_top25_margin": "top_massey_margin_top25_margin",
                        "massey_bucket_top10_share": "top_massey_margin_top10_share",
                        "massey_bucket_top25_share": "top_massey_margin_top25_share",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Top-Massey-opponent schedule-share features",
                artifact_slug="top_massey_opponent_schedule_share_features",
                hypothesis="The distribution of top-Massey opponents faced may better calibrate season difficulty.",
                extend_iteration_38=build_team_alias_builder(
                    build_top_massey_bucket_features,
                    {
                        "massey_bucket_top10_share": "top_massey_share_top10",
                        "massey_bucket_top25_share": "top_massey_share_top25",
                        "massey_bucket_top50_share": "top_massey_share_top50",
                        "massey_bucket_top25_win_pct": "top_massey_share_top25_win_pct",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Quality-win-count and bad-loss-count features",
                artifact_slug="quality_win_count_and_bad_loss_count_features",
                hypothesis="Quality-win and bad-loss counts may compress schedule-adjusted resume information into portable features.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "quality_win_count": "quality_resume_win_count",
                        "bad_loss_count": "quality_resume_bad_loss_count",
                        "quality_balance": "quality_resume_balance",
                        "high_quality_opponent_share": "quality_resume_high_quality_share",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Short-rest and compressed-turnaround features",
                artifact_slug="short_rest_and_compressed_turnaround_features",
                hypothesis="Short-rest and compressed-turnaround performance may matter in Thursday-Saturday and Saturday-Monday paths.",
                extend_iteration_38=build_team_alias_builder(
                    team_source,
                    {
                        "short_rest_share": "compressed_turnaround_short_rest_share",
                        "short_rest_win_pct": "compressed_turnaround_short_rest_win_pct",
                        "short_rest_avg_margin": "compressed_turnaround_short_rest_margin",
                        "back_to_back_share": "compressed_turnaround_back_to_back_share",
                        "games_per_week": "compressed_turnaround_games_per_week",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Selection-Sunday conference-finals exposure features",
                artifact_slug="selection_sunday_conference_finals_exposure_features",
                hypothesis="Conference-finals timing may proxy fatigue, rust, or public overreaction entering the field.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_team_features,
                    {
                        "conf_tourney_last_daynum": "selection_sunday_conf_last_daynum",
                        "conf_tourney_days_before_selection": "selection_sunday_conf_days_before_selection",
                        "conf_tourney_finalist_flag": "selection_sunday_conf_finalist_flag",
                        "conf_tourney_champion_flag": "selection_sunday_conf_champion_flag",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Seed-line historical advancement prior features",
                artifact_slug="seed_line_historical_advancement_prior_features",
                hypothesis="Historical performance by seed line may add a stable prior on expected advancement.",
                extend_iteration_38=build_team_alias_builder(
                    build_seed_line_history_team_features,
                    {
                        "seed_line_prior_mean_wins": "seed_line_prior_mean_wins_current",
                        "seed_line_prior_sweet16_rate": "seed_line_prior_sweet16_rate_current",
                        "seed_line_prior_final_four_rate": "seed_line_prior_final_four_rate_current",
                        "seed_line_prior_championship_rate": "seed_line_prior_championship_rate_current",
                    },
                ),
            ),
        ]
    )
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Seed-pair upset-rate prior features",
                artifact_slug="seed_pair_upset_rate_prior_features",
                hypothesis="Historical upset behavior at the exact seed-pair level may improve bracket priors.",
                extend_iteration_38=build_pair_alias_builder(
                    build_seed_pair_derived_features,
                    {
                        "seed_pair_prior_games": "seed_pair_prior_games_current",
                        "seed_pair_prior_lower_team_win_rate": "seed_pair_prior_lower_team_win_rate_current",
                        "seed_pair_prior_upset_rate": "seed_pair_prior_upset_rate_current",
                        "seed_gap_prior_games": "seed_pair_seed_gap_prior_games_current",
                        "seed_gap_prior_lower_team_win_rate": "seed_pair_seed_gap_prior_lower_team_win_rate_current",
                        "seed_pair_upset_rate_prior": "seed_pair_upset_rate_prior_current",
                        "slot_prior_games": "seed_pair_slot_prior_games_current",
                        "slot_prior_lower_team_win_rate": "seed_pair_slot_prior_lower_team_win_rate_current",
                        "path_chalkiness_prior": "seed_pair_path_chalkiness_prior_current",
                        "slot_prior_favorite_loss_rate": "seed_pair_slot_favorite_loss_rate_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Slot-specific historical upset prior features",
                artifact_slug="slot_specific_historical_upset_prior_features",
                hypothesis="Slot-context upset priors may help beyond generic seed-pair history.",
                extend_iteration_38=build_pair_alias_builder(
                    build_seed_pair_derived_features,
                    {
                        "slot_prior_games": "slot_prior_games_current",
                        "slot_prior_lower_team_win_rate": "slot_prior_lower_team_win_rate_current",
                        "slot_prior_favorite_loss_rate": "slot_prior_favorite_loss_rate_current",
                        "seed_pair_prior_games": "slot_seed_pair_prior_games_current",
                        "seed_pair_prior_lower_team_win_rate": "slot_seed_pair_lower_team_win_rate_current",
                        "seed_gap_prior_games": "slot_seed_gap_prior_games_current",
                        "seed_gap_prior_lower_team_win_rate": "slot_seed_gap_lower_team_win_rate_current",
                        "slot_upset_rate_prior": "slot_upset_rate_prior_current",
                        "path_chalkiness_prior": "slot_path_chalkiness_prior_current",
                        "seed_pair_upset_rate_prior": "slot_seed_pair_upset_rate_prior_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Path-chalkiness prior features",
                artifact_slug="path_chalkiness_prior_features",
                hypothesis="Historical chalkiness by bracket context may act as a useful prior on upset frequency.",
                extend_iteration_38=build_pair_alias_builder(
                    build_seed_pair_derived_features,
                    {
                        "path_chalkiness_prior": "path_chalkiness_prior_current",
                        "seed_pair_upset_rate_prior": "path_seed_pair_upset_rate_prior_current",
                        "slot_upset_rate_prior": "path_slot_upset_rate_prior_current",
                        "seed_pair_prior_games": "path_seed_pair_prior_games_current",
                        "slot_prior_games": "path_slot_prior_games_current",
                        "seed_gap_prior_games": "path_seed_gap_prior_games_current",
                        "seed_pair_prior_lower_team_win_rate": "path_seed_pair_lower_team_win_rate_current",
                        "slot_prior_lower_team_win_rate": "path_slot_lower_team_win_rate_current",
                        "slot_prior_favorite_loss_rate": "path_slot_favorite_loss_rate_current",
                        "seed_pair_prior_upset_rate": "path_seed_pair_prior_upset_rate_current",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Potential same-conference bracket-collision features",
                artifact_slug="potential_same_conference_bracket_collision_features",
                hypothesis="Same-conference collision context may matter when teams know each other unusually well.",
                extend_iteration_38=build_pair_alias_builder(
                    build_same_conference_pair_features,
                    {
                        "pair_same_conference_flag": "collision_same_conference_flag",
                        "pair_same_conference_pod_flag": "collision_same_conference_pod_flag",
                        "pair_same_conference_quadrant_flag": "collision_same_conference_quadrant_flag",
                        "pair_same_conference_round64_flag": "collision_same_conference_round64_flag",
                        "pair_same_conference_round32_flag": "collision_same_conference_round32_flag",
                        "pair_opening_slot_gap": "collision_opening_slot_gap",
                        "pair_regular_season_meetings": "collision_regular_season_meetings",
                        "pair_split_series_flag": "collision_split_series_flag",
                        "pair_days_since_last_meeting": "collision_days_since_last_meeting",
                        "pair_lower_regular_season_win_pct": "collision_lower_regular_season_win_pct",
                    },
                    fill_defaults={"collision_days_since_last_meeting": float(current.SELECTION_SUNDAY_DAYNUM)},
                ),
            ),
            family_driver.FamilySpec(
                name="Program NCAA appearance-streak features",
                artifact_slug="program_ncaa_appearance_streak_features",
                hypothesis="Program appearance streaks may capture institutional tournament baseline.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_ncaa_appearances": "program_appearance_prior_ncaa_appearances",
                        "prior_ncaa_appearance_streak": "program_appearance_prior_streak",
                        "recent_3y_ncaa_appearances": "program_appearance_recent_3y",
                        "prior_tournament_games": "program_appearance_prior_games",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Program seed-stability features",
                artifact_slug="program_seed_stability_features",
                hypothesis="Program seed stability may reflect durable program quality independent of this season's record.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_mean_seed": "program_seed_stability_prior_mean_seed",
                        "prior_best_seed": "program_seed_stability_prior_best_seed",
                        "prior_seed_std": "program_seed_stability_prior_seed_std",
                        "prior_seed_stability": "program_seed_stability_score",
                    },
                    fill_defaults={"program_seed_stability_prior_mean_seed": 17.0, "program_seed_stability_prior_best_seed": 17.0},
                ),
            ),
            family_driver.FamilySpec(
                name="Program historical seed outperformance features",
                artifact_slug="program_historical_seed_outperformance_features",
                hypothesis="Programs that historically outperform seed expectations may deserve a prior bump.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_seed_outperformance": "program_seed_outperformance_prior",
                        "prior_tournament_wins": "program_seed_outperformance_prior_wins",
                        "prior_final_four_appearances": "program_seed_outperformance_prior_final_four",
                        "prior_championships": "program_seed_outperformance_prior_championships",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Program underdog upset propensity features",
                artifact_slug="program_underdog_upset_propensity_features",
                hypothesis="Program-specific upset propensity may matter when seeded as an underdog.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_underdog_upset_rate": "program_underdog_upset_rate_prior",
                        "prior_tournament_games": "program_underdog_prior_tournament_games",
                        "prior_tournament_wins": "program_underdog_prior_tournament_wins",
                        "prior_ncaa_appearances": "program_underdog_prior_ncaa_appearances",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Program favorite hold-serve propensity features",
                artifact_slug="program_favorite_hold_serve_propensity_features",
                hypothesis="Programs that consistently hold serve as favorites may deserve extra prior credit.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_favorite_hold_rate": "program_favorite_hold_rate_prior",
                        "prior_tournament_games": "program_favorite_prior_tournament_games",
                        "prior_tournament_wins": "program_favorite_prior_tournament_wins",
                        "prior_mean_seed": "program_favorite_prior_mean_seed",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Program recent tournament depth features",
                artifact_slug="program_recent_tournament_depth_features",
                hypothesis="Recent program depth may still matter after current-season form is accounted for.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "recent_tournament_depth_score": "program_recent_depth_score",
                        "prior_sweet16_appearances": "program_recent_depth_prior_sweet16",
                        "prior_elite8_appearances": "program_recent_depth_prior_elite8",
                        "prior_final_four_appearances": "program_recent_depth_prior_final_four",
                    },
                ),
            ),
        ]
    )
    specs.extend(
        [
            family_driver.FamilySpec(
                name="Conference auto-bid-versus-at-large proxy features",
                artifact_slug="conference_auto_bid_versus_at_large_proxy_features",
                hypothesis="Conference bid-shape priors may help separate likely auto-bid profiles from at-large leagues.",
                extend_iteration_38=build_team_alias_builder(
                    build_auto_bid_proxy_features,
                    {
                        "auto_bid_proxy_champion_flag": "conference_auto_bid_champion_flag",
                        "auto_bid_proxy_finalist_flag": "conference_auto_bid_finalist_flag",
                        "auto_bid_proxy_prior_bid_count_mean": "conference_auto_bid_prior_bid_count_mean",
                        "auto_bid_proxy_prior_one_bid_rate": "conference_auto_bid_prior_one_bid_rate",
                        "auto_bid_proxy_at_large_support": "conference_auto_bid_at_large_support",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference seed-distribution strength features",
                artifact_slug="conference_seed_distribution_strength_features",
                hypothesis="Conference seed distribution priors may summarize the field strength surrounding each team.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_conference_history_features,
                    {
                        "prior_conference_bid_count_mean": "conference_seed_dist_prior_bid_count_mean",
                        "prior_conference_top4_seed_share_mean": "conference_seed_dist_prior_top4_share",
                        "prior_conference_top8_seed_share_mean": "conference_seed_dist_prior_top8_share",
                        "prior_conference_seed_std_mean": "conference_seed_dist_prior_seed_std",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference parity and top-heaviness features",
                artifact_slug="conference_parity_and_top_heaviness_features",
                hypothesis="Conference parity and top-heaviness priors may influence how robust each resume really is.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_conference_history_features,
                    {
                        "prior_conference_seed_std_mean": "conference_parity_prior_seed_std",
                        "prior_conference_top4_seed_share_mean": "conference_parity_prior_top4_share",
                        "prior_conference_top8_seed_share_mean": "conference_parity_prior_top8_share",
                        "prior_conference_bid_count_mean": "conference_parity_prior_bid_count_mean",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament upset-rate features",
                artifact_slug="conference_tournament_upset_rate_features",
                hypothesis="Conference tournament upset rates may reflect how reliable a league hierarchy really is.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_conference_features,
                    {
                        "conf_tourney_game_count": "conference_tourney_upset_game_count",
                        "conf_tourney_upset_rate": "conference_tourney_upset_rate_current",
                        "conf_tourney_avg_abs_margin": "conference_tourney_upset_avg_abs_margin",
                        "conf_tourney_avg_path_difficulty": "conference_tourney_upset_path_difficulty",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament close-game performance features",
                artifact_slug="conference_tournament_close_game_performance_features",
                hypothesis="Conference tournament close-game surfaces may add a league-level variance prior.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_conference_features,
                    {
                        "conf_tourney_game_count": "conference_tourney_close_game_count",
                        "conf_tourney_close_game_share": "conference_tourney_close_game_share_current",
                        "conf_tourney_avg_abs_margin": "conference_tourney_close_avg_abs_margin",
                        "conf_tourney_avg_path_difficulty": "conference_tourney_close_path_difficulty",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Conference tournament path-difficulty features",
                artifact_slug="conference_tournament_path_difficulty_features",
                hypothesis="Conference tournament path difficulty may proxy the quality of late-week tests before Selection Sunday.",
                extend_iteration_38=build_team_alias_builder(
                    build_conference_tourney_team_features,
                    {
                        "conf_tourney_avg_opp_quality": "conference_tourney_path_avg_opp_quality",
                        "conf_tourney_max_opp_quality": "conference_tourney_path_max_opp_quality",
                        "conf_tourney_games": "conference_tourney_path_games",
                        "conf_tourney_wins": "conference_tourney_path_wins",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Secondary-tournament championship-depth features",
                artifact_slug="secondary_tournament_championship_depth_features",
                hypothesis="Secondary-tournament success may separate fringe teams with postseason competence from empty resumes.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "prior_secondary_championships": "secondary_depth_prior_championships",
                        "prior_secondary_wins": "secondary_depth_prior_wins",
                        "prior_secondary_games": "secondary_depth_prior_games",
                        "prior_secondary_participations": "secondary_depth_prior_participations",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Secondary-tournament participation-streak features",
                artifact_slug="secondary_tournament_participation_streak_features",
                hypothesis="Secondary-tournament streaks may proxy the stability of the fringe-postseason tier.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_program_history_features,
                    {
                        "secondary_participation_streak": "secondary_streak_prior_streak",
                        "recent_3y_secondary_participations": "secondary_streak_recent_3y",
                        "prior_secondary_participations": "secondary_streak_prior_participations",
                        "prior_secondary_games": "secondary_streak_prior_games",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men Massey elite-system subset consensus features",
                artifact_slug="men_massey_elite_system_subset_consensus_features",
                hypothesis="Consensus across elite Massey systems may be more informative than all-system averages.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_massey_history_features,
                    {
                        "massey_elite_mean_rank": "massey_elite_consensus_mean_rank",
                        "massey_elite_rank_std": "massey_elite_consensus_rank_std",
                        "massey_elite_top10_share": "massey_elite_consensus_top10_share",
                        "massey_elite_top25_share": "massey_elite_consensus_top25_share",
                    },
                ),
            ),
            family_driver.FamilySpec(
                name="Men Massey top-N bucket-share features",
                artifact_slug="men_massey_top_n_bucket_share_features",
                hypothesis="Top-N bucket shares may capture how broadly supported a team is across ranking systems.",
                extend_iteration_38=build_team_alias_builder(
                    history.build_pooled_massey_history_features,
                    {
                        "massey_all_system_top10_share": "massey_topn_all_system_top10_share",
                        "massey_all_system_top25_share": "massey_topn_all_system_top25_share",
                        "massey_all_system_top50_share": "massey_topn_all_system_top50_share",
                        "massey_elite_top50_share": "massey_topn_elite_top50_share",
                    },
                ),
            ),
        ]
    )
    return specs


def write_sweep_summary(
    evaluations: list[family_driver.FamilyEvaluation],
    final_benchmark_state: family_driver.BenchmarkState,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_benchmark_name": "iteration_111_official_benchmark",
        "final_benchmark_name": final_benchmark_state.name,
        "final_mean_brier": final_benchmark_state.mean_brier,
        "final_fold_scores": {str(season): score for season, score in final_benchmark_state.fold_scores.items()},
        "accepted_family_names": list(final_benchmark_state.active_family_names),
        "evaluations": [
            {
                "name": evaluation.spec.name,
                "artifact_dir": str(evaluation.artifact_dir),
                "accepted": evaluation.accepted,
                "improved_seasons": list(evaluation.improved_seasons),
                "mean_brier": evaluation.validation_summary["mean_brier"],
                "iteration_38_added_columns": list(evaluation.iteration_38_added_columns),
                "iteration_49_added_columns": list(evaluation.iteration_49_added_columns),
            }
            for evaluation in evaluations
        ],
    }
    SUMMARY_PATH.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    family_specs = family_specifications()
    evaluations, final_benchmark_state = family_driver.run_family_sequence(
        family_specs=family_specs,
        artifact_root=ARTIFACT_DIR,
    )
    write_sweep_summary(evaluations, final_benchmark_state)

    accepted_count = sum(1 for evaluation in evaluations if evaluation.accepted)
    print(f"Completed {len(evaluations)} family evaluations.")
    print(f"Accepted families: {accepted_count}")
    print(f"Final benchmark mean Brier: {final_benchmark_state.mean_brier:.6f}")
    for evaluation in evaluations:
        print(
            f"{evaluation.spec.name}: "
            f"accepted={evaluation.accepted}, "
            f"mean_brier={evaluation.validation_summary['mean_brier']:.6f}, "
            f"improved_seasons={list(evaluation.improved_seasons)}"
        )


if __name__ == "__main__":
    main()
