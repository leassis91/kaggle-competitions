from __future__ import annotations

from functools import lru_cache

import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline


TEAM_KEY_COLUMNS = ["Season", "TeamID", "men_women"]
MATCHUP_KEY_COLUMNS = ["Season", "LowerTeamID", "HigherTeamID", "men_women"]
SELECTION_SUNDAY_DAYNUM = 132
CLOSE_GAME_MARGIN = 5.0
BLOWOUT_MARGIN = 15.0
SHORT_REST_DAYNUM_GAP = 2
LONG_REST_DAYNUM_GAP = 6
TOP_MASSEY_RANK = 25
BAD_MASSEY_RANK = 100


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr, default_value: float) -> pl.Expr:
    return pl.when(denominator > 0.0).then(numerator / denominator).otherwise(default_value)


def team_feature_output_columns(metric_names: list[str]) -> list[str]:
    output_columns: list[str] = []
    for metric_name in metric_names:
        output_columns.extend([f"low_{metric_name}", f"high_{metric_name}", f"delta_{metric_name}"])
    return output_columns


def fill_defaults_for_metrics(metric_names: list[str], default_value: float) -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for metric_name in metric_names:
        fill_defaults[f"low_{metric_name}"] = default_value
        fill_defaults[f"high_{metric_name}"] = default_value
        fill_defaults[f"delta_{metric_name}"] = 0.0
    return fill_defaults


def parse_seed_number(seed_value: str) -> int:
    return int(seed_value[1:3])


@lru_cache(maxsize=1)
def build_pooled_seed_features() -> pl.DataFrame:
    seed_frames = [
        pl.read_csv(config.seeds_path)
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
            pl.col("Seed").map_elements(parse_seed_number, return_dtype=pl.Int64).cast(pl.Float64).alias("seed_number"),
            pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
        )
        for config in baseline.DIVISION_CONFIGS
    ]
    return pl.concat(seed_frames, how="vertical_relaxed")


@lru_cache(maxsize=1)
def build_pooled_massey_snapshot() -> pl.DataFrame:
    return (
        pl.read_csv(baseline.DATA_DIR / "MMasseyOrdinals.csv")
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .with_columns(pl.col("RankingDayNum").max().over("Season").alias("latest_ranking_daynum"))
        .filter(pl.col("RankingDayNum") == pl.col("latest_ranking_daynum"))
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("OrdinalRank").mean().alias("massey_mean_rank"),
            pl.col("OrdinalRank").min().alias("massey_best_rank"),
            pl.col("OrdinalRank").std(ddof=0).fill_null(0.0).alias("massey_rank_std"),
            pl.len().cast(pl.Float64).alias("massey_system_count"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("men_women"))
    )


def build_division_regular_season_games(config: baseline.DivisionConfig) -> pl.DataFrame:
    detailed_results = pl.read_csv(config.regular_season_detailed_results_path).filter(
        pl.col("Season") >= baseline.MIN_POOLED_SEASON
    )
    cities = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}GameCities.csv").filter(pl.col("CRType") == "Regular")
    conferences = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}TeamConferences.csv").filter(
        pl.col("Season") >= baseline.MIN_POOLED_SEASON
    )
    seeds = build_pooled_seed_features().filter(pl.col("men_women") == config.men_women_flag)
    massey = build_pooled_massey_snapshot().filter(pl.col("men_women") == config.men_women_flag)

    regular_season_with_city = detailed_results.join(
        cities.select(
            pl.col("Season").cast(pl.Int64),
            pl.col("DayNum").cast(pl.Int64),
            pl.col("WTeamID").cast(pl.Int64),
            pl.col("LTeamID").cast(pl.Int64),
            pl.col("CityID").cast(pl.Int64),
        ),
        on=["Season", "DayNum", "WTeamID", "LTeamID"],
        how="left",
    )

    winner_perspective = regular_season_with_city.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64).alias("TeamID"),
        pl.col("LTeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
        pl.lit(1.0).alias("Win"),
        pl.col("WScore").cast(pl.Float64).alias("PointsFor"),
        pl.col("LScore").cast(pl.Float64).alias("PointsAgainst"),
        (pl.col("WScore") - pl.col("LScore")).cast(pl.Float64).alias("Margin"),
        pl.col("NumOT").cast(pl.Float64).alias("NumOT"),
        (pl.col("WLoc") == "H").cast(pl.Int8).alias("IsHome"),
        (pl.col("WLoc") == "A").cast(pl.Int8).alias("IsAway"),
        (pl.col("WLoc") == "N").cast(pl.Int8).alias("IsNeutral"),
        pl.col("CityID").cast(pl.Int64),
        pl.col("WFGM").cast(pl.Float64).alias("FGM"),
        pl.col("WFGA").cast(pl.Float64).alias("FGA"),
        pl.col("WFGM3").cast(pl.Float64).alias("FGM3"),
        pl.col("WFGA3").cast(pl.Float64).alias("FGA3"),
        pl.col("WFTM").cast(pl.Float64).alias("FTM"),
        pl.col("WFTA").cast(pl.Float64).alias("FTA"),
        pl.col("WOR").cast(pl.Float64).alias("OR"),
        pl.col("WDR").cast(pl.Float64).alias("DR"),
        pl.col("WAst").cast(pl.Float64).alias("Ast"),
        pl.col("WTO").cast(pl.Float64).alias("TO"),
        pl.col("WStl").cast(pl.Float64).alias("Stl"),
        pl.col("WBlk").cast(pl.Float64).alias("Blk"),
        pl.col("WPF").cast(pl.Float64).alias("PF"),
        pl.col("LFGM").cast(pl.Float64).alias("OppFGM"),
        pl.col("LFGA").cast(pl.Float64).alias("OppFGA"),
        pl.col("LFGM3").cast(pl.Float64).alias("OppFGM3"),
        pl.col("LFGA3").cast(pl.Float64).alias("OppFGA3"),
        pl.col("LFTM").cast(pl.Float64).alias("OppFTM"),
        pl.col("LFTA").cast(pl.Float64).alias("OppFTA"),
        pl.col("LOR").cast(pl.Float64).alias("OppOR"),
        pl.col("LDR").cast(pl.Float64).alias("OppDR"),
        pl.col("LAst").cast(pl.Float64).alias("OppAst"),
        pl.col("LTO").cast(pl.Float64).alias("OppTO"),
        pl.col("LStl").cast(pl.Float64).alias("OppStl"),
        pl.col("LBlk").cast(pl.Float64).alias("OppBlk"),
        pl.col("LPF").cast(pl.Float64).alias("OppPF"),
    )
    loser_perspective = regular_season_with_city.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64).alias("TeamID"),
        pl.col("WTeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
        pl.lit(0.0).alias("Win"),
        pl.col("LScore").cast(pl.Float64).alias("PointsFor"),
        pl.col("WScore").cast(pl.Float64).alias("PointsAgainst"),
        (pl.col("LScore") - pl.col("WScore")).cast(pl.Float64).alias("Margin"),
        pl.col("NumOT").cast(pl.Float64).alias("NumOT"),
        (pl.col("WLoc") == "A").cast(pl.Int8).alias("IsHome"),
        (pl.col("WLoc") == "H").cast(pl.Int8).alias("IsAway"),
        (pl.col("WLoc") == "N").cast(pl.Int8).alias("IsNeutral"),
        pl.col("CityID").cast(pl.Int64),
        pl.col("LFGM").cast(pl.Float64).alias("FGM"),
        pl.col("LFGA").cast(pl.Float64).alias("FGA"),
        pl.col("LFGM3").cast(pl.Float64).alias("FGM3"),
        pl.col("LFGA3").cast(pl.Float64).alias("FGA3"),
        pl.col("LFTM").cast(pl.Float64).alias("FTM"),
        pl.col("LFTA").cast(pl.Float64).alias("FTA"),
        pl.col("LOR").cast(pl.Float64).alias("OR"),
        pl.col("LDR").cast(pl.Float64).alias("DR"),
        pl.col("LAst").cast(pl.Float64).alias("Ast"),
        pl.col("LTO").cast(pl.Float64).alias("TO"),
        pl.col("LStl").cast(pl.Float64).alias("Stl"),
        pl.col("LBlk").cast(pl.Float64).alias("Blk"),
        pl.col("LPF").cast(pl.Float64).alias("PF"),
        pl.col("WFGM").cast(pl.Float64).alias("OppFGM"),
        pl.col("WFGA").cast(pl.Float64).alias("OppFGA"),
        pl.col("WFGM3").cast(pl.Float64).alias("OppFGM3"),
        pl.col("WFGA3").cast(pl.Float64).alias("OppFGA3"),
        pl.col("WFTM").cast(pl.Float64).alias("OppFTM"),
        pl.col("WFTA").cast(pl.Float64).alias("OppFTA"),
        pl.col("WOR").cast(pl.Float64).alias("OppOR"),
        pl.col("WDR").cast(pl.Float64).alias("OppDR"),
        pl.col("WAst").cast(pl.Float64).alias("OppAst"),
        pl.col("WTO").cast(pl.Float64).alias("OppTO"),
        pl.col("WStl").cast(pl.Float64).alias("OppStl"),
        pl.col("WBlk").cast(pl.Float64).alias("OppBlk"),
        pl.col("WPF").cast(pl.Float64).alias("OppPF"),
    )

    team_games = pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")
    team_conferences = conferences.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("TeamID").cast(pl.Int64),
        pl.col("ConfAbbrev").alias("team_conf_abbrev"),
    )
    opponent_conferences = conferences.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("TeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.col("ConfAbbrev").alias("opp_conf_abbrev"),
    )
    opponent_seed_features = seeds.select(
        pl.col("Season"),
        pl.col("TeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.col("seed_number").alias("opp_seed_number"),
    )
    opponent_massey_features = massey.select(
        pl.col("Season"),
        pl.col("TeamID").cast(pl.Int64).alias("OppTeamID"),
        pl.col("massey_mean_rank").alias("opp_massey_mean_rank"),
        pl.col("massey_best_rank").alias("opp_massey_best_rank"),
    )
    team_massey_features = massey.select(
        pl.col("Season"),
        pl.col("TeamID"),
        pl.col("massey_mean_rank"),
        pl.col("massey_best_rank"),
        pl.col("massey_rank_std"),
        pl.col("massey_system_count"),
    )
    team_seed_features = seeds.select(
        pl.col("Season"),
        pl.col("TeamID"),
        pl.col("seed_number"),
    )

    enriched_games = (
        team_games.join(team_conferences, on=["Season", "TeamID"], how="left")
        .join(opponent_conferences, on=["Season", "OppTeamID"], how="left")
        .join(team_seed_features, on=["Season", "TeamID"], how="left")
        .join(opponent_seed_features, on=["Season", "OppTeamID"], how="left")
        .join(team_massey_features, on=["Season", "TeamID"], how="left")
        .join(opponent_massey_features, on=["Season", "OppTeamID"], how="left")
        .sort(["Season", "TeamID", "DayNum", "OppTeamID"])
        .with_columns(
            (pl.col("FGA") - pl.col("OR") + pl.col("TO") + 0.475 * pl.col("FTA")).alias("team_possessions"),
            (pl.col("OppFGA") - pl.col("OppOR") + pl.col("OppTO") + 0.475 * pl.col("OppFTA")).alias("opp_possessions"),
        )
        .with_columns(
            (((pl.col("team_possessions") + pl.col("opp_possessions")) / 2.0) * 40.0 / (40.0 + 5.0 * pl.col("NumOT"))).alias(
                "pace"
            ),
            (100.0 * pl.col("PointsFor") / pl.col("team_possessions")).alias("off_rating"),
            (100.0 * pl.col("PointsAgainst") / pl.col("opp_possessions")).alias("def_rating"),
            (100.0 * pl.col("Margin") / ((pl.col("team_possessions") + pl.col("opp_possessions")) / 2.0)).alias(
                "net_rating"
            ),
            ((pl.col("FGM") + 0.5 * pl.col("FGM3")) / pl.col("FGA")).alias("effective_fg_pct"),
            ((pl.col("OppFGM") + 0.5 * pl.col("OppFGM3")) / pl.col("OppFGA")).alias("opp_effective_fg_pct"),
            (pl.col("PointsFor") / (2.0 * (pl.col("FGA") + 0.44 * pl.col("FTA")))).alias("true_shooting_pct"),
            (pl.col("PointsAgainst") / (2.0 * (pl.col("OppFGA") + 0.44 * pl.col("OppFTA")))).alias("opp_true_shooting_pct"),
            safe_ratio(pl.col("FGM") - pl.col("FGM3"), pl.col("FGA") - pl.col("FGA3"), 0.5).alias("two_point_pct"),
            safe_ratio(pl.col("OppFGM") - pl.col("OppFGM3"), pl.col("OppFGA") - pl.col("OppFGA3"), 0.5).alias(
                "opp_two_point_pct"
            ),
            safe_ratio(pl.col("FGM3"), pl.col("FGA3"), 0.3333).alias("three_point_pct"),
            safe_ratio(pl.col("OppFGM3"), pl.col("OppFGA3"), 0.3333).alias("opp_three_point_pct"),
            safe_ratio(pl.col("FTM"), pl.col("FTA"), 0.7).alias("free_throw_pct"),
            safe_ratio(pl.col("OppFTM"), pl.col("OppFTA"), 0.7).alias("opp_free_throw_pct"),
            safe_ratio(pl.col("FGA3"), pl.col("FGA"), 0.0).alias("three_point_attempt_rate"),
            safe_ratio(pl.col("FGA") - pl.col("FGA3"), pl.col("FGA"), 0.0).alias("two_point_attempt_rate"),
            safe_ratio(pl.col("FTA"), pl.col("FGA"), 0.0).alias("free_throw_rate"),
            safe_ratio(pl.col("Ast"), pl.col("FGM"), 0.0).alias("assist_rate"),
            safe_ratio(pl.col("OppAst"), pl.col("OppFGM"), 0.0).alias("opp_assist_rate"),
            safe_ratio(pl.col("Ast"), pl.col("TO"), 0.0).alias("assist_to_turnover_ratio"),
            safe_ratio(pl.col("TO"), pl.col("team_possessions"), 0.0).alias("turnover_rate"),
            safe_ratio(pl.col("Stl"), pl.col("opp_possessions"), 0.0).alias("steal_rate"),
            safe_ratio(pl.col("Blk"), pl.col("OppFGA") - pl.col("OppFGA3"), 0.0).alias("block_rate"),
            ((pl.col("OppPF") - pl.col("PF")) / ((pl.col("team_possessions") + pl.col("opp_possessions")) / 2.0)).alias(
                "foul_diff_rate"
            ),
            safe_ratio(pl.col("OR"), pl.col("OR") + pl.col("OppDR"), 0.0).alias("offensive_rebound_share"),
            safe_ratio(pl.col("DR"), pl.col("DR") + pl.col("OppOR"), 0.0).alias("defensive_rebound_share"),
            (pl.col("team_conf_abbrev") == pl.col("opp_conf_abbrev")).cast(pl.Int8).alias("is_conference_game"),
            (pl.col("Margin").abs() <= CLOSE_GAME_MARGIN).cast(pl.Int8).alias("is_close_game"),
            (pl.col("Margin") >= BLOWOUT_MARGIN).cast(pl.Int8).alias("is_blowout_win"),
            (pl.col("Margin") <= -BLOWOUT_MARGIN).cast(pl.Int8).alias("is_blowout_loss"),
            (pl.col("NumOT") > 0.0).cast(pl.Int8).alias("is_overtime_game"),
            pl.col("DayNum").diff().over(["Season", "TeamID"]).cast(pl.Float64).alias("day_gap"),
            pl.col("DayNum").rank("ordinal").over(["Season", "TeamID"]).cast(pl.Float64).alias("game_number"),
            pl.when(pl.col("opp_seed_number").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("faced_seeded_opponent"),
            pl.when(pl.col("opp_massey_mean_rank") <= TOP_MASSEY_RANK).then(1).otherwise(0).cast(pl.Int8).alias(
                "faced_top_massey_opponent"
            ),
            pl.when(pl.col("opp_massey_mean_rank") > BAD_MASSEY_RANK).then(1).otherwise(0).cast(pl.Int8).alias(
                "faced_bad_massey_opponent"
            ),
        )
        .with_columns(
            pl.when(pl.col("day_gap") <= 1.0).then(1).otherwise(0).cast(pl.Int8).alias("is_back_to_back"),
            pl.when(pl.col("day_gap") <= SHORT_REST_DAYNUM_GAP).then(1).otherwise(0).cast(pl.Int8).alias("is_short_rest"),
            pl.when(pl.col("day_gap") >= LONG_REST_DAYNUM_GAP).then(1).otherwise(0).cast(pl.Int8).alias("is_long_rest"),
            pl.when(pl.col("team_conf_abbrev").is_not_null())
            .then(pl.col("is_conference_game"))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("has_conference_context"),
            pl.when((pl.col("Win") == 1.0) & ((pl.col("faced_seeded_opponent") == 1) | (pl.col("faced_top_massey_opponent") == 1)))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_quality_win"),
            pl.when((pl.col("Win") == 0.0) & (pl.col("faced_bad_massey_opponent") == 1))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_bad_loss"),
        )
    )
    conference_start_days = (
        enriched_games.group_by(["Season", "TeamID", "men_women"])
        .agg(pl.col("DayNum").filter(pl.col("is_conference_game") == 1).min().alias("first_conference_day"))
    )
    quality_cutoffs = (
        enriched_games.group_by(["Season", "men_women"])
        .agg(
            pl.col("opp_massey_mean_rank").quantile(0.25).alias("top_tier_cutoff"),
            pl.col("opp_massey_mean_rank").quantile(0.75).alias("bottom_tier_cutoff"),
        )
    )
    return (
        enriched_games.join(conference_start_days, on=["Season", "TeamID", "men_women"], how="left")
        .join(quality_cutoffs, on=["Season", "men_women"], how="left")
        .with_columns(
            pl.when(pl.col("first_conference_day").is_not_null() & (pl.col("DayNum") >= pl.col("first_conference_day")))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_conference_phase"),
            pl.when(pl.col("first_conference_day").is_null() | (pl.col("DayNum") < pl.col("first_conference_day")))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_preconference_phase"),
            pl.when(pl.col("opp_massey_mean_rank") <= pl.col("top_tier_cutoff")).then(1).otherwise(0).cast(pl.Int8).alias(
                "faced_top_tier_opponent"
            ),
            pl.when(pl.col("opp_massey_mean_rank") >= pl.col("bottom_tier_cutoff")).then(1).otherwise(0).cast(pl.Int8).alias(
                "faced_bottom_tier_opponent"
            ),
        )
    )


@lru_cache(maxsize=1)
def build_pooled_team_games() -> pl.DataFrame:
    return pl.concat(
        [build_division_regular_season_games(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def late_split_mean(metric_name: str, filter_expression: pl.Expr) -> pl.Expr:
    return pl.col(metric_name).filter(filter_expression).mean().fill_null(0.0)


@lru_cache(maxsize=1)
def build_pooled_team_feature_table() -> pl.DataFrame:
    team_games = build_pooled_team_games()
    team_aggregates = (
        team_games.group_by(TEAM_KEY_COLUMNS)
        .agg(
            pl.len().cast(pl.Float64).alias("games_played"),
            pl.col("Win").mean().alias("win_pct"),
            pl.col("Margin").mean().alias("avg_margin"),
            pl.col("Margin").std(ddof=0).fill_null(0.0).alias("margin_std"),
            pl.col("Margin").abs().mean().alias("avg_abs_margin"),
            pl.col("Margin").abs().std(ddof=0).fill_null(0.0).alias("abs_margin_std"),
            pl.col("day_gap").mean().fill_null(0.0).alias("avg_day_gap"),
            pl.col("day_gap").max().fill_null(0.0).alias("max_day_gap"),
            (7.0 * pl.len() / (pl.col("DayNum").max() - pl.col("DayNum").min() + 1.0)).alias("games_per_week"),
            pl.col("IsNeutral").mean().alias("neutral_game_share"),
            pl.col("Win").filter(pl.col("IsNeutral") == 1).mean().fill_null(0.5).alias("neutral_win_pct"),
            pl.col("Margin").filter(pl.col("IsNeutral") == 1).mean().fill_null(0.0).alias("neutral_avg_margin"),
            pl.col("off_rating").filter(pl.col("IsNeutral") == 1).mean().fill_null(0.0).alias("neutral_off_rating"),
            pl.col("def_rating").filter(pl.col("IsNeutral") == 1).mean().fill_null(0.0).alias("neutral_def_rating"),
            pl.col("Win").filter(pl.col("IsHome") == 1).mean().fill_null(0.5).alias("home_win_pct"),
            pl.col("Margin").filter(pl.col("IsHome") == 1).mean().fill_null(0.0).alias("home_avg_margin"),
            pl.col("off_rating").filter(pl.col("IsHome") == 1).mean().fill_null(0.0).alias("home_off_rating"),
            pl.col("def_rating").filter(pl.col("IsHome") == 1).mean().fill_null(0.0).alias("home_def_rating"),
            pl.col("Win").filter(pl.col("IsAway") == 1).mean().fill_null(0.5).alias("road_win_pct"),
            pl.col("Margin").filter(pl.col("IsAway") == 1).mean().fill_null(0.0).alias("road_avg_margin"),
            pl.col("off_rating").filter(pl.col("IsAway") == 1).mean().fill_null(0.0).alias("road_off_rating"),
            pl.col("def_rating").filter(pl.col("IsAway") == 1).mean().fill_null(0.0).alias("road_def_rating"),
            pl.col("Win").filter(pl.col("is_conference_game") == 1).mean().fill_null(0.5).alias("conference_win_pct"),
            pl.col("Margin").filter(pl.col("is_conference_game") == 1).mean().fill_null(0.0).alias("conference_avg_margin"),
            pl.col("off_rating").filter(pl.col("is_conference_game") == 1).mean().fill_null(0.0).alias("conference_off_rating"),
            pl.col("def_rating").filter(pl.col("is_conference_game") == 1).mean().fill_null(0.0).alias("conference_def_rating"),
            pl.col("Win").filter(pl.col("is_conference_game") == 0).mean().fill_null(0.5).alias("nonconference_win_pct"),
            pl.col("Margin").filter(pl.col("is_conference_game") == 0).mean().fill_null(0.0).alias("nonconference_avg_margin"),
            pl.col("off_rating").filter(pl.col("is_conference_game") == 0).mean().fill_null(0.0).alias("nonconference_off_rating"),
            pl.col("def_rating").filter(pl.col("is_conference_game") == 0).mean().fill_null(0.0).alias("nonconference_def_rating"),
            pl.col("Win").filter(pl.col("is_preconference_phase") == 1).mean().fill_null(0.5).alias("preconference_win_pct"),
            pl.col("Margin").filter(pl.col("is_preconference_phase") == 1).mean().fill_null(0.0).alias(
                "preconference_avg_margin"
            ),
            pl.col("off_rating").filter(pl.col("is_preconference_phase") == 1).mean().fill_null(0.0).alias(
                "preconference_off_rating"
            ),
            pl.col("def_rating").filter(pl.col("is_preconference_phase") == 1).mean().fill_null(0.0).alias(
                "preconference_def_rating"
            ),
            pl.col("Win").filter(pl.col("is_conference_phase") == 1).mean().fill_null(0.5).alias("conference_phase_win_pct"),
            pl.col("Margin").filter(pl.col("is_conference_phase") == 1).mean().fill_null(0.0).alias(
                "conference_phase_avg_margin"
            ),
            pl.col("off_rating").filter(pl.col("is_conference_phase") == 1).mean().fill_null(0.0).alias(
                "conference_phase_off_rating"
            ),
            pl.col("def_rating").filter(pl.col("is_conference_phase") == 1).mean().fill_null(0.0).alias(
                "conference_phase_def_rating"
            ),
            pl.col("day_gap").mean().fill_null(0.0).alias("avg_rest_gap"),
            pl.col("is_back_to_back").mean().alias("back_to_back_share"),
            pl.col("is_short_rest").mean().alias("short_rest_share"),
            pl.col("is_long_rest").mean().alias("long_rest_share"),
            pl.col("Win").filter(pl.col("is_short_rest") == 1).mean().fill_null(0.5).alias("short_rest_win_pct"),
            pl.col("Margin").filter(pl.col("is_short_rest") == 1).mean().fill_null(0.0).alias("short_rest_avg_margin"),
            pl.col("Win").filter(pl.col("is_long_rest") == 1).mean().fill_null(0.5).alias("long_rest_win_pct"),
            pl.col("Margin").filter(pl.col("is_long_rest") == 1).mean().fill_null(0.0).alias("long_rest_avg_margin"),
            pl.col("is_short_rest").sum().cast(pl.Float64).alias("short_rest_game_count"),
            pl.col("is_overtime_game").mean().alias("overtime_game_share"),
            pl.col("Win").filter(pl.col("is_overtime_game") == 1).mean().fill_null(0.5).alias("overtime_win_pct"),
            pl.col("Margin").filter(pl.col("is_overtime_game") == 1).mean().fill_null(0.0).alias("overtime_avg_margin"),
            pl.col("NumOT").filter(pl.col("is_overtime_game") == 1).mean().fill_null(0.0).alias("avg_overtime_periods"),
            pl.col("is_close_game").mean().alias("close_game_share"),
            pl.col("Win").filter(pl.col("is_close_game") == 1).mean().fill_null(0.5).alias("close_game_win_pct"),
            pl.col("Margin").filter(pl.col("is_close_game") == 1).mean().fill_null(0.0).alias("close_game_avg_margin"),
            pl.col("net_rating").filter(pl.col("is_close_game") == 1).mean().fill_null(0.0).alias("close_game_net_rating"),
            pl.col("is_blowout_win").mean().alias("blowout_win_share"),
            pl.col("is_blowout_loss").mean().alias("blowout_loss_share"),
            (pl.col("is_blowout_win") + pl.col("is_blowout_loss")).mean().alias("blowout_game_share"),
            pl.col("Margin").filter((pl.col("is_blowout_win") == 1) | (pl.col("is_blowout_loss") == 1)).mean().fill_null(0.0).alias(
                "blowout_avg_margin"
            ),
            pl.col("pace").mean().alias("pace"),
            pl.col("pace").std(ddof=0).fill_null(0.0).alias("pace_std"),
            pl.col("off_rating").mean().alias("off_rating"),
            pl.col("def_rating").mean().alias("def_rating"),
            pl.col("net_rating").mean().alias("net_rating"),
            pl.col("effective_fg_pct").mean().alias("effective_fg_pct"),
            pl.col("opp_effective_fg_pct").mean().alias("opp_effective_fg_pct"),
            pl.col("true_shooting_pct").mean().alias("true_shooting_pct"),
            pl.col("opp_true_shooting_pct").mean().alias("opp_true_shooting_pct"),
            pl.col("two_point_pct").mean().alias("two_point_pct"),
            pl.col("opp_two_point_pct").mean().alias("opp_two_point_pct"),
            pl.col("three_point_pct").mean().alias("three_point_pct"),
            pl.col("opp_three_point_pct").mean().alias("opp_three_point_pct"),
            pl.col("free_throw_pct").mean().alias("free_throw_pct"),
            pl.col("opp_free_throw_pct").mean().alias("opp_free_throw_pct"),
            pl.col("three_point_attempt_rate").mean().alias("three_point_attempt_rate"),
            pl.col("two_point_attempt_rate").mean().alias("two_point_attempt_rate"),
            pl.col("free_throw_rate").mean().alias("free_throw_rate"),
            pl.col("assist_rate").mean().alias("assist_rate"),
            pl.col("opp_assist_rate").mean().alias("opp_assist_rate"),
            pl.col("assist_to_turnover_ratio").mean().alias("assist_to_turnover_ratio"),
            pl.col("turnover_rate").mean().alias("turnover_rate"),
            pl.col("steal_rate").mean().alias("steal_rate"),
            pl.col("block_rate").mean().alias("block_rate"),
            pl.col("foul_diff_rate").mean().alias("foul_diff_rate"),
            pl.col("offensive_rebound_share").mean().alias("offensive_rebound_share"),
            pl.col("defensive_rebound_share").mean().alias("defensive_rebound_share"),
            pl.col("off_rating").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_off_rating"),
            pl.col("def_rating").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_def_rating"),
            pl.col("effective_fg_pct").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_effective_fg_pct"),
            pl.col("true_shooting_pct").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_true_shooting_pct"),
            pl.col("two_point_pct").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_two_point_pct"),
            pl.col("three_point_pct").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_three_point_pct"),
            pl.col("free_throw_pct").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_free_throw_pct"),
            pl.col("assist_rate").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_assist_rate"),
            pl.col("assist_to_turnover_ratio").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias(
                "late_assist_to_turnover_ratio"
            ),
            pl.col("turnover_rate").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_turnover_rate"),
            pl.col("steal_rate").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_steal_rate"),
            pl.col("block_rate").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias("late_block_rate"),
            pl.col("offensive_rebound_share").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias(
                "late_offensive_rebound_share"
            ),
            pl.col("defensive_rebound_share").filter(pl.col("DayNum") >= 118).mean().fill_null(0.0).alias(
                "late_defensive_rebound_share"
            ),
            pl.col("seed_number").min().alias("seed_number"),
            pl.col("massey_mean_rank").mean().fill_null(0.0).alias("massey_mean_rank"),
            pl.col("massey_best_rank").mean().fill_null(0.0).alias("massey_best_rank"),
            pl.col("massey_rank_std").mean().fill_null(0.0).alias("massey_rank_std"),
            pl.col("massey_system_count").mean().fill_null(0.0).alias("massey_system_count"),
            pl.col("faced_seeded_opponent").mean().alias("seeded_opponent_schedule_share"),
            pl.col("Win").filter(pl.col("faced_seeded_opponent") == 1).mean().fill_null(0.5).alias(
                "seeded_opponent_win_pct"
            ),
            pl.col("Margin").filter(pl.col("faced_seeded_opponent") == 1).mean().fill_null(0.0).alias(
                "seeded_opponent_avg_margin"
            ),
            pl.col("faced_top_massey_opponent").mean().alias("top_massey_schedule_share"),
            pl.col("Win").filter(pl.col("faced_top_massey_opponent") == 1).mean().fill_null(0.5).alias(
                "top_massey_win_pct"
            ),
            pl.col("Margin").filter(pl.col("faced_top_massey_opponent") == 1).mean().fill_null(0.0).alias(
                "top_massey_avg_margin"
            ),
            pl.col("faced_top_tier_opponent").mean().alias("top_tier_schedule_share"),
            pl.col("Win").filter(pl.col("faced_top_tier_opponent") == 1).mean().fill_null(0.5).alias("top_tier_win_pct"),
            pl.col("Margin").filter(pl.col("faced_top_tier_opponent") == 1).mean().fill_null(0.0).alias(
                "top_tier_avg_margin"
            ),
            pl.col("faced_bottom_tier_opponent").mean().alias("bottom_tier_schedule_share"),
            pl.col("Win").filter(pl.col("faced_bottom_tier_opponent") == 1).mean().fill_null(0.5).alias(
                "bottom_tier_win_pct"
            ),
            pl.col("Margin").filter(pl.col("faced_bottom_tier_opponent") == 1).mean().fill_null(0.0).alias(
                "bottom_tier_avg_margin"
            ),
            pl.col("is_quality_win").sum().cast(pl.Float64).alias("quality_win_count"),
            pl.col("is_bad_loss").sum().cast(pl.Float64).alias("bad_loss_count"),
            pl.col("DayNum").max().alias("last_regular_season_daynum"),
        )
        .with_columns(
            (pl.col("home_win_pct") - pl.col("road_win_pct")).alias("home_road_win_pct_gap"),
            (pl.col("home_avg_margin") - pl.col("road_avg_margin")).alias("home_road_margin_gap"),
            (pl.col("home_off_rating") - pl.col("road_off_rating")).alias("home_road_off_rating_gap"),
            (pl.col("road_def_rating") - pl.col("home_def_rating")).alias("home_road_defense_gap"),
            (pl.col("conference_win_pct") - pl.col("nonconference_win_pct")).alias("conference_nonconference_win_pct_gap"),
            (pl.col("conference_avg_margin") - pl.col("nonconference_avg_margin")).alias(
                "conference_nonconference_margin_gap"
            ),
            (pl.col("conference_off_rating") - pl.col("nonconference_off_rating")).alias(
                "conference_nonconference_off_rating_gap"
            ),
            (pl.col("nonconference_def_rating") - pl.col("conference_def_rating")).alias(
                "conference_nonconference_defense_gap"
            ),
            (pl.col("conference_phase_win_pct") - pl.col("preconference_win_pct")).alias("conference_phase_win_pct_delta"),
            (pl.col("conference_phase_avg_margin") - pl.col("preconference_avg_margin")).alias(
                "conference_phase_margin_delta"
            ),
            (pl.col("conference_phase_off_rating") - pl.col("preconference_off_rating")).alias(
                "conference_phase_off_rating_delta"
            ),
            (pl.col("preconference_def_rating") - pl.col("conference_phase_def_rating")).alias(
                "conference_phase_defense_delta"
            ),
            (pl.col("seeded_opponent_schedule_share") + pl.col("top_massey_schedule_share")).alias(
                "high_quality_opponent_share"
            ),
            (pl.col("quality_win_count") - pl.col("bad_loss_count")).alias("quality_balance"),
            (pl.col("pace").filter(pl.col("pace").is_not_null()).mean().over(["Season", "men_women"])).alias("season_avg_pace"),
            (pl.col("off_rating").mean().over(["Season", "men_women"])).alias("season_avg_off_rating"),
            (pl.col("def_rating").mean().over(["Season", "men_women"])).alias("season_avg_def_rating"),
            (pl.col("avg_margin").mean().over(["Season", "men_women"])).alias("season_avg_margin"),
        )
        .with_columns(
            (pl.col("pace") > pl.col("season_avg_pace")).cast(pl.Int8).alias("above_average_pace_flag"),
            (pl.col("off_rating") > pl.col("season_avg_off_rating")).cast(pl.Int8).alias("above_average_offense_flag"),
            (pl.col("def_rating") < pl.col("season_avg_def_rating")).cast(pl.Int8).alias("above_average_defense_flag"),
            (pl.col("avg_margin") > pl.col("season_avg_margin")).cast(pl.Int8).alias("above_average_margin_flag"),
        )
    )
    team_conferences = (
        build_pooled_team_games()
        .select("Season", "TeamID", "men_women", "team_conf_abbrev")
        .unique()
        .filter(pl.col("team_conf_abbrev").is_not_null())
    )
    conference_relative_metrics = (
        team_aggregates.join(team_conferences, on=TEAM_KEY_COLUMNS, how="left")
        .with_columns(
            pl.col("avg_margin").rank("average").over(["Season", "men_women", "team_conf_abbrev"]).alias(
                "conference_margin_rank"
            ),
            pl.len().over(["Season", "men_women", "team_conf_abbrev"]).alias("conference_team_count"),
            pl.col("avg_margin").mean().over(["Season", "men_women", "team_conf_abbrev"]).alias("conference_margin_mean"),
            pl.col("avg_margin").std(ddof=0).over(["Season", "men_women", "team_conf_abbrev"]).fill_null(0.0).alias(
                "conference_margin_std"
            ),
            pl.col("off_rating").mean().over(["Season", "men_women", "team_conf_abbrev"]).alias("conference_offense_mean"),
            pl.col("off_rating").std(ddof=0).over(["Season", "men_women", "team_conf_abbrev"]).fill_null(0.0).alias(
                "conference_offense_std"
            ),
        )
        .with_columns(
            safe_ratio(pl.col("conference_margin_rank") - 1.0, pl.col("conference_team_count") - 1.0, 0.0).alias(
                "conference_margin_percentile"
            ),
            safe_ratio(
                pl.col("avg_margin") - pl.col("conference_margin_mean"),
                pl.col("conference_margin_std"),
                0.0,
            ).alias("conference_margin_zscore"),
            safe_ratio(
                pl.col("off_rating") - pl.col("conference_offense_mean"),
                pl.col("conference_offense_std"),
                0.0,
            ).alias("conference_offense_zscore"),
        )
        .select(
            TEAM_KEY_COLUMNS
            + [
                "conference_margin_percentile",
                "conference_margin_zscore",
                "conference_offense_zscore",
            ]
        )
    )
    return (
        team_aggregates.join(conference_relative_metrics, on=TEAM_KEY_COLUMNS, how="left")
        .with_columns(
            pl.col("conference_margin_percentile").fill_null(0.0),
            pl.col("conference_margin_zscore").fill_null(0.0),
            pl.col("conference_offense_zscore").fill_null(0.0),
            safe_ratio(pl.col("late_off_rating") - pl.col("off_rating"), pl.lit(SELECTION_SUNDAY_DAYNUM), 0.0).alias(
                "off_rating_trend_slope"
            ),
            safe_ratio(pl.col("late_def_rating") - pl.col("def_rating"), pl.lit(SELECTION_SUNDAY_DAYNUM), 0.0).alias(
                "def_rating_trend_slope"
            ),
            safe_ratio(
                pl.col("late_effective_fg_pct") - pl.col("effective_fg_pct"),
                pl.lit(SELECTION_SUNDAY_DAYNUM),
                0.0,
            ).alias("effective_fg_trend_slope"),
            safe_ratio(
                pl.col("late_turnover_rate") - pl.col("turnover_rate"),
                pl.lit(SELECTION_SUNDAY_DAYNUM),
                0.0,
            ).alias("turnover_rate_trend_slope"),
        )
    )


def build_division_pair_results(config: baseline.DivisionConfig) -> pl.DataFrame:
    results = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv").filter(
        pl.col("Season") >= baseline.MIN_POOLED_SEASON
    )
    return results.select(
        pl.col("Season").cast(pl.Int64),
        pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
        pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
        (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Float64).alias("LowerWin"),
        pl.when(pl.col("WTeamID") < pl.col("LTeamID"))
        .then((pl.col("WScore") - pl.col("LScore")).cast(pl.Float64))
        .otherwise((pl.col("LScore") - pl.col("WScore")).cast(pl.Float64))
        .alias("LowerMargin"),
        pl.col("DayNum").cast(pl.Int64),
        (pl.col("WLoc") == "N").cast(pl.Int8).alias("IsNeutral"),
        (pl.col("NumOT") > 0).cast(pl.Int8).alias("IsOvertime"),
        ((pl.col("WScore") - pl.col("LScore")).abs() <= CLOSE_GAME_MARGIN).cast(pl.Int8).alias("IsClose"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )


@lru_cache(maxsize=1)
def build_pooled_regular_season_pair_features() -> pl.DataFrame:
    pair_results = pl.concat(
        [build_division_pair_results(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )
    return (
        pair_results.sort(MATCHUP_KEY_COLUMNS + ["DayNum"])
        .group_by(MATCHUP_KEY_COLUMNS)
        .agg(
            pl.len().cast(pl.Float64).alias("regular_season_meetings"),
            pl.col("LowerWin").mean().alias("lower_regular_season_win_pct"),
            pl.col("LowerMargin").mean().alias("lower_regular_season_avg_margin"),
            pl.col("LowerMargin").last().alias("latest_lower_regular_season_margin"),
            pl.col("DayNum").max().cast(pl.Float64).alias("last_regular_season_meeting_daynum"),
            pl.col("IsNeutral").mean().alias("regular_season_neutral_meeting_share"),
            pl.col("IsOvertime").mean().alias("regular_season_overtime_meeting_share"),
            pl.col("IsClose").mean().alias("regular_season_close_meeting_share"),
        )
        .with_columns(
            (SELECTION_SUNDAY_DAYNUM - pl.col("last_regular_season_meeting_daynum")).alias("days_since_last_meeting"),
            pl.when((pl.col("lower_regular_season_win_pct") > 0.0) & (pl.col("lower_regular_season_win_pct") < 1.0))
            .then(1.0)
            .otherwise(0.0)
            .alias("split_regular_season_series_flag"),
        )
    )


@lru_cache(maxsize=1)
def build_pooled_common_opponent_features() -> pl.DataFrame:
    opponent_results = (
        build_pooled_team_games()
        .group_by(["Season", "TeamID", "OppTeamID", "men_women"])
        .agg(
            pl.col("Win").mean().alias("opponent_win_pct"),
            pl.col("Margin").mean().alias("opponent_avg_margin"),
            pl.col("off_rating").mean().alias("opponent_off_rating"),
            pl.col("def_rating").mean().alias("opponent_def_rating"),
        )
    )
    low_team_features = opponent_results.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("OppTeamID"),
        pl.col("men_women"),
        pl.col("opponent_win_pct").alias("low_common_win_pct"),
        pl.col("opponent_avg_margin").alias("low_common_avg_margin"),
        pl.col("opponent_off_rating").alias("low_common_off_rating"),
        pl.col("opponent_def_rating").alias("low_common_def_rating"),
    )
    high_team_features = opponent_results.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("OppTeamID"),
        pl.col("men_women"),
        pl.col("opponent_win_pct").alias("high_common_win_pct"),
        pl.col("opponent_avg_margin").alias("high_common_avg_margin"),
        pl.col("opponent_off_rating").alias("high_common_off_rating"),
        pl.col("opponent_def_rating").alias("high_common_def_rating"),
    )
    return (
        low_team_features.join(
            high_team_features,
            on=["Season", "OppTeamID", "men_women"],
            how="inner",
        )
        .filter(pl.col("LowerTeamID") < pl.col("HigherTeamID"))
        .group_by(MATCHUP_KEY_COLUMNS)
        .agg(
            pl.len().cast(pl.Float64).alias("common_opponent_count"),
            pl.col("low_common_win_pct").mean().alias("lower_common_win_pct"),
            pl.col("high_common_win_pct").mean().alias("higher_common_win_pct"),
            pl.col("low_common_avg_margin").mean().alias("lower_common_avg_margin"),
            pl.col("high_common_avg_margin").mean().alias("higher_common_avg_margin"),
            pl.col("low_common_off_rating").mean().alias("lower_common_off_rating"),
            pl.col("high_common_off_rating").mean().alias("higher_common_off_rating"),
            pl.col("low_common_def_rating").mean().alias("lower_common_def_rating"),
            pl.col("high_common_def_rating").mean().alias("higher_common_def_rating"),
        )
        .with_columns(
            (pl.col("lower_common_win_pct") - pl.col("higher_common_win_pct")).alias("delta_common_win_pct"),
            (pl.col("lower_common_avg_margin") - pl.col("higher_common_avg_margin")).alias("delta_common_avg_margin"),
            (pl.col("lower_common_off_rating") - pl.col("higher_common_off_rating")).alias("delta_common_off_rating"),
            (pl.col("higher_common_def_rating") - pl.col("lower_common_def_rating")).alias("delta_common_defense_edge"),
        )
    )


@lru_cache(maxsize=1)
def build_pooled_team_city_features() -> pl.DataFrame:
    team_city_counts = (
        build_pooled_team_games()
        .filter(pl.col("CityID").is_not_null())
        .group_by(["Season", "TeamID", "CityID", "men_women"])
        .agg(
            pl.len().cast(pl.Float64).alias("city_game_count"),
            pl.col("Win").mean().alias("city_win_pct"),
            pl.col("IsNeutral").mean().alias("city_neutral_share"),
        )
    )
    return (
        team_city_counts.group_by(TEAM_KEY_COLUMNS)
        .agg(
            pl.len().cast(pl.Float64).alias("unique_city_count"),
            pl.col("city_game_count").max().alias("max_city_game_count"),
            pl.col("city_game_count").mean().alias("avg_city_game_count"),
            pl.col("city_win_pct").mean().alias("avg_city_win_pct"),
            pl.col("city_neutral_share").mean().alias("avg_city_neutral_share"),
            pl.col("city_game_count").filter(pl.col("city_game_count") > 1.0).sum().fill_null(0.0).alias(
                "repeat_city_game_count"
            ),
        )
        .with_columns(
            safe_ratio(pl.col("repeat_city_game_count"), pl.col("max_city_game_count") + pl.col("avg_city_game_count"), 0.0).alias(
                "repeat_city_exposure_score"
            )
        )
    )


def add_team_metric_features(
    matchups: pl.DataFrame,
    metric_names: list[str],
    fill_defaults: dict[str, float],
) -> tuple[pl.DataFrame, list[str]]:
    team_features = build_pooled_team_feature_table()
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
    matchup_frame = (
        matchups.join(low_features, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_features, on=["Season", "HigherTeamID", "men_women"], how="left")
        .with_columns(*[pl.col(column_name).fill_null(default_value) for column_name, default_value in fill_defaults.items() if column_name.startswith("low_") or column_name.startswith("high_")])
        .with_columns(
            *[
                (pl.col(f"low_{metric_name}") - pl.col(f"high_{metric_name}")).alias(f"delta_{metric_name}")
                for metric_name in metric_names
            ]
        )
    )
    return matchup_frame, team_feature_output_columns(metric_names)


def add_pair_metric_features(
    matchups: pl.DataFrame,
    pair_features: pl.DataFrame,
    pair_metric_names: list[str],
    fill_defaults: dict[str, float],
) -> tuple[pl.DataFrame, list[str]]:
    matchup_frame = (
        matchups.join(
            pair_features.select(*MATCHUP_KEY_COLUMNS, *pair_metric_names),
            on=MATCHUP_KEY_COLUMNS,
            how="left",
        )
        .with_columns(*[pl.col(column_name).fill_null(default_value) for column_name, default_value in fill_defaults.items()])
    )
    return matchup_frame, pair_metric_names
