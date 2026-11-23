from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

import pooled_bracket_features as bracket_features
import train_iteration_16_pooled_selection_sunday_baseline as baseline


MEN_COACHES_PATH = baseline.DATA_DIR / "MTeamCoaches.csv"
MEN_TEAMS_PATH = baseline.DATA_DIR / "MTeams.csv"
CONFERENCES_PATH = baseline.DATA_DIR / "Conferences.csv"
ELITE_MASSEY_SYSTEMS = (
    "COL",
    "DOK",
    "MAS",
    "MOR",
    "POM",
    "RTH",
    "SAG",
    "WLK",
    "WOL",
)
MASSEY_LATE_WINDOW_DAYS = 14
DEFAULT_TEAM_FILL = 0.0
DEFAULT_PAIR_FILL = 0.0


def _regular_season_compact_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv"


def _team_conference_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}TeamConferences.csv"


def _conference_tourney_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}ConferenceTourneyGames.csv"


def _secondary_results_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}SecondaryTourneyCompactResults.csv"


def _secondary_teams_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}SecondaryTourneyTeams.csv"


def _tournament_season_cutoffs(config: baseline.DivisionConfig) -> pl.DataFrame:
    return cast(
        pl.DataFrame,
        pl.scan_csv(config.tournament_results_path)
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .group_by("Season")
        .agg((pl.col("DayNum").min() - 1).cast(pl.Int64).alias("selection_sunday_daynum"))
        .collect(),
    )


def _season_team_frame(config: baseline.DivisionConfig) -> pl.DataFrame:
    results = pl.scan_csv(_regular_season_compact_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    winner_rows = results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
    )
    loser_rows = results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
    )
    return cast(
        pl.DataFrame,
        pl.concat([winner_rows, loser_rows], how="vertical_relaxed")
        .unique()
        .sort(["Season", "TeamID"])
        .with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))
        .collect(),
    )


def _empty_team_frame(feature_columns: Sequence[str]) -> pl.DataFrame:
    schema = {
        "Season": pl.Int64,
        "TeamID": pl.Int64,
        "men_women": pl.Int8,
        **{column: pl.Float64 for column in feature_columns},
    }
    return pl.DataFrame(schema=schema)


def _empty_pair_frame(feature_columns: Sequence[str]) -> pl.DataFrame:
    schema = {
        "Season": pl.Int64,
        "LowerTeamID": pl.Int64,
        "HigherTeamID": pl.Int64,
        "men_women": pl.Int8,
        **{column: pl.Float64 for column in feature_columns},
    }
    return pl.DataFrame(schema=schema)


def _team_history_rows(config: baseline.DivisionConfig) -> pl.DataFrame:
    tournament_results = pl.read_csv(config.tournament_results_path).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    seed_lookup = bracket_features.build_seed_team_features(config).select(
        "Season",
        "TeamID",
        pl.col("seed_number").cast(pl.Int64),
    )

    winner_rows = tournament_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("LTeamID").alias("OppTeamID").cast(pl.Int64),
        pl.lit(1).cast(pl.Int8).alias("is_win"),
    )
    loser_rows = tournament_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("WTeamID").alias("OppTeamID").cast(pl.Int64),
        pl.lit(0).cast(pl.Int8).alias("is_win"),
    )

    return (
        pl.concat([winner_rows, loser_rows], how="vertical_relaxed")
        .join(seed_lookup, on=["Season", "TeamID"], how="left")
        .join(
            seed_lookup.rename(
                {
                    "TeamID": "OppTeamID",
                    "seed_number": "opp_seed_number",
                }
            ),
            on=["Season", "OppTeamID"],
            how="left",
        )
        .with_columns(
            (pl.col("seed_number") < pl.col("opp_seed_number")).cast(pl.Int8).alias("is_favorite"),
            (pl.col("seed_number") > pl.col("opp_seed_number")).cast(pl.Int8).alias("is_underdog"),
        )
        .with_columns(
            (pl.col("is_win") * pl.col("is_favorite")).cast(pl.Int8).alias("favorite_win"),
            (pl.col("is_win") * pl.col("is_underdog")).cast(pl.Int8).alias("upset_win"),
        )
        .with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))
    )


def _team_tournament_summary(config: baseline.DivisionConfig) -> pl.DataFrame:
    seed_rows = bracket_features.build_seed_team_features(config).select(
        pl.col("Season").cast(pl.Int64),
        pl.col("TeamID").cast(pl.Int64),
        pl.col("seed_number").cast(pl.Int64),
    )
    tournament_games = _team_history_rows(config)
    game_summary = tournament_games.group_by(["Season", "TeamID", "men_women"]).agg(
        pl.len().cast(pl.Float64).alias("tournament_games"),
        pl.col("is_win").sum().cast(pl.Float64).alias("tournament_wins"),
        pl.col("is_favorite").sum().cast(pl.Float64).alias("favorite_games"),
        pl.col("favorite_win").sum().cast(pl.Float64).alias("favorite_wins"),
        pl.col("is_underdog").sum().cast(pl.Float64).alias("underdog_games"),
        pl.col("upset_win").sum().cast(pl.Float64).alias("underdog_wins"),
    )

    summary = seed_rows.join(game_summary, on=["Season", "TeamID"], how="left").with_columns(
        pl.col("tournament_games").fill_null(0.0),
        pl.col("tournament_wins").fill_null(0.0),
        pl.col("favorite_games").fill_null(0.0),
        pl.col("favorite_wins").fill_null(0.0),
        pl.col("underdog_games").fill_null(0.0),
        pl.col("underdog_wins").fill_null(0.0),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )

    return summary.with_columns(
        pl.lit(1.0).alias("ncaa_appearance"),
        (pl.col("tournament_wins") >= 2.0).cast(pl.Float64).alias("sweet16_appearance"),
        (pl.col("tournament_wins") >= 3.0).cast(pl.Float64).alias("elite8_appearance"),
        (pl.col("tournament_wins") >= 4.0).cast(pl.Float64).alias("final_four_appearance"),
        (pl.col("tournament_wins") >= 5.0).cast(pl.Float64).alias("title_game_appearance"),
        (pl.col("tournament_wins") >= 6.0).cast(pl.Float64).alias("championship"),
    )


def _seed_expectation_table(config: baseline.DivisionConfig) -> dict[int, dict[int, float]]:
    summary = _team_tournament_summary(config)
    seasons = sorted(summary["Season"].unique().to_list())
    expectation_by_season: dict[int, dict[int, float]] = {}

    for season in seasons:
        prior_summary = summary.filter(pl.col("Season") < season)
        if prior_summary.is_empty():
            expectation_by_season[season] = {}
            continue
        expectation_by_season[season] = {
            int(seed_number): float(mean_wins)
            for seed_number, mean_wins in prior_summary.group_by("seed_number")
            .agg(pl.col("tournament_wins").mean().alias("mean_wins"))
            .iter_rows()
        }

    return expectation_by_season


def _program_history_feature_rows(config: baseline.DivisionConfig) -> list[dict[str, float | int]]:
    team_seasons = _season_team_frame(config)
    tournament_summary = _team_tournament_summary(config)
    secondary_summary = _secondary_tournament_team_features(config)
    expectation_by_season = _seed_expectation_table(config)

    team_table = team_seasons.join(
        tournament_summary.select(
            "Season",
            "TeamID",
            "seed_number",
            "ncaa_appearance",
            "tournament_games",
            "tournament_wins",
            "sweet16_appearance",
            "elite8_appearance",
            "final_four_appearance",
            "title_game_appearance",
            "championship",
            "favorite_games",
            "favorite_wins",
            "underdog_games",
            "underdog_wins",
        ),
        on=["Season", "TeamID"],
        how="left",
    ).join(
        secondary_summary.select(
            "Season",
            "TeamID",
            "secondary_participations",
            "secondary_games",
            "secondary_wins",
            "secondary_championships",
        ),
        on=["Season", "TeamID"],
        how="left",
    ).with_columns(
        pl.col("seed_number").cast(pl.Float64),
        pl.col("ncaa_appearance").fill_null(0.0),
        pl.col("tournament_games").fill_null(0.0),
        pl.col("tournament_wins").fill_null(0.0),
        pl.col("sweet16_appearance").fill_null(0.0),
        pl.col("elite8_appearance").fill_null(0.0),
        pl.col("final_four_appearance").fill_null(0.0),
        pl.col("title_game_appearance").fill_null(0.0),
        pl.col("championship").fill_null(0.0),
        pl.col("favorite_games").fill_null(0.0),
        pl.col("favorite_wins").fill_null(0.0),
        pl.col("underdog_games").fill_null(0.0),
        pl.col("underdog_wins").fill_null(0.0),
        pl.col("secondary_participations").fill_null(0.0),
        pl.col("secondary_games").fill_null(0.0),
        pl.col("secondary_wins").fill_null(0.0),
        pl.col("secondary_championships").fill_null(0.0),
    )

    if config.prefix == "M":
        d1_metadata = pl.read_csv(MEN_TEAMS_PATH).select(
            pl.col("TeamID").cast(pl.Int64),
            pl.col("FirstD1Season").cast(pl.Int64),
            pl.col("LastD1Season").cast(pl.Int64),
        )
        team_table = team_table.join(d1_metadata, on="TeamID", how="left")
    else:
        team_table = team_table.with_columns(
            pl.lit(None).cast(pl.Int64).alias("FirstD1Season"),
            pl.lit(None).cast(pl.Int64).alias("LastD1Season"),
        )

    program_rows: list[dict[str, float | int]] = []

    for team_id in sorted(team_table["TeamID"].unique().to_list()):
        team_history = team_table.filter(pl.col("TeamID") == team_id).sort("Season")
        appearance_streak = 0
        secondary_streak = 0
        prior_seed_values: list[float] = []
        prior_ncaa_seasons: list[int] = []
        prior_secondary_seasons: list[int] = []
        prior_recent_depth: list[float] = []
        cumulative_appearance_count = 0.0
        cumulative_tournament_games = 0.0
        cumulative_tournament_wins = 0.0
        cumulative_sweet16_count = 0.0
        cumulative_elite8_count = 0.0
        cumulative_final_four_count = 0.0
        cumulative_title_game_count = 0.0
        cumulative_championship_count = 0.0
        cumulative_seed_outperformance = 0.0
        cumulative_favorite_games = 0.0
        cumulative_favorite_wins = 0.0
        cumulative_underdog_games = 0.0
        cumulative_underdog_wins = 0.0
        cumulative_secondary_participations = 0.0
        cumulative_secondary_games = 0.0
        cumulative_secondary_wins = 0.0
        cumulative_secondary_championships = 0.0

        for row in team_history.iter_rows(named=True):
            season = int(row["Season"])
            seed_number = row["seed_number"]
            first_d1_season = row["FirstD1Season"]
            current_seed_outperformance = 0.0
            if seed_number is not None:
                prior_seed_expectation = expectation_by_season.get(season, {}).get(int(seed_number), 0.0)
                current_seed_outperformance = float(row["tournament_wins"]) - prior_seed_expectation

            program_rows.append(
                {
                    "Season": season,
                    "TeamID": int(row["TeamID"]),
                    "men_women": int(row["men_women"]),
                    "program_d1_tenure_seasons": float(
                        season - int(first_d1_season) if first_d1_season is not None else 0.0
                    ),
                    "recent_division_transition_flag": float(
                        first_d1_season is not None and season - int(first_d1_season) <= 4
                    ),
                    "prior_ncaa_appearances": cumulative_appearance_count,
                    "prior_ncaa_appearance_streak": float(appearance_streak),
                    "recent_3y_ncaa_appearances": float(len([prior_season for prior_season in prior_ncaa_seasons if prior_season > season - 4])),
                    "prior_tournament_games": cumulative_tournament_games,
                    "prior_tournament_wins": cumulative_tournament_wins,
                    "prior_sweet16_appearances": cumulative_sweet16_count,
                    "prior_elite8_appearances": cumulative_elite8_count,
                    "prior_final_four_appearances": cumulative_final_four_count,
                    "prior_title_game_appearances": cumulative_title_game_count,
                    "prior_championships": cumulative_championship_count,
                    "prior_mean_seed": float(np.mean(prior_seed_values)) if prior_seed_values else 0.0,
                    "prior_best_seed": float(np.min(prior_seed_values)) if prior_seed_values else 0.0,
                    "prior_seed_std": float(np.std(prior_seed_values)) if prior_seed_values else 0.0,
                    "prior_seed_stability": float(1.0 / (1.0 + np.std(prior_seed_values))) if prior_seed_values else 0.0,
                    "prior_seed_outperformance": cumulative_seed_outperformance,
                    "prior_favorite_hold_rate": (
                        cumulative_favorite_wins / cumulative_favorite_games if cumulative_favorite_games > 0.0 else 0.0
                    ),
                    "prior_underdog_upset_rate": (
                        cumulative_underdog_wins / cumulative_underdog_games if cumulative_underdog_games > 0.0 else 0.0
                    ),
                    "recent_tournament_depth_score": float(sum(prior_recent_depth[-5:])),
                    "prior_secondary_participations": cumulative_secondary_participations,
                    "prior_secondary_games": cumulative_secondary_games,
                    "prior_secondary_wins": cumulative_secondary_wins,
                    "prior_secondary_championships": cumulative_secondary_championships,
                    "secondary_participation_streak": float(secondary_streak),
                    "recent_3y_secondary_participations": float(
                        len([prior_season for prior_season in prior_secondary_seasons if prior_season > season - 4])
                    ),
                }
            )

            cumulative_appearance_count += float(row["ncaa_appearance"])
            cumulative_tournament_games += float(row["tournament_games"])
            cumulative_tournament_wins += float(row["tournament_wins"])
            cumulative_sweet16_count += float(row["sweet16_appearance"])
            cumulative_elite8_count += float(row["elite8_appearance"])
            cumulative_final_four_count += float(row["final_four_appearance"])
            cumulative_title_game_count += float(row["title_game_appearance"])
            cumulative_championship_count += float(row["championship"])
            cumulative_seed_outperformance += current_seed_outperformance
            cumulative_favorite_games += float(row["favorite_games"])
            cumulative_favorite_wins += float(row["favorite_wins"])
            cumulative_underdog_games += float(row["underdog_games"])
            cumulative_underdog_wins += float(row["underdog_wins"])
            cumulative_secondary_participations += float(row["secondary_participations"])
            cumulative_secondary_games += float(row["secondary_games"])
            cumulative_secondary_wins += float(row["secondary_wins"])
            cumulative_secondary_championships += float(row["secondary_championships"])

            if seed_number is not None:
                prior_seed_values.append(float(seed_number))

            if float(row["ncaa_appearance"]) > 0.0:
                appearance_streak += 1
                prior_ncaa_seasons.append(season)
                prior_recent_depth.append(
                    float(row["sweet16_appearance"])
                    + float(row["elite8_appearance"])
                    + float(row["final_four_appearance"])
                    + float(row["championship"])
                )
            else:
                appearance_streak = 0

            if float(row["secondary_participations"]) > 0.0:
                secondary_streak += 1
                prior_secondary_seasons.append(season)
            else:
                secondary_streak = 0

    return program_rows


def _coach_history_rows() -> list[dict[str, float | int]]:
    coach_rows = pl.read_csv(MEN_COACHES_PATH).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    latest_coach = (
        coach_rows.sort(["Season", "TeamID", "LastDayNum", "FirstDayNum"])
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("CoachName").last().alias("coach_name"),
            pl.len().cast(pl.Float64).alias("coach_stints_in_season"),
            pl.col("FirstDayNum").min().cast(pl.Float64).alias("coach_first_daynum"),
            pl.col("LastDayNum").max().cast(pl.Float64).alias("coach_last_daynum"),
        )
        .with_columns(
            (pl.col("coach_stints_in_season") > 1.0).cast(pl.Float64).alias("midseason_change_flag"),
            (pl.col("coach_last_daynum") - pl.col("coach_first_daynum") + 1.0).alias("coach_coverage_days"),
        )
    )
    tournament_summary = _team_tournament_summary(baseline.DIVISION_CONFIGS[0]).select(
        "Season",
        "TeamID",
        "ncaa_appearance",
        "tournament_games",
        "tournament_wins",
        "sweet16_appearance",
        "final_four_appearance",
        "championship",
    )
    coach_table = latest_coach.join(tournament_summary, on=["Season", "TeamID"], how="left").with_columns(
        pl.col("ncaa_appearance").fill_null(0.0),
        pl.col("tournament_games").fill_null(0.0),
        pl.col("tournament_wins").fill_null(0.0),
        pl.col("sweet16_appearance").fill_null(0.0),
        pl.col("final_four_appearance").fill_null(0.0),
        pl.col("championship").fill_null(0.0),
        pl.lit(1).cast(pl.Int8).alias("men_women"),
    )

    coach_feature_rows: list[dict[str, float | int]] = []
    coach_state: dict[str, dict[str, float]] = {}
    coach_teams_seen: dict[str, set[int]] = {}
    team_coach_state: dict[tuple[int, str], dict[str, float]] = {}

    for row in coach_table.sort(["Season", "TeamID"]).iter_rows(named=True):
        coach_name = str(row["coach_name"])
        team_id = int(row["TeamID"])
        coach_entry = coach_state.setdefault(
            coach_name,
            {
                "prior_team_count": 0.0,
                "prior_tournament_appearances": 0.0,
                "prior_tournament_games": 0.0,
                "prior_tournament_wins": 0.0,
                "prior_sweet16_appearances": 0.0,
                "prior_final_four_appearances": 0.0,
                "prior_championships": 0.0,
            },
        )
        team_entry = team_coach_state.setdefault(
            (team_id, coach_name),
            {
                "prior_team_seasons": 0.0,
                "prior_team_tournament_appearances": 0.0,
            },
        )
        teams_seen = coach_teams_seen.setdefault(coach_name, set())

        coach_feature_rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": team_id,
                "men_women": 1,
                "coach_stints_in_season": float(row["coach_stints_in_season"]),
                "midseason_change_flag": float(row["midseason_change_flag"]),
                "coach_coverage_days": float(row["coach_coverage_days"]),
                "coach_prior_unique_team_count": float(coach_entry["prior_team_count"]),
                "coach_prior_team_seasons": float(team_entry["prior_team_seasons"]),
                "coach_prior_team_tournament_appearances": float(team_entry["prior_team_tournament_appearances"]),
                "coach_prior_tournament_appearances": float(coach_entry["prior_tournament_appearances"]),
                "coach_prior_tournament_games": float(coach_entry["prior_tournament_games"]),
                "coach_prior_tournament_wins": float(coach_entry["prior_tournament_wins"]),
                "coach_prior_sweet16_appearances": float(coach_entry["prior_sweet16_appearances"]),
                "coach_prior_final_four_appearances": float(coach_entry["prior_final_four_appearances"]),
                "coach_prior_championships": float(coach_entry["prior_championships"]),
            }
        )

        if team_id not in teams_seen:
            teams_seen.add(team_id)
            coach_entry["prior_team_count"] = float(coach_entry["prior_team_count"]) + 1.0
        team_entry["prior_team_seasons"] = float(team_entry["prior_team_seasons"]) + 1.0
        team_entry["prior_team_tournament_appearances"] = float(team_entry["prior_team_tournament_appearances"]) + float(
            row["ncaa_appearance"]
        )
        coach_entry["prior_tournament_appearances"] = float(coach_entry["prior_tournament_appearances"]) + float(
            row["ncaa_appearance"]
        )
        coach_entry["prior_tournament_games"] = float(coach_entry["prior_tournament_games"]) + float(row["tournament_games"])
        coach_entry["prior_tournament_wins"] = float(coach_entry["prior_tournament_wins"]) + float(row["tournament_wins"])
        coach_entry["prior_sweet16_appearances"] = float(coach_entry["prior_sweet16_appearances"]) + float(
            row["sweet16_appearance"]
        )
        coach_entry["prior_final_four_appearances"] = float(coach_entry["prior_final_four_appearances"]) + float(
            row["final_four_appearance"]
        )
        coach_entry["prior_championships"] = float(coach_entry["prior_championships"]) + float(row["championship"])

    return coach_feature_rows


def _secondary_tournament_team_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    team_entries = pl.read_csv(_secondary_teams_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    result_rows = pl.read_csv(_secondary_results_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)

    champion_rows = (
        result_rows.group_by(["Season", "SecondaryTourney", "WTeamID"])
        .agg(pl.len().cast(pl.Float64).alias("win_count"))
        .sort(["Season", "SecondaryTourney", "win_count", "WTeamID"], descending=[False, False, True, False])
        .group_by(["Season", "SecondaryTourney"])
        .agg(pl.col("WTeamID").first().alias("TeamID"))
        .with_columns(pl.lit(1.0).alias("secondary_championships"))
    )

    winner_rows = result_rows.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("SecondaryTourney"),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
        pl.lit(1.0).alias("secondary_wins"),
        pl.lit(1.0).alias("secondary_games"),
    )
    loser_rows = result_rows.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("SecondaryTourney"),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
        pl.lit(0.0).alias("secondary_wins"),
        pl.lit(1.0).alias("secondary_games"),
    )
    game_summary = pl.concat([winner_rows, loser_rows], how="vertical_relaxed").group_by(
        ["Season", "SecondaryTourney", "TeamID"]
    ).agg(
        pl.col("secondary_games").sum().alias("secondary_games"),
        pl.col("secondary_wins").sum().alias("secondary_wins"),
    )

    return (
        team_entries.select(
            pl.col("Season").cast(pl.Int64),
            pl.col("SecondaryTourney"),
            pl.col("TeamID").cast(pl.Int64),
        )
        .join(game_summary, on=["Season", "SecondaryTourney", "TeamID"], how="left")
        .join(
            champion_rows,
            on=["Season", "SecondaryTourney", "TeamID"],
            how="left",
        )
        .with_columns(
            pl.col("secondary_games").fill_null(0.0),
            pl.col("secondary_wins").fill_null(0.0),
            pl.col("TeamID").is_not_null().cast(pl.Float64).alias("secondary_participations"),
            pl.col("secondary_championships").fill_null(0.0),
            pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
        )
        .group_by(["Season", "TeamID", "men_women"])
        .agg(
            pl.col("secondary_participations").sum().alias("secondary_participations"),
            pl.col("secondary_games").sum().alias("secondary_games"),
            pl.col("secondary_wins").sum().alias("secondary_wins"),
            pl.col("secondary_championships").sum().alias("secondary_championships"),
        )
    )


def _conference_prior_rows(config: baseline.DivisionConfig) -> list[dict[str, float | int | str]]:
    team_conferences = pl.read_csv(_team_conference_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    tournament_summary = _team_tournament_summary(config)
    secondary_summary = _secondary_tournament_team_features(config)
    conference_tourney_games = pl.read_csv(_conference_tourney_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)

    conference_seed_summary = (
        team_conferences.join(
            tournament_summary.select(
                "Season",
                "TeamID",
                "seed_number",
                "ncaa_appearance",
                "tournament_games",
                "tournament_wins",
                "sweet16_appearance",
                "elite8_appearance",
                "final_four_appearance",
                "championship",
            ),
            on=["Season", "TeamID"],
            how="left",
        )
        .join(
            secondary_summary.select(
                "Season",
                "TeamID",
                "secondary_participations",
                "secondary_games",
                "secondary_wins",
                "secondary_championships",
            ),
            on=["Season", "TeamID"],
            how="left",
        )
        .with_columns(
            pl.col("seed_number").cast(pl.Float64),
            pl.col("ncaa_appearance").fill_null(0.0),
            pl.col("tournament_games").fill_null(0.0),
            pl.col("tournament_wins").fill_null(0.0),
            pl.col("sweet16_appearance").fill_null(0.0),
            pl.col("elite8_appearance").fill_null(0.0),
            pl.col("final_four_appearance").fill_null(0.0),
            pl.col("championship").fill_null(0.0),
            pl.col("secondary_participations").fill_null(0.0),
            pl.col("secondary_games").fill_null(0.0),
            pl.col("secondary_wins").fill_null(0.0),
            pl.col("secondary_championships").fill_null(0.0),
        )
        .group_by(["Season", "ConfAbbrev"])
        .agg(
            pl.col("ncaa_appearance").sum().alias("ncaa_bid_count"),
            pl.col("tournament_games").sum().alias("ncaa_tournament_games"),
            pl.col("tournament_wins").sum().alias("ncaa_tournament_wins"),
            pl.col("sweet16_appearance").sum().alias("ncaa_sweet16_teams"),
            pl.col("elite8_appearance").sum().alias("ncaa_elite8_teams"),
            pl.col("final_four_appearance").sum().alias("ncaa_final_four_teams"),
            pl.col("championship").sum().alias("ncaa_championships"),
            pl.col("seed_number").mean().alias("conference_mean_seed"),
            pl.col("seed_number").min().alias("conference_best_seed"),
            pl.col("seed_number").std(ddof=0).alias("conference_seed_std"),
            (pl.col("seed_number") <= 4.0).cast(pl.Float64).mean().alias("conference_top4_seed_share"),
            (pl.col("seed_number") <= 8.0).cast(pl.Float64).mean().alias("conference_top8_seed_share"),
            pl.col("secondary_participations").sum().alias("conference_secondary_participations"),
            pl.col("secondary_games").sum().alias("conference_secondary_games"),
            pl.col("secondary_wins").sum().alias("conference_secondary_wins"),
            pl.col("secondary_championships").sum().alias("conference_secondary_championships"),
        )
        .with_columns(
            pl.col("conference_seed_std").fill_null(0.0),
            pl.col("conference_top4_seed_share").fill_null(0.0),
            pl.col("conference_top8_seed_share").fill_null(0.0),
        )
    )

    conference_tourney_summary = (
        conference_tourney_games.group_by(["Season", "ConfAbbrev"])
        .agg(pl.len().cast(pl.Float64).alias("conference_tourney_games"))
        .with_columns((pl.col("conference_tourney_games") > 0.0).cast(pl.Float64).alias("conference_tourney_played"))
    )

    conference_summary = conference_seed_summary.join(
        conference_tourney_summary,
        on=["Season", "ConfAbbrev"],
        how="left",
    ).with_columns(
        pl.col("conference_tourney_games").fill_null(0.0),
        pl.col("conference_tourney_played").fill_null(0.0),
        (pl.col("ncaa_tournament_wins") / pl.col("ncaa_tournament_games")).fill_nan(0.0).fill_null(0.0).alias(
            "ncaa_tournament_win_rate"
        ),
        (pl.col("ncaa_bid_count") == 1.0).cast(pl.Float64).alias("one_bid_league_flag"),
    )

    conference_rows: list[dict[str, float | int | str]] = []

    for conf_abbrev in sorted(conference_summary["ConfAbbrev"].unique().to_list()):
        conf_history = conference_summary.filter(pl.col("ConfAbbrev") == conf_abbrev).sort("Season")
        prior_bid_counts: list[float] = []
        prior_seed_stds: list[float] = []
        prior_top4_shares: list[float] = []
        prior_top8_shares: list[float] = []
        cumulative_ncaa_games = 0.0
        cumulative_ncaa_wins = 0.0
        cumulative_sweet16 = 0.0
        cumulative_elite8 = 0.0
        cumulative_final_four = 0.0
        cumulative_championships = 0.0
        cumulative_secondary_participations = 0.0
        cumulative_secondary_wins = 0.0
        cumulative_secondary_championships = 0.0
        cumulative_conference_tourney_games = 0.0
        one_bid_season_count = 0.0

        for row in conf_history.iter_rows(named=True):
            conference_rows.append(
                {
                    "Season": int(row["Season"]),
                    "ConfAbbrev": str(conf_abbrev),
                    "men_women": int(config.men_women_flag),
                    "prior_conference_bid_count_mean": float(np.mean(prior_bid_counts)) if prior_bid_counts else 0.0,
                    "prior_conference_seed_std_mean": float(np.mean(prior_seed_stds)) if prior_seed_stds else 0.0,
                    "prior_conference_top4_seed_share_mean": float(np.mean(prior_top4_shares)) if prior_top4_shares else 0.0,
                    "prior_conference_top8_seed_share_mean": float(np.mean(prior_top8_shares)) if prior_top8_shares else 0.0,
                    "prior_conference_ncaa_games": cumulative_ncaa_games,
                    "prior_conference_ncaa_wins": cumulative_ncaa_wins,
                    "prior_conference_ncaa_win_rate": (
                        cumulative_ncaa_wins / cumulative_ncaa_games if cumulative_ncaa_games > 0.0 else 0.0
                    ),
                    "prior_conference_sweet16_teams": cumulative_sweet16,
                    "prior_conference_elite8_teams": cumulative_elite8,
                    "prior_conference_final_four_teams": cumulative_final_four,
                    "prior_conference_championships": cumulative_championships,
                    "prior_conference_secondary_participations": cumulative_secondary_participations,
                    "prior_conference_secondary_wins": cumulative_secondary_wins,
                    "prior_conference_secondary_championships": cumulative_secondary_championships,
                    "prior_conference_tourney_games": cumulative_conference_tourney_games,
                    "prior_conference_one_bid_rate": (
                        one_bid_season_count / len(prior_bid_counts) if prior_bid_counts else 0.0
                    ),
                }
            )

            prior_bid_counts.append(float(row["ncaa_bid_count"]))
            prior_seed_stds.append(float(row["conference_seed_std"]))
            prior_top4_shares.append(float(row["conference_top4_seed_share"]))
            prior_top8_shares.append(float(row["conference_top8_seed_share"]))
            cumulative_ncaa_games += float(row["ncaa_tournament_games"])
            cumulative_ncaa_wins += float(row["ncaa_tournament_wins"])
            cumulative_sweet16 += float(row["ncaa_sweet16_teams"])
            cumulative_elite8 += float(row["ncaa_elite8_teams"])
            cumulative_final_four += float(row["ncaa_final_four_teams"])
            cumulative_championships += float(row["ncaa_championships"])
            cumulative_secondary_participations += float(row["conference_secondary_participations"])
            cumulative_secondary_wins += float(row["conference_secondary_wins"])
            cumulative_secondary_championships += float(row["conference_secondary_championships"])
            cumulative_conference_tourney_games += float(row["conference_tourney_games"])
            one_bid_season_count += float(row["one_bid_league_flag"])

    return conference_rows


def _seed_pair_history_rows(config: baseline.DivisionConfig) -> list[dict[str, float | int]]:
    pair_context = bracket_features.build_pair_bracket_features(config)
    seed_features = bracket_features.build_seed_team_features(config).select(
        "Season",
        "TeamID",
        pl.col("seed_number").cast(pl.Int64),
    )
    tournament_results = pl.read_csv(config.tournament_results_path).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
    historical_results = (
        tournament_results.select(
            pl.col("Season").cast(pl.Int64),
            pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
            pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
            (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Float64).alias("lower_team_win"),
        )
        .join(pair_context, on=["Season", "LowerTeamID", "HigherTeamID"], how="left")
        .join(
            seed_features.rename(
                {
                    "TeamID": "LowerTeamID",
                    "seed_number": "current_low_seed_number",
                }
            ),
            on=["Season", "LowerTeamID"],
            how="left",
        )
        .join(
            seed_features.rename(
                {
                    "TeamID": "HigherTeamID",
                    "seed_number": "current_high_seed_number",
                }
            ),
            on=["Season", "HigherTeamID"],
            how="left",
        )
        .with_columns(
            (pl.col("current_low_seed_number") < pl.col("current_high_seed_number")).cast(pl.Int8).alias(
                "lower_team_is_seed_favorite"
            ),
            pl.when(pl.col("current_low_seed_number") == pl.col("current_high_seed_number"))
            .then(None)
            .otherwise(
                pl.when(pl.col("current_low_seed_number") < pl.col("current_high_seed_number"))
                .then(pl.col("lower_team_win"))
                .otherwise(1.0 - pl.col("lower_team_win"))
            )
            .alias("seed_favorite_win"),
        )
    )

    pair_rows: list[dict[str, float | int]] = []

    for season in sorted(pair_context["Season"].unique().to_list()):
        prior_history = historical_results.filter(pl.col("Season") < season)
        exact_seed_stats = prior_history.group_by(
            ["low_seed_number", "high_seed_number", "possible_round"]
        ).agg(
            pl.len().cast(pl.Float64).alias("seed_pair_prior_games"),
            pl.col("seed_favorite_win").mean().alias("seed_pair_favorite_win_rate"),
        )
        gap_stats = prior_history.group_by(["possible_round", "delta_seed_number"]).agg(
            pl.len().cast(pl.Float64).alias("seed_gap_prior_games"),
            pl.col("seed_favorite_win").mean().alias("seed_gap_favorite_win_rate"),
        )
        slot_stats = prior_history.group_by(["possible_round", "opening_slot_gap", "same_region_flag"]).agg(
            pl.len().cast(pl.Float64).alias("slot_prior_games"),
            pl.col("seed_favorite_win").mean().alias("slot_favorite_win_rate"),
        )
        season_pairs = pair_context.filter(pl.col("Season") == season)
        season_pair_features = (
            season_pairs.join(
                exact_seed_stats,
                on=["low_seed_number", "high_seed_number", "possible_round"],
                how="left",
            )
            .join(
                gap_stats,
                on=["possible_round", "delta_seed_number"],
                how="left",
            )
            .join(
                slot_stats,
                on=["possible_round", "opening_slot_gap", "same_region_flag"],
                how="left",
            )
            .with_columns(
                pl.col("seed_pair_prior_games").fill_null(0.0),
                pl.col("seed_pair_favorite_win_rate").fill_null(0.5),
                pl.col("seed_gap_prior_games").fill_null(0.0),
                pl.col("seed_gap_favorite_win_rate").fill_null(0.5),
                pl.col("slot_prior_games").fill_null(0.0),
                pl.col("slot_favorite_win_rate").fill_null(0.5),
            )
            .with_columns(
                (pl.col("low_seed_number") < pl.col("high_seed_number")).cast(pl.Int8).alias(
                    "lower_team_is_seed_favorite"
                )
            )
            .with_columns(
                pl.when(pl.col("low_seed_number") == pl.col("high_seed_number"))
                .then(0.5)
                .when(pl.col("lower_team_is_seed_favorite") == 1)
                .then(pl.col("seed_pair_favorite_win_rate"))
                .otherwise(1.0 - pl.col("seed_pair_favorite_win_rate"))
                .alias("seed_pair_prior_lower_team_win_rate"),
                pl.when(pl.col("low_seed_number") == pl.col("high_seed_number"))
                .then(0.5)
                .when(pl.col("lower_team_is_seed_favorite") == 1)
                .then(pl.col("seed_gap_favorite_win_rate"))
                .otherwise(1.0 - pl.col("seed_gap_favorite_win_rate"))
                .alias("seed_gap_prior_lower_team_win_rate"),
                pl.when(pl.col("low_seed_number") == pl.col("high_seed_number"))
                .then(0.5)
                .when(pl.col("lower_team_is_seed_favorite") == 1)
                .then(pl.col("slot_favorite_win_rate"))
                .otherwise(1.0 - pl.col("slot_favorite_win_rate"))
                .alias("slot_prior_lower_team_win_rate"),
                ((pl.col("seed_pair_favorite_win_rate") + pl.col("slot_favorite_win_rate")) / 2.0).alias(
                    "path_chalkiness_prior"
                ),
                (1.0 - pl.col("seed_pair_favorite_win_rate")).alias("seed_pair_upset_rate_prior"),
                (1.0 - pl.col("slot_favorite_win_rate")).alias("slot_upset_rate_prior"),
            )
            .with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))
        )
        pair_rows.extend(season_pair_features.select(
            "Season",
            "LowerTeamID",
            "HigherTeamID",
            "men_women",
            "seed_pair_prior_games",
            "seed_pair_prior_lower_team_win_rate",
            "seed_gap_prior_games",
            "seed_gap_prior_lower_team_win_rate",
            "slot_prior_games",
            "slot_prior_lower_team_win_rate",
            "path_chalkiness_prior",
            "seed_pair_upset_rate_prior",
            "slot_upset_rate_prior",
        ).iter_rows(named=True))

    return pair_rows


def _men_massey_history_features() -> pl.DataFrame:
    team_seasons = _season_team_frame(baseline.DIVISION_CONFIGS[0]).select("Season", "TeamID")
    cutoffs = _tournament_season_cutoffs(baseline.DIVISION_CONFIGS[0])
    ranking_rows = cast(
        pl.DataFrame,
        pl.scan_csv(baseline.DATA_DIR / "MMasseyOrdinals.csv")
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .join(cutoffs.lazy(), on="Season", how="inner")
        .filter(pl.col("RankingDayNum") <= pl.col("selection_sunday_daynum"))
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("RankingDayNum").cast(pl.Int64),
            pl.col("SystemName"),
            pl.col("TeamID").cast(pl.Int64),
            pl.col("OrdinalRank").cast(pl.Float64),
            pl.col("selection_sunday_daynum").cast(pl.Int64),
        )
        .collect(),
    )

    if ranking_rows.is_empty():
        return _empty_team_frame(
            [
                "massey_latest_mean_rank",
                "massey_latest_rank_std",
                "massey_rank_slope",
                "massey_rank_improvement",
                "massey_late_mean_rank",
                "massey_late_rank_std",
                "massey_elite_mean_rank",
                "massey_elite_rank_std",
                "massey_elite_top10_share",
                "massey_elite_top25_share",
                "massey_elite_top50_share",
                "massey_all_system_top10_share",
                "massey_all_system_top25_share",
                "massey_all_system_top50_share",
            ]
        )

    daily_mean = ranking_rows.group_by(["Season", "TeamID", "RankingDayNum"]).agg(
        pl.col("OrdinalRank").mean().alias("mean_rank"),
        pl.col("OrdinalRank").std(ddof=0).alias("rank_std"),
        pl.col("selection_sunday_daynum").first().alias("selection_sunday_daynum"),
    ).sort(["Season", "TeamID", "RankingDayNum"])

    slope_rows: list[dict[str, float | int]] = []
    for frame in daily_mean.partition_by(["Season", "TeamID"], maintain_order=True):
        season = int(frame["Season"][0])
        team_id = int(frame["TeamID"][0])
        day_values = frame["RankingDayNum"].to_numpy().astype(np.float64)
        mean_ranks = frame["mean_rank"].to_numpy().astype(np.float64)
        late_window_start = float(frame["selection_sunday_daynum"][0]) - float(MASSEY_LATE_WINDOW_DAYS)
        latest_mean_rank = float(mean_ranks[-1])
        latest_rank_std = float(frame["rank_std"].fill_null(0.0)[-1])
        centered_days = day_values - day_values.mean()
        denominator = float(np.sum(centered_days**2))
        rank_slope = float(np.sum(centered_days * (mean_ranks - mean_ranks.mean())) / denominator) if denominator > 0.0 else 0.0
        late_mask = day_values >= late_window_start
        late_ranks = mean_ranks[late_mask]
        late_mean_rank = float(np.mean(late_ranks)) if late_ranks.size > 0 else latest_mean_rank
        late_rank_std = float(np.std(late_ranks)) if late_ranks.size > 0 else 0.0
        slope_rows.append(
            {
                "Season": season,
                "TeamID": team_id,
                "massey_latest_mean_rank": latest_mean_rank,
                "massey_latest_rank_std": latest_rank_std,
                "massey_rank_slope": rank_slope,
                "massey_rank_improvement": float(mean_ranks[0] - latest_mean_rank),
                "massey_late_mean_rank": late_mean_rank,
                "massey_late_rank_std": late_rank_std,
            }
        )

    cutoff_days = ranking_rows.group_by("Season").agg(pl.col("RankingDayNum").max().alias("latest_ranking_daynum"))
    latest_ranks = ranking_rows.join(cutoff_days, on="Season", how="left").filter(
        pl.col("RankingDayNum") == pl.col("latest_ranking_daynum")
    )
    elite_ranks = latest_ranks.filter(pl.col("SystemName").is_in(list(ELITE_MASSEY_SYSTEMS)))

    all_system_summary = latest_ranks.group_by(["Season", "TeamID"]).agg(
        (pl.col("OrdinalRank") <= 10.0).cast(pl.Float64).mean().alias("massey_all_system_top10_share"),
        (pl.col("OrdinalRank") <= 25.0).cast(pl.Float64).mean().alias("massey_all_system_top25_share"),
        (pl.col("OrdinalRank") <= 50.0).cast(pl.Float64).mean().alias("massey_all_system_top50_share"),
    )
    elite_summary = elite_ranks.group_by(["Season", "TeamID"]).agg(
        pl.col("OrdinalRank").mean().alias("massey_elite_mean_rank"),
        pl.col("OrdinalRank").std(ddof=0).fill_null(0.0).alias("massey_elite_rank_std"),
        (pl.col("OrdinalRank") <= 10.0).cast(pl.Float64).mean().alias("massey_elite_top10_share"),
        (pl.col("OrdinalRank") <= 25.0).cast(pl.Float64).mean().alias("massey_elite_top25_share"),
        (pl.col("OrdinalRank") <= 50.0).cast(pl.Float64).mean().alias("massey_elite_top50_share"),
    )

    return (
        team_seasons.join(pl.DataFrame(slope_rows), on=["Season", "TeamID"], how="left")
        .join(all_system_summary, on=["Season", "TeamID"], how="left")
        .join(elite_summary, on=["Season", "TeamID"], how="left")
        .with_columns(
            pl.col("massey_latest_mean_rank").fill_null(0.0),
            pl.col("massey_latest_rank_std").fill_null(0.0),
            pl.col("massey_rank_slope").fill_null(0.0),
            pl.col("massey_rank_improvement").fill_null(0.0),
            pl.col("massey_late_mean_rank").fill_null(0.0),
            pl.col("massey_late_rank_std").fill_null(0.0),
            pl.col("massey_elite_mean_rank").fill_null(0.0),
            pl.col("massey_elite_rank_std").fill_null(0.0),
            pl.col("massey_elite_top10_share").fill_null(0.0),
            pl.col("massey_elite_top25_share").fill_null(0.0),
            pl.col("massey_elite_top50_share").fill_null(0.0),
            pl.col("massey_all_system_top10_share").fill_null(0.0),
            pl.col("massey_all_system_top25_share").fill_null(0.0),
            pl.col("massey_all_system_top50_share").fill_null(0.0),
            pl.lit(1).cast(pl.Int8).alias("men_women"),
        )
    )


PROGRAM_HISTORY_FEATURE_COLUMNS = [
    "program_d1_tenure_seasons",
    "recent_division_transition_flag",
    "prior_ncaa_appearances",
    "prior_ncaa_appearance_streak",
    "recent_3y_ncaa_appearances",
    "prior_tournament_games",
    "prior_tournament_wins",
    "prior_sweet16_appearances",
    "prior_elite8_appearances",
    "prior_final_four_appearances",
    "prior_title_game_appearances",
    "prior_championships",
    "prior_mean_seed",
    "prior_best_seed",
    "prior_seed_std",
    "prior_seed_stability",
    "prior_seed_outperformance",
    "prior_favorite_hold_rate",
    "prior_underdog_upset_rate",
    "recent_tournament_depth_score",
    "prior_secondary_participations",
    "prior_secondary_games",
    "prior_secondary_wins",
    "prior_secondary_championships",
    "secondary_participation_streak",
    "recent_3y_secondary_participations",
]
COACH_HISTORY_FEATURE_COLUMNS = [
    "coach_stints_in_season",
    "midseason_change_flag",
    "coach_coverage_days",
    "coach_prior_unique_team_count",
    "coach_prior_team_seasons",
    "coach_prior_team_tournament_appearances",
    "coach_prior_tournament_appearances",
    "coach_prior_tournament_games",
    "coach_prior_tournament_wins",
    "coach_prior_sweet16_appearances",
    "coach_prior_final_four_appearances",
    "coach_prior_championships",
]
CONFERENCE_HISTORY_FEATURE_COLUMNS = [
    "prior_conference_bid_count_mean",
    "prior_conference_seed_std_mean",
    "prior_conference_top4_seed_share_mean",
    "prior_conference_top8_seed_share_mean",
    "prior_conference_ncaa_games",
    "prior_conference_ncaa_wins",
    "prior_conference_ncaa_win_rate",
    "prior_conference_sweet16_teams",
    "prior_conference_elite8_teams",
    "prior_conference_final_four_teams",
    "prior_conference_championships",
    "prior_conference_secondary_participations",
    "prior_conference_secondary_wins",
    "prior_conference_secondary_championships",
    "prior_conference_tourney_games",
    "prior_conference_one_bid_rate",
]
SEED_PAIR_HISTORY_FEATURE_COLUMNS = [
    "seed_pair_prior_games",
    "seed_pair_prior_lower_team_win_rate",
    "seed_gap_prior_games",
    "seed_gap_prior_lower_team_win_rate",
    "slot_prior_games",
    "slot_prior_lower_team_win_rate",
    "path_chalkiness_prior",
    "seed_pair_upset_rate_prior",
    "slot_upset_rate_prior",
]
MASSEY_HISTORY_FEATURE_COLUMNS = [
    "massey_latest_mean_rank",
    "massey_latest_rank_std",
    "massey_rank_slope",
    "massey_rank_improvement",
    "massey_late_mean_rank",
    "massey_late_rank_std",
    "massey_elite_mean_rank",
    "massey_elite_rank_std",
    "massey_elite_top10_share",
    "massey_elite_top25_share",
    "massey_elite_top50_share",
    "massey_all_system_top10_share",
    "massey_all_system_top25_share",
    "massey_all_system_top50_share",
]


@lru_cache(maxsize=1)
def build_pooled_program_history_features() -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for config in baseline.DIVISION_CONFIGS:
        rows.extend(_program_history_feature_rows(config))
    return pl.DataFrame(rows) if rows else _empty_team_frame(PROGRAM_HISTORY_FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def build_men_coach_history_features() -> pl.DataFrame:
    rows = _coach_history_rows()
    return pl.DataFrame(rows) if rows else _empty_team_frame(COACH_HISTORY_FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def build_pooled_conference_history_features() -> pl.DataFrame:
    conference_rows: list[dict[str, float | int | str]] = []
    for config in baseline.DIVISION_CONFIGS:
        conference_rows.extend(_conference_prior_rows(config))

    if not conference_rows:
        return _empty_team_frame(CONFERENCE_HISTORY_FEATURE_COLUMNS)

    conference_history = pl.DataFrame(conference_rows)
    team_frames = []
    for config in baseline.DIVISION_CONFIGS:
        team_conferences = pl.read_csv(_team_conference_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON).select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
            "ConfAbbrev",
            pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
        )
        team_frames.append(
            team_conferences.join(
                conference_history.filter(pl.col("men_women") == config.men_women_flag),
                on=["Season", "ConfAbbrev", "men_women"],
                how="left",
            ).drop("ConfAbbrev")
        )
    return pl.concat(team_frames, how="vertical_relaxed").with_columns(
        *[pl.col(column).fill_null(0.0) for column in CONFERENCE_HISTORY_FEATURE_COLUMNS]
    )


@lru_cache(maxsize=1)
def build_pooled_secondary_tournament_history_features() -> pl.DataFrame:
    frames = [_secondary_tournament_team_features(config) for config in baseline.DIVISION_CONFIGS]
    return pl.concat(frames, how="vertical_relaxed")


@lru_cache(maxsize=1)
def build_pooled_seed_pair_history_features() -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for config in baseline.DIVISION_CONFIGS:
        rows.extend(_seed_pair_history_rows(config))
    return pl.DataFrame(rows) if rows else _empty_pair_frame(SEED_PAIR_HISTORY_FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def build_pooled_massey_history_features() -> pl.DataFrame:
    men_features = _men_massey_history_features()
    women_teams = _season_team_frame(baseline.DIVISION_CONFIGS[1]).with_columns(
        *[pl.lit(0.0).cast(pl.Float64).alias(column) for column in MASSEY_HISTORY_FEATURE_COLUMNS]
    )
    return pl.concat(
        [
            men_features.select("Season", "TeamID", "men_women", *MASSEY_HISTORY_FEATURE_COLUMNS),
            women_teams.select("Season", "TeamID", "men_women", *MASSEY_HISTORY_FEATURE_COLUMNS),
        ],
        how="vertical_relaxed",
    )


def team_feature_view_columns(feature_columns: Sequence[str]) -> list[str]:
    view_columns: list[str] = []
    for column in feature_columns:
        view_columns.extend([f"low_{column}", f"high_{column}", f"delta_{column}"])
    return view_columns


def add_team_feature_views(
    matchup_frame: pl.DataFrame,
    team_features: pl.DataFrame,
    feature_columns: Sequence[str],
    fill_defaults: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    fill_defaults = fill_defaults or {}
    low_feature_frame = team_features.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        *[pl.col(column).alias(f"low_{column}") for column in feature_columns],
    )
    high_feature_frame = team_features.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        *[pl.col(column).alias(f"high_{column}") for column in feature_columns],
    )

    enriched_frame = matchup_frame.join(
        low_feature_frame,
        on=["Season", "LowerTeamID", "men_women"],
        how="left",
    ).join(
        high_feature_frame,
        on=["Season", "HigherTeamID", "men_women"],
        how="left",
    )

    enriched_frame = enriched_frame.with_columns(
        *[
            pl.col(f"low_{column}").fill_null(fill_defaults.get(column, DEFAULT_TEAM_FILL))
            for column in feature_columns
        ],
        *[
            pl.col(f"high_{column}").fill_null(fill_defaults.get(column, DEFAULT_TEAM_FILL))
            for column in feature_columns
        ],
    ).with_columns(
        *[
            (pl.col(f"low_{column}") - pl.col(f"high_{column}")).alias(f"delta_{column}")
            for column in feature_columns
        ]
    )

    return enriched_frame, team_feature_view_columns(feature_columns)


def add_pair_feature_views(
    matchup_frame: pl.DataFrame,
    pair_features: pl.DataFrame,
    feature_columns: Sequence[str],
    fill_defaults: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    fill_defaults = fill_defaults or {}
    enriched_frame = matchup_frame.join(
        pair_features.select("Season", "LowerTeamID", "HigherTeamID", "men_women", *feature_columns),
        on=["Season", "LowerTeamID", "HigherTeamID", "men_women"],
        how="left",
    ).with_columns(
        *[
            pl.col(column).fill_null(fill_defaults.get(column, DEFAULT_PAIR_FILL))
            for column in feature_columns
        ]
    )
    return enriched_frame, list(feature_columns)
