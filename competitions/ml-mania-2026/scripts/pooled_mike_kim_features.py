from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr

import train_iteration_16_pooled_selection_sunday_baseline as baseline


LATE_SEASON_DAYNUM_CUTOFF = 118
QUALITY_RIDGE_PENALTY = 5.0


def regular_season_compact_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv"


def build_team_game_perspectives(config: baseline.DivisionConfig) -> pl.DataFrame:
    results = pl.read_csv(regular_season_compact_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)

    winner_perspective = results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("LTeamID").alias("OppTeamID").cast(pl.Int64),
        pl.lit(1).cast(pl.Int8).alias("IsWin"),
        (pl.col("WLoc") == "A").cast(pl.Int8).alias("IsAway"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )

    loser_perspective = results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("WTeamID").alias("OppTeamID").cast(pl.Int64),
        pl.lit(0).cast(pl.Int8).alias("IsWin"),
        (pl.col("WLoc") == "H").cast(pl.Int8).alias("IsAway"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )

    return pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")


def build_team_form_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    team_games = build_team_game_perspectives(config).with_columns(
        (pl.col("DayNum") > LATE_SEASON_DAYNUM_CUTOFF).cast(pl.Int8).alias("IsLateSeason"),
    )

    return team_games.group_by(["Season", "TeamID", "men_women"]).agg(
        (pl.col("IsWin") * pl.col("IsLateSeason")).sum().cast(pl.Float64).alias("late_season_wins"),
        pl.col("IsLateSeason").sum().cast(pl.Float64).alias("late_season_games"),
        (pl.col("IsWin") * pl.col("IsAway")).sum().cast(pl.Float64).alias("away_wins"),
        pl.col("IsAway").sum().cast(pl.Float64).alias("away_games"),
    ).with_columns(
        (pl.col("late_season_wins") / (1.0 + pl.col("late_season_games"))).alias("win_ratio_14d"),
        pl.when(pl.col("away_games") > 0.0)
        .then(pl.col("away_wins") / pl.col("away_games"))
        .otherwise(0.5)
        .alias("away_win_pct"),
    ).select(
        "Season",
        "TeamID",
        "men_women",
        "win_ratio_14d",
        "away_win_pct",
    )


def build_matchup_laplace_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    results = pl.read_csv(regular_season_compact_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)

    return results.select(
        pl.col("Season").cast(pl.Int64),
        pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
        pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
        (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Int8).alias("LowerWin"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    ).group_by(["Season", "LowerTeamID", "HigherTeamID", "men_women"]).agg(
        pl.col("LowerWin").sum().cast(pl.Float64).alias("lower_h2h_wins"),
        pl.len().cast(pl.Float64).alias("h2h_meetings"),
    ).with_columns(
        ((pl.col("lower_h2h_wins") + 1.0) / (pl.col("h2h_meetings") + 2.0)).alias("h2h_laplace_win_pct")
    ).select(
        "Season",
        "LowerTeamID",
        "HigherTeamID",
        "men_women",
        "h2h_meetings",
        "h2h_laplace_win_pct",
    )


def solve_filtered_quality_for_season(
    season_results: pl.DataFrame,
    seeded_team_ids: set[int],
) -> pl.DataFrame:
    winner_team_ids = season_results["WTeamID"].to_numpy()
    loser_team_ids = season_results["LTeamID"].to_numpy()
    winner_scores = season_results["WScore"].to_numpy().astype(np.float64)
    loser_scores = season_results["LScore"].to_numpy().astype(np.float64)

    upset_winners = {
        int(winner_team_id)
        for winner_team_id, loser_team_id in zip(winner_team_ids, loser_team_ids, strict=True)
        if int(loser_team_id) in seeded_team_ids
    }
    relevant_team_ids = set(seeded_team_ids) | upset_winners
    bucket_team_id = 0

    filtered_rows: list[tuple[int, int, float]] = []
    for winner_team_id, loser_team_id, winner_score, loser_score in zip(
        winner_team_ids,
        loser_team_ids,
        winner_scores,
        loser_scores,
        strict=True,
    ):
        winner_team_id = int(winner_team_id)
        loser_team_id = int(loser_team_id)
        if winner_team_id not in relevant_team_ids and loser_team_id not in relevant_team_ids:
            continue

        filtered_rows.append(
            (
                winner_team_id if winner_team_id in relevant_team_ids else bucket_team_id,
                loser_team_id if loser_team_id in relevant_team_ids else bucket_team_id,
                float(winner_score - loser_score),
            )
        )

    if not filtered_rows:
        return pl.DataFrame({"Season": [], "TeamID": [], "mike_quality": []}, schema={"Season": pl.Int64, "TeamID": pl.Int64, "mike_quality": pl.Float64})

    mapped_winner_ids = np.array([row[0] for row in filtered_rows], dtype=np.int64)
    mapped_loser_ids = np.array([row[1] for row in filtered_rows], dtype=np.int64)
    margins = np.array([row[2] for row in filtered_rows], dtype=np.float64)

    team_ids = np.array(sorted({bucket_team_id} | relevant_team_ids), dtype=np.int64)
    team_index = {team_id: index for index, team_id in enumerate(team_ids.tolist())}
    game_count = len(filtered_rows)
    team_count = len(team_ids)

    row_indices = np.repeat(np.arange(game_count), 2)
    column_indices = np.empty(game_count * 2, dtype=np.int64)
    column_indices[0::2] = [team_index[int(team_id)] for team_id in mapped_winner_ids]
    column_indices[1::2] = [team_index[int(team_id)] for team_id in mapped_loser_ids]
    values = np.tile(np.array([1.0, -1.0]), game_count)

    design_matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(game_count, team_count),
    )
    ridge_matrix = sparse.eye(team_count, format="csr") * np.sqrt(QUALITY_RIDGE_PENALTY)
    solution = lsqr(
        sparse.vstack([design_matrix, ridge_matrix], format="csr"),
        np.concatenate([margins, np.zeros(team_count, dtype=np.float64)]),
        atol=1e-8,
        btol=1e-8,
    )[0]

    quality_frame = pl.DataFrame(
        {
            "Season": np.full(team_count, int(season_results["Season"][0]), dtype=np.int64),
            "TeamID": team_ids,
            "mike_quality": solution,
        }
    ).filter(pl.col("TeamID") != bucket_team_id)

    if quality_frame.height == 0:
        return quality_frame

    quality_mean = quality_frame["mike_quality"].mean()
    return quality_frame.with_columns((pl.col("mike_quality") - quality_mean).alias("mike_quality"))


def build_filtered_quality_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = pl.read_csv(regular_season_compact_path(config)).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON).select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64),
        pl.col("WScore").cast(pl.Float64),
        pl.col("LScore").cast(pl.Float64),
    )
    seed_rows = pl.read_csv(config.seeds_path).filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON).select(
        pl.col("Season").cast(pl.Int64),
        pl.col("TeamID").cast(pl.Int64),
    )

    seed_by_season: dict[int, set[int]] = {}
    for season, team_id in seed_rows.iter_rows():
        seed_by_season.setdefault(int(season), set()).add(int(team_id))

    quality_frames: list[pl.DataFrame] = []
    for season in sorted(regular_season_results["Season"].unique().to_list()):
        season_results = regular_season_results.filter(pl.col("Season") == season)
        season_seeded = seed_by_season.get(int(season), set())
        quality_frames.append(solve_filtered_quality_for_season(season_results, season_seeded))

    quality = pl.concat(quality_frames, how="vertical_relaxed") if quality_frames else pl.DataFrame()
    if quality.is_empty():
        return pl.DataFrame({"Season": [], "TeamID": [], "men_women": [], "mike_quality": []}, schema={"Season": pl.Int64, "TeamID": pl.Int64, "men_women": pl.Int8, "mike_quality": pl.Float64})
    return quality.with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))


def build_pooled_team_form_features() -> pl.DataFrame:
    return pl.concat(
        [build_team_form_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def build_pooled_matchup_laplace_features() -> pl.DataFrame:
    return pl.concat(
        [build_matchup_laplace_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def build_pooled_filtered_quality_features() -> pl.DataFrame:
    return pl.concat(
        [build_filtered_quality_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )
