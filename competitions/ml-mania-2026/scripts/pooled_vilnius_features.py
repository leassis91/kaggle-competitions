from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ELO_BASE_RATING = 1000.0
ELO_WIDTH = 400.0
ELO_K_FACTOR = 100.0
QUALITY_BUCKET_TEAM_ID = 0


def regular_season_compact_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv"


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_WIDTH))


def build_current_season_elo_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = (
        pl.read_csv(regular_season_compact_path(config))
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("DayNum").cast(pl.Int64),
            pl.col("WTeamID").cast(pl.Int64),
            pl.col("LTeamID").cast(pl.Int64),
        )
        .sort(["Season", "DayNum", "WTeamID", "LTeamID"])
    )

    elo_rows: list[dict[str, float | int]] = []
    current_season: int | None = None
    season_ratings: dict[int, float] = {}

    def flush_season_ratings(season: int | None) -> None:
        if season is None:
            return
        for team_id, final_rating in season_ratings.items():
            elo_rows.append(
                {
                    "Season": season,
                    "TeamID": team_id,
                    "men_women": config.men_women_flag,
                    "vilnius_elo": final_rating,
                }
            )

    for season, _day_num, winner_team_id, loser_team_id in regular_season_results.iter_rows():
        season = int(season)
        winner_team_id = int(winner_team_id)
        loser_team_id = int(loser_team_id)

        if current_season != season:
            flush_season_ratings(current_season)
            current_season = season
            season_ratings = {}

        winner_rating = season_ratings.get(winner_team_id, ELO_BASE_RATING)
        loser_rating = season_ratings.get(loser_team_id, ELO_BASE_RATING)
        winner_expected_score = expected_score(winner_rating, loser_rating)
        rating_delta = ELO_K_FACTOR * (1.0 - winner_expected_score)

        season_ratings[winner_team_id] = winner_rating + rating_delta
        season_ratings[loser_team_id] = loser_rating - rating_delta

    flush_season_ratings(current_season)
    return pl.DataFrame(elo_rows)


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
                winner_team_id if winner_team_id in relevant_team_ids else QUALITY_BUCKET_TEAM_ID,
                loser_team_id if loser_team_id in relevant_team_ids else QUALITY_BUCKET_TEAM_ID,
                float(winner_score - loser_score),
            )
        )

    if not filtered_rows:
        return pl.DataFrame(
            {"Season": [], "TeamID": [], "vilnius_quality": []},
            schema={"Season": pl.Int64, "TeamID": pl.Int64, "vilnius_quality": pl.Float64},
        )

    team_ids = np.array(sorted({QUALITY_BUCKET_TEAM_ID} | relevant_team_ids), dtype=np.int64)
    team_index = {team_id: index for index, team_id in enumerate(team_ids.tolist())}
    game_count = len(filtered_rows)
    team_count = len(team_ids)

    row_indices = np.repeat(np.arange(game_count), 2)
    column_indices = np.empty(game_count * 2, dtype=np.int64)
    column_indices[0::2] = [team_index[row[0]] for row in filtered_rows]
    column_indices[1::2] = [team_index[row[1]] for row in filtered_rows]
    values = np.tile(np.array([1.0, -1.0]), game_count)
    margins = np.array([row[2] for row in filtered_rows], dtype=np.float64)

    design_matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(game_count, team_count),
    )
    bucket_anchor = sparse.csr_matrix(
        ([1.0], ([0], [team_index[QUALITY_BUCKET_TEAM_ID]])),
        shape=(1, team_count),
    )

    solution = lsqr(
        sparse.vstack([design_matrix, bucket_anchor], format="csr"),
        np.concatenate([margins, np.array([0.0], dtype=np.float64)]),
        atol=1e-8,
        btol=1e-8,
    )[0]

    quality_frame = pl.DataFrame(
        {
            "Season": np.full(team_count, int(season_results["Season"][0]), dtype=np.int64),
            "TeamID": team_ids,
            "vilnius_quality": solution,
        }
    ).filter(pl.col("TeamID") != QUALITY_BUCKET_TEAM_ID)

    if quality_frame.height == 0:
        return quality_frame

    return quality_frame.with_columns(
        (pl.col("vilnius_quality") - pl.col("vilnius_quality").mean()).alias("vilnius_quality")
    )


def build_filtered_quality_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = (
        pl.read_csv(regular_season_compact_path(config))
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("WTeamID").cast(pl.Int64),
            pl.col("LTeamID").cast(pl.Int64),
            pl.col("WScore").cast(pl.Float64),
            pl.col("LScore").cast(pl.Float64),
        )
    )
    seed_rows = (
        pl.read_csv(config.seeds_path)
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
        )
    )

    seed_by_season: dict[int, set[int]] = {}
    for season, team_id in seed_rows.iter_rows():
        seed_by_season.setdefault(int(season), set()).add(int(team_id))

    quality_frames: list[pl.DataFrame] = []
    for season in sorted(regular_season_results["Season"].unique().to_list()):
        season_results = regular_season_results.filter(pl.col("Season") == season)
        quality_frames.append(
            solve_filtered_quality_for_season(
                season_results=season_results,
                seeded_team_ids=seed_by_season.get(int(season), set()),
            )
        )

    quality = pl.concat(quality_frames, how="vertical_relaxed") if quality_frames else pl.DataFrame()
    if quality.is_empty():
        return pl.DataFrame(
            {"Season": [], "TeamID": [], "men_women": [], "vilnius_quality": []},
            schema={
                "Season": pl.Int64,
                "TeamID": pl.Int64,
                "men_women": pl.Int8,
                "vilnius_quality": pl.Float64,
            },
        )

    return quality.with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))


def build_pooled_elo_features() -> pl.DataFrame:
    return pl.concat(
        [build_current_season_elo_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def build_pooled_filtered_quality_features() -> pl.DataFrame:
    return pl.concat(
        [build_filtered_quality_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )
