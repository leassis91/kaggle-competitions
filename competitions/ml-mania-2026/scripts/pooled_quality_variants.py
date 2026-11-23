from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr

import train_iteration_16_pooled_selection_sunday_baseline as baseline


QUALITY_BUCKET_TEAM_ID = 0
RECENCY_WEIGHT_FLOOR = 0.5
RECENCY_WEIGHT_RANGE = 1.0


def regular_season_compact_path(config: baseline.DivisionConfig) -> Path:
    return baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv"


def build_relevant_team_ids(
    winner_team_ids: np.ndarray,
    loser_team_ids: np.ndarray,
    seeded_team_ids: set[int],
    relevance_hops: int,
) -> set[int]:
    relevant_team_ids = set(seeded_team_ids)
    if relevance_hops <= 0 or not relevant_team_ids:
        return relevant_team_ids

    for _hop in range(relevance_hops):
        expanded_team_ids = set(relevant_team_ids)
        for winner_team_id, loser_team_id in zip(winner_team_ids, loser_team_ids, strict=True):
            winner_team_id = int(winner_team_id)
            loser_team_id = int(loser_team_id)
            if loser_team_id in relevant_team_ids:
                expanded_team_ids.add(winner_team_id)

        if expanded_team_ids == relevant_team_ids:
            break
        relevant_team_ids = expanded_team_ids

    return relevant_team_ids


def season_recency_weights(day_numbers: np.ndarray) -> np.ndarray:
    if day_numbers.size == 0:
        return np.array([], dtype=np.float64)

    max_day_number = float(day_numbers.max())
    if max_day_number <= 0.0:
        return np.ones(day_numbers.size, dtype=np.float64)

    return RECENCY_WEIGHT_FLOOR + RECENCY_WEIGHT_RANGE * (day_numbers.astype(np.float64) / max_day_number)


def solve_filtered_quality_for_season(
    season_results: pl.DataFrame,
    seeded_team_ids: set[int],
    output_column: str,
    margin_cap: float | None = None,
    use_recency_weights: bool = False,
    relevance_hops: int = 1,
    ridge_lambda: float = 0.0,
) -> pl.DataFrame:
    winner_team_ids = season_results["WTeamID"].to_numpy()
    loser_team_ids = season_results["LTeamID"].to_numpy()
    winner_scores = season_results["WScore"].to_numpy().astype(np.float64)
    loser_scores = season_results["LScore"].to_numpy().astype(np.float64)
    day_numbers = season_results["DayNum"].to_numpy().astype(np.float64)

    relevant_team_ids = build_relevant_team_ids(
        winner_team_ids=winner_team_ids,
        loser_team_ids=loser_team_ids,
        seeded_team_ids=seeded_team_ids,
        relevance_hops=relevance_hops,
    )

    filtered_rows: list[tuple[int, int, float, float]] = []
    raw_recency_weights = season_recency_weights(day_numbers) if use_recency_weights else np.ones(day_numbers.size, dtype=np.float64)
    for winner_team_id, loser_team_id, winner_score, loser_score, sample_weight in zip(
        winner_team_ids,
        loser_team_ids,
        winner_scores,
        loser_scores,
        raw_recency_weights,
        strict=True,
    ):
        winner_team_id = int(winner_team_id)
        loser_team_id = int(loser_team_id)
        if winner_team_id not in relevant_team_ids and loser_team_id not in relevant_team_ids:
            continue

        margin = float(winner_score - loser_score)
        if margin_cap is not None:
            margin = float(np.clip(margin, -margin_cap, margin_cap))

        filtered_rows.append(
            (
                winner_team_id if winner_team_id in relevant_team_ids else QUALITY_BUCKET_TEAM_ID,
                loser_team_id if loser_team_id in relevant_team_ids else QUALITY_BUCKET_TEAM_ID,
                margin,
                float(sample_weight),
            )
        )

    if not filtered_rows:
        return pl.DataFrame(
            {"Season": [], "TeamID": [], output_column: []},
            schema={"Season": pl.Int64, "TeamID": pl.Int64, output_column: pl.Float64},
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
    sample_weights = np.array([row[3] for row in filtered_rows], dtype=np.float64)
    sqrt_weights = np.sqrt(sample_weights)

    design_matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(game_count, team_count),
    )
    weighted_design_matrix = design_matrix.multiply(sqrt_weights[:, None])
    weighted_margins = margins * sqrt_weights
    bucket_anchor = sparse.csr_matrix(
        ([1.0], ([0], [team_index[QUALITY_BUCKET_TEAM_ID]])),
        shape=(1, team_count),
    )
    solve_matrix_blocks = [weighted_design_matrix, bucket_anchor]
    solve_targets = [weighted_margins, np.array([0.0], dtype=np.float64)]

    if ridge_lambda > 0.0 and team_count > 1:
        ridge_value = float(np.sqrt(ridge_lambda))
        ridge_row_indices = np.arange(team_count - 1, dtype=np.int64)
        ridge_column_indices = np.arange(1, team_count, dtype=np.int64)
        ridge_values = np.full(team_count - 1, ridge_value, dtype=np.float64)
        ridge_penalty = sparse.csr_matrix(
            (ridge_values, (ridge_row_indices, ridge_column_indices)),
            shape=(team_count - 1, team_count),
        )
        solve_matrix_blocks.append(ridge_penalty)
        solve_targets.append(np.zeros(team_count - 1, dtype=np.float64))

    solution = lsqr(
        sparse.vstack(solve_matrix_blocks, format="csr"),
        np.concatenate(solve_targets),
        atol=1e-8,
        btol=1e-8,
    )[0]

    quality_frame = pl.DataFrame(
        {
            "Season": np.full(team_count, int(season_results["Season"][0]), dtype=np.int64),
            "TeamID": team_ids,
            output_column: solution,
        }
    ).filter(pl.col("TeamID") != QUALITY_BUCKET_TEAM_ID)

    if quality_frame.height == 0:
        return quality_frame

    return quality_frame.with_columns((pl.col(output_column) - pl.col(output_column).mean()).alias(output_column))


def build_filtered_quality_features(
    config: baseline.DivisionConfig,
    output_column: str,
    margin_cap: float | None = None,
    use_recency_weights: bool = False,
    relevance_hops: int = 1,
    ridge_lambda: float = 0.0,
) -> pl.DataFrame:
    regular_season_results = (
        pl.read_csv(regular_season_compact_path(config))
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("DayNum").cast(pl.Int64),
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
                output_column=output_column,
                margin_cap=margin_cap,
                use_recency_weights=use_recency_weights,
                relevance_hops=relevance_hops,
                ridge_lambda=ridge_lambda,
            )
        )

    quality = pl.concat(quality_frames, how="vertical_relaxed") if quality_frames else pl.DataFrame()
    if quality.is_empty():
        return pl.DataFrame(
            {"Season": [], "TeamID": [], "men_women": [], output_column: []},
            schema={
                "Season": pl.Int64,
                "TeamID": pl.Int64,
                "men_women": pl.Int8,
                output_column: pl.Float64,
            },
        )

    return quality.with_columns(pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"))


def build_pooled_filtered_quality_features(
    output_column: str,
    margin_cap: float | None = None,
    use_recency_weights: bool = False,
    relevance_hops: int = 1,
    ridge_lambda: float = 0.0,
) -> pl.DataFrame:
    return pl.concat(
        [
            build_filtered_quality_features(
                config,
                output_column=output_column,
                margin_cap=margin_cap,
                use_recency_weights=use_recency_weights,
                relevance_hops=relevance_hops,
                ridge_lambda=ridge_lambda,
            )
            for config in baseline.DIVISION_CONFIGS
        ],
        how="vertical_relaxed",
    )
