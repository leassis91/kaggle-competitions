from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_19_pooled_selection_sunday_team_quality")
QUALITY_RIDGE_PENALTY = 5.0


def solve_team_quality_for_season(season_results: pl.DataFrame, men_women_flag: int) -> pl.DataFrame:
    winner_team_ids = season_results["WTeamID"].to_numpy()
    loser_team_ids = season_results["LTeamID"].to_numpy()
    margins = (season_results["WScore"] - season_results["LScore"]).to_numpy().astype(np.float64)

    team_ids = np.unique(np.concatenate([winner_team_ids, loser_team_ids]))
    team_index = {team_id: index for index, team_id in enumerate(team_ids.tolist())}
    game_count = len(margins)
    team_count = len(team_ids)

    row_indices = np.repeat(np.arange(game_count), 2)
    column_indices = np.empty(game_count * 2, dtype=np.int64)
    column_indices[0::2] = [team_index[int(team_id)] for team_id in winner_team_ids]
    column_indices[1::2] = [team_index[int(team_id)] for team_id in loser_team_ids]
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
    centered_solution = solution - solution.mean()

    return pl.DataFrame(
        {
            "Season": np.full(team_count, int(season_results["Season"][0]), dtype=np.int64),
            "TeamID": team_ids.astype(np.int64),
            "current_team_quality": centered_solution,
            "men_women": np.full(team_count, men_women_flag, dtype=np.int8),
        }
    )


def build_current_season_team_quality_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv").select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64),
        pl.col("WScore").cast(pl.Float64),
        pl.col("LScore").cast(pl.Float64),
    )

    quality_frames: list[pl.DataFrame] = []
    for season in sorted(regular_season_results["Season"].unique().to_list()):
        if season < baseline.MIN_POOLED_SEASON:
            continue
        season_results = regular_season_results.filter(pl.col("Season") == season)
        quality_frames.append(solve_team_quality_for_season(season_results, config.men_women_flag))

    return pl.concat(quality_frames, how="vertical_relaxed")


def build_pooled_matchups() -> tuple[pl.DataFrame, list[str]]:
    pooled_team_features = pl.concat(
        [baseline.build_current_season_team_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    ).join(
        pl.concat(
            [build_current_season_team_quality_features(config) for config in baseline.DIVISION_CONFIGS],
            how="vertical_relaxed",
        ),
        on=["Season", "TeamID", "men_women"],
        how="left",
    )

    pooled_tournament_results = pl.concat(
        [
            pl.read_csv(config.tournament_results_path)
            .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
            .select(
                pl.col("Season").cast(pl.Int64),
                pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
                pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
                (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Int8).alias("Target"),
                pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
            )
            for config in baseline.DIVISION_CONFIGS
        ],
        how="vertical_relaxed",
    )

    low_team_features = pooled_team_features.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        *[
            pl.col(column).alias(f"low_{column}")
            for column in pooled_team_features.columns
            if column not in {"Season", "TeamID"}
        ],
    )
    high_team_features = pooled_team_features.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        *[
            pl.col(column).alias(f"high_{column}")
            for column in pooled_team_features.columns
            if column not in {"Season", "TeamID"}
        ],
    )

    fill_zero_columns = [
        "low_avg_score",
        "low_avg_fga",
        "low_avg_or",
        "low_avg_dr",
        "low_avg_blk",
        "low_avg_pf",
        "low_avg_opp_fga",
        "low_avg_opp_blk",
        "low_avg_opp_pf",
        "low_avg_point_diff",
        "high_avg_score",
        "high_avg_fga",
        "high_avg_or",
        "high_avg_dr",
        "high_avg_blk",
        "high_avg_pf",
        "high_avg_opp_fga",
        "high_avg_opp_blk",
        "high_avg_opp_pf",
        "high_avg_point_diff",
        "low_current_team_quality",
        "high_current_team_quality",
    ]

    matchup_frame = (
        pooled_tournament_results.join(
            low_team_features,
            on=["Season", "LowerTeamID"],
            how="left",
        )
        .join(
            high_team_features,
            on=["Season", "HigherTeamID"],
            how="left",
        )
        .with_columns(
            pl.col("low_seed").fill_null(baseline.LOW_HISTORY_DEFAULT_SEED),
            pl.col("high_seed").fill_null(baseline.LOW_HISTORY_DEFAULT_SEED),
            *[pl.col(column).fill_null(0.0) for column in fill_zero_columns],
        )
    )

    matchup_frame = matchup_frame.with_columns(
        (pl.col("high_seed") - pl.col("low_seed")).alias("delta_seed"),
        (pl.col("low_avg_score") - pl.col("high_avg_score")).alias("delta_avg_score"),
        (pl.col("low_avg_fga") - pl.col("high_avg_fga")).alias("delta_avg_fga"),
        (pl.col("low_avg_or") - pl.col("high_avg_or")).alias("delta_avg_or"),
        (pl.col("low_avg_dr") - pl.col("high_avg_dr")).alias("delta_avg_dr"),
        (pl.col("low_avg_blk") - pl.col("high_avg_blk")).alias("delta_avg_blk"),
        (pl.col("low_avg_pf") - pl.col("high_avg_pf")).alias("delta_avg_pf"),
        (pl.col("low_avg_opp_fga") - pl.col("high_avg_opp_fga")).alias("delta_avg_opp_fga"),
        (pl.col("low_avg_opp_blk") - pl.col("high_avg_opp_blk")).alias("delta_avg_opp_blk"),
        (pl.col("low_avg_opp_pf") - pl.col("high_avg_opp_pf")).alias("delta_avg_opp_pf"),
        (pl.col("low_avg_point_diff") - pl.col("high_avg_point_diff")).alias("delta_avg_point_diff"),
        (pl.col("low_current_team_quality") - pl.col("high_current_team_quality")).alias("delta_current_team_quality"),
    )

    feature_columns = [
        "men_women",
        "low_seed",
        "high_seed",
        "delta_seed",
        "low_avg_score",
        "low_avg_fga",
        "low_avg_or",
        "low_avg_dr",
        "low_avg_blk",
        "low_avg_pf",
        "low_avg_opp_fga",
        "low_avg_opp_blk",
        "low_avg_opp_pf",
        "low_avg_point_diff",
        "high_avg_score",
        "high_avg_fga",
        "high_avg_or",
        "high_avg_dr",
        "high_avg_blk",
        "high_avg_pf",
        "high_avg_opp_fga",
        "high_avg_opp_blk",
        "high_avg_opp_pf",
        "high_avg_point_diff",
        "delta_avg_score",
        "delta_avg_fga",
        "delta_avg_or",
        "delta_avg_dr",
        "delta_avg_blk",
        "delta_avg_pf",
        "delta_avg_opp_fga",
        "delta_avg_opp_blk",
        "delta_avg_opp_pf",
        "delta_avg_point_diff",
        "low_current_team_quality",
        "high_current_team_quality",
        "delta_current_team_quality",
    ]

    return matchup_frame.select("Season", "LowerTeamID", "HigherTeamID", "Target", *feature_columns), feature_columns


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_matchups()
    validation_summary = baseline.run_validation(matchups, feature_columns)
    write_validation_summary(validation_summary, feature_columns)

    print(f"Mean validation Brier: {validation_summary['mean_brier']:.6f}")
    fold_summaries = cast(list[dict[str, float | int]], validation_summary["folds"])
    for fold_summary in fold_summaries:
        print(
            f"Season {fold_summary['season']}: "
            f"combined={fold_summary['combined_brier']:.6f}, "
            f"men={fold_summary['men_brier']:.6f}, "
            f"women={fold_summary['women_brier']:.6f}"
        )


if __name__ == "__main__":
    main()
