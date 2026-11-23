from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_17_pooled_selection_sunday_massey")


def build_current_season_massey_features() -> pl.DataFrame:
    return cast(
        pl.DataFrame,
        pl.scan_csv(baseline.DATA_DIR / "MMasseyOrdinals.csv")
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .with_columns(pl.col("RankingDayNum").max().over("Season").alias("LatestRankingDayNum"))
        .filter(pl.col("RankingDayNum") == pl.col("LatestRankingDayNum"))
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("OrdinalRank").mean().alias("current_massey_mean_rank"),
            pl.col("OrdinalRank").min().alias("current_massey_best_rank"),
            pl.col("OrdinalRank").count().alias("current_massey_system_count"),
        )
        .collect(),
    )


def build_pooled_matchups() -> tuple[pl.DataFrame, list[str]]:
    pooled_team_features = pl.concat(
        [baseline.build_current_season_team_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    ).join(
        build_current_season_massey_features(),
        on=["Season", "TeamID"],
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
        "low_current_massey_mean_rank",
        "low_current_massey_best_rank",
        "low_current_massey_system_count",
        "high_current_massey_mean_rank",
        "high_current_massey_best_rank",
        "high_current_massey_system_count",
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
        (pl.col("high_current_massey_mean_rank") - pl.col("low_current_massey_mean_rank")).alias("delta_current_massey_mean_rank"),
        (pl.col("high_current_massey_best_rank") - pl.col("low_current_massey_best_rank")).alias("delta_current_massey_best_rank"),
        (pl.col("low_current_massey_system_count") - pl.col("high_current_massey_system_count")).alias("delta_current_massey_system_count"),
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
        "low_current_massey_mean_rank",
        "low_current_massey_best_rank",
        "low_current_massey_system_count",
        "high_current_massey_mean_rank",
        "high_current_massey_best_rank",
        "high_current_massey_system_count",
        "delta_current_massey_mean_rank",
        "delta_current_massey_best_rank",
        "delta_current_massey_system_count",
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
        "information_cutoff": "Selection Sunday with best available current-season Massey snapshot",
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
