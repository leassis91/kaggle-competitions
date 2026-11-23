from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_18_pooled_selection_sunday_elo")
ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 75.0


def expected_score(rating_difference: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def build_current_season_elo_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = pl.read_csv(baseline.DATA_DIR / f"{config.prefix}RegularSeasonCompactResults.csv").select(
        pl.col("Season").cast(pl.Int64),
        pl.col("DayNum").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64),
        pl.col("WLoc"),
    )

    elo_rows: list[dict[str, float | int]] = []
    current_season: int | None = None
    season_ratings: dict[int, float] = {}

    def flush_season_features(season: int | None) -> None:
        if season is None or season < baseline.MIN_POOLED_SEASON:
            return
        for team_id, final_rating in season_ratings.items():
            elo_rows.append(
                {
                    "Season": season,
                    "TeamID": team_id,
                    "current_elo": final_rating,
                    "men_women": config.men_women_flag,
                }
            )

    for season, _day_num, winner_team_id, loser_team_id, winner_location in regular_season_results.iter_rows():
        if current_season != season:
            flush_season_features(current_season)
            current_season = season
            season_ratings = {}

        winner_rating = season_ratings.get(winner_team_id, ELO_BASE_RATING)
        loser_rating = season_ratings.get(loser_team_id, ELO_BASE_RATING)
        winner_adjustment = ELO_HOME_ADVANTAGE if winner_location == "H" else 0.0
        loser_adjustment = ELO_HOME_ADVANTAGE if winner_location == "A" else 0.0
        winner_expected_score = expected_score((winner_rating + winner_adjustment) - (loser_rating + loser_adjustment))
        elo_delta = ELO_K_FACTOR * (1.0 - winner_expected_score)

        season_ratings[winner_team_id] = winner_rating + elo_delta
        season_ratings[loser_team_id] = loser_rating - elo_delta

    flush_season_features(current_season)
    return pl.DataFrame(elo_rows)


def build_pooled_matchups() -> tuple[pl.DataFrame, list[str]]:
    pooled_team_features = pl.concat(
        [baseline.build_current_season_team_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    ).join(
        pl.concat(
            [build_current_season_elo_features(config) for config in baseline.DIVISION_CONFIGS],
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
        "low_current_elo",
        "high_current_elo",
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
        (pl.col("low_current_elo") - pl.col("high_current_elo")).alias("delta_current_elo"),
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
        "low_current_elo",
        "high_current_elo",
        "delta_current_elo",
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
