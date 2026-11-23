from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import orjson
import polars as pl


DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts/iteration_16_pooled_selection_sunday_baseline")
VALIDATION_FOLD_COUNT = 5
LOW_HISTORY_DEFAULT_SEED = 17.0
MIN_POOLED_SEASON = 2010

LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
}


@dataclass(frozen=True)
class DivisionConfig:
    name: str
    prefix: str
    men_women_flag: int

    @property
    def regular_season_detailed_results_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}RegularSeasonDetailedResults.csv"

    @property
    def tournament_results_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}NCAATourneyCompactResults.csv"

    @property
    def seeds_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}NCAATourneySeeds.csv"


DIVISION_CONFIGS = (
    DivisionConfig(name="men", prefix="M", men_women_flag=1),
    DivisionConfig(name="women", prefix="W", men_women_flag=0),
)


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=7,
        n_jobs=4,
        verbosity=-1,
    )


def build_selected_team_games(config: DivisionConfig) -> pl.LazyFrame:
    detailed_results = pl.scan_csv(config.regular_season_detailed_results_path).filter(pl.col("Season") >= MIN_POOLED_SEASON)
    overtime_adjustment = ((pl.lit(40.0) + 5.0 * pl.col("NumOT").cast(pl.Float64)) / 40.0).alias("OvertimeAdjustment")

    adjusted_results = detailed_results.with_columns(overtime_adjustment)

    winner_perspective = adjusted_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
        (pl.col("WScore").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("Score"),
        (pl.col("WFGA").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("FGA"),
        (pl.col("WOR").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OR"),
        (pl.col("WDR").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("DR"),
        (pl.col("WBlk").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("Blk"),
        (pl.col("WPF").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("PF"),
        (pl.col("LFGA").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentFGA"),
        (pl.col("LBlk").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentBlk"),
        (pl.col("LPF").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentPF"),
        ((pl.col("WScore") - pl.col("LScore")).cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("PointDiff"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )

    loser_perspective = adjusted_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
        (pl.col("LScore").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("Score"),
        (pl.col("LFGA").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("FGA"),
        (pl.col("LOR").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OR"),
        (pl.col("LDR").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("DR"),
        (pl.col("LBlk").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("Blk"),
        (pl.col("LPF").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("PF"),
        (pl.col("WFGA").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentFGA"),
        (pl.col("WBlk").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentBlk"),
        (pl.col("WPF").cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("OpponentPF"),
        ((pl.col("LScore") - pl.col("WScore")).cast(pl.Float64) / pl.col("OvertimeAdjustment")).alias("PointDiff"),
        pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
    )

    return pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")


def build_current_season_team_features(config: DivisionConfig) -> pl.DataFrame:
    team_games = build_selected_team_games(config)

    selected_stats = cast(
        pl.DataFrame,
        team_games.group_by(["Season", "TeamID", "men_women"])
        .agg(
            pl.col("Score").mean().alias("avg_score"),
            pl.col("FGA").mean().alias("avg_fga"),
            pl.col("OR").mean().alias("avg_or"),
            pl.col("DR").mean().alias("avg_dr"),
            pl.col("Blk").mean().alias("avg_blk"),
            pl.col("PF").mean().alias("avg_pf"),
            pl.col("OpponentFGA").mean().alias("avg_opp_fga"),
            pl.col("OpponentBlk").mean().alias("avg_opp_blk"),
            pl.col("OpponentPF").mean().alias("avg_opp_pf"),
            pl.col("PointDiff").mean().alias("avg_point_diff"),
        )
        .collect(),
    )

    seeds = cast(
        pl.DataFrame,
        pl.scan_csv(config.seeds_path)
        .filter(pl.col("Season") >= MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
            pl.col("Seed").str.slice(1, 2).cast(pl.Float64).alias("seed"),
        )
        .collect(),
    )

    return selected_stats.join(seeds, on=["Season", "TeamID"], how="left")


def build_pooled_matchups() -> tuple[pl.DataFrame, list[str]]:
    pooled_team_features = pl.concat(
        [build_current_season_team_features(config) for config in DIVISION_CONFIGS],
        how="vertical_relaxed",
    )

    pooled_tournament_results = pl.concat(
        [
            pl.read_csv(config.tournament_results_path)
            .filter(pl.col("Season") >= MIN_POOLED_SEASON)
            .select(
                pl.col("Season").cast(pl.Int64),
                pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID").cast(pl.Int64),
                pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID").cast(pl.Int64),
                (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Int8).alias("Target"),
                pl.lit(config.men_women_flag).cast(pl.Int8).alias("men_women"),
            )
            for config in DIVISION_CONFIGS
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
            pl.col("low_seed").fill_null(LOW_HISTORY_DEFAULT_SEED),
            pl.col("high_seed").fill_null(LOW_HISTORY_DEFAULT_SEED),
            *[
                pl.col(column).fill_null(0.0)
                for column in matchup_frame_feature_columns()
                if column not in {"low_seed", "high_seed"}
            ],
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
    ]

    return matchup_frame.select("Season", "LowerTeamID", "HigherTeamID", "Target", *feature_columns), feature_columns


def matchup_frame_feature_columns() -> list[str]:
    return [
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
        "low_seed",
        "high_seed",
    ]


def latest_completed_validation_seasons() -> list[int]:
    men_seasons = set(pl.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")["Season"].to_list())
    women_seasons = set(pl.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")["Season"].to_list())
    return sorted(men_seasons & women_seasons)[-VALIDATION_FOLD_COUNT:]


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        training_rows = matchups.filter(pl.col("Season") < validation_season)
        validation_rows = matchups.filter(pl.col("Season") == validation_season)

        model = build_model()
        model.fit(training_rows.select(feature_columns).to_numpy(), training_rows["Target"].to_numpy())
        probabilities = model.predict_proba(validation_rows.select(feature_columns).to_numpy())[:, 1]
        targets = validation_rows["Target"].to_numpy()

        men_mask = validation_rows["men_women"].to_numpy() == 1
        women_mask = ~men_mask

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": brier_score(probabilities, targets),
                "men_brier": brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": brier_score(probabilities[women_mask], targets[women_mask]),
            }
        )

    return {
        "validation_seasons": validation_seasons,
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_matchups()
    validation_summary = run_validation(matchups, feature_columns)
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
