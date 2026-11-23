from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl

import train_iteration_04_men_ensemble as baseline


ARTIFACT_DIR = Path("artifacts/iteration_13_men_selected_detailed")


def build_selected_detailed_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    detailed_results = pl.scan_csv(Path("data") / f"{config.prefix}RegularSeasonDetailedResults.csv")

    winner_perspective = detailed_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("WFGA").cast(pl.Float64).alias("FieldGoalAttempts"),
        pl.col("WOR").cast(pl.Float64).alias("OffensiveRebounds"),
        pl.col("WDR").cast(pl.Float64).alias("DefensiveRebounds"),
        pl.col("WBlk").cast(pl.Float64).alias("Blocks"),
        pl.col("WPF").cast(pl.Float64).alias("PersonalFouls"),
        pl.col("LFGA").cast(pl.Float64).alias("OpponentFieldGoalAttempts"),
        pl.col("LBlk").cast(pl.Float64).alias("OpponentBlocks"),
        pl.col("LPF").cast(pl.Float64).alias("OpponentPersonalFouls"),
    )

    loser_perspective = detailed_results.select(
        pl.col("Season").cast(pl.Int64),
        pl.col("LTeamID").alias("TeamID").cast(pl.Int64),
        pl.col("LFGA").cast(pl.Float64).alias("FieldGoalAttempts"),
        pl.col("LOR").cast(pl.Float64).alias("OffensiveRebounds"),
        pl.col("LDR").cast(pl.Float64).alias("DefensiveRebounds"),
        pl.col("LBlk").cast(pl.Float64).alias("Blocks"),
        pl.col("LPF").cast(pl.Float64).alias("PersonalFouls"),
        pl.col("WFGA").cast(pl.Float64).alias("OpponentFieldGoalAttempts"),
        pl.col("WBlk").cast(pl.Float64).alias("OpponentBlocks"),
        pl.col("WPF").cast(pl.Float64).alias("OpponentPersonalFouls"),
    )

    return (
        pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("FieldGoalAttempts").mean().alias("prev_avg_fga"),
            pl.col("OffensiveRebounds").mean().alias("prev_avg_or"),
            pl.col("DefensiveRebounds").mean().alias("prev_avg_dr"),
            pl.col("Blocks").mean().alias("prev_avg_blk"),
            pl.col("PersonalFouls").mean().alias("prev_avg_pf"),
            pl.col("OpponentFieldGoalAttempts").mean().alias("prev_avg_opp_fga"),
            pl.col("OpponentBlocks").mean().alias("prev_avg_opp_blk"),
            pl.col("OpponentPersonalFouls").mean().alias("prev_avg_opp_pf"),
        )
        .collect()
    )


def build_preseason_team_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_features = baseline.build_regular_season_features(config)
    tournament_features = baseline.build_previous_tournament_features(config)
    combined_features = regular_season_features.join(
        tournament_features,
        on=["Season", "TeamID"],
        how="left",
    )

    massey_features = baseline.build_massey_features(config)
    if massey_features is not None:
        combined_features = combined_features.join(massey_features, on=["Season", "TeamID"], how="left")

    if config.name == "men":
        combined_features = combined_features.join(
            build_selected_detailed_features(config),
            on=["Season", "TeamID"],
            how="left",
        )

    if config.prefix == "M":
        team_details = pl.read_csv(config.teams_path).select(
            pl.col("TeamID"),
            pl.col("FirstD1Season").cast(pl.Int64),
            pl.col("LastD1Season").cast(pl.Int64),
        )
        combined_features = combined_features.join(team_details, on="TeamID", how="left")

    combined_features = combined_features.sort(["TeamID", "Season"]).with_columns(
        pl.col("prev_win_pct").rolling_mean(window_size=3, min_samples=1).over("TeamID").alias("rolling_3_win_pct"),
        pl.col("prev_avg_margin")
        .rolling_mean(window_size=3, min_samples=1)
        .over("TeamID")
        .alias("rolling_3_avg_margin"),
        pl.col("prev_avg_opp_win_pct")
        .rolling_mean(window_size=3, min_samples=1)
        .over("TeamID")
        .alias("rolling_3_avg_opp_win_pct"),
        pl.col("prev_tourney_wins")
        .fill_null(0.0)
        .rolling_mean(window_size=3, min_samples=1)
        .over("TeamID")
        .alias("rolling_3_tourney_wins"),
        pl.col("prev_games").cum_sum().over("TeamID").alias("career_games_through_prev_season"),
        pl.col("prev_wins").cum_sum().over("TeamID").alias("career_wins_through_prev_season"),
    )

    combined_features = combined_features.with_columns(
        (pl.col("Season") + 1).alias("FeatureSeason"),
        baseline.safe_rate("career_wins_through_prev_season", "career_games_through_prev_season").alias("career_win_pct"),
    )

    if config.prefix == "M":
        combined_features = combined_features.with_columns(
            (pl.col("FeatureSeason") - pl.col("FirstD1Season")).cast(pl.Float64).alias("program_tenure"),
            (pl.col("LastD1Season") >= pl.col("FeatureSeason")).cast(pl.Int8).alias("team_active_flag"),
        )

    return combined_features


def main() -> None:
    baseline.ARTIFACT_DIR = ARTIFACT_DIR
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    preseason_features_by_division: dict[str, pl.DataFrame] = {}
    matchups_by_division: dict[str, pl.DataFrame] = {}
    feature_columns_by_division: dict[str, list[str]] = {}

    for config in baseline.DIVISION_CONFIGS:
        preseason_features = build_preseason_team_features(config)
        matchups, feature_columns = baseline.build_training_matchups(config, preseason_features)
        preseason_features_by_division[config.name] = preseason_features
        matchups_by_division[config.name] = matchups
        feature_columns_by_division[config.name] = feature_columns

    validation_summary = baseline.run_validation(matchups_by_division, feature_columns_by_division)
    submission = baseline.build_submission(
        preseason_features_by_division=preseason_features_by_division,
        matchups_by_division=matchups_by_division,
        feature_columns_by_division=feature_columns_by_division,
    )

    submission.write_csv(ARTIFACT_DIR / "submission.csv")
    baseline.write_validation_summary(validation_summary)

    print(f"Generated at UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Mean validation Brier: {validation_summary['mean_brier']:.6f}")
    for fold_summary in validation_summary["folds"]:
        print(
            f"Season {fold_summary['season']}: "
            f"combined={fold_summary['combined_brier']:.6f}, "
            f"men={fold_summary['men_brier']:.6f}, "
            f"women={fold_summary['women_brier']:.6f}"
        )


if __name__ == "__main__":
    main()
