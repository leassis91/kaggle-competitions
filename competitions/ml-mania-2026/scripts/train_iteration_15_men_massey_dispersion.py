from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import orjson
import polars as pl

import train_iteration_04_men_ensemble as baseline


ARTIFACT_DIR = Path("artifacts/iteration_15_men_massey_dispersion")


def build_extended_massey_features(config: baseline.DivisionConfig) -> pl.DataFrame | None:
    if not config.use_massey:
        return None

    return (
        pl.scan_csv(Path("data") / "MMasseyOrdinals.csv")
        .filter(pl.col("RankingDayNum") == 133)
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("OrdinalRank").mean().alias("prev_massey_mean_rank"),
            pl.col("OrdinalRank").min().alias("prev_massey_best_rank"),
            pl.col("OrdinalRank").max().alias("prev_massey_worst_rank"),
            pl.col("OrdinalRank").median().alias("prev_massey_median_rank"),
            pl.col("OrdinalRank").std(ddof=0).fill_null(0.0).alias("prev_massey_rank_std"),
            (pl.col("OrdinalRank") <= 25).mean().alias("prev_massey_top25_share"),
            pl.col("OrdinalRank").count().alias("prev_massey_system_count"),
        )
        .with_columns(
            (pl.col("prev_massey_worst_rank") - pl.col("prev_massey_best_rank")).alias("prev_massey_rank_range")
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

    massey_features = build_extended_massey_features(config)
    if massey_features is not None:
        combined_features = combined_features.join(massey_features, on=["Season", "TeamID"], how="left")

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


def write_validation_summary(validation_summary: dict[str, object]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "baseline_lightgbm_params": baseline.BASELINE_LIGHTGBM_PARAMS,
        "men_smooth_lightgbm_params": baseline.MEN_SMOOTH_LIGHTGBM_PARAMS,
        "probability_shrink_by_division": baseline.PROBABILITY_SHRINK_BY_DIVISION,
        "men_ensemble_weights": baseline.MEN_ENSEMBLE_WEIGHTS,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    baseline.ARTIFACT_DIR = ARTIFACT_DIR
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups_by_division: dict[str, pl.DataFrame] = {}
    feature_columns_by_division: dict[str, list[str]] = {}

    for config in baseline.DIVISION_CONFIGS:
        preseason_features = build_preseason_team_features(config)
        matchups, feature_columns = baseline.build_training_matchups(config, preseason_features)
        matchups_by_division[config.name] = matchups
        feature_columns_by_division[config.name] = feature_columns

    validation_summary = baseline.run_validation(matchups_by_division, feature_columns_by_division)
    write_validation_summary(validation_summary)

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
