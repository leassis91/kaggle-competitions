from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr

import train_iteration_04_men_ensemble as baseline


ARTIFACT_DIR = Path("artifacts/iteration_12_prior_team_quality")
QUALITY_RIDGE_PENALTY = 5.0


def solve_team_quality_for_season(season_results: pl.DataFrame) -> pl.DataFrame:
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
            "prev_team_quality": centered_solution,
        }
    )


def build_team_quality_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    regular_season_results = pl.read_csv(config.regular_season_results_path).select(
        pl.col("Season").cast(pl.Int64),
        pl.col("WTeamID").cast(pl.Int64),
        pl.col("LTeamID").cast(pl.Int64),
        pl.col("WScore").cast(pl.Float64),
        pl.col("LScore").cast(pl.Float64),
    )

    quality_frames: list[pl.DataFrame] = []
    for season in sorted(regular_season_results["Season"].unique().to_list()):
        season_results = regular_season_results.filter(pl.col("Season") == season)
        quality_frames.append(solve_team_quality_for_season(season_results))

    return pl.concat(quality_frames, how="vertical")


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

    team_quality_features = build_team_quality_features(config)
    combined_features = combined_features.join(team_quality_features, on=["Season", "TeamID"], how="left")

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
