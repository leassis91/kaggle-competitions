from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import orjson
import polars as pl


DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts/iteration_11_men_feature_ensemble")
VALIDATION_FOLD_COUNT = 5
LOW_HISTORY_DEFAULT_SEED = 17.0
PROBABILITY_SHRINK_BY_DIVISION = {
    "men": 0.80,
    "women": 0.80,
}
MEN_ENSEMBLE_WEIGHTS = {
    "baseline": 0.30,
    "smooth": 0.70,
}
MEN_FEATURE_VARIANT_WEIGHTS = {
    "base": 0.60,
    "second_order": 0.20,
    "zscore": 0.20,
}


@dataclass(frozen=True)
class DivisionConfig:
    name: str
    prefix: str
    team_min_id: int
    team_max_id: int
    use_massey: bool = False

    @property
    def regular_season_results_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}RegularSeasonCompactResults.csv"

    @property
    def tournament_results_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}NCAATourneyCompactResults.csv"

    @property
    def seeds_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}NCAATourneySeeds.csv"

    @property
    def teams_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}Teams.csv"

    @property
    def conferences_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}TeamConferences.csv"


DIVISION_CONFIGS = (
    DivisionConfig(name="men", prefix="M", team_min_id=1000, team_max_id=1999, use_massey=True),
    DivisionConfig(name="women", prefix="W", team_min_id=3000, team_max_id=3999),
)

BASELINE_LIGHTGBM_PARAMS = {
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

MEN_SMOOTH_LIGHTGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 5.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
}


def safe_rate(numerator: str, denominator: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(0.0)
    )


def season_zscore(column_name: str) -> pl.Expr:
    season_mean = pl.col(column_name).mean().over("Season")
    season_std = pl.col(column_name).std(ddof=0).over("Season")
    return (
        pl.when(season_std > 0)
        .then((pl.col(column_name) - season_mean) / season_std)
        .otherwise(0.0)
        .alias(f"{column_name}_z")
    )


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def shrink_probabilities(probabilities: np.ndarray, division_name: str) -> np.ndarray:
    shrink_factor = PROBABILITY_SHRINK_BY_DIVISION[division_name]
    return np.clip(0.5 + shrink_factor * (probabilities - 0.5), 0.0, 1.0)


def blend_men_probabilities(
    baseline_probabilities: np.ndarray,
    smooth_probabilities: np.ndarray,
) -> np.ndarray:
    return (
        MEN_ENSEMBLE_WEIGHTS["baseline"] * baseline_probabilities
        + MEN_ENSEMBLE_WEIGHTS["smooth"] * smooth_probabilities
    )


def build_team_games(regular_season_results_path: Path) -> pl.LazyFrame:
    results = pl.scan_csv(regular_season_results_path)

    winner_perspective = results.select(
        pl.col("Season"),
        pl.col("WTeamID").alias("TeamID"),
        pl.col("LTeamID").alias("OppTeamID"),
        pl.lit(1).cast(pl.Int8).alias("IsWin"),
        pl.col("WScore").cast(pl.Float64).alias("PointsFor"),
        pl.col("LScore").cast(pl.Float64).alias("PointsAgainst"),
        (pl.col("WScore") - pl.col("LScore")).cast(pl.Float64).alias("Margin"),
        (pl.col("WLoc") == "H").cast(pl.Int8).alias("IsHome"),
        (pl.col("WLoc") == "A").cast(pl.Int8).alias("IsAway"),
        (pl.col("WLoc") == "N").cast(pl.Int8).alias("IsNeutral"),
    )

    loser_perspective = results.select(
        pl.col("Season"),
        pl.col("LTeamID").alias("TeamID"),
        pl.col("WTeamID").alias("OppTeamID"),
        pl.lit(0).cast(pl.Int8).alias("IsWin"),
        pl.col("LScore").cast(pl.Float64).alias("PointsFor"),
        pl.col("WScore").cast(pl.Float64).alias("PointsAgainst"),
        (pl.col("LScore") - pl.col("WScore")).cast(pl.Float64).alias("Margin"),
        (pl.col("WLoc") == "A").cast(pl.Int8).alias("IsHome"),
        (pl.col("WLoc") == "H").cast(pl.Int8).alias("IsAway"),
        (pl.col("WLoc") == "N").cast(pl.Int8).alias("IsNeutral"),
    )

    return pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")


def build_regular_season_features(config: DivisionConfig, men_feature_variant: str = "base") -> pl.DataFrame:
    team_games = build_team_games(config.regular_season_results_path)

    basic_team_stats = (
        team_games.group_by(["Season", "TeamID"])
        .agg(
            pl.len().alias("prev_games"),
            pl.col("IsWin").sum().alias("prev_wins"),
            pl.col("PointsFor").mean().alias("prev_avg_points_for"),
            pl.col("PointsAgainst").mean().alias("prev_avg_points_against"),
            pl.col("Margin").mean().alias("prev_avg_margin"),
            pl.col("IsHome").sum().alias("prev_home_games"),
            (pl.col("IsWin") * pl.col("IsHome")).sum().alias("prev_home_wins"),
            pl.col("IsAway").sum().alias("prev_away_games"),
            (pl.col("IsWin") * pl.col("IsAway")).sum().alias("prev_away_wins"),
            pl.col("IsNeutral").sum().alias("prev_neutral_games"),
            (pl.col("IsWin") * pl.col("IsNeutral")).sum().alias("prev_neutral_wins"),
        )
        .with_columns(
            (pl.col("prev_games") - pl.col("prev_wins")).alias("prev_losses"),
            (pl.col("prev_wins") / pl.col("prev_games")).alias("prev_win_pct"),
            safe_rate("prev_home_wins", "prev_home_games").alias("prev_home_win_pct"),
            safe_rate("prev_away_wins", "prev_away_games").alias("prev_away_win_pct"),
            safe_rate("prev_neutral_wins", "prev_neutral_games").alias("prev_neutral_win_pct"),
        )
    )

    opponent_strength = (
        team_games.join(
            basic_team_stats.select(
                pl.col("Season"),
                pl.col("TeamID").alias("OppTeamID"),
                pl.col("prev_win_pct").alias("prev_avg_opp_win_pct_component"),
                pl.col("prev_avg_margin").alias("prev_avg_opp_margin_component"),
            ),
            on=["Season", "OppTeamID"],
            how="left",
        )
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("prev_avg_opp_win_pct_component").mean().alias("prev_avg_opp_win_pct"),
            pl.col("prev_avg_opp_margin_component").mean().alias("prev_avg_opp_margin"),
        )
    )

    second_order_opponent_strength = None
    if config.use_massey and men_feature_variant == "second_order":
        second_order_opponent_strength = (
            team_games.join(
                opponent_strength.select(
                    pl.col("Season"),
                    pl.col("TeamID").alias("OppTeamID"),
                    pl.col("prev_avg_opp_win_pct").alias("prev_avg_opp_opp_win_pct"),
                    pl.col("prev_avg_opp_margin").alias("prev_avg_opp_opp_margin"),
                ),
                on=["Season", "OppTeamID"],
                how="left",
            )
            .group_by(["Season", "TeamID"])
            .agg(
                pl.col("prev_avg_opp_opp_win_pct").mean().alias("prev_avg_opp_opp_win_pct"),
                pl.col("prev_avg_opp_opp_margin").mean().alias("prev_avg_opp_opp_margin"),
            )
        )

    conference_assignments = pl.scan_csv(config.conferences_path)

    conference_strength = (
        basic_team_stats.join(conference_assignments, on=["Season", "TeamID"], how="left")
        .group_by(["Season", "ConfAbbrev"])
        .agg(
            pl.col("prev_win_pct").mean().alias("prev_conf_avg_win_pct"),
            pl.col("prev_avg_margin").mean().alias("prev_conf_avg_margin"),
            pl.len().alias("prev_conf_team_count"),
        )
    )

    team_conference_features = (
        conference_assignments.join(conference_strength, on=["Season", "ConfAbbrev"], how="left")
        .select(
            pl.col("Season"),
            pl.col("TeamID"),
            pl.col("ConfAbbrev").alias("prev_conf_abbrev"),
            pl.col("prev_conf_avg_win_pct"),
            pl.col("prev_conf_avg_margin"),
            pl.col("prev_conf_team_count"),
        )
    )

    team_features = basic_team_stats.join(opponent_strength, on=["Season", "TeamID"], how="left")

    if second_order_opponent_strength is not None:
        team_features = team_features.join(second_order_opponent_strength, on=["Season", "TeamID"], how="left")

    if config.use_massey and men_feature_variant == "zscore":
        team_features = team_features.with_columns(
            season_zscore("prev_win_pct"),
            season_zscore("prev_avg_points_for"),
            season_zscore("prev_avg_points_against"),
            season_zscore("prev_avg_margin"),
            season_zscore("prev_avg_opp_win_pct"),
        )

    team_features = team_features.join(
        team_conference_features,
        on=["Season", "TeamID"],
        how="left",
    )

    return team_features.collect()


def build_previous_tournament_features(config: DivisionConfig) -> pl.DataFrame:
    tournament_results = pl.scan_csv(config.tournament_results_path)

    winner_perspective = tournament_results.select(
        pl.col("Season"),
        pl.col("WTeamID").alias("TeamID"),
        pl.lit(1).cast(pl.Int8).alias("TournamentWin"),
        (pl.col("WScore") - pl.col("LScore")).cast(pl.Float64).alias("TournamentMargin"),
    )

    loser_perspective = tournament_results.select(
        pl.col("Season"),
        pl.col("LTeamID").alias("TeamID"),
        pl.lit(0).cast(pl.Int8).alias("TournamentWin"),
        (pl.col("LScore") - pl.col("WScore")).cast(pl.Float64).alias("TournamentMargin"),
    )

    tournament_team_stats = (
        pl.concat([winner_perspective, loser_perspective], how="vertical_relaxed")
        .group_by(["Season", "TeamID"])
        .agg(
            pl.len().alias("prev_tourney_games"),
            pl.col("TournamentWin").sum().alias("prev_tourney_wins"),
            pl.col("TournamentMargin").mean().alias("prev_tourney_avg_margin"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("prev_tourney_appearance"))
        .collect()
    )

    seeds = (
        pl.scan_csv(config.seeds_path)
        .select(
            pl.col("Season"),
            pl.col("TeamID"),
            pl.col("Seed").str.slice(1, 2).cast(pl.Float64).alias("prev_seed"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("prev_seeded"))
        .collect()
    )

    return tournament_team_stats.join(seeds, on=["Season", "TeamID"], how="full").with_columns(
        pl.coalesce(["Season", "Season_right"]).alias("Season"),
        pl.coalesce(["TeamID", "TeamID_right"]).alias("TeamID"),
    ).drop(["Season_right", "TeamID_right"])


def build_massey_features(config: DivisionConfig) -> pl.DataFrame | None:
    if not config.use_massey:
        return None

    return (
        pl.scan_csv(DATA_DIR / "MMasseyOrdinals.csv")
        .filter(pl.col("RankingDayNum") == 133)
        .group_by(["Season", "TeamID"])
        .agg(
            pl.col("OrdinalRank").mean().alias("prev_massey_mean_rank"),
            pl.col("OrdinalRank").min().alias("prev_massey_best_rank"),
            pl.col("OrdinalRank").count().alias("prev_massey_system_count"),
        )
        .collect()
    )


def build_preseason_team_features(config: DivisionConfig, men_feature_variant: str = "base") -> pl.DataFrame:
    regular_season_features = build_regular_season_features(config, men_feature_variant=men_feature_variant)
    tournament_features = build_previous_tournament_features(config)
    combined_features = regular_season_features.join(
        tournament_features,
        on=["Season", "TeamID"],
        how="left",
    )

    massey_features = build_massey_features(config)
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
        safe_rate("career_wins_through_prev_season", "career_games_through_prev_season").alias("career_win_pct"),
    )

    if config.prefix == "M":
        combined_features = combined_features.with_columns(
            (pl.col("FeatureSeason") - pl.col("FirstD1Season")).cast(pl.Float64).alias("program_tenure"),
            (pl.col("LastD1Season") >= pl.col("FeatureSeason")).cast(pl.Int8).alias("team_active_flag"),
        )

    return combined_features


def build_training_matchups(
    config: DivisionConfig,
    preseason_features: pl.DataFrame,
    men_feature_variant: str = "base",
) -> tuple[pl.DataFrame, list[str]]:
    tournament_results = pl.read_csv(config.tournament_results_path).select(
        pl.col("Season").cast(pl.Int64),
        pl.min_horizontal("WTeamID", "LTeamID").alias("LowerTeamID"),
        pl.max_horizontal("WTeamID", "LTeamID").alias("HigherTeamID"),
        (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Int8).alias("Target"),
    )

    low_team_features = preseason_features.select(
        pl.col("FeatureSeason").alias("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        *[pl.col(column).alias(f"low_{column}") for column in preseason_features.columns if column not in {"FeatureSeason", "TeamID"}],
    )
    high_team_features = preseason_features.select(
        pl.col("FeatureSeason").alias("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        *[pl.col(column).alias(f"high_{column}") for column in preseason_features.columns if column not in {"FeatureSeason", "TeamID"}],
    )

    matchup_frame = tournament_results.join(
        low_team_features,
        on=["Season", "LowerTeamID"],
        how="left",
    ).join(
        high_team_features,
        on=["Season", "HigherTeamID"],
        how="left",
    )

    matchup_frame = matchup_frame.with_columns(
        pl.col("low_prev_games").is_not_null().cast(pl.Int8).alias("low_has_history"),
        pl.col("high_prev_games").is_not_null().cast(pl.Int8).alias("high_has_history"),
        pl.when(pl.col("low_prev_conf_abbrev").is_not_null() & pl.col("high_prev_conf_abbrev").is_not_null())
        .then((pl.col("low_prev_conf_abbrev") == pl.col("high_prev_conf_abbrev")).cast(pl.Int8))
        .otherwise(0)
        .alias("same_prev_conference"),
    )

    fill_zero_columns = [
        column
        for column in matchup_frame.columns
        if column.startswith("low_") or column.startswith("high_")
        if not column.endswith("_conf_abbrev")
    ]
    seed_columns = [
        "low_prev_seed",
        "high_prev_seed",
    ]

    matchup_frame = matchup_frame.with_columns(
        *[
            pl.col(column).fill_null(LOW_HISTORY_DEFAULT_SEED if column in seed_columns else 0.0)
            for column in fill_zero_columns
        ]
    )

    delta_feature_pairs = [
        "prev_games",
        "prev_wins",
        "prev_win_pct",
        "prev_avg_points_for",
        "prev_avg_points_against",
        "prev_avg_margin",
        "prev_home_win_pct",
        "prev_away_win_pct",
        "prev_neutral_win_pct",
        "prev_avg_opp_win_pct",
        "prev_avg_opp_margin",
        "prev_conf_avg_win_pct",
        "prev_conf_avg_margin",
        "prev_conf_team_count",
        "prev_tourney_games",
        "prev_tourney_wins",
        "prev_tourney_avg_margin",
        "prev_tourney_appearance",
        "prev_seed",
        "prev_seeded",
        "rolling_3_win_pct",
        "rolling_3_avg_margin",
        "rolling_3_avg_opp_win_pct",
        "rolling_3_tourney_wins",
        "career_games_through_prev_season",
        "career_wins_through_prev_season",
        "career_win_pct",
    ]

    if config.use_massey:
        if men_feature_variant == "second_order":
            delta_feature_pairs.extend(
                [
                    "prev_avg_opp_opp_win_pct",
                    "prev_avg_opp_opp_margin",
                ]
            )
        if men_feature_variant == "zscore":
            delta_feature_pairs.extend(
                [
                    "prev_win_pct_z",
                    "prev_avg_points_for_z",
                    "prev_avg_points_against_z",
                    "prev_avg_margin_z",
                    "prev_avg_opp_win_pct_z",
                ]
            )
        delta_feature_pairs.extend(
            [
                "prev_massey_mean_rank",
                "prev_massey_best_rank",
                "prev_massey_system_count",
            ]
        )

    if config.prefix == "M":
        delta_feature_pairs.extend(["program_tenure", "team_active_flag"])

    matchup_frame = matchup_frame.with_columns(
        *[
            (pl.col(f"low_{feature_name}") - pl.col(f"high_{feature_name}")).alias(f"delta_{feature_name}")
            for feature_name in delta_feature_pairs
        ]
    )

    feature_columns = [
        column
        for column in matchup_frame.columns
        if column not in {"Season", "Target", "LowerTeamID", "HigherTeamID"}
        and not column.endswith("_conf_abbrev")
    ]

    return matchup_frame.select(
        "Season",
        "LowerTeamID",
        "HigherTeamID",
        "Target",
        *feature_columns,
    ), feature_columns


def build_submission_matchups(
    config: DivisionConfig,
    preseason_features: pl.DataFrame,
    feature_columns: list[str],
) -> pl.DataFrame:
    sample_submission = pl.read_csv(DATA_DIR / "SampleSubmissionStage2.csv").with_columns(
        pl.col("ID").str.split_exact("_", 2).alias("id_parts")
    ).select(
        pl.col("ID"),
        pl.col("id_parts").struct.field("field_0").cast(pl.Int64).alias("Season"),
        pl.col("id_parts").struct.field("field_1").cast(pl.Int64).alias("LowerTeamID"),
        pl.col("id_parts").struct.field("field_2").cast(pl.Int64).alias("HigherTeamID"),
    )

    sample_submission = sample_submission.filter(
        pl.col("LowerTeamID").is_between(config.team_min_id, config.team_max_id)
    )

    low_team_features = preseason_features.select(
        pl.col("FeatureSeason").alias("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        *[pl.col(column).alias(f"low_{column}") for column in preseason_features.columns if column not in {"FeatureSeason", "TeamID"}],
    )
    high_team_features = preseason_features.select(
        pl.col("FeatureSeason").alias("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        *[pl.col(column).alias(f"high_{column}") for column in preseason_features.columns if column not in {"FeatureSeason", "TeamID"}],
    )

    submission_matchups = sample_submission.join(
        low_team_features,
        on=["Season", "LowerTeamID"],
        how="left",
    ).join(
        high_team_features,
        on=["Season", "HigherTeamID"],
        how="left",
    )

    submission_matchups = submission_matchups.with_columns(
        pl.col("low_prev_games").is_not_null().cast(pl.Int8).alias("low_has_history"),
        pl.col("high_prev_games").is_not_null().cast(pl.Int8).alias("high_has_history"),
        pl.when(pl.col("low_prev_conf_abbrev").is_not_null() & pl.col("high_prev_conf_abbrev").is_not_null())
        .then((pl.col("low_prev_conf_abbrev") == pl.col("high_prev_conf_abbrev")).cast(pl.Int8))
        .otherwise(0)
        .alias("same_prev_conference"),
    )

    fill_zero_columns = [
        column
        for column in submission_matchups.columns
        if column.startswith("low_") or column.startswith("high_")
        if not column.endswith("_conf_abbrev")
    ]
    submission_matchups = submission_matchups.with_columns(
        *[
            pl.col(column).fill_null(LOW_HISTORY_DEFAULT_SEED if column in {"low_prev_seed", "high_prev_seed"} else 0.0)
            for column in fill_zero_columns
        ]
    )

    derived_columns = [column for column in feature_columns if column.startswith("delta_")]
    submission_matchups = submission_matchups.with_columns(
        *[
            (pl.col(column.replace("delta_", "low_")) - pl.col(column.replace("delta_", "high_"))).alias(column)
            for column in derived_columns
        ]
    )

    return submission_matchups.select("ID", *feature_columns)


def train_variant_models(
    training_matrix: np.ndarray,
    training_target: np.ndarray,
    division_name: str,
) -> dict[str, lgb.LGBMClassifier]:
    if division_name == "men":
        models = {
            "baseline": lgb.LGBMClassifier(**BASELINE_LIGHTGBM_PARAMS),
            "smooth": lgb.LGBMClassifier(**MEN_SMOOTH_LIGHTGBM_PARAMS),
        }
    else:
        models = {
            "baseline": lgb.LGBMClassifier(**BASELINE_LIGHTGBM_PARAMS),
        }

    for model in models.values():
        model.fit(training_matrix, training_target)
    return models


def predict_variant_probabilities(
    models: dict[str, lgb.LGBMClassifier],
    feature_matrix: np.ndarray,
    division_name: str,
) -> np.ndarray:
    if division_name == "men":
        return shrink_probabilities(
            blend_men_probabilities(
                models["baseline"].predict_proba(feature_matrix)[:, 1],
                models["smooth"].predict_proba(feature_matrix)[:, 1],
            ),
            division_name=division_name,
        )

    return shrink_probabilities(
        models["baseline"].predict_proba(feature_matrix)[:, 1],
        division_name=division_name,
    )


def train_division_model(
    matchups_by_variant: dict[str, pl.DataFrame],
    feature_columns_by_variant: dict[str, list[str]],
    validation_season: int,
    division_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if division_name == "men":
        blended_probabilities: np.ndarray | None = None
        validation_targets: np.ndarray | None = None

        for variant_name, weight in MEN_FEATURE_VARIANT_WEIGHTS.items():
            variant_matchups = matchups_by_variant[variant_name]
            feature_columns = feature_columns_by_variant[variant_name]
            training_rows = variant_matchups.filter(pl.col("Season") < validation_season)
            validation_rows = variant_matchups.filter(pl.col("Season") == validation_season)
            models = train_variant_models(
                training_matrix=training_rows.select(feature_columns).to_numpy(),
                training_target=training_rows["Target"].to_numpy(),
                division_name=division_name,
            )
            variant_probabilities = predict_variant_probabilities(
                models=models,
                feature_matrix=validation_rows.select(feature_columns).to_numpy(),
                division_name=division_name,
            )
            if blended_probabilities is None:
                blended_probabilities = np.zeros_like(variant_probabilities)
                validation_targets = validation_rows["Target"].to_numpy()
            blended_probabilities = blended_probabilities + weight * variant_probabilities

        return np.clip(blended_probabilities, 0.0, 1.0), validation_targets

    feature_columns = feature_columns_by_variant["base"]
    matchups = matchups_by_variant["base"]
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)
    models = train_variant_models(
        training_matrix=training_rows.select(feature_columns).to_numpy(),
        training_target=training_rows["Target"].to_numpy(),
        division_name=division_name,
    )
    probabilities = predict_variant_probabilities(
        models=models,
        feature_matrix=validation_rows.select(feature_columns).to_numpy(),
        division_name=division_name,
    )
    return probabilities, validation_rows["Target"].to_numpy()


def train_final_division_models(
    matchups_by_variant: dict[str, pl.DataFrame],
    feature_columns_by_variant: dict[str, list[str]],
    target_season: int,
    division_name: str,
) -> dict[str, dict[str, lgb.LGBMClassifier]]:
    variant_names = MEN_FEATURE_VARIANT_WEIGHTS.keys() if division_name == "men" else ("base",)
    trained_models: dict[str, dict[str, lgb.LGBMClassifier]] = {}

    for variant_name in variant_names:
        feature_columns = feature_columns_by_variant[variant_name]
        matchups = matchups_by_variant[variant_name]
        training_rows = matchups.filter(pl.col("Season") < target_season)
        trained_models[variant_name] = train_variant_models(
            training_matrix=training_rows.select(feature_columns).to_numpy(),
            training_target=training_rows["Target"].to_numpy(),
            division_name=division_name,
        )

    return trained_models


def latest_completed_validation_seasons() -> list[int]:
    men_seasons = set(pl.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")["Season"].to_list())
    women_seasons = set(pl.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")["Season"].to_list())
    return sorted(men_seasons & women_seasons)[-VALIDATION_FOLD_COUNT:]


def run_validation(
    matchups_by_division: dict[str, dict[str, pl.DataFrame]],
    feature_columns_by_division: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    validation_seasons = latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        all_probabilities: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        division_scores: dict[str, float] = {}

        for config in DIVISION_CONFIGS:
            probabilities, targets = train_division_model(
                matchups_by_variant=matchups_by_division[config.name],
                feature_columns_by_variant=feature_columns_by_division[config.name],
                validation_season=validation_season,
                division_name=config.name,
            )
            division_scores[config.name] = brier_score(probabilities, targets)
            all_probabilities.append(probabilities)
            all_targets.append(targets)

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": brier_score(np.concatenate(all_probabilities), np.concatenate(all_targets)),
                "men_brier": division_scores["men"],
                "women_brier": division_scores["women"],
            }
        )

    return {
        "validation_seasons": validation_seasons,
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def build_submission(
    preseason_features_by_division: dict[str, dict[str, pl.DataFrame]],
    matchups_by_division: dict[str, dict[str, pl.DataFrame]],
    feature_columns_by_division: dict[str, dict[str, list[str]]],
) -> pl.DataFrame:
    division_submissions: list[pl.DataFrame] = []

    for config in DIVISION_CONFIGS:
        final_models = train_final_division_models(
            matchups_by_variant=matchups_by_division[config.name],
            feature_columns_by_variant=feature_columns_by_division[config.name],
            target_season=2026,
            division_name=config.name,
        )

        if config.name == "men":
            probabilities = None
            submission_ids = None

            for variant_name, weight in MEN_FEATURE_VARIANT_WEIGHTS.items():
                submission_matchups = build_submission_matchups(
                    config=config,
                    preseason_features=preseason_features_by_division[config.name][variant_name],
                    feature_columns=feature_columns_by_division[config.name][variant_name],
                )
                variant_probabilities = predict_variant_probabilities(
                    models=final_models[variant_name],
                    feature_matrix=submission_matchups.select(feature_columns_by_division[config.name][variant_name]).to_numpy(),
                    division_name=config.name,
                )
                if probabilities is None:
                    probabilities = np.zeros_like(variant_probabilities)
                    submission_ids = submission_matchups.select("ID")
                probabilities = probabilities + weight * variant_probabilities

            division_submissions.append(
                submission_ids.with_columns(pl.Series("Pred", np.clip(probabilities, 0.0, 1.0)))
            )
        else:
            submission_matchups = build_submission_matchups(
                config=config,
                preseason_features=preseason_features_by_division[config.name]["base"],
                feature_columns=feature_columns_by_division[config.name]["base"],
            )
            probabilities = predict_variant_probabilities(
                models=final_models["base"],
                feature_matrix=submission_matchups.select(feature_columns_by_division[config.name]["base"]).to_numpy(),
                division_name=config.name,
            )
            division_submissions.append(
                submission_matchups.select("ID").with_columns(pl.Series("Pred", np.clip(probabilities, 0.0, 1.0)))
            )

        joblib.dump(
            {
                "models": final_models,
                "feature_columns": feature_columns_by_division[config.name],
                "baseline_lightgbm_params": BASELINE_LIGHTGBM_PARAMS,
                "men_smooth_lightgbm_params": MEN_SMOOTH_LIGHTGBM_PARAMS if config.name == "men" else None,
                "probability_shrink_by_division": PROBABILITY_SHRINK_BY_DIVISION,
                "men_ensemble_weights": MEN_ENSEMBLE_WEIGHTS if config.name == "men" else None,
                "men_feature_variant_weights": MEN_FEATURE_VARIANT_WEIGHTS if config.name == "men" else None,
            },
            ARTIFACT_DIR / f"{config.name}_lightgbm.joblib",
        )

    return pl.concat(division_submissions, how="vertical").sort("ID")


def write_validation_summary(validation_summary: dict[str, object]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "baseline_lightgbm_params": BASELINE_LIGHTGBM_PARAMS,
        "men_smooth_lightgbm_params": MEN_SMOOTH_LIGHTGBM_PARAMS,
        "probability_shrink_by_division": PROBABILITY_SHRINK_BY_DIVISION,
        "men_ensemble_weights": MEN_ENSEMBLE_WEIGHTS,
        "men_feature_variant_weights": MEN_FEATURE_VARIANT_WEIGHTS,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    preseason_features_by_division: dict[str, dict[str, pl.DataFrame]] = {}
    matchups_by_division: dict[str, dict[str, pl.DataFrame]] = {}
    feature_columns_by_division: dict[str, dict[str, list[str]]] = {}

    for config in DIVISION_CONFIGS:
        if config.name == "men":
            preseason_features_by_division[config.name] = {}
            matchups_by_division[config.name] = {}
            feature_columns_by_division[config.name] = {}
            for variant_name in MEN_FEATURE_VARIANT_WEIGHTS:
                preseason_features = build_preseason_team_features(config, men_feature_variant=variant_name)
                matchups, feature_columns = build_training_matchups(
                    config,
                    preseason_features,
                    men_feature_variant=variant_name,
                )
                preseason_features_by_division[config.name][variant_name] = preseason_features
                matchups_by_division[config.name][variant_name] = matchups
                feature_columns_by_division[config.name][variant_name] = feature_columns
        else:
            preseason_features = build_preseason_team_features(config)
            matchups, feature_columns = build_training_matchups(config, preseason_features)
            preseason_features_by_division[config.name] = {"base": preseason_features}
            matchups_by_division[config.name] = {"base": matchups}
            feature_columns_by_division[config.name] = {"base": feature_columns}

    validation_summary = run_validation(matchups_by_division, feature_columns_by_division)
    submission = build_submission(
        preseason_features_by_division=preseason_features_by_division,
        matchups_by_division=matchups_by_division,
        feature_columns_by_division=feature_columns_by_division,
    )

    submission.write_csv(ARTIFACT_DIR / "submission.csv")
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
