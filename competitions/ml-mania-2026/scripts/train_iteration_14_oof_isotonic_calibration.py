from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import orjson
import polars as pl
from sklearn.isotonic import IsotonicRegression

import train_iteration_04_men_ensemble as baseline


ARTIFACT_DIR = Path("artifacts/iteration_14_oof_isotonic_calibration")


def raw_division_probabilities(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    division_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)
    training_matrix = training_rows.select(feature_columns).to_numpy()
    training_target = training_rows["Target"].to_numpy()
    validation_matrix = validation_rows.select(feature_columns).to_numpy()

    if division_name == "men":
        baseline_model = lgb.LGBMClassifier(**baseline.BASELINE_LIGHTGBM_PARAMS)
        smooth_model = lgb.LGBMClassifier(**baseline.MEN_SMOOTH_LIGHTGBM_PARAMS)
        baseline_model.fit(training_matrix, training_target)
        smooth_model.fit(training_matrix, training_target)
        probabilities = baseline.blend_men_probabilities(
            baseline_model.predict_proba(validation_matrix)[:, 1],
            smooth_model.predict_proba(validation_matrix)[:, 1],
        )
    else:
        model = lgb.LGBMClassifier(**baseline.BASELINE_LIGHTGBM_PARAMS)
        model.fit(training_matrix, training_target)
        probabilities = model.predict_proba(validation_matrix)[:, 1]

    return probabilities, validation_rows["Target"].to_numpy()


def fit_oof_isotonic_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    division_name: str,
    target_season: int,
) -> IsotonicRegression | None:
    training_seasons = sorted(
        season for season in matchups["Season"].unique().to_list() if season < target_season
    )
    oof_probabilities: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        probabilities, targets = raw_division_probabilities(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
            division_name=division_name,
        )
        oof_probabilities.append(probabilities)
        oof_targets.append(targets)

    if not oof_probabilities:
        return None

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.concatenate(oof_probabilities), np.concatenate(oof_targets))
    return calibrator


def apply_calibration(probabilities: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    if calibrator is None:
        return probabilities
    return np.clip(calibrator.transform(probabilities), 0.0, 1.0)


def run_validation(matchups_by_division: dict[str, pl.DataFrame], feature_columns_by_division: dict[str, list[str]]) -> dict[str, object]:
    validation_seasons = baseline.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        all_probabilities: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        division_scores: dict[str, float] = {}

        for config in baseline.DIVISION_CONFIGS:
            raw_probabilities, targets = raw_division_probabilities(
                matchups=matchups_by_division[config.name],
                feature_columns=feature_columns_by_division[config.name],
                validation_season=validation_season,
                division_name=config.name,
            )
            calibrator = fit_oof_isotonic_calibrator(
                matchups=matchups_by_division[config.name],
                feature_columns=feature_columns_by_division[config.name],
                division_name=config.name,
                target_season=validation_season,
            )
            probabilities = apply_calibration(raw_probabilities, calibrator)
            division_scores[config.name] = baseline.brier_score(probabilities, targets)
            all_probabilities.append(probabilities)
            all_targets.append(targets)

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": baseline.brier_score(np.concatenate(all_probabilities), np.concatenate(all_targets)),
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
    preseason_features_by_division: dict[str, pl.DataFrame],
    matchups_by_division: dict[str, pl.DataFrame],
    feature_columns_by_division: dict[str, list[str]],
) -> pl.DataFrame:
    division_submissions: list[pl.DataFrame] = []

    for config in baseline.DIVISION_CONFIGS:
        final_models = baseline.train_final_division_models(
            matchups=matchups_by_division[config.name],
            feature_columns=feature_columns_by_division[config.name],
            target_season=2026,
            division_name=config.name,
        )
        calibrator = fit_oof_isotonic_calibrator(
            matchups=matchups_by_division[config.name],
            feature_columns=feature_columns_by_division[config.name],
            division_name=config.name,
            target_season=2026,
        )
        submission_matchups = baseline.build_submission_matchups(
            config=config,
            preseason_features=preseason_features_by_division[config.name],
            feature_columns=feature_columns_by_division[config.name],
        )
        submission_matrix = submission_matchups.select(feature_columns_by_division[config.name]).to_numpy()

        if config.name == "men":
            raw_probabilities = baseline.blend_men_probabilities(
                final_models["baseline"].predict_proba(submission_matrix)[:, 1],
                final_models["smooth"].predict_proba(submission_matrix)[:, 1],
            )
        else:
            raw_probabilities = final_models["baseline"].predict_proba(submission_matrix)[:, 1]

        probabilities = apply_calibration(raw_probabilities, calibrator)
        division_submissions.append(
            submission_matchups.select("ID").with_columns(pl.Series("Pred", probabilities))
        )
        joblib.dump(
            {
                "models": final_models,
                "calibrator": calibrator,
                "feature_columns": feature_columns_by_division[config.name],
                "baseline_lightgbm_params": baseline.BASELINE_LIGHTGBM_PARAMS,
                "men_smooth_lightgbm_params": baseline.MEN_SMOOTH_LIGHTGBM_PARAMS if config.name == "men" else None,
                "men_ensemble_weights": baseline.MEN_ENSEMBLE_WEIGHTS if config.name == "men" else None,
            },
            ARTIFACT_DIR / f"{config.name}_lightgbm.joblib",
        )

    return pl.concat(division_submissions, how="vertical").sort("ID")


def write_validation_summary(validation_summary: dict[str, object]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "baseline_lightgbm_params": baseline.BASELINE_LIGHTGBM_PARAMS,
        "men_smooth_lightgbm_params": baseline.MEN_SMOOTH_LIGHTGBM_PARAMS,
        "men_ensemble_weights": baseline.MEN_ENSEMBLE_WEIGHTS,
        "probability_calibration": "nested_oof_isotonic_by_division",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    preseason_features_by_division: dict[str, pl.DataFrame] = {}
    matchups_by_division: dict[str, pl.DataFrame] = {}
    feature_columns_by_division: dict[str, list[str]] = {}

    for config in baseline.DIVISION_CONFIGS:
        preseason_features = baseline.build_preseason_team_features(config)
        matchups, feature_columns = baseline.build_training_matchups(config, preseason_features)
        preseason_features_by_division[config.name] = preseason_features
        matchups_by_division[config.name] = matchups
        feature_columns_by_division[config.name] = feature_columns

    validation_summary = run_validation(matchups_by_division, feature_columns_by_division)
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
