from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import optuna
import orjson
import polars as pl
from sklearn.linear_model import LogisticRegression

import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_32_vilnius_filtered_quality as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_34_optuna_tune_iteration_32")
TRIAL_COUNT = 20
FROZEN_N_ESTIMATORS = 400
OPTUNA_SEED = 7
NON_ACCEPTED_PENALTY = 1.0


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads((Path("artifacts/iteration_32_vilnius_filtered_quality") / "validation_summary.json").read_bytes()),
    )


def benchmark_fold_scores() -> dict[int, float]:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    folds = cast(list[dict[str, float | int]], validation["folds"])
    return {int(fold["season"]): float(fold["combined_brier"]) for fold in folds}


def benchmark_mean_brier() -> float:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    return cast(float, validation["mean_brier"])


def build_model(params: dict[str, Any]) -> Any:
    return lgb.LGBMRegressor(**params)


def raw_margin_predictions(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)

    model = build_model(params)
    model.fit(training_rows.select(feature_columns).to_numpy(), training_rows["TargetMargin"].to_numpy())
    predictions = model.predict(validation_rows.select(feature_columns).to_numpy())
    return predictions, validation_rows["Target"].to_numpy(), validation_rows


def build_calibration_matrix(predictions: np.ndarray, validation_rows: pl.DataFrame) -> np.ndarray:
    possible_round = validation_rows["possible_round"].to_numpy()
    delta_seed_number = validation_rows["delta_seed_number"].to_numpy()
    same_region_flag = validation_rows["same_region_flag"].to_numpy()
    any_play_in_flag = validation_rows["any_play_in_flag"].to_numpy()

    return np.column_stack(
        [
            predictions,
            possible_round,
            delta_seed_number,
            same_region_flag,
            any_play_in_flag,
            predictions * possible_round,
            predictions * delta_seed_number,
            predictions * same_region_flag,
        ]
    )


def fit_margin_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
    params: dict[str, Any],
) -> LogisticRegression | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets, validation_rows = raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
            params=params,
        )
        oof_matrices.append(build_calibration_matrix(predictions, validation_rows))
        oof_targets.append(targets)

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=OPTUNA_SEED, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def apply_margin_calibration(
    predictions: np.ndarray,
    validation_rows: pl.DataFrame,
    calibrator: LogisticRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(1.0 / (1.0 + np.exp(-predictions / 10.0)), 0.0, 1.0)

    return np.clip(calibrator.predict_proba(build_calibration_matrix(predictions, validation_rows))[:, 1], 0.0, 1.0)


def run_validation(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=validation_season,
            params=params,
        )
        calibrator = fit_margin_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
            params=params,
        )
        probabilities = apply_margin_calibration(raw_predictions, validation_rows, calibrator)
        men_mask = validation_rows["men_women"].to_numpy() == 1
        women_mask = ~men_mask

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": pooled_base.brier_score(probabilities, targets),
                "men_brier": pooled_base.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": pooled_base.brier_score(probabilities[women_mask], targets[women_mask]),
            }
        )

    return {
        "validation_seasons": validation_seasons,
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "n_estimators": FROZEN_N_ESTIMATORS,
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 25, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "random_state": OPTUNA_SEED,
        "n_jobs": 4,
        "verbosity": -1,
    }


def improvement_summary(validation_summary: dict[str, object]) -> tuple[int, list[int]]:
    fold_scores = benchmark_fold_scores()
    candidate_folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    improved_seasons = [
        int(fold["season"])
        for fold in candidate_folds
        if float(fold["combined_brier"]) < fold_scores[int(fold["season"])]
    ]
    return len(improved_seasons), improved_seasons


def objective_factory(matchups: pl.DataFrame, feature_columns: list[str]):
    current_best_mean_brier = benchmark_mean_brier()

    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial)
        validation_summary = run_validation(matchups, feature_columns, params)
        mean_brier = cast(float, validation_summary["mean_brier"])
        improved_fold_count, improved_seasons = improvement_summary(validation_summary)
        accepted = mean_brier < current_best_mean_brier and improved_fold_count >= 4
        objective_value = mean_brier if accepted else current_best_mean_brier + NON_ACCEPTED_PENALTY + mean_brier

        trial.set_user_attr("validation_summary", validation_summary)
        trial.set_user_attr("improved_fold_count", improved_fold_count)
        trial.set_user_attr("improved_seasons", improved_seasons)
        trial.set_user_attr("accepted", accepted)
        trial.set_user_attr("raw_mean_brier", mean_brier)
        return objective_value

    return objective


def serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    return {
        "number": trial.number,
        "params": trial.params,
        "objective_value": trial.value,
        "accepted": bool(trial.user_attrs["accepted"]),
        "raw_mean_brier": float(trial.user_attrs["raw_mean_brier"]),
        "improved_fold_count": int(trial.user_attrs["improved_fold_count"]),
        "improved_seasons": cast(list[int], trial.user_attrs["improved_seasons"]),
        "validation_summary": cast(dict[str, object], trial.user_attrs["validation_summary"]),
    }


def best_accepted_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    accepted_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and bool(trial.user_attrs.get("accepted", False))]
    if not accepted_trials:
        return None
    return min(accepted_trials, key=lambda trial: float(trial.user_attrs["raw_mean_brier"]))


def best_raw_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    return min(completed_trials, key=lambda trial: float(trial.user_attrs["raw_mean_brier"]))


def write_tuning_summary(
    study: optuna.Study,
    selected_trial: optuna.trial.FrozenTrial | None,
    feature_columns: list[str],
) -> None:
    summary_payload: dict[str, object] = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "iteration_32_vilnius_filtered_quality",
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin_plus_bracket_context",
        "trial_count": TRIAL_COUNT,
        "frozen_n_estimators": FROZEN_N_ESTIMATORS,
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
        "feature_columns": feature_columns,
        "trials": [serialize_trial(trial) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
        "best_raw_trial": serialize_trial(best_raw_trial(study)),
        "selected_trial": serialize_trial(selected_trial) if selected_trial is not None else None,
        "accepted": selected_trial is not None,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = benchmark.build_pooled_margin_matchups()
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective_factory(matchups, feature_columns), n_trials=TRIAL_COUNT)
    selected_trial = best_accepted_trial(study)
    write_tuning_summary(study, selected_trial, feature_columns)

    if selected_trial is None:
        raw_trial = best_raw_trial(study)
        raw_mean_brier = float(raw_trial.user_attrs["raw_mean_brier"])
        print(f"No accepted trial found across {TRIAL_COUNT} trials.")
        print(f"Best raw mean validation Brier: {raw_mean_brier:.6f}")
        print(f"Best raw params: {raw_trial.params}")
        return

    raw_mean_brier = float(selected_trial.user_attrs["raw_mean_brier"])
    improved_fold_count = int(selected_trial.user_attrs["improved_fold_count"])
    print(f"Accepted mean validation Brier: {raw_mean_brier:.6f}")
    print(f"Improved folds: {improved_fold_count}/5")
    print(f"Best params: {selected_trial.params}")
    validation_summary = cast(dict[str, object], selected_trial.user_attrs["validation_summary"])
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
