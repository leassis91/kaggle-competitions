from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from sklearn.linear_model import LogisticRegression

import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_38_vilnius_plus_capped_quality as feature_stack
import train_iteration_41_iteration_38_optuna_params as tuned


ARTIFACT_DIR = Path("artifacts/iteration_43_iteration_38_optuna_blend_conservative")
MARGIN_BLEND_WEIGHTS = {
    "baseline": 0.8,
    "tuned": 0.2,
}


def raw_margin_predictions(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    baseline_predictions, targets, validation_rows = feature_stack.benchmark.benchmark.baseline.raw_margin_predictions(
        matchups=matchups,
        feature_columns=feature_columns,
        validation_season=validation_season,
    )
    tuned_predictions, _tuned_targets, _tuned_rows = tuned.raw_margin_predictions(
        matchups=matchups,
        feature_columns=feature_columns,
        validation_season=validation_season,
    )
    blended_predictions = (
        MARGIN_BLEND_WEIGHTS["baseline"] * baseline_predictions
        + MARGIN_BLEND_WEIGHTS["tuned"] * tuned_predictions
    )
    return blended_predictions, targets, validation_rows


def build_calibration_matrix(predictions: np.ndarray, validation_rows: pl.DataFrame) -> np.ndarray:
    return tuned.build_calibration_matrix(predictions, validation_rows)


def fit_margin_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
) -> LogisticRegression | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets, validation_rows = raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
        )
        oof_matrices.append(build_calibration_matrix(predictions, validation_rows))
        oof_targets.append(targets)

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
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


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=validation_season,
        )
        calibrator = fit_margin_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
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


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": {
            "baseline": feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
            "tuned": tuned.LIGHTGBM_PARAMS,
            "margin_blend_weights": MARGIN_BLEND_WEIGHTS,
        },
        "feature_columns": feature_columns,
        "minimum_pooled_season": pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_blended_margin_plus_bracket_context",
        "feature_family": "iteration_38_feature_stack_with_conservative_baseline_tuned_margin_blend",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = feature_stack.build_pooled_margin_matchups()
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
