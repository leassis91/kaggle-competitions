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
import train_iteration_44_division_specific_margin_models as baseline


ARTIFACT_DIR = Path("artifacts/iteration_45_division_specific_calibration")


def filtered_calibration_matrix(
    predictions: np.ndarray,
    validation_rows: pl.DataFrame,
    men_women_flag: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = validation_rows["men_women"].to_numpy() == men_women_flag
    return baseline.build_calibration_matrix(predictions, validation_rows)[mask], mask


def fit_margin_calibrator(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    target_season: int,
    men_women_flag: int,
) -> LogisticRegression | None:
    training_seasons = sorted(season for season in matchups["Season"].unique().to_list() if season < target_season)
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets, validation_rows = baseline.raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=calibration_season,
        )
        calibration_matrix, mask = filtered_calibration_matrix(predictions, validation_rows, men_women_flag)
        if calibration_matrix.size == 0:
            continue
        oof_matrices.append(calibration_matrix)
        oof_targets.append(targets[mask])

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def apply_margin_calibration(
    predictions: np.ndarray,
    validation_rows: pl.DataFrame,
    men_calibrator: LogisticRegression | None,
    women_calibrator: LogisticRegression | None,
) -> np.ndarray:
    probabilities = np.zeros(predictions.shape[0], dtype=np.float64)
    men_mask = validation_rows["men_women"].to_numpy() == 1
    women_mask = ~men_mask

    if men_calibrator is None:
        probabilities[men_mask] = np.clip(1.0 / (1.0 + np.exp(-predictions[men_mask] / 10.0)), 0.0, 1.0)
    else:
        men_matrix = baseline.build_calibration_matrix(predictions, validation_rows)[men_mask]
        probabilities[men_mask] = np.clip(men_calibrator.predict_proba(men_matrix)[:, 1], 0.0, 1.0)

    if women_calibrator is None:
        probabilities[women_mask] = np.clip(1.0 / (1.0 + np.exp(-predictions[women_mask] / 10.0)), 0.0, 1.0)
    else:
        women_matrix = baseline.build_calibration_matrix(predictions, validation_rows)[women_mask]
        probabilities[women_mask] = np.clip(women_calibrator.predict_proba(women_matrix)[:, 1], 0.0, 1.0)

    return probabilities


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = baseline.raw_margin_predictions(
            matchups=matchups,
            feature_columns=feature_columns,
            validation_season=validation_season,
        )
        men_calibrator = fit_margin_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
            men_women_flag=1,
        )
        women_calibrator = fit_margin_calibrator(
            matchups=matchups,
            feature_columns=feature_columns,
            target_season=validation_season,
            men_women_flag=0,
        )
        probabilities = apply_margin_calibration(raw_predictions, validation_rows, men_calibrator, women_calibrator)
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
            "men": baseline.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
            "women": baseline.tuned.LIGHTGBM_PARAMS,
        },
        "feature_columns": feature_columns,
        "minimum_pooled_season": pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_division_specific_logistic_from_division_specific_margin_plus_bracket_context",
        "feature_family": "iteration_44_plus_division_specific_calibration",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = baseline.feature_stack.build_pooled_margin_matchups()
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
