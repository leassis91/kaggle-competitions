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
import train_iteration_53_iteration_44_iteration_50_blend as model_53
import train_iteration_56_iteration_53_division_specific_calibration as model_56


ARTIFACT_DIR = Path("artifacts/iteration_59_iteration_53_56_probability_stack")


def model_53_probabilities(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    benchmark_predictions, targets, validation_rows = model_53.benchmark.raw_margin_predictions(
        matchups=benchmark_matchups,
        feature_columns=benchmark_feature_columns,
        validation_season=validation_season,
    )
    alternative_predictions, _alternative_targets, _alternative_rows = model_53.benchmark.raw_margin_predictions(
        matchups=alternative_matchups,
        feature_columns=alternative_feature_columns,
        validation_season=validation_season,
    )
    blended_predictions = (
        model_53.MARGIN_BLEND_WEIGHTS["benchmark"] * benchmark_predictions
        + model_53.MARGIN_BLEND_WEIGHTS["alternative"] * alternative_predictions
    )
    calibrator = model_53.fit_margin_calibrator(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
        target_season=validation_season,
    )
    probabilities = model_53.apply_margin_calibration(blended_predictions, validation_rows, calibrator)
    return probabilities, targets, validation_rows


def model_56_probabilities(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    raw_predictions, targets, validation_rows = model_56.benchmark.raw_margin_predictions(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
        validation_season=validation_season,
    )
    men_calibrator = model_56.fit_margin_calibrator(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
        target_season=validation_season,
        men_women_flag=1,
    )
    women_calibrator = model_56.fit_margin_calibrator(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
        target_season=validation_season,
        men_women_flag=0,
    )
    probabilities = model_56.apply_margin_calibration(
        predictions=raw_predictions,
        validation_rows=validation_rows,
        men_calibrator=men_calibrator,
        women_calibrator=women_calibrator,
    )
    return probabilities, targets, validation_rows


def build_meta_matrix(
    probabilities_53: np.ndarray,
    probabilities_56: np.ndarray,
    validation_rows: pl.DataFrame,
) -> np.ndarray:
    return np.column_stack(
        [
            probabilities_53,
            probabilities_56,
            validation_rows["men_women"].to_numpy(),
            validation_rows["possible_round"].to_numpy(),
            validation_rows["delta_seed_number"].to_numpy(),
            validation_rows["same_region_flag"].to_numpy(),
            validation_rows["any_play_in_flag"].to_numpy(),
            probabilities_56 - probabilities_53,
        ]
    )


def fit_meta_calibrator(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    target_season: int,
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        probabilities_53, targets, validation_rows = model_53_probabilities(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=calibration_season,
        )
        probabilities_56, _targets_56, _validation_rows_56 = model_56_probabilities(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=calibration_season,
        )
        oof_matrices.append(build_meta_matrix(probabilities_53, probabilities_56, validation_rows))
        oof_targets.append(targets)

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def apply_meta_calibration(
    probabilities_53: np.ndarray,
    probabilities_56: np.ndarray,
    validation_rows: pl.DataFrame,
    calibrator: LogisticRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(probabilities_53, 0.0, 1.0)
    meta_matrix = build_meta_matrix(probabilities_53, probabilities_56, validation_rows)
    return np.clip(calibrator.predict_proba(meta_matrix)[:, 1], 0.0, 1.0)


def run_validation(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        probabilities_53, targets, validation_rows = model_53_probabilities(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=validation_season,
        )
        probabilities_56, _targets_56, _validation_rows_56 = model_56_probabilities(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=validation_season,
        )
        calibrator = fit_meta_calibrator(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            target_season=validation_season,
        )
        probabilities = apply_meta_calibration(
            probabilities_53=probabilities_53,
            probabilities_56=probabilities_56,
            validation_rows=validation_rows,
            calibrator=calibrator,
        )
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


def write_validation_summary(
    validation_summary: dict[str, object],
    benchmark_feature_columns: list[str],
    alternative_feature_columns: list[str],
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": {
            "model_53": {
                "benchmark": {
                    "men": model_53.benchmark.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                    "women": model_53.benchmark.tuned.LIGHTGBM_PARAMS,
                },
                "alternative": {
                    "men": model_53.alternative.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                    "women": model_53.alternative.benchmark.tuned.LIGHTGBM_PARAMS,
                },
                "margin_blend_weights": model_53.MARGIN_BLEND_WEIGHTS,
            },
        },
        "benchmark_feature_columns": benchmark_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "minimum_pooled_season": pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_stack_from_iteration_53_and_iteration_56_probabilities",
        "feature_family": "iteration_53_iteration_56_probability_stack",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_matchups, benchmark_feature_columns = model_53.benchmark.feature_stack.build_pooled_margin_matchups()
    alternative_matchups, alternative_feature_columns = model_53.alternative.build_pooled_margin_matchups()
    validation_summary = run_validation(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    write_validation_summary(
        validation_summary=validation_summary,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
    )

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
