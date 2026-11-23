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
import train_iteration_44_division_specific_margin_models as benchmark
import train_iteration_50_capped_seed_closure_quality as alternative


ARTIFACT_DIR = Path("artifacts/iteration_53_iteration_44_iteration_50_blend")
MARGIN_BLEND_WEIGHTS = {
    "benchmark": 0.75,
    "alternative": 0.25,
}


def raw_margin_predictions(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    benchmark_predictions, targets, validation_rows = benchmark.raw_margin_predictions(
        matchups=benchmark_matchups,
        feature_columns=benchmark_feature_columns,
        validation_season=validation_season,
    )
    alternative_predictions, _alternative_targets, _alternative_rows = benchmark.raw_margin_predictions(
        matchups=alternative_matchups,
        feature_columns=alternative_feature_columns,
        validation_season=validation_season,
    )
    blended_predictions = (
        MARGIN_BLEND_WEIGHTS["benchmark"] * benchmark_predictions
        + MARGIN_BLEND_WEIGHTS["alternative"] * alternative_predictions
    )
    return blended_predictions, targets, validation_rows


def fit_margin_calibrator(
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
        predictions, targets, validation_rows = raw_margin_predictions(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=calibration_season,
        )
        oof_matrices.append(benchmark.build_calibration_matrix(predictions, validation_rows))
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
    calibration_matrix = benchmark.build_calibration_matrix(predictions, validation_rows)
    return np.clip(calibrator.predict_proba(calibration_matrix)[:, 1], 0.0, 1.0)


def run_validation(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = raw_margin_predictions(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=validation_season,
        )
        calibrator = fit_margin_calibrator(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
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
            "benchmark": {
                "men": benchmark.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                "women": benchmark.tuned.LIGHTGBM_PARAMS,
            },
            "alternative": {
                "men": alternative.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                "women": alternative.benchmark.tuned.LIGHTGBM_PARAMS,
            },
            "margin_blend_weights": MARGIN_BLEND_WEIGHTS,
        },
        "benchmark_feature_columns": benchmark_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "minimum_pooled_season": pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_blended_division_specific_margin_plus_bracket_context",
        "feature_family": "iteration_44_iteration_50_conservative_margin_blend",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_matchups, benchmark_feature_columns = benchmark.feature_stack.build_pooled_margin_matchups()
    alternative_matchups, alternative_feature_columns = alternative.build_pooled_margin_matchups()
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
