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
import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_46_division_aware_calibration_matrix as division_aware
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as search


ARTIFACT_DIR = Path("artifacts/iteration_91_iteration_87_official_calibration_women_default_weight_search")
DEFAULT_CONFIG = {
    "men_iteration_71_probability_weight": 0.29,
    "women_gate_mode": "low_gap",
    "women_gate_iteration_71_probability_weight": 1.0,
}
BENCHMARK_CONFIG = {
    "women_gap_threshold": 0.054,
    "women_default_iteration_71_probability_weight": 0.12,
}
GRID_CONFIGS = [
    {
        "women_gap_threshold": women_gap_threshold,
        "women_default_iteration_71_probability_weight": women_default_iteration_71_probability_weight,
    }
    for women_gap_threshold in [0.0525, 0.0530, 0.0535, 0.0540, 0.0545, 0.0550]
    for women_default_iteration_71_probability_weight in [0.00, 0.04, 0.08, 0.12, 0.16]
]


def official_row_mask(validation_rows: pl.DataFrame) -> np.ndarray:
    return validation_rows["first_four_pair_flag"].to_numpy() == 0


def fit_iteration_69_calibrator(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    target_season: int,
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        source_predictions, targets, validation_rows = search.raw_margin_predictions(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=calibration_season,
        )
        keep_mask = official_row_mask(validation_rows)
        raw_predictions = search.iteration_69_raw_margin(source_predictions)
        calibration_matrix = benchmark.benchmark.build_calibration_matrix(raw_predictions, validation_rows)
        oof_matrices.append(calibration_matrix[keep_mask])
        oof_targets.append(targets[keep_mask])

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def fit_iteration_71_calibrator(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    target_season: int,
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        source_predictions, targets, validation_rows = search.raw_margin_predictions(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=calibration_season,
        )
        keep_mask = official_row_mask(validation_rows)
        raw_predictions = search.iteration_71_raw_margin(source_predictions)
        calibration_matrix = division_aware.build_calibration_matrix(raw_predictions, validation_rows)
        oof_matrices.append(calibration_matrix[keep_mask])
        oof_targets.append(targets[keep_mask])

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def build_cached_fold_inputs(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
) -> list[dict[str, object]]:
    cached_fold_inputs: list[dict[str, object]] = []

    for validation_season in pooled_base.latest_completed_validation_seasons():
        source_predictions, targets, validation_rows = search.raw_margin_predictions(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            validation_season=validation_season,
        )
        iteration_69_calibrator = fit_iteration_69_calibrator(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            target_season=validation_season,
        )
        iteration_71_calibrator = fit_iteration_71_calibrator(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            target_season=validation_season,
        )
        iteration_69_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_69_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_69_calibrator,
            calibration_mode="iteration_69",
        )
        iteration_71_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_71_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_71_calibrator,
            calibration_mode="iteration_71",
        )
        cached_fold_inputs.append(
            {
                "season": validation_season,
                "targets": targets,
                "validation_rows": validation_rows,
                "iteration_69_probabilities": iteration_69_probabilities,
                "iteration_71_probabilities": iteration_71_probabilities,
            }
        )

    return cached_fold_inputs


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, float],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    men_iteration_71_probability_weight = float(DEFAULT_CONFIG["men_iteration_71_probability_weight"])
    men_iteration_69_probability_weight = 1.0 - men_iteration_71_probability_weight
    women_default_iteration_71_probability_weight = float(config["women_default_iteration_71_probability_weight"])
    women_default_iteration_69_probability_weight = 1.0 - women_default_iteration_71_probability_weight
    women_gate_iteration_71_probability_weight = float(DEFAULT_CONFIG["women_gate_iteration_71_probability_weight"])
    women_gate_iteration_69_probability_weight = 1.0 - women_gate_iteration_71_probability_weight
    women_gap_threshold = float(config["women_gap_threshold"])

    for cached_fold_input in cached_fold_inputs:
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])

        men_women = validation_rows["men_women"].to_numpy()
        official_mask = official_row_mask(validation_rows)
        men_mask = (men_women == 1) & official_mask
        women_mask = (men_women == 0) & official_mask
        women_probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)
        women_gate_mask = (men_women == 0) & (women_probability_gap <= women_gap_threshold)
        women_default_mask = (men_women == 0) & ~women_gate_mask

        probabilities = np.empty_like(iteration_69_probabilities)
        probabilities[men_women == 1] = (
            men_iteration_69_probability_weight * iteration_69_probabilities[men_women == 1]
            + men_iteration_71_probability_weight * iteration_71_probabilities[men_women == 1]
        )
        probabilities[women_default_mask] = (
            women_default_iteration_69_probability_weight * iteration_69_probabilities[women_default_mask]
            + women_default_iteration_71_probability_weight * iteration_71_probabilities[women_default_mask]
        )
        probabilities[women_gate_mask] = (
            women_gate_iteration_69_probability_weight * iteration_69_probabilities[women_gate_mask]
            + women_gate_iteration_71_probability_weight * iteration_71_probabilities[women_gate_mask]
        )

        fold_summaries.append(
            {
                "season": season,
                "combined_brier": pooled_base.brier_score(probabilities[official_mask], targets[official_mask]),
                "men_brier": pooled_base.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": pooled_base.brier_score(probabilities[women_mask], targets[women_mask]),
                "official_row_count": int(official_mask.sum()),
                "excluded_first_four_count": int((~official_mask).sum()),
            }
        )

    return {
        "validation_seasons": [int(fold_summary["season"]) for fold_summary in fold_summaries],
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def improvement_summary(
    validation_summary: dict[str, object],
    benchmark_fold_score_by_season: dict[int, float],
) -> tuple[int, list[int]]:
    candidate_folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    improved_seasons = [
        int(fold["season"])
        for fold in candidate_folds
        if float(fold["combined_brier"]) < benchmark_fold_score_by_season[int(fold["season"])]
    ]
    return len(improved_seasons), improved_seasons


def serialize_result(
    config: dict[str, float],
    validation_summary: dict[str, object],
    benchmark_mean_brier: float,
    benchmark_fold_score_by_season: dict[int, float],
) -> dict[str, object]:
    improved_fold_count, improved_seasons = improvement_summary(validation_summary, benchmark_fold_score_by_season)
    mean_brier = cast(float, validation_summary["mean_brier"])
    accepted = mean_brier < benchmark_mean_brier and improved_fold_count >= 4
    return {
        "config": config,
        "accepted": accepted,
        "raw_mean_brier": mean_brier,
        "improved_fold_count": improved_fold_count,
        "improved_seasons": improved_seasons,
        "validation_summary": validation_summary,
    }


def best_accepted_result(results: list[dict[str, object]]) -> dict[str, object] | None:
    accepted_results = [result for result in results if bool(result["accepted"])]
    if not accepted_results:
        return None
    return min(accepted_results, key=lambda result: float(result["raw_mean_brier"]))


def best_raw_result(results: list[dict[str, object]]) -> dict[str, object]:
    return min(results, key=lambda result: float(result["raw_mean_brier"]))


def write_search_summary(
    results: list[dict[str, object]],
    selected_result: dict[str, object] | None,
    benchmark_summary: dict[str, object],
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_feature_columns: list[str],
    pooled_feature_columns: list[str],
    alternative_feature_columns: list[str],
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "iteration_87_women_default_weight_search_with_official_first_four_exclusion_and_calibration",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "women_default_weight_search_with_first_four_excluded_calibrator_training",
        "benchmark_config": BENCHMARK_CONFIG,
        "benchmark_mean_brier": cast(float, benchmark_summary["mean_brier"]),
        "benchmark_fold_scores": {
            str(int(fold["season"])): float(fold["combined_brier"])
            for fold in cast(list[dict[str, float | int]], benchmark_summary["folds"])
        },
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "fixed_weights": {
            "iteration_38": search.ITERATION_38_WEIGHT,
            "iteration_49": search.ITERATION_49_WEIGHT,
            "iteration_41": search.ITERATION_41_WEIGHT,
            "default_config": DEFAULT_CONFIG,
        },
        "results": results,
        "best_raw_result": best_raw_result(results),
        "selected_result": selected_result,
        "accepted": selected_result is not None,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_benchmark_matchups, benchmark_benchmark_feature_columns = (
        benchmark.benchmark.feature_stack.build_pooled_margin_matchups()
    )
    benchmark_alternative_matchups, benchmark_alternative_feature_columns = (
        benchmark.alternative.build_pooled_margin_matchups()
    )
    pooled_matchups, pooled_feature_columns = pooled_feature_stack.build_pooled_margin_matchups()
    alternative_matchups, alternative_feature_columns = alternative.build_pooled_margin_matchups()

    cached_fold_inputs = build_cached_fold_inputs(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    benchmark_summary = run_validation(cached_fold_inputs=cached_fold_inputs, config=BENCHMARK_CONFIG)
    benchmark_mean_brier = cast(float, benchmark_summary["mean_brier"])
    benchmark_fold_score_by_season = {
        int(fold["season"]): float(fold["combined_brier"])
        for fold in cast(list[dict[str, float | int]], benchmark_summary["folds"])
    }

    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        validation_summary = run_validation(cached_fold_inputs=cached_fold_inputs, config=config)
        results.append(
            serialize_result(
                config,
                validation_summary,
                benchmark_mean_brier,
                benchmark_fold_score_by_season,
            )
        )

    selected_result = best_accepted_result(results)
    write_search_summary(
        results=results,
        selected_result=selected_result,
        benchmark_summary=benchmark_summary,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
    )

    print(f"Official-score benchmark mean validation Brier: {benchmark_summary['mean_brier']:.6f}")

    if selected_result is None:
        best_result = best_raw_result(results)
        print("No accepted configuration found.")
        print(f"Best raw official-score mean validation Brier: {best_result['raw_mean_brier']:.6f}")
        print(f"Best raw config: {best_result['config']}")
        return

    print("Accepted configuration found.")
    print(f"Best official-score mean validation Brier: {selected_result['raw_mean_brier']:.6f}")
    print(f"Best config: {selected_result['config']}")


if __name__ == "__main__":
    main()
