from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from sklearn.linear_model import LogisticRegression

import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_46_division_aware_calibration_matrix as division_aware
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as base_search
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as benchmark_routing
import train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend as benchmark_model


ARTIFACT_DIR = Path("artifacts/iteration_114_iteration_111_official_source_weight_shortlist_search")
GRID_CONFIGS = [
    {"iteration_38_weight": 0.020, "iteration_49_weight": 0.015, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.015, "iteration_49_weight": 0.015, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.025, "iteration_49_weight": 0.015, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.020, "iteration_49_weight": 0.010, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.020, "iteration_49_weight": 0.020, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.020, "iteration_49_weight": 0.015, "iteration_41_weight": 0.000},
    {"iteration_38_weight": 0.015, "iteration_49_weight": 0.010, "iteration_41_weight": 0.010},
    {"iteration_38_weight": 0.025, "iteration_49_weight": 0.020, "iteration_41_weight": 0.010},
]


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads(
            (
                Path("artifacts/iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend")
                / "validation_summary.json"
            ).read_bytes()
        ),
    )


def benchmark_fold_scores() -> dict[int, float]:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    folds = cast(list[dict[str, float | int]], validation["folds"])
    return {int(fold["season"]): float(fold["combined_brier"]) for fold in folds}


def benchmark_mean_brier() -> float:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    return cast(float, validation["mean_brier"])


def iteration_69_raw_margin(source_predictions: dict[str, np.ndarray], config: dict[str, float]) -> np.ndarray:
    benchmark_weight = 1.0 - config["iteration_38_weight"] - config["iteration_49_weight"] - config["iteration_41_weight"]
    return (
        benchmark_weight * source_predictions["benchmark_margin"]
        + config["iteration_38_weight"] * source_predictions["iteration_38_margin"]
        + config["iteration_49_weight"] * source_predictions["iteration_49_margin"]
        + config["iteration_41_weight"] * source_predictions["iteration_41_margin"]
    )


def iteration_71_raw_margin(source_predictions: dict[str, np.ndarray], config: dict[str, float]) -> np.ndarray:
    benchmark_weight = 1.0 - config["iteration_38_weight"] - config["iteration_49_weight"]
    return (
        benchmark_weight * source_predictions["benchmark_margin"]
        + config["iteration_38_weight"] * source_predictions["iteration_38_margin"]
        + config["iteration_49_weight"] * source_predictions["iteration_49_margin"]
    )


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
    config: dict[str, float],
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        source_predictions, targets, validation_rows = base_search.raw_margin_predictions(
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
        raw_predictions = iteration_69_raw_margin(source_predictions, config)
        oof_matrices.append(benchmark.benchmark.build_calibration_matrix(raw_predictions, validation_rows))
        oof_targets.append(targets)

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
    config: dict[str, float],
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        source_predictions, targets, validation_rows = base_search.raw_margin_predictions(
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
        raw_predictions = iteration_71_raw_margin(source_predictions, config)
        oof_matrices.append(division_aware.build_calibration_matrix(raw_predictions, validation_rows))
        oof_targets.append(targets)

    if not oof_matrices:
        return None

    calibrator = LogisticRegression(random_state=7, max_iter=1000)
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def calibrated_probabilities(
    raw_predictions: np.ndarray,
    validation_rows: pl.DataFrame,
    calibrator: LogisticRegression | None,
    calibration_mode: str,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(1.0 / (1.0 + np.exp(-raw_predictions / 10.0)), 0.0, 1.0)
    if calibration_mode == "iteration_69":
        calibration_matrix = benchmark.benchmark.build_calibration_matrix(raw_predictions, validation_rows)
    else:
        calibration_matrix = division_aware.build_calibration_matrix(raw_predictions, validation_rows)
    return np.clip(calibrator.predict_proba(calibration_matrix)[:, 1], 0.0, 1.0)


def build_cached_fold_inputs(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    config: dict[str, float],
) -> list[dict[str, object]]:
    cached_fold_inputs: list[dict[str, object]] = []

    for validation_season in base_search.pooled_base.latest_completed_validation_seasons():
        source_predictions, targets, validation_rows = base_search.raw_margin_predictions(
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
            config=config,
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
            config=config,
        )
        iteration_69_probabilities = calibrated_probabilities(
            raw_predictions=iteration_69_raw_margin(source_predictions, config),
            validation_rows=validation_rows,
            calibrator=iteration_69_calibrator,
            calibration_mode="iteration_69",
        )
        iteration_71_probabilities = calibrated_probabilities(
            raw_predictions=iteration_71_raw_margin(source_predictions, config),
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


def improvement_summary(validation_summary: dict[str, object]) -> tuple[int, list[int]]:
    fold_scores = benchmark_fold_scores()
    candidate_folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    improved_seasons = [
        int(fold["season"])
        for fold in candidate_folds
        if float(fold["combined_brier"]) < fold_scores[int(fold["season"])]
    ]
    return len(improved_seasons), improved_seasons


def serialize_result(config: dict[str, float], validation_summary: dict[str, object]) -> dict[str, object]:
    improved_fold_count, improved_seasons = improvement_summary(validation_summary)
    mean_brier = cast(float, validation_summary["mean_brier"])
    accepted = mean_brier < benchmark_mean_brier() and improved_fold_count >= 4
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
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_feature_columns: list[str],
    pooled_feature_columns: list[str],
    alternative_feature_columns: list[str],
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "iteration_111_official_source_weight_shortlist_search",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "iteration_111_routing_with_rebuilt_source_blends_shortlist",
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "fixed_routing_config": benchmark_model.CONFIG,
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

    results: list[dict[str, object]] = []
    for index, config in enumerate(GRID_CONFIGS, start=1):
        cached_fold_inputs = build_cached_fold_inputs(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            config=config,
        )
        validation_summary = benchmark_routing.run_validation(
            cached_fold_inputs=cached_fold_inputs,
            config=benchmark_model.CONFIG,
        )
        result = serialize_result(config=config, validation_summary=validation_summary)
        results.append(result)
        print(
            f"[{index}/{len(GRID_CONFIGS)}] config={config} mean_brier={result['raw_mean_brier']:.6f} "
            f"improved_folds={result['improved_fold_count']}"
        )

    selected_result = best_accepted_result(results)
    write_search_summary(
        results=results,
        selected_result=selected_result,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
    )

    print(f"Official-score benchmark mean validation Brier: {benchmark_mean_brier():.6f}")

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
