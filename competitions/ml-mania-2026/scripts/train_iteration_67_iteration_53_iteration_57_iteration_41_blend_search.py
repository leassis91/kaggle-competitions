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
import train_iteration_41_iteration_38_optuna_params as pooled_tuned
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_57_iteration_44_iteration_50_men_only_blend as alternative


ARTIFACT_DIR = Path("artifacts/iteration_67_iteration_53_iteration_57_iteration_41_blend_search")
GRID_CONFIGS = [
    {"iteration_57_weight": 0.10, "iteration_41_weight": 0.01},
    {"iteration_57_weight": 0.10, "iteration_41_weight": 0.02},
    {"iteration_57_weight": 0.10, "iteration_41_weight": 0.03},
    {"iteration_57_weight": 0.20, "iteration_41_weight": 0.01},
    {"iteration_57_weight": 0.20, "iteration_41_weight": 0.02},
    {"iteration_57_weight": 0.20, "iteration_41_weight": 0.03},
    {"iteration_57_weight": 0.30, "iteration_41_weight": 0.01},
    {"iteration_57_weight": 0.30, "iteration_41_weight": 0.02},
    {"iteration_57_weight": 0.30, "iteration_41_weight": 0.03},
]


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads(
            (
                Path("artifacts/iteration_53_iteration_44_iteration_50_blend") / "validation_summary.json"
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


def raw_margin_predictions(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    validation_season: int,
    config: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    benchmark_predictions, targets, validation_rows = benchmark.raw_margin_predictions(
        benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_feature_columns=benchmark_benchmark_feature_columns,
        alternative_matchups=benchmark_alternative_matchups,
        alternative_feature_columns=benchmark_alternative_feature_columns,
        validation_season=validation_season,
    )
    alternative_predictions, _alternative_targets, _alternative_rows = alternative.raw_margin_predictions(
        benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_feature_columns=benchmark_benchmark_feature_columns,
        alternative_matchups=benchmark_alternative_matchups,
        alternative_feature_columns=benchmark_alternative_feature_columns,
        validation_season=validation_season,
    )
    iteration_41_predictions, _iteration_41_targets, _iteration_41_rows = pooled_tuned.raw_margin_predictions(
        matchups=pooled_matchups,
        feature_columns=pooled_feature_columns,
        validation_season=validation_season,
    )
    iteration_57_weight = float(config["iteration_57_weight"])
    iteration_41_weight = float(config["iteration_41_weight"])
    benchmark_weight = 1.0 - iteration_57_weight - iteration_41_weight
    blended_predictions = (
        benchmark_weight * benchmark_predictions
        + iteration_57_weight * alternative_predictions
        + iteration_41_weight * iteration_41_predictions
    )
    return blended_predictions, targets, validation_rows


def fit_margin_calibrator(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    target_season: int,
    config: dict[str, float],
) -> LogisticRegression | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        predictions, targets, validation_rows = raw_margin_predictions(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            validation_season=calibration_season,
            config=config,
        )
        oof_matrices.append(benchmark.benchmark.build_calibration_matrix(predictions, validation_rows))
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
    calibration_matrix = benchmark.benchmark.build_calibration_matrix(predictions, validation_rows)
    return np.clip(calibrator.predict_proba(calibration_matrix)[:, 1], 0.0, 1.0)


def run_validation(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    config: dict[str, float],
) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        raw_predictions, targets, validation_rows = raw_margin_predictions(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            validation_season=validation_season,
            config=config,
        )
        calibrator = fit_margin_calibrator(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            target_season=validation_season,
            config=config,
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
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "iteration_53_iteration_57_iteration_41_margin_blend_search",
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_four_source_margin_blend_plus_bracket_context",
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
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

    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        validation_summary = run_validation(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            config=config,
        )
        results.append(serialize_result(config, validation_summary))

    selected_result = best_accepted_result(results)
    write_search_summary(
        results=results,
        selected_result=selected_result,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
    )

    if selected_result is None:
        best_result = best_raw_result(results)
        print("No accepted configuration found.")
        print(f"Best raw mean validation Brier: {best_result['raw_mean_brier']:.6f}")
        print(f"Best raw config: {best_result['config']}")
        return

    print("Accepted configuration found.")
    print(f"Best mean validation Brier: {selected_result['raw_mean_brier']:.6f}")
    print(f"Best config: {selected_result['config']}")


if __name__ == "__main__":
    main()
