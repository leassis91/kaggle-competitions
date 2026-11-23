from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson

import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as search


ARTIFACT_DIR = Path("artifacts/iteration_75_iteration_73_probability_blend_boundary_search")
GRID_CONFIGS = [
    {"iteration_71_probability_weight": 0.251},
    {"iteration_71_probability_weight": 0.2525},
    {"iteration_71_probability_weight": 0.254},
    {"iteration_71_probability_weight": 0.255},
    {"iteration_71_probability_weight": 0.256},
    {"iteration_71_probability_weight": 0.2575},
    {"iteration_71_probability_weight": 0.259},
]


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads(
            (
                Path("artifacts/iteration_73_iteration_69_iteration_71_probability_blend")
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
        "feature_family": "iteration_73_probability_blend_boundary_search",
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "boundary_blended_probabilities_from_iteration_69_and_iteration_71",
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "fixed_weights": {
            "iteration_38": search.ITERATION_38_WEIGHT,
            "iteration_49": search.ITERATION_49_WEIGHT,
            "iteration_41": search.ITERATION_41_WEIGHT,
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

    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        validation_summary = search.run_validation(
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
        results.append(serialize_result(config, validation_summary))

    selected_result = best_accepted_result(results)
    write_search_summary(
        results=results,
        selected_result=selected_result,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
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
