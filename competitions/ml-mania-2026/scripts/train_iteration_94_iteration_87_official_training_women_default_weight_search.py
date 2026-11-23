from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_76_iteration_73_division_specific_probability_blend_search as division_specific_search
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search


ARTIFACT_DIR = Path("artifacts/iteration_94_iteration_87_official_training_women_default_weight_search")


def filter_official_matchups(matchups: pl.DataFrame) -> pl.DataFrame:
    return matchups.filter(pl.col("first_four_pair_flag") == 0)


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
        "feature_family": "iteration_87_women_default_weight_search_with_first_four_excluded_training",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "women_default_weight_search_with_first_four_excluded_training_and_validation",
        "benchmark_config": official_search.BENCHMARK_CONFIG,
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
            "default_config": official_search.DEFAULT_CONFIG,
        },
        "results": results,
        "best_raw_result": official_search.best_raw_result(results),
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

    filtered_benchmark_benchmark_matchups = filter_official_matchups(benchmark_benchmark_matchups)
    filtered_benchmark_alternative_matchups = filter_official_matchups(benchmark_alternative_matchups)
    filtered_pooled_matchups = filter_official_matchups(pooled_matchups)
    filtered_alternative_matchups = filter_official_matchups(alternative_matchups)

    cached_fold_inputs = division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=filtered_benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=filtered_benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=filtered_pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=filtered_alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    benchmark_summary = official_search.run_validation(
        cached_fold_inputs=cached_fold_inputs,
        config=official_search.BENCHMARK_CONFIG,
    )
    benchmark_mean_brier = cast(float, benchmark_summary["mean_brier"])
    benchmark_fold_score_by_season = {
        int(fold["season"]): float(fold["combined_brier"])
        for fold in cast(list[dict[str, float | int]], benchmark_summary["folds"])
    }

    results: list[dict[str, object]] = []
    for config in official_search.GRID_CONFIGS:
        validation_summary = official_search.run_validation(cached_fold_inputs=cached_fold_inputs, config=config)
        results.append(
            official_search.serialize_result(
                config,
                validation_summary,
                benchmark_mean_brier,
                benchmark_fold_score_by_season,
            )
        )

    selected_result = official_search.best_accepted_result(results)
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
        best_result = official_search.best_raw_result(results)
        print("No accepted configuration found.")
        print(f"Best raw official-score mean validation Brier: {best_result['raw_mean_brier']:.6f}")
        print(f"Best raw config: {best_result['config']}")
        return

    print("Accepted configuration found.")
    print(f"Best official-score mean validation Brier: {selected_result['raw_mean_brier']:.6f}")
    print(f"Best config: {selected_result['config']}")


if __name__ == "__main__":
    main()
