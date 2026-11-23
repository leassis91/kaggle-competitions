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
import train_iteration_80_iteration_77_women_gap_gate_search as women_gap_search


ARTIFACT_DIR = Path("artifacts/iteration_83_iteration_82_women_gap_gate")
CONFIG = {
    "women_gate_mode": "low_gap",
    "women_gap_threshold": 0.055,
    "women_gate_iteration_71_probability_weight": 0.35,
}


def write_validation_summary(
    validation_summary: dict[str, object],
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_feature_columns: list[str],
    pooled_feature_columns: list[str],
    alternative_feature_columns: list[str],
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": {
            "benchmark": {
                "men": benchmark.benchmark.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                "women": benchmark.benchmark.tuned.LIGHTGBM_PARAMS,
            },
            "iteration_38": pooled_feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
            "iteration_49": {
                "men": alternative.benchmark.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                "women": alternative.benchmark.tuned.LIGHTGBM_PARAMS,
            },
            "fixed_weights": {
                "iteration_38": search.ITERATION_38_WEIGHT,
                "iteration_49": search.ITERATION_49_WEIGHT,
                "iteration_41": search.ITERATION_41_WEIGHT,
                "men_iteration_71_probability_weight": women_gap_search.DEFAULT_CONFIG["men_iteration_71_probability_weight"],
                "women_default_iteration_71_probability_weight": women_gap_search.DEFAULT_CONFIG[
                    "women_iteration_71_probability_weight"
                ],
                "women_gap_mode": CONFIG["women_gate_mode"],
                "women_gap_threshold": CONFIG["women_gap_threshold"],
                "women_gate_iteration_71_probability_weight": CONFIG["women_gate_iteration_71_probability_weight"],
            },
        },
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "minimum_pooled_season": search.pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "fixed_extended_women_gap_gated_division_specific_probability_blend_from_iteration_69_and_iteration_71",
        "feature_family": "iteration_82_fixed_women_gap_gate_probability_blend",
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

    cached_fold_inputs = women_gap_search.division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    validation_summary = women_gap_search.run_validation(cached_fold_inputs=cached_fold_inputs, config=CONFIG)
    write_validation_summary(
        validation_summary=validation_summary,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
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
