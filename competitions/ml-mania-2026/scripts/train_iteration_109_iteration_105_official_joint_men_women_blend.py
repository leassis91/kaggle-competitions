from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson

import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_105_iteration_102_official_joint_men_women_fine_search as accepted_search


ARTIFACT_DIR = Path("artifacts/iteration_109_iteration_105_official_joint_men_women_blend")
CONFIG = {
    "men_gap_threshold": 0.04,
    "men_low_gap_iteration_71_probability_weight": 0.44,
    "men_high_gap_iteration_71_probability_weight": 0.36,
    "women_extreme_gap_iteration_71_probability_weight": 0.26,
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
                "iteration_38": accepted_search.search.ITERATION_38_WEIGHT,
                "iteration_49": accepted_search.search.ITERATION_49_WEIGHT,
                "iteration_41": accepted_search.search.ITERATION_41_WEIGHT,
                "men_gap_threshold": CONFIG["men_gap_threshold"],
                "men_low_gap_iteration_71_probability_weight": CONFIG["men_low_gap_iteration_71_probability_weight"],
                "men_high_gap_iteration_71_probability_weight": CONFIG["men_high_gap_iteration_71_probability_weight"],
                "women_low_gap_threshold": accepted_search.DEFAULT_CONFIG["women_low_gap_threshold"],
                "women_mid_gap_threshold": accepted_search.DEFAULT_CONFIG["women_mid_gap_threshold"],
                "women_high_gap_threshold": accepted_search.DEFAULT_CONFIG["women_high_gap_threshold"],
                "women_low_gap_iteration_71_probability_weight": accepted_search.DEFAULT_CONFIG[
                    "women_low_gap_iteration_71_probability_weight"
                ],
                "women_mid_gap_iteration_71_probability_weight": accepted_search.DEFAULT_CONFIG[
                    "women_mid_gap_iteration_71_probability_weight"
                ],
                "women_high_gap_iteration_71_probability_weight": accepted_search.DEFAULT_CONFIG[
                    "women_high_gap_iteration_71_probability_weight"
                ],
                "women_extreme_gap_iteration_71_probability_weight": CONFIG[
                    "women_extreme_gap_iteration_71_probability_weight"
                ],
            },
        },
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "minimum_pooled_season": accepted_search.search.pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "fixed_official_joint_men_gap_and_women_extreme_gap_probability_blend",
        "feature_family": "iteration_105_fixed_official_joint_men_women_probability_blend",
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

    cached_fold_inputs = accepted_search.division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    validation_summary = accepted_search.run_validation(cached_fold_inputs=cached_fold_inputs, config=CONFIG)
    write_validation_summary(
        validation_summary=validation_summary,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
    )

    print(f"Mean official-score validation Brier: {validation_summary['mean_brier']:.6f}")
    fold_summaries = cast(list[dict[str, float | int]], validation_summary["folds"])
    for fold_summary in fold_summaries:
        print(
            f"Season {fold_summary['season']}: "
            f"combined={fold_summary['combined_brier']:.6f}, "
            f"men={fold_summary['men_brier']:.6f}, "
            f"women={fold_summary['women_brier']:.6f}, "
            f"official_rows={fold_summary['official_row_count']}, "
            f"excluded_first_four={fold_summary['excluded_first_four_count']}"
        )


if __name__ == "__main__":
    main()
