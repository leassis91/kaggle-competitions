from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl

import official_current_block_rate_benchmark as active_benchmark
import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search


ARTIFACT_DIR = Path("artifacts/iteration_119_iteration_116_block_rate_men_context_router_search")
GRID_CONFIGS = [
    {
        "men_context_gap_threshold": men_context_gap_threshold,
        "men_context_round_cutoff": men_context_round_cutoff,
        "men_context_seed_gap_max": men_context_seed_gap_max,
        "men_context_positive_block_late_required": men_context_positive_block_late_required,
        "men_context_iteration_71_probability_weight": men_context_iteration_71_probability_weight,
    }
    for men_context_gap_threshold in [0.015, 0.020, 0.025]
    for men_context_round_cutoff in [2, 4, 7]
    for men_context_seed_gap_max in [1, 2]
    for men_context_positive_block_late_required in [False, True]
    for men_context_iteration_71_probability_weight in [0.65, 0.80, 1.00]
]


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    men_context_gap_threshold = cast(float, config["men_context_gap_threshold"])
    men_context_round_cutoff = cast(int, config["men_context_round_cutoff"])
    men_context_seed_gap_max = cast(int, config["men_context_seed_gap_max"])
    men_context_positive_block_late_required = cast(bool, config["men_context_positive_block_late_required"])
    men_context_iteration_71_probability_weight = cast(float, config["men_context_iteration_71_probability_weight"])
    men_context_iteration_69_probability_weight = 1.0 - men_context_iteration_71_probability_weight

    for cached_fold_input in cached_fold_inputs:
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])

        probabilities = active_benchmark.benchmark_probabilities(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
        )
        official_mask = official_search.official_row_mask(validation_rows)
        men_rows_mask = validation_rows["men_women"].to_numpy() == 1
        probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)
        context_mask = men_rows_mask & (probability_gap <= men_context_gap_threshold) & (
            np.abs(validation_rows["delta_seed_number"].to_numpy()) <= men_context_seed_gap_max
        ) & (validation_rows["possible_round"].to_numpy() <= men_context_round_cutoff)

        if men_context_positive_block_late_required:
            context_mask = context_mask & (validation_rows["delta_block_feature_late"].to_numpy() > 0.0)

        probabilities[context_mask] = (
            men_context_iteration_69_probability_weight * iteration_69_probabilities[context_mask]
            + men_context_iteration_71_probability_weight * iteration_71_probabilities[context_mask]
        )

        men_official_mask = men_rows_mask & official_mask
        women_official_mask = (~men_rows_mask) & official_mask
        fold_summaries.append(
            {
                "season": season,
                "combined_brier": pooled_base.brier_score(probabilities[official_mask], targets[official_mask]),
                "men_brier": pooled_base.brier_score(probabilities[men_official_mask], targets[men_official_mask]),
                "women_brier": pooled_base.brier_score(probabilities[women_official_mask], targets[women_official_mask]),
                "official_row_count": int(official_mask.sum()),
                "excluded_first_four_count": int((~official_mask).sum()),
            }
        )

    return {
        "validation_seasons": [int(fold_summary["season"]) for fold_summary in fold_summaries],
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def improvement_summary(validation_summary: dict[str, object]) -> tuple[int, list[int]]:
    fold_scores = active_benchmark.benchmark_fold_scores()
    candidate_folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    improved_seasons = [
        int(fold["season"])
        for fold in candidate_folds
        if float(fold["combined_brier"]) < fold_scores[int(fold["season"])]
    ]
    return len(improved_seasons), improved_seasons


def serialize_result(config: dict[str, object], validation_summary: dict[str, object]) -> dict[str, object]:
    improved_fold_count, improved_seasons = improvement_summary(validation_summary)
    mean_brier = cast(float, validation_summary["mean_brier"])
    accepted = mean_brier < active_benchmark.benchmark_mean_brier() and improved_fold_count >= 4
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
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": active_benchmark.FAMILY_NAME,
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "block_rate_benchmark_with_men_context_router",
        "benchmark_mean_brier": active_benchmark.benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in active_benchmark.benchmark_fold_scores().items()},
        **active_benchmark.benchmark_feature_columns(),
        "results": results,
        "best_raw_result": best_raw_result(results),
        "selected_result": selected_result,
        "accepted": selected_result is not None,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cached_fold_inputs = active_benchmark.cached_fold_inputs()
    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        typed_config = cast(dict[str, object], config)
        validation_summary = run_validation(cached_fold_inputs=cached_fold_inputs, config=typed_config)
        results.append(serialize_result(typed_config, validation_summary))

    selected_result = best_accepted_result(results)
    write_search_summary(results=results, selected_result=selected_result)

    print(f"Active block-rate benchmark mean validation Brier: {active_benchmark.benchmark_mean_brier():.6f}")
    print(f"Best raw candidate mean validation Brier: {best_raw_result(results)['raw_mean_brier']:.6f}")
    print(f"Accepted candidate found: {selected_result is not None}")
    if selected_result is not None:
        print(orjson.dumps(selected_result, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
