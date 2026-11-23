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
import train_iteration_49_seed_closure_quality as alternative
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_76_iteration_73_division_specific_probability_blend_search as division_specific_search
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search


ARTIFACT_DIR = Path("artifacts/iteration_93_iteration_90_constrained_probability_calibrator_search")
BENCHMARK_CONFIG = official_search.BENCHMARK_CONFIG
GRID_CONFIGS = [
    {"feature_set": feature_set, "regularization_c": regularization_c}
    for feature_set in ["prob_gap_division", "logit_gap_division", "logit_gap_bracket"]
    for regularization_c in [0.05, 0.1, 0.2, 0.5, 1.0]
]


def benchmark_probabilities(
    validation_rows: pl.DataFrame,
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
) -> np.ndarray:
    men_women = validation_rows["men_women"].to_numpy()
    women_probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)

    probabilities = np.empty_like(iteration_69_probabilities)
    men_mask = men_women == 1
    women_gate_mask = (men_women == 0) & (women_probability_gap <= float(BENCHMARK_CONFIG["women_gap_threshold"]))
    women_default_mask = (men_women == 0) & ~women_gate_mask

    probabilities[men_mask] = 0.71 * iteration_69_probabilities[men_mask] + 0.29 * iteration_71_probabilities[men_mask]
    probabilities[women_default_mask] = (
        (1.0 - float(BENCHMARK_CONFIG["women_default_iteration_71_probability_weight"]))
        * iteration_69_probabilities[women_default_mask]
        + float(BENCHMARK_CONFIG["women_default_iteration_71_probability_weight"])
        * iteration_71_probabilities[women_default_mask]
    )
    probabilities[women_gate_mask] = iteration_71_probabilities[women_gate_mask]
    return probabilities


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_meta_matrix(
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
    validation_rows: pl.DataFrame,
    feature_set: str,
) -> np.ndarray:
    men_women = validation_rows["men_women"].to_numpy()
    possible_round = validation_rows["possible_round"].to_numpy()
    delta_seed_number = validation_rows["delta_seed_number"].to_numpy()
    same_region_flag = validation_rows["same_region_flag"].to_numpy()
    probability_gap = iteration_71_probabilities - iteration_69_probabilities
    absolute_probability_gap = np.abs(probability_gap)

    if feature_set == "prob_gap_division":
        return np.column_stack(
            [
                iteration_69_probabilities,
                iteration_71_probabilities,
                probability_gap,
                absolute_probability_gap,
                men_women,
                absolute_probability_gap * men_women,
            ]
        )

    iteration_69_logit = safe_logit(iteration_69_probabilities)
    iteration_71_logit = safe_logit(iteration_71_probabilities)
    logit_gap = iteration_71_logit - iteration_69_logit
    absolute_logit_gap = np.abs(logit_gap)

    if feature_set == "logit_gap_division":
        return np.column_stack(
            [
                iteration_69_logit,
                iteration_71_logit,
                logit_gap,
                absolute_logit_gap,
                men_women,
                absolute_logit_gap * men_women,
            ]
        )

    return np.column_stack(
        [
            iteration_69_logit,
            iteration_71_logit,
            logit_gap,
            absolute_logit_gap,
            men_women,
            possible_round,
            delta_seed_number,
            same_region_flag,
            absolute_logit_gap * men_women,
            absolute_logit_gap * possible_round,
        ]
    )


def fit_meta_calibrator(
    prior_fold_inputs: list[dict[str, object]],
    config: dict[str, str | float],
) -> LogisticRegression | None:
    if not prior_fold_inputs:
        return None

    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for cached_fold_input in prior_fold_inputs:
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])
        official_mask = official_search.official_row_mask(validation_rows)
        oof_matrices.append(
            build_meta_matrix(
                iteration_69_probabilities=iteration_69_probabilities,
                iteration_71_probabilities=iteration_71_probabilities,
                validation_rows=validation_rows,
                feature_set=cast(str, config["feature_set"]),
            )[official_mask]
        )
        oof_targets.append(targets[official_mask])

    calibrator = LogisticRegression(random_state=7, max_iter=1000, C=cast(float, config["regularization_c"]))
    calibrator.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return calibrator


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, str | float],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    for fold_index, cached_fold_input in enumerate(cached_fold_inputs):
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])

        official_mask = official_search.official_row_mask(validation_rows)
        men_mask = (validation_rows["men_women"].to_numpy() == 1) & official_mask
        women_mask = (validation_rows["men_women"].to_numpy() == 0) & official_mask

        calibrator = fit_meta_calibrator(cached_fold_inputs[:fold_index], config)
        if calibrator is None:
            probabilities = benchmark_probabilities(
                validation_rows=validation_rows,
                iteration_69_probabilities=iteration_69_probabilities,
                iteration_71_probabilities=iteration_71_probabilities,
            )
        else:
            meta_matrix = build_meta_matrix(
                iteration_69_probabilities=iteration_69_probabilities,
                iteration_71_probabilities=iteration_71_probabilities,
                validation_rows=validation_rows,
                feature_set=cast(str, config["feature_set"]),
            )
            probabilities = np.clip(calibrator.predict_proba(meta_matrix)[:, 1], 0.0, 1.0)

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
    config: dict[str, str | float],
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
        "feature_family": "iteration_90_constrained_probability_calibrator_search",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "constrained_logistic_calibrator_on_iteration_69_and_iteration_71_probabilities",
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

    cached_fold_inputs = division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )
    benchmark_summary = official_search.run_validation(cached_fold_inputs=cached_fold_inputs, config=BENCHMARK_CONFIG)
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
