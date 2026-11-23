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


ARTIFACT_DIR = Path("artifacts/iteration_106_iteration_105_official_benchmark_probability_calibrator_search")
BENCHMARK_CONFIG = {
    "men_gap_threshold": 0.040,
    "men_low_gap_iteration_71_probability_weight": 0.44,
    "men_high_gap_iteration_71_probability_weight": 0.36,
    "women_low_gap_threshold": 0.054,
    "women_mid_gap_threshold": 0.065,
    "women_high_gap_threshold": 0.080,
    "women_low_gap_iteration_71_probability_weight": 1.0,
    "women_mid_gap_iteration_71_probability_weight": 0.0,
    "women_high_gap_iteration_71_probability_weight": 0.0,
    "women_extreme_gap_iteration_71_probability_weight": 0.26,
}
GRID_CONFIGS = [
    {
        "feature_set": feature_set,
        "regularization_c": regularization_c,
        "official_only_calibration": official_only_calibration,
    }
    for feature_set in ["benchmark_only", "benchmark_gap_context", "logit_gap_context"]
    for regularization_c in [0.01, 0.03, 0.1, 0.3, 1.0]
    for official_only_calibration in [False, True]
]


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads(
            (
                Path("artifacts/iteration_105_iteration_102_official_joint_men_women_fine_search")
                / "validation_summary.json"
            ).read_bytes()
        ),
    )


def benchmark_fold_scores() -> dict[int, float]:
    selected_result = cast(dict[str, object], load_benchmark_summary()["selected_result"])
    validation = cast(dict[str, object], selected_result["validation_summary"])
    folds = cast(list[dict[str, float | int]], validation["folds"])
    return {int(fold["season"]): float(fold["combined_brier"]) for fold in folds}


def benchmark_mean_brier() -> float:
    selected_result = cast(dict[str, object], load_benchmark_summary()["selected_result"])
    return cast(float, selected_result["raw_mean_brier"])


def benchmark_probabilities(
    validation_rows: pl.DataFrame,
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
) -> np.ndarray:
    men_gap_threshold = float(BENCHMARK_CONFIG["men_gap_threshold"])
    men_low_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["men_low_gap_iteration_71_probability_weight"])
    men_low_gap_iteration_69_probability_weight = 1.0 - men_low_gap_iteration_71_probability_weight
    men_high_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["men_high_gap_iteration_71_probability_weight"])
    men_high_gap_iteration_69_probability_weight = 1.0 - men_high_gap_iteration_71_probability_weight

    women_low_gap_threshold = float(BENCHMARK_CONFIG["women_low_gap_threshold"])
    women_mid_gap_threshold = float(BENCHMARK_CONFIG["women_mid_gap_threshold"])
    women_high_gap_threshold = float(BENCHMARK_CONFIG["women_high_gap_threshold"])
    women_low_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["women_low_gap_iteration_71_probability_weight"])
    women_low_gap_iteration_69_probability_weight = 1.0 - women_low_gap_iteration_71_probability_weight
    women_mid_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["women_mid_gap_iteration_71_probability_weight"])
    women_mid_gap_iteration_69_probability_weight = 1.0 - women_mid_gap_iteration_71_probability_weight
    women_high_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["women_high_gap_iteration_71_probability_weight"])
    women_high_gap_iteration_69_probability_weight = 1.0 - women_high_gap_iteration_71_probability_weight
    women_extreme_gap_iteration_71_probability_weight = float(BENCHMARK_CONFIG["women_extreme_gap_iteration_71_probability_weight"])
    women_extreme_gap_iteration_69_probability_weight = 1.0 - women_extreme_gap_iteration_71_probability_weight

    men_rows_mask = validation_rows["men_women"].to_numpy() == 1
    women_rows_mask = ~men_rows_mask
    probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)
    men_low_gap_mask = men_rows_mask & (probability_gap <= men_gap_threshold)
    men_high_gap_mask = men_rows_mask & ~men_low_gap_mask
    women_low_gap_mask = women_rows_mask & (probability_gap <= women_low_gap_threshold)
    women_mid_gap_mask = women_rows_mask & (probability_gap > women_low_gap_threshold) & (
        probability_gap <= women_mid_gap_threshold
    )
    women_high_gap_mask = women_rows_mask & (probability_gap > women_mid_gap_threshold) & (
        probability_gap <= women_high_gap_threshold
    )
    women_extreme_gap_mask = women_rows_mask & (probability_gap > women_high_gap_threshold)

    probabilities = np.empty_like(iteration_69_probabilities)
    probabilities[men_low_gap_mask] = (
        men_low_gap_iteration_69_probability_weight * iteration_69_probabilities[men_low_gap_mask]
        + men_low_gap_iteration_71_probability_weight * iteration_71_probabilities[men_low_gap_mask]
    )
    probabilities[men_high_gap_mask] = (
        men_high_gap_iteration_69_probability_weight * iteration_69_probabilities[men_high_gap_mask]
        + men_high_gap_iteration_71_probability_weight * iteration_71_probabilities[men_high_gap_mask]
    )
    probabilities[women_low_gap_mask] = (
        women_low_gap_iteration_69_probability_weight * iteration_69_probabilities[women_low_gap_mask]
        + women_low_gap_iteration_71_probability_weight * iteration_71_probabilities[women_low_gap_mask]
    )
    probabilities[women_mid_gap_mask] = (
        women_mid_gap_iteration_69_probability_weight * iteration_69_probabilities[women_mid_gap_mask]
        + women_mid_gap_iteration_71_probability_weight * iteration_71_probabilities[women_mid_gap_mask]
    )
    probabilities[women_high_gap_mask] = (
        women_high_gap_iteration_69_probability_weight * iteration_69_probabilities[women_high_gap_mask]
        + women_high_gap_iteration_71_probability_weight * iteration_71_probabilities[women_high_gap_mask]
    )
    probabilities[women_extreme_gap_mask] = (
        women_extreme_gap_iteration_69_probability_weight * iteration_69_probabilities[women_extreme_gap_mask]
        + women_extreme_gap_iteration_71_probability_weight * iteration_71_probabilities[women_extreme_gap_mask]
    )
    return probabilities


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped_probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped_probabilities / (1.0 - clipped_probabilities))


def build_calibration_matrix(
    benchmark_probability: np.ndarray,
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

    if feature_set == "benchmark_only":
        return benchmark_probability.reshape(-1, 1)

    if feature_set == "benchmark_gap_context":
        return np.column_stack(
            [
                benchmark_probability,
                probability_gap,
                absolute_probability_gap,
                men_women,
                possible_round,
                delta_seed_number,
                same_region_flag,
                benchmark_probability * men_women,
                benchmark_probability * possible_round,
            ]
        )

    benchmark_logit = safe_logit(benchmark_probability)
    return np.column_stack(
        [
            benchmark_probability,
            benchmark_logit,
            probability_gap,
            absolute_probability_gap,
            men_women,
            possible_round,
            delta_seed_number,
            same_region_flag,
            benchmark_logit * men_women,
            benchmark_logit * possible_round,
            absolute_probability_gap * men_women,
        ]
    )


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
    return division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
    )


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    feature_set = cast(str, config["feature_set"])
    regularization_c = cast(float, config["regularization_c"])
    official_only_calibration = cast(bool, config["official_only_calibration"])

    for fold_index, cached_fold_input in enumerate(cached_fold_inputs):
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])
        validation_benchmark_probability = benchmark_probabilities(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
        )
        validation_matrix = build_calibration_matrix(
            benchmark_probability=validation_benchmark_probability,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
            validation_rows=validation_rows,
            feature_set=feature_set,
        )

        calibration_matrices: list[np.ndarray] = []
        calibration_targets: list[np.ndarray] = []
        for prior_cached_fold_input in cached_fold_inputs[:fold_index]:
            prior_targets = cast(np.ndarray, prior_cached_fold_input["targets"])
            prior_validation_rows = cast(pl.DataFrame, prior_cached_fold_input["validation_rows"])
            prior_iteration_69_probabilities = cast(np.ndarray, prior_cached_fold_input["iteration_69_probabilities"])
            prior_iteration_71_probabilities = cast(np.ndarray, prior_cached_fold_input["iteration_71_probabilities"])
            prior_benchmark_probability = benchmark_probabilities(
                validation_rows=prior_validation_rows,
                iteration_69_probabilities=prior_iteration_69_probabilities,
                iteration_71_probabilities=prior_iteration_71_probabilities,
            )
            calibration_mask = official_search.official_row_mask(prior_validation_rows) if official_only_calibration else np.ones_like(prior_targets, dtype=bool)
            calibration_matrices.append(
                build_calibration_matrix(
                    benchmark_probability=prior_benchmark_probability[calibration_mask],
                    iteration_69_probabilities=prior_iteration_69_probabilities[calibration_mask],
                    iteration_71_probabilities=prior_iteration_71_probabilities[calibration_mask],
                    validation_rows=prior_validation_rows.filter(pl.Series(calibration_mask)),
                    feature_set=feature_set,
                )
            )
            calibration_targets.append(prior_targets[calibration_mask])

        if not calibration_matrices:
            probabilities = validation_benchmark_probability
        else:
            calibrator = LogisticRegression(
                random_state=7,
                max_iter=1000,
                C=regularization_c,
            )
            calibrator.fit(np.vstack(calibration_matrices), np.concatenate(calibration_targets))
            probabilities = np.clip(calibrator.predict_proba(validation_matrix)[:, 1], 0.0, 1.0)

        official_mask = official_search.official_row_mask(validation_rows)
        men_mask = (validation_rows["men_women"].to_numpy() == 1) & official_mask
        women_mask = ~men_mask & official_mask
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


def improvement_summary(validation_summary: dict[str, object]) -> tuple[int, list[int]]:
    fold_scores = benchmark_fold_scores()
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
        "feature_family": "iteration_105_official_benchmark_probability_calibrator_search",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "nested_oof_logistic_recalibration_on_iteration_105_benchmark_probability",
        "benchmark_config": BENCHMARK_CONFIG,
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
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

    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        typed_config = cast(dict[str, object], config)
        validation_summary = run_validation(cached_fold_inputs=cached_fold_inputs, config=typed_config)
        results.append(serialize_result(typed_config, validation_summary))

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
