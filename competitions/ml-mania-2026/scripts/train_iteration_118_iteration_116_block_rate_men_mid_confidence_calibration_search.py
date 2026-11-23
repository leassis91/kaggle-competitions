from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import polars as pl
from sklearn.linear_model import LogisticRegression

import official_current_block_rate_benchmark as active_benchmark
import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search


ARTIFACT_DIR = Path("artifacts/iteration_118_iteration_116_block_rate_men_mid_confidence_calibration_search")
GRID_CONFIGS = [
    {
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "feature_set": feature_set,
        "regularization_c": regularization_c,
    }
    for confidence_lower, confidence_upper in [(0.30, 0.70), (0.35, 0.65), (0.40, 0.60)]
    for feature_set in ["benchmark_gap_round_seed", "logit_gap_round_seed"]
    for regularization_c in [0.03, 0.10, 0.30, 1.00]
]


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
    probability_gap = iteration_71_probabilities - iteration_69_probabilities
    absolute_probability_gap = np.abs(probability_gap)
    possible_round = validation_rows["possible_round"].to_numpy()
    absolute_seed_gap = np.abs(validation_rows["delta_seed_number"].to_numpy())

    if feature_set == "benchmark_gap_round_seed":
        return np.column_stack(
            [
                benchmark_probability,
                probability_gap,
                absolute_probability_gap,
                possible_round,
                absolute_seed_gap,
                benchmark_probability * absolute_probability_gap,
            ]
        )

    benchmark_logit = safe_logit(benchmark_probability)
    return np.column_stack(
        [
            benchmark_probability,
            benchmark_logit,
            probability_gap,
            absolute_probability_gap,
            possible_round,
            absolute_seed_gap,
            benchmark_logit * absolute_probability_gap,
            benchmark_logit * possible_round,
        ]
    )


def men_mid_confidence_mask(
    validation_rows: pl.DataFrame,
    benchmark_probability: np.ndarray,
    confidence_lower: float,
    confidence_upper: float,
) -> np.ndarray:
    official_mask = official_search.official_row_mask(validation_rows)
    men_mask = validation_rows["men_women"].to_numpy() == 1
    return (
        official_mask
        & men_mask
        & (benchmark_probability >= confidence_lower)
        & (benchmark_probability <= confidence_upper)
    )


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    confidence_lower = cast(float, config["confidence_lower"])
    confidence_upper = cast(float, config["confidence_upper"])
    feature_set = cast(str, config["feature_set"])
    regularization_c = cast(float, config["regularization_c"])

    for fold_index, cached_fold_input in enumerate(cached_fold_inputs):
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, cached_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, cached_fold_input["iteration_71_probabilities"])

        benchmark_probability = active_benchmark.benchmark_probabilities(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
        )
        calibration_mask = men_mid_confidence_mask(
            validation_rows=validation_rows,
            benchmark_probability=benchmark_probability,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
        )
        validation_matrix = build_calibration_matrix(
            benchmark_probability=benchmark_probability[calibration_mask],
            iteration_69_probabilities=iteration_69_probabilities[calibration_mask],
            iteration_71_probabilities=iteration_71_probabilities[calibration_mask],
            validation_rows=validation_rows.filter(pl.Series(calibration_mask)),
            feature_set=feature_set,
        )

        calibration_matrices: list[np.ndarray] = []
        calibration_targets: list[np.ndarray] = []
        for prior_cached_fold_input in cached_fold_inputs[:fold_index]:
            prior_targets = cast(np.ndarray, prior_cached_fold_input["targets"])
            prior_validation_rows = cast(pl.DataFrame, prior_cached_fold_input["validation_rows"])
            prior_iteration_69_probabilities = cast(np.ndarray, prior_cached_fold_input["iteration_69_probabilities"])
            prior_iteration_71_probabilities = cast(np.ndarray, prior_cached_fold_input["iteration_71_probabilities"])
            prior_benchmark_probability = active_benchmark.benchmark_probabilities(
                validation_rows=prior_validation_rows,
                iteration_69_probabilities=prior_iteration_69_probabilities,
                iteration_71_probabilities=prior_iteration_71_probabilities,
            )
            prior_calibration_mask = men_mid_confidence_mask(
                validation_rows=prior_validation_rows,
                benchmark_probability=prior_benchmark_probability,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
            )
            if prior_calibration_mask.sum() == 0:
                continue
            calibration_matrices.append(
                build_calibration_matrix(
                    benchmark_probability=prior_benchmark_probability[prior_calibration_mask],
                    iteration_69_probabilities=prior_iteration_69_probabilities[prior_calibration_mask],
                    iteration_71_probabilities=prior_iteration_71_probabilities[prior_calibration_mask],
                    validation_rows=prior_validation_rows.filter(pl.Series(prior_calibration_mask)),
                    feature_set=feature_set,
                )
            )
            calibration_targets.append(prior_targets[prior_calibration_mask])

        probabilities = benchmark_probability.copy()
        if calibration_matrices and calibration_mask.sum() > 0:
            stacked_targets = np.concatenate(calibration_targets)
            if np.unique(stacked_targets).size >= 2:
                calibrator = LogisticRegression(
                    random_state=7,
                    max_iter=1000,
                    C=regularization_c,
                )
                calibrator.fit(np.vstack(calibration_matrices), stacked_targets)
                probabilities[calibration_mask] = np.clip(calibrator.predict_proba(validation_matrix)[:, 1], 0.0, 1.0)

        official_mask = official_search.official_row_mask(validation_rows)
        men_mask = (validation_rows["men_women"].to_numpy() == 1) & official_mask
        women_mask = (~(validation_rows["men_women"].to_numpy() == 1)) & official_mask
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
        "probability_mapping": "block_rate_benchmark_with_men_mid_confidence_calibration",
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
