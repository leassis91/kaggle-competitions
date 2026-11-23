from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import orjson
import polars as pl
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_21_pooled_selection_sunday_margin_regression as pooled_regression
import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_41_iteration_38_optuna_params as pooled_tuned
import train_iteration_53_iteration_44_iteration_50_blend as benchmark
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as search
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search


ARTIFACT_DIR = Path("artifacts/iteration_100_iteration_99_official_leaf_logistic_calibrator_search")
GRID_CONFIGS = [
    {"regularization_c": regularization_c}
    for regularization_c in [0.01, 0.03, 0.1, 0.3, 1.0]
]


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads(
            (
                Path("artifacts/iteration_99_iteration_97_official_joint_men_women_gap_search")
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


def fit_margin_model(
    matchups: pl.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    training_rows = matchups.filter(pl.col("Season") < validation_season)
    validation_rows = matchups.filter(pl.col("Season") == validation_season)
    training_matrix = training_rows.select(feature_columns).to_numpy()
    validation_matrix = validation_rows.select(feature_columns).to_numpy()

    model = lgb.LGBMRegressor(**params)
    model.fit(training_matrix, training_rows["TargetMargin"].to_numpy())
    predictions = cast(np.ndarray, model.predict(validation_matrix))
    leaves = np.asarray(model.booster_.predict(validation_matrix, pred_leaf=True), dtype=np.int32)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    expected_tree_count = int(params["n_estimators"])
    if leaves.shape[1] < expected_tree_count:
        leaves = np.pad(
            leaves,
            ((0, 0), (0, expected_tree_count - leaves.shape[1])),
            constant_values=-1,
        )
    return predictions, leaves, validation_rows


def assert_same_matchup_rows(reference_rows: pl.DataFrame, candidate_rows: pl.DataFrame) -> None:
    reference_keys = reference_rows.select("Season", "LowerTeamID", "HigherTeamID", "men_women")
    candidate_keys = candidate_rows.select("Season", "LowerTeamID", "HigherTeamID", "men_women")
    if not reference_keys.equals(candidate_keys):
        raise ValueError("Matchup row ordering mismatch across feature stacks.")


def season_source_features(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    validation_season: int,
) -> dict[str, object]:
    benchmark_baseline_margin, benchmark_baseline_leaves, benchmark_rows = fit_margin_model(
        matchups=benchmark_matchups,
        feature_columns=benchmark_feature_columns,
        validation_season=validation_season,
        params=pooled_regression.LIGHTGBM_PARAMS,
    )
    benchmark_tuned_margin, benchmark_tuned_leaves, benchmark_tuned_rows = fit_margin_model(
        matchups=benchmark_matchups,
        feature_columns=benchmark_feature_columns,
        validation_season=validation_season,
        params=pooled_tuned.LIGHTGBM_PARAMS,
    )
    alternative_baseline_margin, alternative_baseline_leaves, alternative_rows = fit_margin_model(
        matchups=alternative_matchups,
        feature_columns=alternative_feature_columns,
        validation_season=validation_season,
        params=pooled_regression.LIGHTGBM_PARAMS,
    )
    alternative_tuned_margin, alternative_tuned_leaves, alternative_tuned_rows = fit_margin_model(
        matchups=alternative_matchups,
        feature_columns=alternative_feature_columns,
        validation_season=validation_season,
        params=pooled_tuned.LIGHTGBM_PARAMS,
    )
    pooled_baseline_margin, pooled_baseline_leaves, pooled_rows = fit_margin_model(
        matchups=pooled_matchups,
        feature_columns=pooled_feature_columns,
        validation_season=validation_season,
        params=pooled_regression.LIGHTGBM_PARAMS,
    )
    pooled_tuned_margin, pooled_tuned_leaves, pooled_tuned_rows = fit_margin_model(
        matchups=pooled_matchups,
        feature_columns=pooled_feature_columns,
        validation_season=validation_season,
        params=pooled_tuned.LIGHTGBM_PARAMS,
    )

    assert_same_matchup_rows(benchmark_rows, benchmark_tuned_rows)
    assert_same_matchup_rows(benchmark_rows, alternative_rows)
    assert_same_matchup_rows(benchmark_rows, alternative_tuned_rows)
    assert_same_matchup_rows(benchmark_rows, pooled_rows)
    assert_same_matchup_rows(benchmark_rows, pooled_tuned_rows)

    men_mask = benchmark_rows["men_women"].to_numpy() == 1
    benchmark_selected_margin = benchmark_baseline_margin.copy()
    benchmark_selected_margin[~men_mask] = benchmark_tuned_margin[~men_mask]
    benchmark_selected_leaves = benchmark_baseline_leaves.copy()
    benchmark_selected_leaves[~men_mask] = benchmark_tuned_leaves[~men_mask]

    alternative_selected_margin = alternative_baseline_margin.copy()
    alternative_selected_margin[~men_mask] = alternative_tuned_margin[~men_mask]
    alternative_selected_leaves = alternative_baseline_leaves.copy()
    alternative_selected_leaves[~men_mask] = alternative_tuned_leaves[~men_mask]

    blended_benchmark_margin = (
        benchmark.MARGIN_BLEND_WEIGHTS["benchmark"] * benchmark_selected_margin
        + benchmark.MARGIN_BLEND_WEIGHTS["alternative"] * alternative_selected_margin
    )
    source_predictions = {
        "benchmark_margin": blended_benchmark_margin,
        "iteration_38_margin": pooled_baseline_margin,
        "iteration_41_margin": pooled_tuned_margin,
        "iteration_49_margin": alternative_selected_margin,
    }
    iteration_69_raw_margin = search.iteration_69_raw_margin(source_predictions)
    iteration_71_raw_margin = search.iteration_71_raw_margin(source_predictions)

    leaf_matrix = np.column_stack(
        [
            benchmark_selected_leaves,
            alternative_selected_leaves,
            pooled_baseline_leaves,
            pooled_tuned_leaves,
        ]
    )
    dense_matrix = np.column_stack(
        [
            blended_benchmark_margin,
            alternative_selected_margin,
            pooled_baseline_margin,
            pooled_tuned_margin,
            iteration_69_raw_margin,
            iteration_71_raw_margin,
            iteration_71_raw_margin - iteration_69_raw_margin,
            np.abs(iteration_71_raw_margin - iteration_69_raw_margin),
            benchmark_rows["men_women"].to_numpy(),
            benchmark_rows["possible_round"].to_numpy(),
            benchmark_rows["delta_seed_number"].to_numpy(),
            benchmark_rows["same_region_flag"].to_numpy(),
            benchmark_rows["any_play_in_flag"].to_numpy(),
        ]
    )

    return {
        "season": validation_season,
        "targets": benchmark_rows["Target"].to_numpy(),
        "validation_rows": benchmark_rows,
        "leaf_matrix": leaf_matrix,
        "dense_matrix": dense_matrix,
    }


def transform_sparse_matrix(
    encoder: OneHotEncoder,
    leaf_matrix: np.ndarray,
    dense_matrix: np.ndarray,
) -> sparse.csr_matrix:
    encoded_leaf_matrix = encoder.transform(leaf_matrix)
    dense_sparse_matrix = sparse.csr_matrix(dense_matrix.astype(np.float64))
    return cast(sparse.csr_matrix, sparse.hstack([encoded_leaf_matrix, dense_sparse_matrix], format="csr"))


def build_cached_fold_inputs(
    benchmark_matchups: pl.DataFrame,
    benchmark_feature_columns: list[str],
    alternative_matchups: pl.DataFrame,
    alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
) -> list[dict[str, object]]:
    cached_fold_inputs: list[dict[str, object]] = []

    for validation_season in pooled_base.latest_completed_validation_seasons():
        training_seasons = sorted(
            season for season in benchmark_matchups["Season"].unique().to_list() if season < validation_season
        )
        calibration_leaf_matrices: list[np.ndarray] = []
        calibration_dense_matrices: list[np.ndarray] = []
        calibration_targets: list[np.ndarray] = []

        for calibration_season in training_seasons[1:]:
            calibration_features = season_source_features(
                benchmark_matchups=benchmark_matchups,
                benchmark_feature_columns=benchmark_feature_columns,
                alternative_matchups=alternative_matchups,
                alternative_feature_columns=alternative_feature_columns,
                pooled_matchups=pooled_matchups,
                pooled_feature_columns=pooled_feature_columns,
                validation_season=calibration_season,
            )
            calibration_leaf_matrices.append(cast(np.ndarray, calibration_features["leaf_matrix"]))
            calibration_dense_matrices.append(cast(np.ndarray, calibration_features["dense_matrix"]))
            calibration_targets.append(cast(np.ndarray, calibration_features["targets"]))

        if not calibration_leaf_matrices:
            raise ValueError("Leaf calibrator requires at least one prior calibration season.")

        validation_features = season_source_features(
            benchmark_matchups=benchmark_matchups,
            benchmark_feature_columns=benchmark_feature_columns,
            alternative_matchups=alternative_matchups,
            alternative_feature_columns=alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            validation_season=validation_season,
        )
        calibration_leaf_matrix = np.vstack(calibration_leaf_matrices)
        calibration_dense_matrix = np.vstack(calibration_dense_matrices)
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(calibration_leaf_matrix)

        cached_fold_inputs.append(
            {
                "season": validation_season,
                "targets": cast(np.ndarray, validation_features["targets"]),
                "validation_rows": cast(pl.DataFrame, validation_features["validation_rows"]),
                "calibration_targets": np.concatenate(calibration_targets),
                "calibration_matrix": transform_sparse_matrix(
                    encoder=encoder,
                    leaf_matrix=calibration_leaf_matrix,
                    dense_matrix=calibration_dense_matrix,
                ),
                "validation_matrix": transform_sparse_matrix(
                    encoder=encoder,
                    leaf_matrix=cast(np.ndarray, validation_features["leaf_matrix"]),
                    dense_matrix=cast(np.ndarray, validation_features["dense_matrix"]),
                ),
            }
        )

    return cached_fold_inputs


def run_validation(
    cached_fold_inputs: list[dict[str, object]],
    config: dict[str, float],
) -> dict[str, object]:
    fold_summaries: list[dict[str, float | int]] = []

    for cached_fold_input in cached_fold_inputs:
        season = cast(int, cached_fold_input["season"])
        targets = cast(np.ndarray, cached_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, cached_fold_input["validation_rows"])
        calibration_targets = cast(np.ndarray, cached_fold_input["calibration_targets"])
        calibration_matrix = cast(sparse.csr_matrix, cached_fold_input["calibration_matrix"])
        validation_matrix = cast(sparse.csr_matrix, cached_fold_input["validation_matrix"])

        calibrator = LogisticRegression(
            random_state=7,
            solver="liblinear",
            max_iter=1000,
            C=float(config["regularization_c"]),
        )
        calibrator.fit(calibration_matrix, calibration_targets)
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
    benchmark_feature_columns: list[str],
    alternative_feature_columns: list[str],
    pooled_feature_columns: list[str],
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "iteration_99_official_leaf_logistic_calibrator_search",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "nested_oof_leaf_logistic_calibrator_from_iteration_53_38_41_49_sources",
        "benchmark_mean_brier": benchmark_mean_brier(),
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_fold_scores().items()},
        "benchmark_feature_columns": benchmark_feature_columns,
        "alternative_feature_columns": alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "fixed_weights": {
            "iteration_38": search.ITERATION_38_WEIGHT,
            "iteration_49": search.ITERATION_49_WEIGHT,
            "iteration_41": search.ITERATION_41_WEIGHT,
        },
        "grid_configs": GRID_CONFIGS,
        "results": results,
        "best_raw_result": best_raw_result(results),
        "selected_result": selected_result,
        "accepted": selected_result is not None,
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_matchups, benchmark_feature_columns = benchmark.benchmark.feature_stack.build_pooled_margin_matchups()
    alternative_matchups, alternative_feature_columns = benchmark.alternative.build_pooled_margin_matchups()
    pooled_matchups, pooled_feature_columns = pooled_feature_stack.build_pooled_margin_matchups()

    cached_fold_inputs = build_cached_fold_inputs(
        benchmark_matchups=benchmark_matchups,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_matchups=alternative_matchups,
        alternative_feature_columns=alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
    )

    results: list[dict[str, object]] = []
    for config in GRID_CONFIGS:
        validation_summary = run_validation(cached_fold_inputs=cached_fold_inputs, config=config)
        results.append(serialize_result(config, validation_summary))

    selected_result = best_accepted_result(results)
    write_search_summary(
        results=results,
        selected_result=selected_result,
        benchmark_feature_columns=benchmark_feature_columns,
        alternative_feature_columns=alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
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
