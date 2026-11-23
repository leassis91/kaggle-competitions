from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import orjson
import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_26_pooled_selection_sunday_bracket_calibration as pooled_baseline
import train_iteration_38_vilnius_plus_capped_quality as pooled_feature_stack
import train_iteration_41_iteration_38_optuna_params as pooled_tuned
import train_iteration_49_seed_closure_quality as seeded_closure
import train_iteration_53_iteration_44_iteration_50_blend as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_70_nested_meta_margin_blender")
META_LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 300,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 5.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
}


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


def build_meta_model() -> Any:
    return cast(Any, lgb).LGBMClassifier(**META_LIGHTGBM_PARAMS)


def source_predictions_for_season(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    seeded_closure_matchups: pl.DataFrame,
    seeded_closure_feature_columns: list[str],
    validation_season: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, pl.DataFrame]:
    benchmark_predictions, targets, validation_rows = benchmark.raw_margin_predictions(
        benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_feature_columns=benchmark_benchmark_feature_columns,
        alternative_matchups=benchmark_alternative_matchups,
        alternative_feature_columns=benchmark_alternative_feature_columns,
        validation_season=validation_season,
    )
    iteration_38_predictions, _iteration_38_targets, _iteration_38_rows = pooled_baseline.raw_margin_predictions(
        matchups=pooled_matchups,
        feature_columns=pooled_feature_columns,
        validation_season=validation_season,
    )
    iteration_41_predictions, _iteration_41_targets, _iteration_41_rows = pooled_tuned.raw_margin_predictions(
        matchups=pooled_matchups,
        feature_columns=pooled_feature_columns,
        validation_season=validation_season,
    )
    iteration_49_predictions, _iteration_49_targets, _iteration_49_rows = benchmark.benchmark.raw_margin_predictions(
        matchups=seeded_closure_matchups,
        feature_columns=seeded_closure_feature_columns,
        validation_season=validation_season,
    )
    return {
        "benchmark_margin": benchmark_predictions,
        "iteration_38_margin": iteration_38_predictions,
        "iteration_41_margin": iteration_41_predictions,
        "iteration_49_margin": iteration_49_predictions,
    }, targets, validation_rows


def build_meta_matrix(source_predictions: dict[str, np.ndarray], validation_rows: pl.DataFrame) -> np.ndarray:
    benchmark_margin = source_predictions["benchmark_margin"]
    iteration_38_margin = source_predictions["iteration_38_margin"]
    iteration_41_margin = source_predictions["iteration_41_margin"]
    iteration_49_margin = source_predictions["iteration_49_margin"]
    men_women = validation_rows["men_women"].to_numpy()
    possible_round = validation_rows["possible_round"].to_numpy()
    delta_seed_number = validation_rows["delta_seed_number"].to_numpy()
    same_region_flag = validation_rows["same_region_flag"].to_numpy()
    any_play_in_flag = validation_rows["any_play_in_flag"].to_numpy()

    return np.column_stack(
        [
            benchmark_margin,
            iteration_38_margin,
            iteration_41_margin,
            iteration_49_margin,
            iteration_38_margin - benchmark_margin,
            iteration_41_margin - benchmark_margin,
            iteration_49_margin - benchmark_margin,
            men_women,
            possible_round,
            delta_seed_number,
            same_region_flag,
            any_play_in_flag,
            benchmark_margin * possible_round,
            benchmark_margin * delta_seed_number,
            iteration_38_margin * men_women,
            iteration_41_margin * men_women,
            iteration_49_margin * men_women,
        ]
    )


def fit_meta_model(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    seeded_closure_matchups: pl.DataFrame,
    seeded_closure_feature_columns: list[str],
    target_season: int,
) -> Any | None:
    training_seasons = sorted(
        season for season in benchmark_benchmark_matchups["Season"].unique().to_list() if season < target_season
    )
    oof_matrices: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for calibration_season in training_seasons[1:]:
        source_predictions, targets, validation_rows = source_predictions_for_season(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            seeded_closure_matchups=seeded_closure_matchups,
            seeded_closure_feature_columns=seeded_closure_feature_columns,
            validation_season=calibration_season,
        )
        oof_matrices.append(build_meta_matrix(source_predictions, validation_rows))
        oof_targets.append(targets)

    if not oof_matrices:
        return None

    model = build_meta_model()
    model.fit(np.vstack(oof_matrices), np.concatenate(oof_targets))
    return model


def run_validation(
    benchmark_benchmark_matchups: pl.DataFrame,
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_matchups: pl.DataFrame,
    benchmark_alternative_feature_columns: list[str],
    pooled_matchups: pl.DataFrame,
    pooled_feature_columns: list[str],
    seeded_closure_matchups: pl.DataFrame,
    seeded_closure_feature_columns: list[str],
) -> dict[str, object]:
    validation_seasons = pooled_base.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        source_predictions, targets, validation_rows = source_predictions_for_season(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            seeded_closure_matchups=seeded_closure_matchups,
            seeded_closure_feature_columns=seeded_closure_feature_columns,
            validation_season=validation_season,
        )
        meta_model = fit_meta_model(
            benchmark_benchmark_matchups=benchmark_benchmark_matchups,
            benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
            benchmark_alternative_matchups=benchmark_alternative_matchups,
            benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
            pooled_matchups=pooled_matchups,
            pooled_feature_columns=pooled_feature_columns,
            seeded_closure_matchups=seeded_closure_matchups,
            seeded_closure_feature_columns=seeded_closure_feature_columns,
            target_season=validation_season,
        )
        if meta_model is None:
            probabilities = np.clip(
                1.0 / (1.0 + np.exp(-source_predictions["benchmark_margin"] / 10.0)),
                0.0,
                1.0,
            )
        else:
            probabilities = np.clip(
                meta_model.predict_proba(build_meta_matrix(source_predictions, validation_rows))[:, 1],
                0.0,
                1.0,
            )
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


def write_validation_summary(
    validation_summary: dict[str, object],
    benchmark_benchmark_feature_columns: list[str],
    benchmark_alternative_feature_columns: list[str],
    pooled_feature_columns: list[str],
    seeded_closure_feature_columns: list[str],
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
            "iteration_41": pooled_tuned.LIGHTGBM_PARAMS,
            "iteration_49": {
                "men": seeded_closure.benchmark.feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
                "women": seeded_closure.benchmark.tuned.LIGHTGBM_PARAMS,
            },
            "meta": META_LIGHTGBM_PARAMS,
        },
        "benchmark_benchmark_feature_columns": benchmark_benchmark_feature_columns,
        "benchmark_alternative_feature_columns": benchmark_alternative_feature_columns,
        "pooled_feature_columns": pooled_feature_columns,
        "seeded_closure_feature_columns": seeded_closure_feature_columns,
        "minimum_pooled_season": pooled_base.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "direct_from_nested_oof_meta_lightgbm_classifier",
        "feature_family": "nested_meta_margin_blender_from_iteration_53_38_41_49",
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
    seeded_closure_matchups, seeded_closure_feature_columns = seeded_closure.build_pooled_margin_matchups()

    validation_summary = run_validation(
        benchmark_benchmark_matchups=benchmark_benchmark_matchups,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_matchups=benchmark_alternative_matchups,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_matchups=pooled_matchups,
        pooled_feature_columns=pooled_feature_columns,
        seeded_closure_matchups=seeded_closure_matchups,
        seeded_closure_feature_columns=seeded_closure_feature_columns,
    )
    write_validation_summary(
        validation_summary=validation_summary,
        benchmark_benchmark_feature_columns=benchmark_benchmark_feature_columns,
        benchmark_alternative_feature_columns=benchmark_alternative_feature_columns,
        pooled_feature_columns=pooled_feature_columns,
        seeded_closure_feature_columns=seeded_closure_feature_columns,
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
