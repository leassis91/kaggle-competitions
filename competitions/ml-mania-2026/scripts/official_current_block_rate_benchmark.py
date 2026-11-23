from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import orjson
import polars as pl

import official_feature_family_driver as family_driver
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as official_router
import train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend as official_benchmark
import train_iteration_116_pending_feature_family_sweep as pending_feature_family_sweep


BLOCK_RATE_ARTIFACT_DIR = Path(
    "artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_block_rate_features"
)
CACHE_ARTIFACT_DIR = Path("artifacts/official_current_block_rate_benchmark")
FAMILY_NAME = "Current-season block-rate features"


def load_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads((BLOCK_RATE_ARTIFACT_DIR / "validation_summary.json").read_bytes()),
    )


def benchmark_fold_scores() -> dict[int, float]:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    folds = cast(list[dict[str, float | int]], validation["folds"])
    return {int(fold["season"]): float(fold["combined_brier"]) for fold in folds}


def benchmark_mean_brier() -> float:
    validation = cast(dict[str, object], load_benchmark_summary()["validation"])
    return cast(float, validation["mean_brier"])


@lru_cache(maxsize=1)
def block_rate_family_spec() -> family_driver.FamilySpec:
    return next(
        family_spec
        for family_spec in pending_feature_family_sweep.family_specifications()
        if family_spec.name == FAMILY_NAME
    )


@lru_cache(maxsize=1)
def active_matchup_tables() -> family_driver.OfficialMatchupTables:
    base_benchmark_state = family_driver.build_base_benchmark_state()
    matchup_tables, _, _ = family_driver.build_family_matchup_tables(
        benchmark_state=base_benchmark_state,
        family_spec=block_rate_family_spec(),
    )
    return matchup_tables


def benchmark_feature_columns() -> dict[str, list[str]]:
    matchup_tables = active_matchup_tables()
    return {
        "benchmark_benchmark_feature_columns": matchup_tables.benchmark_benchmark_bundle.feature_columns,
        "benchmark_alternative_feature_columns": matchup_tables.benchmark_alternative_bundle.feature_columns,
        "pooled_feature_columns": matchup_tables.pooled_bundle.feature_columns,
        "alternative_feature_columns": matchup_tables.alternative_bundle.feature_columns,
    }


def cached_fold_inputs() -> list[dict[str, object]]:
    CACHE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ARTIFACT_DIR / "cached_fold_inputs.joblib"
    if cache_path.exists():
        return cast(list[dict[str, object]], joblib.load(cache_path))

    cached_inputs = family_driver.build_cached_fold_inputs(active_matchup_tables())
    joblib.dump(cached_inputs, cache_path)
    return cached_inputs


def benchmark_probabilities(
    validation_rows: pl.DataFrame,
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
) -> np.ndarray:
    men_rows_mask = validation_rows["men_women"].to_numpy() == 1
    women_rows_mask = ~men_rows_mask
    probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)

    men_low_gap_threshold = float(official_router.DEFAULT_CONFIG["men_low_gap_threshold"])
    men_low_gap_iteration_71_probability_weight = float(
        official_router.DEFAULT_CONFIG["men_low_gap_iteration_71_probability_weight"]
    )
    men_low_gap_iteration_69_probability_weight = 1.0 - men_low_gap_iteration_71_probability_weight
    men_mid_gap_threshold = float(official_benchmark.CONFIG["men_mid_gap_threshold"])
    men_mid_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["men_mid_gap_iteration_71_probability_weight"])
    men_mid_gap_iteration_69_probability_weight = 1.0 - men_mid_gap_iteration_71_probability_weight
    men_high_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["men_high_gap_iteration_71_probability_weight"])
    men_high_gap_iteration_69_probability_weight = 1.0 - men_high_gap_iteration_71_probability_weight

    women_low_gap_threshold = float(official_router.DEFAULT_CONFIG["women_low_gap_threshold"])
    women_mid_gap_threshold = float(official_router.DEFAULT_CONFIG["women_mid_gap_threshold"])
    women_high_gap_threshold = float(official_benchmark.CONFIG["women_high_gap_threshold"])
    women_low_gap_iteration_71_probability_weight = float(
        official_router.DEFAULT_CONFIG["women_low_gap_iteration_71_probability_weight"]
    )
    women_low_gap_iteration_69_probability_weight = 1.0 - women_low_gap_iteration_71_probability_weight
    women_mid_gap_iteration_71_probability_weight = float(
        official_router.DEFAULT_CONFIG["women_mid_gap_iteration_71_probability_weight"]
    )
    women_mid_gap_iteration_69_probability_weight = 1.0 - women_mid_gap_iteration_71_probability_weight
    women_high_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["women_high_gap_iteration_71_probability_weight"])
    women_high_gap_iteration_69_probability_weight = 1.0 - women_high_gap_iteration_71_probability_weight
    women_extreme_gap_iteration_71_probability_weight = float(
        official_benchmark.CONFIG["women_extreme_gap_iteration_71_probability_weight"]
    )
    women_extreme_gap_iteration_69_probability_weight = 1.0 - women_extreme_gap_iteration_71_probability_weight

    men_low_gap_mask = men_rows_mask & (probability_gap <= men_low_gap_threshold)
    men_mid_gap_mask = men_rows_mask & (probability_gap > men_low_gap_threshold) & (
        probability_gap <= men_mid_gap_threshold
    )
    men_high_gap_mask = men_rows_mask & ~(men_low_gap_mask | men_mid_gap_mask)

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
    probabilities[men_mid_gap_mask] = (
        men_mid_gap_iteration_69_probability_weight * iteration_69_probabilities[men_mid_gap_mask]
        + men_mid_gap_iteration_71_probability_weight * iteration_71_probabilities[men_mid_gap_mask]
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
