from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable
import warnings

import numpy as np
import orjson
import polars as pl


REPO_ROOT = Path("/Users/mariofilho/ml/random/ml-mania-2026")
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUT_DIR = REPO_ROOT / "tmp"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import official_feature_family_driver as family_driver
import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as search
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as official_router
import train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend as official_benchmark
import train_iteration_116_pending_feature_family_sweep as feature_sweep


OUTPUT_ROWS_PATH = OUTPUT_DIR / "error-analysis-inner-rows.parquet"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "error-analysis-summary.json"


def latest_prior_tournament_season(available_seasons: list[int], outer_season: int) -> int:
    eligible_seasons = [season for season in available_seasons if season < outer_season]
    if not eligible_seasons:
        raise ValueError(f"No eligible inner validation season before outer season {outer_season}.")
    return max(eligible_seasons)


def qbucket(values: np.ndarray, series_name: str) -> tuple[np.ndarray, list[float]]:
    quantiles = np.quantile(values, [0.2, 0.4, 0.6, 0.8]).tolist()
    thresholds = [float(value) for value in quantiles]
    labels = np.empty(values.shape[0], dtype=object)
    labels[values <= thresholds[0]] = f"{series_name}_q1"
    labels[(values > thresholds[0]) & (values <= thresholds[1])] = f"{series_name}_q2"
    labels[(values > thresholds[1]) & (values <= thresholds[2])] = f"{series_name}_q3"
    labels[(values > thresholds[2]) & (values <= thresholds[3])] = f"{series_name}_q4"
    labels[values > thresholds[3]] = f"{series_name}_q5"
    return labels, thresholds


def router_regimes(
    men_mask: np.ndarray,
    probability_gap: np.ndarray,
) -> np.ndarray:
    men_low_gap_threshold = float(official_router.DEFAULT_CONFIG["men_low_gap_threshold"])
    men_mid_gap_threshold = float(official_benchmark.CONFIG["men_mid_gap_threshold"])
    women_low_gap_threshold = float(official_router.DEFAULT_CONFIG["women_low_gap_threshold"])
    women_mid_gap_threshold = float(official_router.DEFAULT_CONFIG["women_mid_gap_threshold"])
    women_high_gap_threshold = float(official_benchmark.CONFIG["women_high_gap_threshold"])

    women_mask = ~men_mask
    regimes = np.empty(probability_gap.shape[0], dtype=object)
    regimes[men_mask & (probability_gap <= men_low_gap_threshold)] = "men_low"
    regimes[men_mask & (probability_gap > men_low_gap_threshold) & (probability_gap <= men_mid_gap_threshold)] = "men_mid"
    regimes[men_mask & (probability_gap > men_mid_gap_threshold)] = "men_high"
    regimes[women_mask & (probability_gap <= women_low_gap_threshold)] = "women_low"
    regimes[
        women_mask
        & (probability_gap > women_low_gap_threshold)
        & (probability_gap <= women_mid_gap_threshold)
    ] = "women_mid"
    regimes[
        women_mask
        & (probability_gap > women_mid_gap_threshold)
        & (probability_gap <= women_high_gap_threshold)
    ] = "women_high"
    regimes[women_mask & (probability_gap > women_high_gap_threshold)] = "women_extreme"
    return regimes


def final_probabilities(
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
    validation_rows: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    men_low_gap_threshold = float(official_router.DEFAULT_CONFIG["men_low_gap_threshold"])
    men_low_gap_iteration_71_probability_weight = float(official_router.DEFAULT_CONFIG["men_low_gap_iteration_71_probability_weight"])
    men_low_gap_iteration_69_probability_weight = 1.0 - men_low_gap_iteration_71_probability_weight
    men_mid_gap_threshold = float(official_benchmark.CONFIG["men_mid_gap_threshold"])
    men_mid_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["men_mid_gap_iteration_71_probability_weight"])
    men_mid_gap_iteration_69_probability_weight = 1.0 - men_mid_gap_iteration_71_probability_weight
    men_high_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["men_high_gap_iteration_71_probability_weight"])
    men_high_gap_iteration_69_probability_weight = 1.0 - men_high_gap_iteration_71_probability_weight

    women_low_gap_iteration_71_probability_weight = float(official_router.DEFAULT_CONFIG["women_low_gap_iteration_71_probability_weight"])
    women_low_gap_iteration_69_probability_weight = 1.0 - women_low_gap_iteration_71_probability_weight
    women_mid_gap_iteration_71_probability_weight = float(official_router.DEFAULT_CONFIG["women_mid_gap_iteration_71_probability_weight"])
    women_mid_gap_iteration_69_probability_weight = 1.0 - women_mid_gap_iteration_71_probability_weight
    women_high_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["women_high_gap_iteration_71_probability_weight"])
    women_high_gap_iteration_69_probability_weight = 1.0 - women_high_gap_iteration_71_probability_weight
    women_extreme_gap_iteration_71_probability_weight = float(official_benchmark.CONFIG["women_extreme_gap_iteration_71_probability_weight"])
    women_extreme_gap_iteration_69_probability_weight = 1.0 - women_extreme_gap_iteration_71_probability_weight
    women_low_gap_threshold = float(official_router.DEFAULT_CONFIG["women_low_gap_threshold"])
    women_mid_gap_threshold = float(official_router.DEFAULT_CONFIG["women_mid_gap_threshold"])
    women_high_gap_threshold = float(official_benchmark.CONFIG["women_high_gap_threshold"])

    men_mask = validation_rows["men_women"].to_numpy() == 1
    women_mask = ~men_mask
    probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)

    men_low_gap_mask = men_mask & (probability_gap <= men_low_gap_threshold)
    men_mid_gap_mask = men_mask & (probability_gap > men_low_gap_threshold) & (probability_gap <= men_mid_gap_threshold)
    men_high_gap_mask = men_mask & ~(men_low_gap_mask | men_mid_gap_mask)
    women_low_gap_mask = women_mask & (probability_gap <= women_low_gap_threshold)
    women_mid_gap_mask = women_mask & (probability_gap > women_low_gap_threshold) & (probability_gap <= women_mid_gap_threshold)
    women_high_gap_mask = women_mask & (probability_gap > women_mid_gap_threshold) & (probability_gap <= women_high_gap_threshold)
    women_extreme_gap_mask = women_mask & (probability_gap > women_high_gap_threshold)

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

    return probabilities, probability_gap


def build_active_matchup_tables() -> family_driver.OfficialMatchupTables:
    benchmark_state = family_driver.build_base_benchmark_state()
    block_rate_spec = next(
        spec
        for spec in feature_sweep.family_specifications()
        if spec.name == "Current-season block-rate features"
    )
    candidate_tables, _added_38, _added_49 = family_driver.build_family_matchup_tables(
        benchmark_state=benchmark_state,
        family_spec=block_rate_spec,
    )
    return candidate_tables


def build_inner_rows() -> pl.DataFrame:
    matchup_tables = build_active_matchup_tables()
    available_seasons = sorted(
        int(season)
        for season in matchup_tables.benchmark_benchmark_bundle.matchups["Season"].unique().to_list()
    )

    row_frames: list[pl.DataFrame] = []
    for outer_season in pooled_base.latest_completed_validation_seasons():
        inner_validation_season = latest_prior_tournament_season(available_seasons, outer_season)
        print(f"Scoring outer season {outer_season} with inner validation season {inner_validation_season}...")

        source_predictions, targets, validation_rows = search.raw_margin_predictions(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            validation_season=inner_validation_season,
        )
        iteration_69_calibrator = search.fit_iteration_69_calibrator(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            target_season=inner_validation_season,
        )
        iteration_71_calibrator = search.fit_iteration_71_calibrator(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            target_season=inner_validation_season,
        )
        iteration_69_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_69_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_69_calibrator,
            calibration_mode="iteration_69",
        )
        iteration_71_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_71_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_71_calibrator,
            calibration_mode="iteration_71",
        )
        probabilities, probability_gap = final_probabilities(
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
            validation_rows=validation_rows,
        )

        men_mask = validation_rows["men_women"].to_numpy() == 1
        regime_labels = router_regimes(men_mask=men_mask, probability_gap=probability_gap)
        official_mask = official_search.official_row_mask(validation_rows)

        row_frame = validation_rows.with_columns(
            pl.lit(outer_season).alias("outer_season"),
            pl.lit(inner_validation_season).alias("inner_validation_season"),
            pl.Series("target", targets),
            pl.Series("iteration_69_probability", iteration_69_probabilities),
            pl.Series("iteration_71_probability", iteration_71_probabilities),
            pl.Series("prediction", probabilities),
            pl.Series("probability_gap", probability_gap),
            pl.Series("router_regime", regime_labels.tolist()),
            pl.Series("official_row_flag", official_mask.astype(np.int8)),
        ).with_columns(
            ((pl.col("prediction") - pl.col("target")) ** 2).alias("error_sq"),
            ((pl.col("iteration_69_probability") - pl.col("target")) ** 2).alias("iteration_69_error_sq"),
            ((pl.col("iteration_71_probability") - pl.col("target")) ** 2).alias("iteration_71_error_sq"),
            (pl.col("prediction") - pl.col("target")).alias("residual"),
            (pl.col("prediction") * 10.0).floor().clip(0, 9).cast(pl.Int64).alias("confidence_decile"),
            pl.col("delta_seed_number").abs().alias("abs_delta_seed_number"),
        ).with_columns(
            pl.when(pl.col("abs_delta_seed_number") == 0)
            .then(pl.lit("seed_gap_0"))
            .when(pl.col("abs_delta_seed_number") == 1)
            .then(pl.lit("seed_gap_1"))
            .when(pl.col("abs_delta_seed_number") == 2)
            .then(pl.lit("seed_gap_2"))
            .when(pl.col("abs_delta_seed_number") <= 4)
            .then(pl.lit("seed_gap_3_4"))
            .otherwise(pl.lit("seed_gap_5_plus"))
            .alias("abs_seed_gap_bucket"),
            pl.when(pl.col("probability_gap") <= 0.02)
            .then(pl.lit("gap_le_0p02"))
            .when(pl.col("probability_gap") <= 0.04)
            .then(pl.lit("gap_0p02_0p04"))
            .when(pl.col("probability_gap") <= 0.065)
            .then(pl.lit("gap_0p04_0p065"))
            .when(pl.col("probability_gap") <= 0.08)
            .then(pl.lit("gap_0p065_0p08"))
            .otherwise(pl.lit("gap_gt_0p08"))
            .alias("probability_gap_bucket"),
        )
        row_frames.append(row_frame.filter(pl.col("official_row_flag") == 1))

    combined = pl.concat(row_frames, how="vertical_relaxed")

    late_bucket_labels, late_thresholds = qbucket(
        combined["delta_block_feature_late"].to_numpy(),
        "delta_block_feature_late",
    )
    neutral_bucket_labels, neutral_thresholds = qbucket(
        combined["delta_block_feature_neutral"].to_numpy(),
        "delta_block_feature_neutral",
    )
    overall_bucket_labels, overall_thresholds = qbucket(
        combined["delta_block_feature_overall"].to_numpy(),
        "delta_block_feature_overall",
    )

    combined = combined.with_columns(
        pl.Series("delta_block_feature_late_bucket", late_bucket_labels.tolist()),
        pl.Series("delta_block_feature_neutral_bucket", neutral_bucket_labels.tolist()),
        pl.Series("delta_block_feature_overall_bucket", overall_bucket_labels.tolist()),
    )

    combined.write_parquet(OUTPUT_ROWS_PATH)

    combined = combined.with_columns(
        pl.when(pl.col("men_women") == 1).then(pl.lit("men")).otherwise(pl.lit("women")).alias("division_label"),
    )
    return combined


def summarize_slice_family(rows: pl.DataFrame, group_columns: Iterable[str], family_name: str) -> pl.DataFrame:
    overall_by_outer = rows.group_by("outer_season").agg(
        pl.col("error_sq").mean().alias("outer_brier"),
    )

    per_outer = rows.group_by(["outer_season", *group_columns]).agg(
        pl.len().alias("row_count"),
        pl.col("error_sq").mean().alias("slice_brier"),
        pl.col("iteration_69_error_sq").mean().alias("iteration_69_slice_brier"),
        pl.col("iteration_71_error_sq").mean().alias("iteration_71_slice_brier"),
        pl.col("prediction").mean().alias("mean_prediction"),
        pl.col("target").mean().alias("positive_rate"),
        pl.col("residual").mean().alias("mean_residual"),
        pl.col("residual").abs().mean().alias("mean_abs_residual"),
    ).join(
        overall_by_outer,
        on="outer_season",
        how="left",
    ).with_columns(
        (pl.col("slice_brier") - pl.col("outer_brier")).alias("outer_excess_brier"),
    )

    aggregated = per_outer.group_by(list(group_columns)).agg(
        pl.col("row_count").sum().alias("total_rows"),
        pl.col("outer_season").n_unique().alias("outer_season_count"),
        pl.col("slice_brier").mean().alias("mean_slice_brier"),
        pl.col("outer_excess_brier").mean().alias("mean_outer_excess_brier"),
        pl.col("outer_excess_brier").max().alias("max_outer_excess_brier"),
        pl.col("iteration_69_slice_brier").mean().alias("mean_iteration_69_slice_brier"),
        pl.col("iteration_71_slice_brier").mean().alias("mean_iteration_71_slice_brier"),
        pl.col("mean_prediction").mean().alias("mean_prediction"),
        pl.col("positive_rate").mean().alias("positive_rate"),
        pl.col("mean_residual").mean().alias("mean_residual"),
        pl.col("mean_abs_residual").mean().alias("mean_abs_residual"),
    ).with_columns(
        (pl.col("mean_slice_brier") - pl.col("mean_iteration_69_slice_brier")).alias("router_regret_vs_69"),
        (pl.col("mean_slice_brier") - pl.col("mean_iteration_71_slice_brier")).alias("router_regret_vs_71"),
        pl.lit(family_name).alias("slice_family"),
    )

    slice_expr = pl.concat_str(
        [
            pl.concat_str(
                [pl.lit(f"{column}="), pl.col(column).cast(pl.Utf8)],
                separator="",
            )
            for column in group_columns
        ],
        separator=" | ",
    )
    return aggregated.with_columns(slice_expr.alias("slice_label")).select(
        "slice_family",
        "slice_label",
        "total_rows",
        "outer_season_count",
        "mean_slice_brier",
        "mean_outer_excess_brier",
        "max_outer_excess_brier",
        "mean_iteration_69_slice_brier",
        "mean_iteration_71_slice_brier",
        "router_regret_vs_69",
        "router_regret_vs_71",
        "mean_prediction",
        "positive_rate",
        "mean_residual",
        "mean_abs_residual",
    )


def slice_tables(rows: pl.DataFrame) -> pl.DataFrame:
    families = [
        ("division", ["division_label"]),
        ("router_regime", ["router_regime"]),
        ("possible_round", ["possible_round"]),
        ("abs_seed_gap_bucket", ["abs_seed_gap_bucket"]),
        ("confidence_decile", ["confidence_decile"]),
        ("probability_gap_bucket", ["probability_gap_bucket"]),
        ("block_late_bucket", ["delta_block_feature_late_bucket"]),
        ("block_neutral_bucket", ["delta_block_feature_neutral_bucket"]),
        ("block_overall_bucket", ["delta_block_feature_overall_bucket"]),
        ("division_x_router_regime", ["division_label", "router_regime"]),
        ("division_x_possible_round", ["division_label", "possible_round"]),
        ("router_regime_x_gap_bucket", ["router_regime", "probability_gap_bucket"]),
        ("block_late_x_seed_gap", ["delta_block_feature_late_bucket", "abs_seed_gap_bucket"]),
    ]

    tables = [summarize_slice_family(rows, group_columns, family_name) for family_name, group_columns in families]
    return pl.concat(tables, how="vertical_relaxed").filter(
        (pl.col("total_rows") >= 25) & (pl.col("outer_season_count") >= 2)
    ).sort(
        ["mean_outer_excess_brier", "total_rows"],
        descending=[True, True],
    )


def top_findings(slice_summary: pl.DataFrame) -> dict[str, object]:
    worst_overall = slice_summary.select(
        "slice_family",
        "slice_label",
        "total_rows",
        "outer_season_count",
        "mean_slice_brier",
        "mean_outer_excess_brier",
        "router_regret_vs_69",
        "router_regret_vs_71",
        "mean_prediction",
        "positive_rate",
    ).head(10)

    block_related = slice_summary.filter(
        pl.col("slice_family").is_in(["block_late_bucket", "block_neutral_bucket", "block_overall_bucket", "block_late_x_seed_gap"])
    ).select(
        "slice_family",
        "slice_label",
        "total_rows",
        "outer_season_count",
        "mean_slice_brier",
        "mean_outer_excess_brier",
        "router_regret_vs_69",
        "router_regret_vs_71",
        "mean_prediction",
        "positive_rate",
    ).head(10)

    return {
        "worst_overall": worst_overall.to_dicts(),
        "worst_block_related": block_related.to_dicts(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ROWS_PATH.exists():
        rows = pl.read_parquet(OUTPUT_ROWS_PATH)
        if "division_label" not in rows.columns:
            rows = rows.with_columns(
                pl.when(pl.col("men_women") == 1).then(pl.lit("men")).otherwise(pl.lit("women")).alias("division_label"),
            )
    else:
        rows = build_inner_rows()
    slice_summary = slice_tables(rows)

    pooled_brier = float(rows["error_sq"].mean())
    pooled_69_brier = float(rows["iteration_69_error_sq"].mean())
    pooled_71_brier = float(rows["iteration_71_error_sq"].mean())
    by_division = rows.group_by("division_label").agg(
        pl.len().alias("row_count"),
        pl.col("error_sq").mean().alias("brier"),
        pl.col("iteration_69_error_sq").mean().alias("iteration_69_brier"),
        pl.col("iteration_71_error_sq").mean().alias("iteration_71_brier"),
    ).sort("division_label")
    by_regime = rows.group_by("router_regime").agg(
        pl.len().alias("row_count"),
        pl.col("error_sq").mean().alias("brier"),
        pl.col("iteration_69_error_sq").mean().alias("iteration_69_brier"),
        pl.col("iteration_71_error_sq").mean().alias("iteration_71_brier"),
        pl.col("probability_gap").mean().alias("mean_probability_gap"),
    ).sort("brier", descending=True)

    summary_payload = {
        "pooled_inner_brier": pooled_brier,
        "pooled_iteration_69_brier": pooled_69_brier,
        "pooled_iteration_71_brier": pooled_71_brier,
        "row_count": rows.height,
        "outer_seasons": rows["outer_season"].unique().sort().to_list(),
        "inner_validation_seasons": rows["inner_validation_season"].unique().sort().to_list(),
        "division_summary": by_division.to_dicts(),
        "router_regime_summary": by_regime.to_dicts(),
        "top_findings": top_findings(slice_summary),
    }
    OUTPUT_SUMMARY_PATH.write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))

    print(f"Row artifact: {OUTPUT_ROWS_PATH}")
    print(f"Summary artifact: {OUTPUT_SUMMARY_PATH}")
    print(f"Pooled inner Brier: {pooled_brier:.6f}")
    print(f"Iteration 69 pooled inner Brier: {pooled_69_brier:.6f}")
    print(f"Iteration 71 pooled inner Brier: {pooled_71_brier:.6f}")
    print("Division summary:")
    print(by_division)
    print("Worst slice candidates:")
    print(slice_summary.select("slice_family", "slice_label", "total_rows", "mean_slice_brier", "mean_outer_excess_brier").head(10))


if __name__ == "__main__":
    main()
