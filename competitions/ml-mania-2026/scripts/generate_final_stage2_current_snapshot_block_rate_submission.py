from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import orjson
import polars as pl

import official_current_block_rate_benchmark as block_rate_benchmark
import official_feature_family_driver as family_driver
import pooled_bracket_features as bracket_features
import pooled_mike_kim_features as mike_features
import pooled_quality_variants as quality_variants
import pooled_vilnius_features as vilnius_features
import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_24_pooled_selection_sunday_bracket_features as bracket_base
import train_iteration_38_vilnius_plus_capped_quality as capped_quality_iteration
import train_iteration_49_seed_closure_quality as seed_closure_iteration
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as probability_blend
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as official_router
import train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend as official_benchmark


TARGET_SEASON = 2026
EXPECTED_SUBMISSION_ROWS = 132133
SUBMISSION_PATH = Path("submissions/2026_stage2_current_snapshot_block_rate.csv")
ARTIFACT_DIR = Path("artifacts/final_stage2_current_snapshot_block_rate")
MANIFEST_PATH = ARTIFACT_DIR / "submission_manifest.json"
VALIDATION_PATH = ARTIFACT_DIR / "validation_summary.json"
CURRENT_SNAPSHOT_BENCHMARK_PATH = Path(
    "artifacts/iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search/validation_summary.json"
)


def load_current_snapshot_benchmark() -> dict[str, Any]:
    payload = cast(dict[str, Any], orjson.loads(CURRENT_SNAPSHOT_BENCHMARK_PATH.read_bytes()))
    return cast(dict[str, Any], payload["current_snapshot_no_rule_validation"])


def build_division_lookup() -> pl.DataFrame:
    men_lookup = pl.read_csv(pooled_base.DATA_DIR / "MTeams.csv").select(
        pl.col("TeamID").cast(pl.Int64),
        pl.lit(1).cast(pl.Int8).alias("men_women"),
    )
    women_lookup = pl.read_csv(pooled_base.DATA_DIR / "WTeams.csv").select(
        pl.col("TeamID").cast(pl.Int64),
        pl.lit(0).cast(pl.Int8).alias("men_women"),
    )
    return pl.concat([men_lookup, women_lookup], how="vertical_relaxed")


def parse_submission_template() -> pl.DataFrame:
    sample_submission = pl.read_csv(pooled_base.DATA_DIR / "SampleSubmissionStage2.csv").with_columns(
        pl.col("ID").str.split_exact("_", 2).alias("id_parts")
    ).select(
        pl.col("ID"),
        pl.col("id_parts").struct.field("field_0").cast(pl.Int64).alias("Season"),
        pl.col("id_parts").struct.field("field_1").cast(pl.Int64).alias("LowerTeamID"),
        pl.col("id_parts").struct.field("field_2").cast(pl.Int64).alias("HigherTeamID"),
    )

    division_lookup = build_division_lookup()
    template_with_division = (
        sample_submission.join(
            division_lookup.rename({"TeamID": "LowerTeamID", "men_women": "lower_men_women"}),
            on="LowerTeamID",
            how="left",
        )
        .join(
            division_lookup.rename({"TeamID": "HigherTeamID", "men_women": "higher_men_women"}),
            on="HigherTeamID",
            how="left",
        )
    )

    invalid_rows = template_with_division.filter(
        pl.col("lower_men_women").is_null()
        | pl.col("higher_men_women").is_null()
        | (pl.col("lower_men_women") != pl.col("higher_men_women"))
    )
    if not invalid_rows.is_empty():
        raise ValueError("Sample submission contains rows with missing or mixed men/women division membership.")

    return template_with_division.select(
        "ID",
        "Season",
        "LowerTeamID",
        "HigherTeamID",
        pl.col("lower_men_women").alias("men_women"),
    )


def collect_snapshot_facts(submission_template: pl.DataFrame) -> dict[str, Any]:
    men_seeds_2026 = pl.read_csv(pooled_base.DATA_DIR / "MNCAATourneySeeds.csv").filter(pl.col("Season") == TARGET_SEASON)
    women_seeds_2026 = pl.read_csv(pooled_base.DATA_DIR / "WNCAATourneySeeds.csv").filter(pl.col("Season") == TARGET_SEASON)
    men_daynum = pl.read_csv(pooled_base.DATA_DIR / "MRegularSeasonCompactResults.csv").filter(
        pl.col("Season") == TARGET_SEASON
    )["DayNum"].max()
    women_daynum = pl.read_csv(pooled_base.DATA_DIR / "WRegularSeasonCompactResults.csv").filter(
        pl.col("Season") == TARGET_SEASON
    )["DayNum"].max()
    men_massey_summary = cast(
        pl.DataFrame,
        pl.scan_csv(pooled_base.DATA_DIR / "MMasseyOrdinals.csv")
        .filter(pl.col("Season") == TARGET_SEASON)
        .select(pl.col("RankingDayNum").max().cast(pl.Int64).alias("max_daynum"))
        .collect()
    )
    men_massey_daynum = cast(int | None, men_massey_summary.get_column("max_daynum")[0])

    if submission_template.height != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SUBMISSION_ROWS} submission rows, found {submission_template.height} in SampleSubmissionStage2.csv."
        )
    if men_seeds_2026.is_empty() or women_seeds_2026.is_empty():
        raise ValueError("Missing 2026 men or women NCAA tournament seeds.")
    if men_daynum != 132 or women_daynum != 132:
        raise ValueError(
            f"Expected 2026 regular-season results through DayNum 132, found men={men_daynum}, women={women_daynum}."
        )

    return {
        "submission_rows": submission_template.height,
        "men_seed_rows_2026": men_seeds_2026.height,
        "women_seed_rows_2026": women_seeds_2026.height,
        "men_regular_season_max_daynum_2026": men_daynum,
        "women_regular_season_max_daynum_2026": women_daynum,
        "men_massey_max_daynum_2026": men_massey_daynum,
    }


def build_pooled_current_season_team_features() -> pl.DataFrame:
    return pl.concat(
        [pooled_base.build_current_season_team_features(config) for config in pooled_base.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )


def add_team_quality_views(
    matchup_frame: pl.DataFrame,
    quality_frame: pl.DataFrame,
    source_column: str,
    low_column: str,
    high_column: str,
    delta_column: str,
) -> pl.DataFrame:
    low_quality = quality_frame.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col(source_column).alias(low_column),
    )
    high_quality = quality_frame.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col(source_column).alias(high_column),
    )
    return (
        matchup_frame.join(low_quality, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_quality, on=["Season", "HigherTeamID", "men_women"], how="left")
        .with_columns(
            pl.col(low_column).fill_null(0.0),
            pl.col(high_column).fill_null(0.0),
        )
        .with_columns((pl.col(low_column) - pl.col(high_column)).alias(delta_column))
    )


def build_base_submission_matchups(submission_template: pl.DataFrame) -> pl.DataFrame:
    pooled_team_features = build_pooled_current_season_team_features()
    base_team_feature_columns = [
        column
        for column in pooled_team_features.columns
        if column not in {"Season", "TeamID", "men_women"}
    ]
    low_team_features = pooled_team_features.select(
        pl.col("Season"),
        pl.col("men_women"),
        pl.col("TeamID").alias("LowerTeamID"),
        *[pl.col(column).alias(f"low_{column}") for column in base_team_feature_columns],
    )
    high_team_features = pooled_team_features.select(
        pl.col("Season"),
        pl.col("men_women"),
        pl.col("TeamID").alias("HigherTeamID"),
        *[pl.col(column).alias(f"high_{column}") for column in base_team_feature_columns],
    )
    filled_matchups = (
        submission_template.join(low_team_features, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_team_features, on=["Season", "HigherTeamID", "men_women"], how="left")
        .with_columns(
            pl.lit(None).cast(pl.Float64).alias("Target"),
            pl.lit(None).cast(pl.Float64).alias("TargetMargin"),
            pl.col("low_seed").fill_null(pooled_base.LOW_HISTORY_DEFAULT_SEED),
            pl.col("high_seed").fill_null(pooled_base.LOW_HISTORY_DEFAULT_SEED),
            *[
                pl.col(column).fill_null(0.0)
                for column in (
                    f"low_{column}"
                    for column in base_team_feature_columns
                    if column != "seed"
                )
            ],
            *[
                pl.col(column).fill_null(0.0)
                for column in (
                    f"high_{column}"
                    for column in base_team_feature_columns
                    if column != "seed"
                )
            ],
        )
        .with_columns(
            (pl.col("high_seed") - pl.col("low_seed")).alias("delta_seed"),
            (pl.col("low_avg_score") - pl.col("high_avg_score")).alias("delta_avg_score"),
            (pl.col("low_avg_fga") - pl.col("high_avg_fga")).alias("delta_avg_fga"),
            (pl.col("low_avg_or") - pl.col("high_avg_or")).alias("delta_avg_or"),
            (pl.col("low_avg_dr") - pl.col("high_avg_dr")).alias("delta_avg_dr"),
            (pl.col("low_avg_blk") - pl.col("high_avg_blk")).alias("delta_avg_blk"),
            (pl.col("low_avg_pf") - pl.col("high_avg_pf")).alias("delta_avg_pf"),
            (pl.col("low_avg_opp_fga") - pl.col("high_avg_opp_fga")).alias("delta_avg_opp_fga"),
            (pl.col("low_avg_opp_blk") - pl.col("high_avg_opp_blk")).alias("delta_avg_opp_blk"),
            (pl.col("low_avg_opp_pf") - pl.col("high_avg_opp_pf")).alias("delta_avg_opp_pf"),
            (pl.col("low_avg_point_diff") - pl.col("high_avg_point_diff")).alias("delta_avg_point_diff"),
        )
    )
    return filled_matchups


def build_iteration_38_submission_matchups(submission_template: pl.DataFrame) -> pl.DataFrame:
    base_matchups = build_base_submission_matchups(submission_template)
    bracket_matchups = (
        base_matchups.join(
            bracket_features.build_pooled_pair_bracket_features(),
            on=["Season", "LowerTeamID", "HigherTeamID"],
            how="left",
        ).with_columns(*[pl.col(column).fill_null(0) for column in bracket_base.BRACKET_FEATURE_COLUMNS])
    )
    interaction_matchups = bracket_matchups.with_columns(
        (pl.col("possible_round") * pl.col("delta_seed_number")).alias("possible_round_x_delta_seed_number"),
        (pl.col("same_region_flag") * pl.col("delta_seed_number")).alias("same_region_x_delta_seed_number"),
        (pl.col("same_final_four_side_flag") * pl.col("delta_seed_number")).alias(
            "same_final_four_side_x_delta_seed_number"
        ),
        (pl.col("opening_slot_gap") * pl.col("delta_seed_number")).alias("opening_slot_gap_x_delta_seed_number"),
        (pl.col("possible_round") * pl.col("opening_slot_gap")).alias("possible_round_x_opening_slot_gap"),
        (pl.col("possible_round") * pl.col("any_play_in_flag")).alias("possible_round_x_any_play_in_flag"),
        (pl.col("possible_round") * pl.col("same_region_flag")).alias("possible_round_x_same_region_flag"),
    )
    mike_quality = mike_features.build_pooled_filtered_quality_features()
    with_mike_quality = add_team_quality_views(
        matchup_frame=interaction_matchups,
        quality_frame=mike_quality,
        source_column="mike_quality",
        low_column="low_mike_quality",
        high_column="high_mike_quality",
        delta_column="delta_mike_quality",
    )
    vilnius_quality = vilnius_features.build_pooled_filtered_quality_features()
    with_vilnius_quality = add_team_quality_views(
        matchup_frame=with_mike_quality,
        quality_frame=vilnius_quality,
        source_column="vilnius_quality",
        low_column="low_vilnius_quality",
        high_column="high_vilnius_quality",
        delta_column="delta_vilnius_quality",
    )
    capped_quality = quality_variants.build_pooled_filtered_quality_features(
        output_column=capped_quality_iteration.CAPPED_QUALITY_COLUMN,
        margin_cap=capped_quality_iteration.CAPPED_MARGIN,
    )
    return add_team_quality_views(
        matchup_frame=with_vilnius_quality,
        quality_frame=capped_quality,
        source_column=capped_quality_iteration.CAPPED_QUALITY_COLUMN,
        low_column="low_capped_margin_quality",
        high_column="high_capped_margin_quality",
        delta_column="delta_capped_margin_quality",
    )


def build_iteration_49_submission_matchups(submission_template: pl.DataFrame) -> pl.DataFrame:
    iteration_38_matchups = build_iteration_38_submission_matchups(submission_template)
    seed_closure_quality = quality_variants.build_pooled_filtered_quality_features(
        output_column=seed_closure_iteration.QUALITY_COLUMN,
        relevance_hops=seed_closure_iteration.RELEVANCE_HOPS,
    )
    return add_team_quality_views(
        matchup_frame=iteration_38_matchups,
        quality_frame=seed_closure_quality,
        source_column=seed_closure_iteration.QUALITY_COLUMN,
        low_column="low_seed_closure_quality",
        high_column="high_seed_closure_quality",
        delta_column="delta_seed_closure_quality",
    )


def append_submission_rows(historical_matchups: pl.DataFrame, submission_matchups: pl.DataFrame) -> pl.DataFrame:
    historical_with_id = historical_matchups.with_columns(pl.lit(None).cast(pl.Utf8).alias("ID"))
    ordered_columns = historical_with_id.columns
    return pl.concat([historical_with_id, submission_matchups.select(ordered_columns)], how="vertical_relaxed")


def build_augmented_matchup_tables(
    submission_template: pl.DataFrame,
) -> tuple[family_driver.OfficialMatchupTables, dict[str, list[str]]]:
    benchmark_state = family_driver.build_base_benchmark_state()
    family_spec = block_rate_benchmark.block_rate_family_spec()
    active_tables, _iteration_38_added_columns, _iteration_49_added_columns = family_driver.build_family_matchup_tables(
        benchmark_state=benchmark_state,
        family_spec=family_spec,
    )

    submission_iteration_38 = build_iteration_38_submission_matchups(submission_template)
    submission_iteration_49 = build_iteration_49_submission_matchups(submission_template)
    submission_iteration_38, submission_iteration_38_feature_columns = family_spec.extend_iteration_38(
        submission_iteration_38,
        benchmark_state.iteration_38_bundle.feature_columns,
    )
    iteration_49_builder = family_spec.extend_iteration_49 or family_spec.extend_iteration_38
    submission_iteration_49, submission_iteration_49_feature_columns = iteration_49_builder(
        submission_iteration_49,
        benchmark_state.iteration_49_bundle.feature_columns,
    )

    expected_pooled_columns = active_tables.pooled_bundle.feature_columns
    expected_alternative_columns = active_tables.alternative_bundle.feature_columns
    if submission_iteration_38_feature_columns != expected_pooled_columns:
        raise ValueError("Submission iteration 38 feature columns do not match the validated block-rate benchmark columns.")
    if submission_iteration_49_feature_columns != expected_alternative_columns:
        raise ValueError("Submission iteration 49 feature columns do not match the validated block-rate benchmark columns.")

    augmented_tables = family_driver.OfficialMatchupTables(
        benchmark_benchmark_bundle=family_driver.MatchupBundle(
            matchups=append_submission_rows(active_tables.benchmark_benchmark_bundle.matchups, submission_iteration_38),
            feature_columns=expected_pooled_columns,
        ),
        benchmark_alternative_bundle=family_driver.MatchupBundle(
            matchups=append_submission_rows(active_tables.benchmark_alternative_bundle.matchups, submission_iteration_49),
            feature_columns=expected_alternative_columns,
        ),
        pooled_bundle=family_driver.MatchupBundle(
            matchups=append_submission_rows(active_tables.pooled_bundle.matchups, submission_iteration_38),
            feature_columns=expected_pooled_columns,
        ),
        alternative_bundle=family_driver.MatchupBundle(
            matchups=append_submission_rows(active_tables.alternative_bundle.matchups, submission_iteration_49),
            feature_columns=expected_alternative_columns,
        ),
    )
    return augmented_tables, {
        "benchmark_benchmark_feature_columns": expected_pooled_columns,
        "benchmark_alternative_feature_columns": expected_alternative_columns,
        "pooled_feature_columns": expected_pooled_columns,
        "alternative_feature_columns": expected_alternative_columns,
    }


def reproduce_validation() -> dict[str, Any]:
    validation_summary = cast(
        dict[str, Any],
        official_router.run_validation(
            cached_fold_inputs=block_rate_benchmark.cached_fold_inputs(),
            config=official_benchmark.CONFIG,
        ),
    )
    expected_summary = load_current_snapshot_benchmark()
    expected_fold_scores = {
        int(fold["season"]): float(fold["combined_brier"])
        for fold in cast(list[dict[str, Any]], expected_summary["folds"])
    }
    reproduced_fold_scores = {
        int(fold["season"]): float(fold["combined_brier"])
        for fold in cast(list[dict[str, Any]], validation_summary["folds"])
    }
    max_fold_abs_diff = max(
        abs(reproduced_fold_scores[season] - expected_fold_scores[season]) for season in expected_fold_scores
    )
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "expected_artifact": str(CURRENT_SNAPSHOT_BENCHMARK_PATH),
        "expected_mean_brier": float(expected_summary["mean_brier"]),
        "reproduced_mean_brier": float(validation_summary["mean_brier"]),
        "mean_brier_abs_diff": abs(float(validation_summary["mean_brier"]) - float(expected_summary["mean_brier"])),
        "expected_fold_scores": {str(season): score for season, score in expected_fold_scores.items()},
        "reproduced_fold_scores": {str(season): score for season, score in reproduced_fold_scores.items()},
        "max_fold_abs_diff": max_fold_abs_diff,
        "matches_expected": abs(float(validation_summary["mean_brier"]) - float(expected_summary["mean_brier"])) < 1e-12
        and max_fold_abs_diff < 1e-12,
        "validation": validation_summary,
    }
    VALIDATION_PATH.write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))
    if not cast(bool, summary_payload["matches_expected"]):
        raise ValueError("Refreshed-data validation reproduction did not match the saved current-snapshot benchmark.")
    return summary_payload


def generate_submission(
    submission_template: pl.DataFrame,
    augmented_tables: family_driver.OfficialMatchupTables,
) -> pl.DataFrame:
    source_predictions, _targets, validation_rows = probability_blend.raw_margin_predictions(
        benchmark_benchmark_matchups=augmented_tables.benchmark_benchmark_bundle.matchups,
        benchmark_benchmark_feature_columns=augmented_tables.benchmark_benchmark_bundle.feature_columns,
        benchmark_alternative_matchups=augmented_tables.benchmark_alternative_bundle.matchups,
        benchmark_alternative_feature_columns=augmented_tables.benchmark_alternative_bundle.feature_columns,
        pooled_matchups=augmented_tables.pooled_bundle.matchups,
        pooled_feature_columns=augmented_tables.pooled_bundle.feature_columns,
        alternative_matchups=augmented_tables.alternative_bundle.matchups,
        alternative_feature_columns=augmented_tables.alternative_bundle.feature_columns,
        validation_season=TARGET_SEASON,
    )
    iteration_69_calibrator = probability_blend.fit_iteration_69_calibrator(
        benchmark_benchmark_matchups=augmented_tables.benchmark_benchmark_bundle.matchups,
        benchmark_benchmark_feature_columns=augmented_tables.benchmark_benchmark_bundle.feature_columns,
        benchmark_alternative_matchups=augmented_tables.benchmark_alternative_bundle.matchups,
        benchmark_alternative_feature_columns=augmented_tables.benchmark_alternative_bundle.feature_columns,
        pooled_matchups=augmented_tables.pooled_bundle.matchups,
        pooled_feature_columns=augmented_tables.pooled_bundle.feature_columns,
        alternative_matchups=augmented_tables.alternative_bundle.matchups,
        alternative_feature_columns=augmented_tables.alternative_bundle.feature_columns,
        target_season=TARGET_SEASON,
    )
    iteration_71_calibrator = probability_blend.fit_iteration_71_calibrator(
        benchmark_benchmark_matchups=augmented_tables.benchmark_benchmark_bundle.matchups,
        benchmark_benchmark_feature_columns=augmented_tables.benchmark_benchmark_bundle.feature_columns,
        benchmark_alternative_matchups=augmented_tables.benchmark_alternative_bundle.matchups,
        benchmark_alternative_feature_columns=augmented_tables.benchmark_alternative_bundle.feature_columns,
        pooled_matchups=augmented_tables.pooled_bundle.matchups,
        pooled_feature_columns=augmented_tables.pooled_bundle.feature_columns,
        alternative_matchups=augmented_tables.alternative_bundle.matchups,
        alternative_feature_columns=augmented_tables.alternative_bundle.feature_columns,
        target_season=TARGET_SEASON,
    )
    iteration_69_probabilities = probability_blend.calibrated_probabilities(
        raw_predictions=probability_blend.iteration_69_raw_margin(source_predictions),
        validation_rows=validation_rows,
        calibrator=iteration_69_calibrator,
        calibration_mode="iteration_69",
    )
    iteration_71_probabilities = probability_blend.calibrated_probabilities(
        raw_predictions=probability_blend.iteration_71_raw_margin(source_predictions),
        validation_rows=validation_rows,
        calibrator=iteration_71_calibrator,
        calibration_mode="iteration_71",
    )
    probabilities = np.clip(
        block_rate_benchmark.benchmark_probabilities(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
        ),
        0.0,
        1.0,
    )
    prediction_rows = validation_rows.select("ID").with_columns(pl.Series("Pred", probabilities))
    final_submission = submission_template.select("ID").join(prediction_rows, on="ID", how="left")

    if final_submission.height != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"Expected {EXPECTED_SUBMISSION_ROWS} final submission rows, found {final_submission.height}.")
    if final_submission["Pred"].null_count() != 0:
        raise ValueError("Final submission contains null predictions.")
    if not bool(final_submission["Pred"].is_finite().all()):
        raise ValueError("Final submission contains non-finite predictions.")
    if final_submission["ID"].n_unique() != final_submission.height:
        raise ValueError("Final submission contains duplicate IDs.")
    if final_submission["ID"].to_list() != submission_template["ID"].to_list():
        raise ValueError("Final submission IDs do not match the sample submission exactly.")

    return final_submission


def write_manifest(
    snapshot_facts: dict[str, Any],
    validation_summary: dict[str, Any],
    final_submission: pl.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    prediction_array = final_submission["Pred"].to_numpy()
    manifest = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_season": TARGET_SEASON,
        "source_script": "scripts/generate_final_stage2_current_snapshot_block_rate_submission.py",
        "submission_path": str(SUBMISSION_PATH),
        "validation_path": str(VALIDATION_PATH),
        "expected_current_snapshot_benchmark_artifact": str(CURRENT_SNAPSHOT_BENCHMARK_PATH),
        "block_rate_benchmark_artifact": str(block_rate_benchmark.BLOCK_RATE_ARTIFACT_DIR / "validation_summary.json"),
        "snapshot_facts": snapshot_facts,
        "validation_reproduction": {
            "expected_mean_brier": validation_summary["expected_mean_brier"],
            "reproduced_mean_brier": validation_summary["reproduced_mean_brier"],
            "matches_expected": validation_summary["matches_expected"],
            "expected_fold_scores": validation_summary["expected_fold_scores"],
            "reproduced_fold_scores": validation_summary["reproduced_fold_scores"],
        },
        "submission_stats": {
            "row_count": final_submission.height,
            "prediction_min": float(np.min(prediction_array)),
            "prediction_max": float(np.max(prediction_array)),
            "prediction_mean": float(np.mean(prediction_array)),
        },
        "feature_columns": feature_columns,
        "official_router_config": official_benchmark.CONFIG,
        "official_default_router_config": official_router.DEFAULT_CONFIG,
    }
    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    submission_template = parse_submission_template()
    snapshot_facts = collect_snapshot_facts(submission_template)
    validation_summary = reproduce_validation()
    augmented_tables, feature_columns = build_augmented_matchup_tables(submission_template)
    final_submission = generate_submission(submission_template, augmented_tables)
    final_submission.write_csv(SUBMISSION_PATH)
    write_manifest(snapshot_facts, validation_summary, final_submission, feature_columns)

    prediction_min = cast(float, final_submission.get_column("Pred").min())
    prediction_max = cast(float, final_submission.get_column("Pred").max())
    prediction_mean = cast(float, final_submission.get_column("Pred").mean())

    print(f"Reproduced refreshed-data mean Brier: {validation_summary['reproduced_mean_brier']:.12f}")
    print(f"Wrote submission: {SUBMISSION_PATH}")
    print(
        "Prediction summary: "
        f"min={prediction_min:.6f}, "
        f"max={prediction_max:.6f}, "
        f"mean={prediction_mean:.6f}"
    )


if __name__ == "__main__":
    main()
