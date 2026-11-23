from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import train_iteration_21_pooled_selection_sunday_margin_regression as baseline
import train_iteration_24_pooled_selection_sunday_bracket_features as bracket_base


ARTIFACT_DIR = Path("artifacts/iteration_25_pooled_selection_sunday_bracket_interactions")
INTERACTION_FEATURE_COLUMNS = [
    "possible_round_x_delta_seed_number",
    "same_region_x_delta_seed_number",
    "same_final_four_side_x_delta_seed_number",
    "opening_slot_gap_x_delta_seed_number",
    "possible_round_x_opening_slot_gap",
    "possible_round_x_any_play_in_flag",
    "possible_round_x_same_region_flag",
]


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = bracket_base.build_pooled_margin_matchups()
    matchup_frame = matchups.with_columns(
        (pl.col("possible_round") * pl.col("delta_seed_number")).alias("possible_round_x_delta_seed_number"),
        (pl.col("same_region_flag") * pl.col("delta_seed_number")).alias("same_region_x_delta_seed_number"),
        (pl.col("same_final_four_side_flag") * pl.col("delta_seed_number")).alias("same_final_four_side_x_delta_seed_number"),
        (pl.col("opening_slot_gap") * pl.col("delta_seed_number")).alias("opening_slot_gap_x_delta_seed_number"),
        (pl.col("possible_round") * pl.col("opening_slot_gap")).alias("possible_round_x_opening_slot_gap"),
        (pl.col("possible_round") * pl.col("any_play_in_flag")).alias("possible_round_x_any_play_in_flag"),
        (pl.col("possible_round") * pl.col("same_region_flag")).alias("possible_round_x_same_region_flag"),
    )

    return matchup_frame, feature_columns + INTERACTION_FEATURE_COLUMNS


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin",
        "feature_family": "iteration_24_plus_bracket_interactions",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_margin_matchups()
    validation_summary = baseline.run_validation(matchups, feature_columns)
    write_validation_summary(validation_summary, feature_columns)

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
