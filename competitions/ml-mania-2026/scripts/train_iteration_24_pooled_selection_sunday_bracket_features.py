from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import pooled_bracket_features as bracket
import train_iteration_21_pooled_selection_sunday_margin_regression as baseline


ARTIFACT_DIR = Path("artifacts/iteration_24_pooled_selection_sunday_bracket_features")
BRACKET_FEATURE_COLUMNS = [
    "low_seed_number",
    "high_seed_number",
    "delta_seed_number",
    "low_seed_region_code",
    "high_seed_region_code",
    "same_region_flag",
    "same_final_four_side_flag",
    "low_play_in_flag",
    "high_play_in_flag",
    "any_play_in_flag",
    "both_play_in_flag",
    "shared_seed_line_flag",
    "low_opening_slot_index",
    "high_opening_slot_index",
    "opening_slot_gap",
    "same_first_weekend_pod_flag",
    "same_regional_quadrant_flag",
    "possible_round",
    "first_four_pair_flag",
    "round64_pair_flag",
    "round32_pair_flag",
    "sweet16_pair_flag",
    "elite8_pair_flag",
    "final_four_pair_flag",
    "title_game_pair_flag",
]


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = baseline.build_pooled_margin_matchups()
    matchup_frame = matchups.join(
        bracket.build_pooled_pair_bracket_features(),
        on=["Season", "LowerTeamID", "HigherTeamID"],
        how="left",
    ).with_columns(*[pl.col(column).fill_null(0) for column in BRACKET_FEATURE_COLUMNS])

    return matchup_frame, feature_columns + BRACKET_FEATURE_COLUMNS


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
        "feature_family": "iteration_21_plus_bracket_geometry_and_seed_structure",
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
