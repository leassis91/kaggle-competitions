from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson

import train_iteration_28_mike_filtered_quality as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_33_vilnius_selected_stat_subset")
REMOVED_FEATURE_COLUMNS = {
    "low_avg_or",
    "high_avg_or",
    "delta_avg_or",
    "low_avg_dr",
    "high_avg_dr",
    "delta_avg_dr",
}


def build_pooled_margin_matchups() -> tuple[benchmark.pl.DataFrame, list[str]]:
    matchups, feature_columns = benchmark.build_pooled_margin_matchups()
    selected_feature_columns = [column for column in feature_columns if column not in REMOVED_FEATURE_COLUMNS]
    return matchups, selected_feature_columns


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": benchmark.baseline.baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": benchmark.baseline.baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin_plus_bracket_context",
        "feature_family": "iteration_28_with_vilnius_selected_stat_subset",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_margin_matchups()
    validation_summary = benchmark.baseline.run_validation(matchups, feature_columns)
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
