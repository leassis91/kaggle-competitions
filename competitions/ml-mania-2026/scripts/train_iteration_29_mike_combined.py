from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import train_iteration_26_pooled_selection_sunday_bracket_calibration as baseline
import train_iteration_27_mike_form_matchup_priors as mike_form
import train_iteration_28_mike_filtered_quality as mike_quality


ARTIFACT_DIR = Path("artifacts/iteration_29_mike_combined")


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    form_matchups, form_feature_columns = mike_form.build_pooled_margin_matchups()
    quality_matchups, quality_feature_columns = mike_quality.build_pooled_margin_matchups()
    quality_columns = [column for column in mike_quality.QUALITY_FEATURE_COLUMNS]

    matchup_frame = form_matchups.join(
        quality_matchups.select("Season", "LowerTeamID", "HigherTeamID", *quality_columns),
        on=["Season", "LowerTeamID", "HigherTeamID"],
        how="left",
    ).with_columns(*[pl.col(column).fill_null(0.0) for column in quality_columns])

    return matchup_frame, form_feature_columns + quality_columns


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": baseline.baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin_plus_bracket_context",
        "feature_family": "iteration_26_plus_all_portable_mike_kim_features",
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
