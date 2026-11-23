from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import pooled_quality_variants as quality_variants
import train_iteration_38_vilnius_plus_capped_quality as feature_stack
import train_iteration_44_division_specific_margin_models as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_52_ridge_capped_quality")
QUALITY_COLUMN = "ridge_capped_quality"
MARGIN_CAP = 15.0
RIDGE_LAMBDA = 1.0
QUALITY_FEATURE_COLUMNS = [
    "low_ridge_capped_quality",
    "high_ridge_capped_quality",
    "delta_ridge_capped_quality",
]


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = feature_stack.build_pooled_margin_matchups()
    quality = quality_variants.build_pooled_filtered_quality_features(
        output_column=QUALITY_COLUMN,
        margin_cap=MARGIN_CAP,
        ridge_lambda=RIDGE_LAMBDA,
    )
    low_quality = quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col(QUALITY_COLUMN).alias("low_ridge_capped_quality"),
    )
    high_quality = quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col(QUALITY_COLUMN).alias("high_ridge_capped_quality"),
    )

    matchup_frame = (
        matchups.join(low_quality, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_quality, on=["Season", "HigherTeamID", "men_women"], how="left")
        .with_columns(
            pl.col("low_ridge_capped_quality").fill_null(0.0),
            pl.col("high_ridge_capped_quality").fill_null(0.0),
        )
        .with_columns(
            (pl.col("low_ridge_capped_quality") - pl.col("high_ridge_capped_quality")).alias(
                "delta_ridge_capped_quality"
            )
        )
    )

    return matchup_frame, feature_columns + QUALITY_FEATURE_COLUMNS


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": {
            "men": feature_stack.benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
            "women": benchmark.tuned.LIGHTGBM_PARAMS,
        },
        "feature_columns": feature_columns,
        "minimum_pooled_season": feature_stack.benchmark.benchmark.baseline.baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_division_specific_margin_plus_bracket_context",
        "feature_family": "iteration_44_plus_ridge_capped_quality_view",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_margin_matchups()
    validation_summary = benchmark.run_validation(matchups, feature_columns)
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
