from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import pooled_quality_variants as quality_variants
import train_iteration_32_vilnius_filtered_quality as benchmark


ARTIFACT_DIR = Path("artifacts/iteration_40_vilnius_plus_quality_views")
CAPPED_QUALITY_COLUMN = "capped_margin_quality"
RECENCY_QUALITY_COLUMN = "recency_weighted_quality"
CAPPED_MARGIN = 15.0
QUALITY_FEATURE_COLUMNS = [
    "low_capped_margin_quality",
    "high_capped_margin_quality",
    "delta_capped_margin_quality",
    "low_recency_weighted_quality",
    "high_recency_weighted_quality",
    "delta_recency_weighted_quality",
]


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = benchmark.build_pooled_margin_matchups()
    capped_quality = quality_variants.build_pooled_filtered_quality_features(
        output_column=CAPPED_QUALITY_COLUMN,
        margin_cap=CAPPED_MARGIN,
    )
    recency_quality = quality_variants.build_pooled_filtered_quality_features(
        output_column=RECENCY_QUALITY_COLUMN,
        use_recency_weights=True,
    )

    low_capped = capped_quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col(CAPPED_QUALITY_COLUMN).alias("low_capped_margin_quality"),
    )
    high_capped = capped_quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col(CAPPED_QUALITY_COLUMN).alias("high_capped_margin_quality"),
    )
    low_recency = recency_quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col(RECENCY_QUALITY_COLUMN).alias("low_recency_weighted_quality"),
    )
    high_recency = recency_quality.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col(RECENCY_QUALITY_COLUMN).alias("high_recency_weighted_quality"),
    )

    matchup_frame = (
        matchups.join(low_capped, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_capped, on=["Season", "HigherTeamID", "men_women"], how="left")
        .join(low_recency, on=["Season", "LowerTeamID", "men_women"], how="left")
        .join(high_recency, on=["Season", "HigherTeamID", "men_women"], how="left")
        .with_columns(
            pl.col("low_capped_margin_quality").fill_null(0.0),
            pl.col("high_capped_margin_quality").fill_null(0.0),
            pl.col("low_recency_weighted_quality").fill_null(0.0),
            pl.col("high_recency_weighted_quality").fill_null(0.0),
        )
        .with_columns(
            (pl.col("low_capped_margin_quality") - pl.col("high_capped_margin_quality")).alias(
                "delta_capped_margin_quality"
            ),
            (
                pl.col("low_recency_weighted_quality") - pl.col("high_recency_weighted_quality")
            ).alias("delta_recency_weighted_quality"),
        )
    )

    return matchup_frame, feature_columns + QUALITY_FEATURE_COLUMNS


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": benchmark.benchmark.baseline.baseline.LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": benchmark.benchmark.baseline.baseline.baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "nested_oof_logistic_from_predicted_margin_plus_bracket_context",
        "feature_family": "iteration_32_plus_capped_and_recency_quality_views",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = build_pooled_margin_matchups()
    validation_summary = benchmark.benchmark.baseline.run_validation(matchups, feature_columns)
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
