from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import cast

import orjson
import polars as pl

import pooled_mike_kim_features as mike
import train_iteration_26_pooled_selection_sunday_bracket_calibration as baseline


ARTIFACT_DIR = Path("artifacts/iteration_27_mike_form_matchup_priors")
MIKE_FEATURE_COLUMNS = [
    "low_win_ratio_14d",
    "high_win_ratio_14d",
    "low_away_win_pct",
    "high_away_win_pct",
    "delta_win_ratio_14d",
    "delta_away_win_pct",
    "h2h_meetings",
    "h2h_laplace_win_pct",
]


def build_pooled_margin_matchups() -> tuple[pl.DataFrame, list[str]]:
    matchups, feature_columns = baseline.bracket_interactions.build_pooled_margin_matchups()
    team_form = mike.build_pooled_team_form_features()
    low_team_form = team_form.select(
        pl.col("Season"),
        pl.col("TeamID").alias("LowerTeamID"),
        pl.col("men_women"),
        pl.col("win_ratio_14d").alias("low_win_ratio_14d"),
        pl.col("away_win_pct").alias("low_away_win_pct"),
    )
    high_team_form = team_form.select(
        pl.col("Season"),
        pl.col("TeamID").alias("HigherTeamID"),
        pl.col("men_women"),
        pl.col("win_ratio_14d").alias("high_win_ratio_14d"),
        pl.col("away_win_pct").alias("high_away_win_pct"),
    )

    matchup_frame = (
        matchups.join(
            low_team_form,
            on=["Season", "LowerTeamID", "men_women"],
            how="left",
        )
        .join(
            high_team_form,
            on=["Season", "HigherTeamID", "men_women"],
            how="left",
        )
        .join(
            mike.build_pooled_matchup_laplace_features(),
            on=["Season", "LowerTeamID", "HigherTeamID", "men_women"],
            how="left",
        )
        .with_columns(
            pl.col("low_win_ratio_14d").fill_null(0.0),
            pl.col("high_win_ratio_14d").fill_null(0.0),
            pl.col("low_away_win_pct").fill_null(0.5),
            pl.col("high_away_win_pct").fill_null(0.5),
            pl.col("h2h_meetings").fill_null(0.0),
            pl.col("h2h_laplace_win_pct").fill_null(0.5),
        )
        .with_columns(
            (pl.col("low_win_ratio_14d") - pl.col("high_win_ratio_14d")).alias("delta_win_ratio_14d"),
            (pl.col("low_away_win_pct") - pl.col("high_away_win_pct")).alias("delta_away_win_pct"),
        )
    )

    return matchup_frame, feature_columns + MIKE_FEATURE_COLUMNS


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
        "feature_family": "iteration_26_plus_mike_form_away_and_matchup_priors",
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
