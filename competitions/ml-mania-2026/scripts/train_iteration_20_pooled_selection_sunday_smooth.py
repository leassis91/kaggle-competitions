from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import orjson
import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline


ARTIFACT_DIR = Path("artifacts/iteration_20_pooled_selection_sunday_smooth")
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 5.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
}


def build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LIGHTGBM_PARAMS)


def run_validation(matchups: pl.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    validation_seasons = baseline.latest_completed_validation_seasons()
    fold_summaries: list[dict[str, float | int]] = []

    for validation_season in validation_seasons:
        training_rows = matchups.filter(pl.col("Season") < validation_season)
        validation_rows = matchups.filter(pl.col("Season") == validation_season)

        model = build_model()
        model.fit(training_rows.select(feature_columns).to_numpy(), training_rows["Target"].to_numpy())
        probabilities = model.predict_proba(validation_rows.select(feature_columns).to_numpy())[:, 1]
        targets = validation_rows["Target"].to_numpy()

        men_mask = validation_rows["men_women"].to_numpy() == 1
        women_mask = ~men_mask

        fold_summaries.append(
            {
                "season": validation_season,
                "combined_brier": baseline.brier_score(probabilities, targets),
                "men_brier": baseline.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": baseline.brier_score(probabilities[women_mask], targets[women_mask]),
            }
        )

    return {
        "validation_seasons": validation_seasons,
        "mean_brier": float(np.mean([fold_summary["combined_brier"] for fold_summary in fold_summaries])),
        "folds": fold_summaries,
    }


def write_validation_summary(validation_summary: dict[str, object], feature_columns: list[str]) -> None:
    summary_payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "validation": validation_summary,
        "lightgbm_params": LIGHTGBM_PARAMS,
        "feature_columns": feature_columns,
        "minimum_pooled_season": baseline.MIN_POOLED_SEASON,
        "information_cutoff": "Selection Sunday",
    }
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matchups, feature_columns = baseline.build_pooled_matchups()
    validation_summary = run_validation(matchups, feature_columns)
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
