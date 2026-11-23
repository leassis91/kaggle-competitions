from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Callable, cast

import orjson
import polars as pl

import official_feature_family_current_season as current_season
import official_feature_family_driver as family_driver


ARTIFACT_DIR = Path("artifacts/iteration_117_iteration_116_neutral_site_strength_search")

build_pooled_team_feature_table = cast(
    Callable[[], pl.DataFrame],
    getattr(current_season, "build_pooled_team_feature_table"),
)
team_feature_output_columns = cast(
    Callable[[list[str]], list[str]],
    getattr(current_season, "team_feature_output_columns"),
)


def neutral_fill_defaults(metric_names: list[str]) -> dict[str, float]:
    fill_defaults: dict[str, float] = {}
    for metric_name in metric_names:
        default_value = 0.5 if "win" in metric_name else 0.0
        fill_defaults[f"low_{metric_name}"] = default_value
        fill_defaults[f"high_{metric_name}"] = default_value
        fill_defaults[f"delta_{metric_name}"] = 0.0
    return fill_defaults


def neutral_bundle_builder(
    metric_names: list[str],
) -> Callable[[pl.DataFrame, list[str]], tuple[pl.DataFrame, list[str]]]:
    def builder(matchups: pl.DataFrame, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
        _ = feature_columns
        team_features = build_pooled_team_feature_table().with_columns(
            (pl.col("games_played") * pl.col("neutral_game_share")).alias("neutral_game_count"),
            (pl.col("neutral_off_rating") - pl.col("neutral_def_rating")).alias("neutral_net_rating"),
            (pl.col("neutral_win_pct") - pl.col("win_pct")).alias("neutral_win_delta_vs_all"),
            (pl.col("neutral_avg_margin") - pl.col("avg_margin")).alias("neutral_margin_delta_vs_all"),
            (pl.col("neutral_off_rating") - pl.col("off_rating")).alias("neutral_off_delta_vs_all"),
            (pl.col("neutral_def_rating") - pl.col("def_rating")).alias("neutral_def_delta_vs_all"),
            ((pl.col("neutral_off_rating") - pl.col("neutral_def_rating")) - pl.col("net_rating")).alias(
                "neutral_net_delta_vs_all"
            ),
        )
        low_features = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("LowerTeamID"),
            pl.col("men_women"),
            *[pl.col(metric_name).alias(f"low_{metric_name}") for metric_name in metric_names],
        )
        high_features = team_features.select(
            pl.col("Season"),
            pl.col("TeamID").alias("HigherTeamID"),
            pl.col("men_women"),
            *[pl.col(metric_name).alias(f"high_{metric_name}") for metric_name in metric_names],
        )
        fill_defaults = neutral_fill_defaults(metric_names)
        matchup_frame = (
            matchups.join(low_features, on=["Season", "LowerTeamID", "men_women"], how="left")
            .join(high_features, on=["Season", "HigherTeamID", "men_women"], how="left")
            .with_columns(
                *[
                    pl.col(column_name).fill_null(default_value)
                    for column_name, default_value in fill_defaults.items()
                    if column_name.startswith("low_") or column_name.startswith("high_")
                ]
            )
            .with_columns(
                *[
                    (pl.col(f"low_{metric_name}") - pl.col(f"high_{metric_name}")).alias(f"delta_{metric_name}")
                    for metric_name in metric_names
                ]
            )
        )
        return matchup_frame, team_feature_output_columns(metric_names)

    return builder


FAMILY_SPECS = [
    family_driver.FamilySpec(
        name="current_season_neutral_site_only_strength_core_volume_features",
        artifact_slug="current_season_neutral_site_only_strength_core_volume_features",
        hypothesis="Neutral-site win, margin, ratings, and volume may capture portable court-independent form.",
        extend_iteration_38=neutral_bundle_builder(
            metric_names=[
                "neutral_game_count",
                "neutral_game_share",
                "neutral_win_pct",
                "neutral_avg_margin",
                "neutral_off_rating",
                "neutral_def_rating",
            ],
        ),
    ),
    family_driver.FamilySpec(
        name="current_season_neutral_site_only_strength_efficiency_features",
        artifact_slug="current_season_neutral_site_only_strength_efficiency_features",
        hypothesis="Neutral-site versus overall deltas may preserve the mean gain while reducing fold-specific overreactions.",
        extend_iteration_38=neutral_bundle_builder(
            metric_names=[
                "neutral_game_count",
                "neutral_win_pct",
                "neutral_avg_margin",
                "neutral_net_rating",
                "neutral_win_delta_vs_all",
                "neutral_margin_delta_vs_all",
                "neutral_net_delta_vs_all",
            ],
        ),
    ),
    family_driver.FamilySpec(
        name="current_season_neutral_site_only_strength_volatility_features",
        artifact_slug="current_season_neutral_site_only_strength_volatility_features",
        hypothesis="Offense-defense neutral deltas may be less noisy than raw neutral splits in the missed folds.",
        extend_iteration_38=neutral_bundle_builder(
            metric_names=[
                "neutral_game_count",
                "neutral_game_share",
                "neutral_off_rating",
                "neutral_def_rating",
                "neutral_net_rating",
                "neutral_off_delta_vs_all",
                "neutral_def_delta_vs_all",
                "neutral_net_delta_vs_all",
            ],
        ),
    ),
    family_driver.FamilySpec(
        name="current_season_neutral_site_only_strength_balanced_features",
        artifact_slug="current_season_neutral_site_only_strength_balanced_features",
        hypothesis="A balanced neutral-site bundle combining volume, ratings, pace, and efficiency may preserve the mean gain while fixing the missed folds.",
        extend_iteration_38=neutral_bundle_builder(
            metric_names=[
                "neutral_game_count",
                "neutral_game_share",
                "neutral_win_pct",
                "neutral_avg_margin",
                "neutral_off_rating",
                "neutral_def_rating",
                "neutral_net_rating",
                "neutral_win_delta_vs_all",
                "neutral_margin_delta_vs_all",
                "neutral_net_delta_vs_all",
            ],
        ),
    ),
]


def best_accepted_evaluation(
    evaluations: list[family_driver.FamilyEvaluation],
) -> family_driver.FamilyEvaluation | None:
    accepted_evaluations = [evaluation for evaluation in evaluations if evaluation.accepted]
    if not accepted_evaluations:
        return None
    return min(
        accepted_evaluations,
        key=lambda evaluation: cast(float, evaluation.validation_summary["mean_brier"]),
    )


def best_raw_evaluation(
    evaluations: list[family_driver.FamilyEvaluation],
) -> family_driver.FamilyEvaluation:
    return min(
        evaluations,
        key=lambda evaluation: cast(float, evaluation.validation_summary["mean_brier"]),
    )


def write_search_summary(
    benchmark_state: family_driver.BenchmarkState,
    evaluations: list[family_driver.FamilyEvaluation],
    selected_evaluation: family_driver.FamilyEvaluation | None,
) -> None:
    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": "current_season_neutral_site_only_strength_features_search",
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "probability_mapping": "fixed_iteration_111_official_router",
        "benchmark_name": benchmark_state.name,
        "benchmark_mean_brier": benchmark_state.mean_brier,
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_state.fold_scores.items()},
        "results": [
            {
                "name": evaluation.spec.name,
                "artifact_dir": str(evaluation.artifact_dir),
                "accepted": evaluation.accepted,
                "mean_brier": cast(float, evaluation.validation_summary["mean_brier"]),
                "improved_seasons": list(evaluation.improved_seasons),
            }
            for evaluation in evaluations
        ],
        "best_raw_result": {
            "name": best_raw_evaluation(evaluations).spec.name,
            "mean_brier": cast(float, best_raw_evaluation(evaluations).validation_summary["mean_brier"]),
            "improved_seasons": list(best_raw_evaluation(evaluations).improved_seasons),
        },
        "selected_result": None
        if selected_evaluation is None
        else {
            "name": selected_evaluation.spec.name,
            "mean_brier": cast(float, selected_evaluation.validation_summary["mean_brier"]),
            "improved_seasons": list(selected_evaluation.improved_seasons),
            "artifact_dir": str(selected_evaluation.artifact_dir),
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    benchmark_state = family_driver.build_base_benchmark_state()
    evaluations = [
        family_driver.evaluate_family(
            family_spec=family_spec,
            benchmark_state=benchmark_state,
        )
        for family_spec in FAMILY_SPECS
    ]
    selected_evaluation = best_accepted_evaluation(evaluations)
    write_search_summary(
        benchmark_state=benchmark_state,
        evaluations=evaluations,
        selected_evaluation=selected_evaluation,
    )

    if selected_evaluation is None:
        best_raw = best_raw_evaluation(evaluations)
        print("No accepted neutral-site configuration found.")
        print(f"Best raw mean official-score Brier: {cast(float, best_raw.validation_summary['mean_brier']):.6f}")
        print(f"Best raw config: {best_raw.spec.name}")
        print(f"Improved seasons: {list(best_raw.improved_seasons)}")
        return

    print("Accepted neutral-site configuration found.")
    print(f"Best mean official-score Brier: {cast(float, selected_evaluation.validation_summary['mean_brier']):.6f}")
    print(f"Selected config: {selected_evaluation.spec.name}")
    print(f"Improved seasons: {list(selected_evaluation.improved_seasons)}")


if __name__ == "__main__":
    main()
