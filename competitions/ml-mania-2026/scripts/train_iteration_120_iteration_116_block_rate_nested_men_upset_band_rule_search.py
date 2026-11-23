from __future__ import annotations

import datetime
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import orjson
import polars as pl

import official_current_block_rate_benchmark as active_benchmark
import official_feature_family_driver as family_driver
import train_iteration_16_pooled_selection_sunday_baseline as pooled_base
import train_iteration_72_iteration_69_iteration_71_probability_blend_search as search
import train_iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search as official_search
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as official_router
import train_iteration_116_pending_feature_family_sweep as pending_feature_family_sweep


ARTIFACT_DIR = Path("artifacts/iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search")
CACHE_PATH = ARTIFACT_DIR / "all_season_cached_fold_inputs.joblib"
OUTER_VALIDATION_SEASONS = tuple(pooled_base.latest_completed_validation_seasons())
ALL_VALIDATION_SEASONS = tuple(season for season in range(2011, 2026) if season != 2020)
MIN_ACTIVATED_ROWS = 30
MIN_ACTIVATED_SEASONS = 3


@dataclass(frozen=True)
class CandidateRule:
    name: str
    direction_filter: str
    gap_threshold: float | None
    round_scope: str
    seed_band: str
    iteration_71_weight: float | None

    @property
    def complexity(self) -> int:
        if self.name == "no_rule":
            return 0
        return 1 + int(self.direction_filter != "none") + int(self.round_scope != "lte_3") + int(self.seed_band != "none")


def active_matchup_tables() -> family_driver.OfficialMatchupTables:
    benchmark_state = family_driver.build_base_benchmark_state()
    block_rate_spec = next(
        spec
        for spec in pending_feature_family_sweep.family_specifications()
        if spec.name == active_benchmark.FAMILY_NAME
    )
    matchup_tables, _, _ = family_driver.build_family_matchup_tables(
        benchmark_state=benchmark_state,
        family_spec=block_rate_spec,
    )
    return matchup_tables


def build_all_season_cached_fold_inputs() -> list[dict[str, object]]:
    if CACHE_PATH.exists():
        return cast(list[dict[str, object]], joblib.load(CACHE_PATH))

    matchup_tables = active_matchup_tables()
    cached_fold_inputs: list[dict[str, object]] = []
    for validation_season in ALL_VALIDATION_SEASONS:
        source_predictions, targets, validation_rows = search.raw_margin_predictions(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            validation_season=validation_season,
        )
        iteration_69_calibrator = search.fit_iteration_69_calibrator(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            target_season=validation_season,
        )
        iteration_71_calibrator = search.fit_iteration_71_calibrator(
            benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
            benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
            benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
            benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
            pooled_matchups=matchup_tables.pooled_bundle.matchups,
            pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
            alternative_matchups=matchup_tables.alternative_bundle.matchups,
            alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
            target_season=validation_season,
        )
        iteration_69_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_69_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_69_calibrator,
            calibration_mode="iteration_69",
        )
        iteration_71_probabilities = search.calibrated_probabilities(
            raw_predictions=search.iteration_71_raw_margin(source_predictions),
            validation_rows=validation_rows,
            calibrator=iteration_71_calibrator,
            calibration_mode="iteration_71",
        )
        cached_fold_inputs.append(
            {
                "season": validation_season,
                "targets": targets,
                "validation_rows": validation_rows,
                "iteration_69_probabilities": iteration_69_probabilities,
                "iteration_71_probabilities": iteration_71_probabilities,
            }
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cached_fold_inputs, CACHE_PATH)
    return cached_fold_inputs


def candidate_rules() -> list[CandidateRule]:
    rules = [CandidateRule("no_rule", "none", None, "none", "none", None)]
    for direction_filter in ["none", "iteration_71_higher"]:
        for gap_threshold in [0.015, 0.020, 0.025]:
            for round_scope in ["eq_2", "lte_2", "lte_3"]:
                for seed_band in [
                    "none",
                    "upset_core",
                    "top_mid_upset_core",
                    "near_peer_upset_core",
                    "seed_gap_1",
                    "seed_gap_3_4",
                ]:
                    for iteration_71_weight in [0.60, 0.70, 0.75, 0.80]:
                        rules.append(
                            CandidateRule(
                                name="men_directional_upset_band_override",
                                direction_filter=direction_filter,
                                gap_threshold=gap_threshold,
                                round_scope=round_scope,
                                seed_band=seed_band,
                                iteration_71_weight=iteration_71_weight,
                            )
                        )
    return rules


def sorted_seed_pairs(validation_rows: pl.DataFrame) -> np.ndarray:
    low_seed = validation_rows["low_seed"].to_numpy().astype(np.int64)
    high_seed = validation_rows["high_seed"].to_numpy().astype(np.int64)
    min_seed = np.minimum(low_seed, high_seed)
    max_seed = np.maximum(low_seed, high_seed)
    return np.array([f"{low}_vs_{high}" for low, high in zip(min_seed, max_seed, strict=True)], dtype=object)


def candidate_activation_mask(
    validation_rows: pl.DataFrame,
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
    candidate_rule: CandidateRule,
) -> np.ndarray:
    if candidate_rule.name == "no_rule":
        return np.zeros(validation_rows.height, dtype=bool)

    official_mask = official_search.official_row_mask(validation_rows)
    men_rows_mask = validation_rows["men_women"].to_numpy() == 1
    probability_gap = np.abs(iteration_71_probabilities - iteration_69_probabilities)
    mask = official_mask & men_rows_mask & (probability_gap <= float(official_router.DEFAULT_CONFIG["men_low_gap_threshold"]))

    gap_threshold = cast(float, candidate_rule.gap_threshold)
    mask &= probability_gap <= gap_threshold

    if candidate_rule.direction_filter == "iteration_71_higher":
        mask &= iteration_71_probabilities > iteration_69_probabilities

    possible_round = validation_rows["possible_round"].to_numpy()
    if candidate_rule.round_scope == "eq_2":
        mask &= possible_round == 2
    elif candidate_rule.round_scope == "lte_2":
        mask &= possible_round <= 2
    elif candidate_rule.round_scope == "lte_3":
        mask &= possible_round <= 3

    sorted_seed_pair = sorted_seed_pairs(validation_rows)
    abs_seed_gap = np.abs(validation_rows["delta_seed_number"].to_numpy()).astype(np.int64)
    if candidate_rule.seed_band == "upset_core":
        mask &= np.isin(sorted_seed_pair, ["5_vs_12", "6_vs_11", "7_vs_10", "8_vs_9"])
    elif candidate_rule.seed_band == "top_mid_upset_core":
        mask &= np.isin(sorted_seed_pair, ["5_vs_12", "6_vs_11"])
    elif candidate_rule.seed_band == "near_peer_upset_core":
        mask &= np.isin(sorted_seed_pair, ["7_vs_10", "8_vs_9"])
    elif candidate_rule.seed_band == "seed_gap_1":
        mask &= abs_seed_gap == 1
    elif candidate_rule.seed_band == "seed_gap_3_4":
        mask &= np.isin(abs_seed_gap, [3, 4])

    return mask


def apply_candidate_rule(
    validation_rows: pl.DataFrame,
    iteration_69_probabilities: np.ndarray,
    iteration_71_probabilities: np.ndarray,
    candidate_rule: CandidateRule,
) -> tuple[np.ndarray, np.ndarray]:
    benchmark_probabilities = active_benchmark.benchmark_probabilities(
        validation_rows=validation_rows,
        iteration_69_probabilities=iteration_69_probabilities,
        iteration_71_probabilities=iteration_71_probabilities,
    )
    activation_mask = candidate_activation_mask(
        validation_rows=validation_rows,
        iteration_69_probabilities=iteration_69_probabilities,
        iteration_71_probabilities=iteration_71_probabilities,
        candidate_rule=candidate_rule,
    )
    if candidate_rule.name == "no_rule":
        return benchmark_probabilities, activation_mask

    iteration_71_weight = cast(float, candidate_rule.iteration_71_weight)
    iteration_69_weight = 1.0 - iteration_71_weight
    probabilities = benchmark_probabilities.copy()
    probabilities[activation_mask] = (
        iteration_69_weight * iteration_69_probabilities[activation_mask]
        + iteration_71_weight * iteration_71_probabilities[activation_mask]
    )
    return probabilities, activation_mask


def evaluate_candidate_on_folds(
    fold_inputs: list[dict[str, object]],
    candidate_rule: CandidateRule,
) -> dict[str, object]:
    fold_summaries: list[dict[str, object]] = []
    activated_row_count = 0
    activated_seasons: list[int] = []
    activated_targets: list[np.ndarray] = []

    for fold_input in fold_inputs:
        season = cast(int, fold_input["season"])
        targets = cast(np.ndarray, fold_input["targets"])
        validation_rows = cast(pl.DataFrame, fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, fold_input["iteration_71_probabilities"])
        official_mask = official_search.official_row_mask(validation_rows)
        men_mask = (validation_rows["men_women"].to_numpy() == 1) & official_mask
        women_mask = (~(validation_rows["men_women"].to_numpy() == 1)) & official_mask

        probabilities, activation_mask = apply_candidate_rule(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
            candidate_rule=candidate_rule,
        )
        activated_official_mask = activation_mask & official_mask
        activated_count = int(activated_official_mask.sum())
        activated_row_count += activated_count
        if activated_count > 0:
            activated_seasons.append(season)
            activated_targets.append(targets[activated_official_mask])

        fold_summaries.append(
            {
                "season": season,
                "combined_brier": pooled_base.brier_score(probabilities[official_mask], targets[official_mask]),
                "men_brier": pooled_base.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": pooled_base.brier_score(probabilities[women_mask], targets[women_mask]),
                "activated_row_count": activated_count,
            }
        )

    if activated_targets:
        unique_activated_classes = int(np.unique(np.concatenate(activated_targets)).size)
    else:
        unique_activated_classes = 0

    return {
        "candidate": asdict(candidate_rule),
        "candidate_complexity": candidate_rule.complexity,
        "mean_brier": float(np.mean([fold_combined_brier(fold) for fold in fold_summaries])),
        "folds": fold_summaries,
        "activated_row_count": activated_row_count,
        "activated_season_count": len(set(activated_seasons)),
        "unique_activated_classes": unique_activated_classes,
        "eligible": candidate_rule.name == "no_rule"
        or (
            activated_row_count >= MIN_ACTIVATED_ROWS
            and len(set(activated_seasons)) >= MIN_ACTIVATED_SEASONS
            and unique_activated_classes >= 2
        ),
    }


def candidate_payload(evaluation: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], evaluation["candidate"])


def candidate_rule_from_payload(payload: dict[str, object]) -> CandidateRule:
    return CandidateRule(
        name=cast(str, payload["name"]),
        direction_filter=cast(str, payload["direction_filter"]),
        gap_threshold=cast(float | None, payload["gap_threshold"]),
        round_scope=cast(str, payload["round_scope"]),
        seed_band=cast(str, payload["seed_band"]),
        iteration_71_weight=cast(float | None, payload["iteration_71_weight"]),
    )


def fold_season(fold_summary: dict[str, object]) -> int:
    return cast(int, fold_summary["season"])


def fold_combined_brier(fold_summary: dict[str, object]) -> float:
    return cast(float, fold_summary["combined_brier"])


def evaluation_mean_brier(evaluation: dict[str, object]) -> float:
    return cast(float, evaluation["mean_brier"])


def benchmark_fold_summary_map(fold_inputs: list[dict[str, object]]) -> dict[int, float]:
    summary_map: dict[int, float] = {}
    no_rule = candidate_rules()[0]
    evaluation = evaluate_candidate_on_folds(fold_inputs, no_rule)
    for fold in cast(list[dict[str, object]], evaluation["folds"]):
        summary_map[fold_season(fold)] = fold_combined_brier(fold)
    return summary_map


def select_candidate_rule(
    inner_fold_inputs: list[dict[str, object]],
    rules: list[CandidateRule],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluations = [evaluate_candidate_on_folds(inner_fold_inputs, candidate_rule) for candidate_rule in rules]
    no_rule_evaluation = next(
        evaluation for evaluation in evaluations if cast(str, candidate_payload(evaluation)["name"]) == "no_rule"
    )

    for evaluation in evaluations:
        fold_score_by_season = {
            fold_season(fold): fold_combined_brier(fold)
            for fold in cast(list[dict[str, object]], no_rule_evaluation["folds"])
        }
        improved_seasons = [
            fold_season(fold)
            for fold in cast(list[dict[str, object]], evaluation["folds"])
            if fold_combined_brier(fold) < fold_score_by_season[fold_season(fold)]
        ]
        evaluation["improved_fold_count_vs_no_rule"] = len(improved_seasons)
        evaluation["improved_seasons_vs_no_rule"] = improved_seasons

    eligible_evaluations = [evaluation for evaluation in evaluations if bool(evaluation["eligible"])]
    best_evaluation = min(
        eligible_evaluations,
        key=lambda evaluation: (
            evaluation_mean_brier(evaluation),
            cast(int, evaluation["candidate_complexity"]),
            -int(evaluation["activated_row_count"]),
        ),
    )
    if evaluation_mean_brier(best_evaluation) >= evaluation_mean_brier(no_rule_evaluation):
        best_evaluation = no_rule_evaluation

    return best_evaluation, evaluations


def current_snapshot_baseline_summary(fold_inputs: list[dict[str, object]]) -> dict[str, object]:
    return evaluate_candidate_on_folds(
        [fold_input for fold_input in fold_inputs if cast(int, fold_input["season"]) in OUTER_VALIDATION_SEASONS],
        candidate_rules()[0],
    )


def run_nested_validation() -> dict[str, object]:
    fold_inputs = build_all_season_cached_fold_inputs()
    rules = candidate_rules()
    current_snapshot_baseline = current_snapshot_baseline_summary(fold_inputs)
    baseline_fold_scores = {
        fold_season(fold): fold_combined_brier(fold)
        for fold in cast(list[dict[str, object]], current_snapshot_baseline["folds"])
    }

    outer_fold_summaries: list[dict[str, object]] = []
    inner_selection_details: list[dict[str, object]] = []
    for outer_season in OUTER_VALIDATION_SEASONS:
        inner_fold_inputs = [fold_input for fold_input in fold_inputs if cast(int, fold_input["season"]) < outer_season]
        outer_fold_input = next(fold_input for fold_input in fold_inputs if cast(int, fold_input["season"]) == outer_season)

        selected_evaluation, inner_evaluations = select_candidate_rule(inner_fold_inputs, rules)
        inner_selection_details.append(
            {
                "outer_season": outer_season,
                "selected_candidate": selected_evaluation["candidate"],
                "selected_candidate_mean_brier": selected_evaluation["mean_brier"],
                "selected_candidate_improved_fold_count_vs_no_rule": selected_evaluation["improved_fold_count_vs_no_rule"],
                "selected_candidate_improved_seasons_vs_no_rule": selected_evaluation["improved_seasons_vs_no_rule"],
                "candidate_results": inner_evaluations,
            }
        )

        targets = cast(np.ndarray, outer_fold_input["targets"])
        validation_rows = cast(pl.DataFrame, outer_fold_input["validation_rows"])
        iteration_69_probabilities = cast(np.ndarray, outer_fold_input["iteration_69_probabilities"])
        iteration_71_probabilities = cast(np.ndarray, outer_fold_input["iteration_71_probabilities"])
        official_mask = official_search.official_row_mask(validation_rows)
        men_mask = (validation_rows["men_women"].to_numpy() == 1) & official_mask
        women_mask = (~(validation_rows["men_women"].to_numpy() == 1)) & official_mask

        selected_candidate = candidate_rule_from_payload(candidate_payload(selected_evaluation))
        probabilities, activation_mask = apply_candidate_rule(
            validation_rows=validation_rows,
            iteration_69_probabilities=iteration_69_probabilities,
            iteration_71_probabilities=iteration_71_probabilities,
            candidate_rule=selected_candidate,
        )
        outer_fold_summaries.append(
            {
                "season": outer_season,
                "combined_brier": pooled_base.brier_score(probabilities[official_mask], targets[official_mask]),
                "men_brier": pooled_base.brier_score(probabilities[men_mask], targets[men_mask]),
                "women_brier": pooled_base.brier_score(probabilities[women_mask], targets[women_mask]),
                "activated_row_count": int((activation_mask & official_mask).sum()),
                "selected_candidate": asdict(selected_candidate),
                "baseline_combined_brier": baseline_fold_scores[outer_season],
                "improved_vs_current_snapshot_no_rule": pooled_base.brier_score(probabilities[official_mask], targets[official_mask])
                < baseline_fold_scores[outer_season],
            }
        )

    mean_brier = float(np.mean([fold_combined_brier(fold) for fold in outer_fold_summaries]))
    improved_seasons = [
        fold_season(fold)
        for fold in outer_fold_summaries
        if bool(fold["improved_vs_current_snapshot_no_rule"])
    ]
    baseline_mean_brier = evaluation_mean_brier(current_snapshot_baseline)

    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": active_benchmark.FAMILY_NAME,
        "information_cutoff": "Selection Sunday",
        "validation_scope": "official_stage2_excluding_first_four",
        "outer_validation_seasons": list(OUTER_VALIDATION_SEASONS),
        "inner_training_validation_seasons": list(ALL_VALIDATION_SEASONS),
        "candidate_rule_count": len(rules),
        "candidate_support_guard": {
            "min_activated_rows": MIN_ACTIVATED_ROWS,
            "min_activated_seasons": MIN_ACTIVATED_SEASONS,
            "required_activated_class_count": 2,
        },
        "data_snapshot_note": {
            "accepted_block_rate_artifact_path": str(active_benchmark.BLOCK_RATE_ARTIFACT_DIR / "validation_summary.json"),
            "accepted_block_rate_artifact_generated_at_utc": "2026-03-16T15:04:00+00:00 (journal entry)",
            "current_data_files_modified_after_that_artifact": True,
            "comparison_baseline": "current_snapshot_rebuilt_block_rate_no_rule",
        },
        "current_snapshot_no_rule_validation": current_snapshot_baseline,
        "nested_rule_validation": {
            "mean_brier": mean_brier,
            "baseline_mean_brier": baseline_mean_brier,
            "improved_fold_count": len(improved_seasons),
            "improved_seasons": improved_seasons,
            "accepted": mean_brier < baseline_mean_brier and len(improved_seasons) >= 4,
            "folds": outer_fold_summaries,
        },
        "outer_selection_details": inner_selection_details,
        "benchmark_feature_columns": active_benchmark.benchmark_feature_columns(),
    }


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_nested_validation()
    (ARTIFACT_DIR / "validation_summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    nested_validation = cast(dict[str, object], summary["nested_rule_validation"])
    print(
        f"Current-snapshot no-rule mean validation Brier: "
        f"{cast(dict[str, object], summary['current_snapshot_no_rule_validation'])['mean_brier']:.6f}"
    )
    print(f"Nested rule-search mean validation Brier: {cast(float, nested_validation['mean_brier']):.6f}")
    print(f"Accepted candidate found: {cast(bool, nested_validation['accepted'])}")
    print(f"Improved seasons: {cast(list[int], nested_validation['improved_seasons'])}")


if __name__ == "__main__":
    main()
