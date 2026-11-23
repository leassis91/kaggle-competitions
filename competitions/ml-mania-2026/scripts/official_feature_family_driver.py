from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import orjson
import polars as pl

import train_iteration_38_vilnius_plus_capped_quality as iteration_38
import train_iteration_49_seed_closure_quality as iteration_49
import train_iteration_76_iteration_73_division_specific_probability_blend_search as division_specific_search
import train_iteration_110_iteration_109_official_men_post_threshold_women_tail_search as official_router
import train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend as official_benchmark


BASE_BENCHMARK_ARTIFACT_DIR = Path("artifacts/iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/official_feature_family_driver")

MatchupBuilder = Callable[[pl.DataFrame, list[str]], tuple[pl.DataFrame, list[str]]]


@dataclass(frozen=True)
class MatchupBundle:
    matchups: pl.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class OfficialMatchupTables:
    benchmark_benchmark_bundle: MatchupBundle
    benchmark_alternative_bundle: MatchupBundle
    pooled_bundle: MatchupBundle
    alternative_bundle: MatchupBundle


@dataclass(frozen=True)
class FamilySpec:
    name: str
    artifact_slug: str
    hypothesis: str
    extend_iteration_38: MatchupBuilder
    extend_iteration_49: MatchupBuilder | None = None


@dataclass(frozen=True)
class BenchmarkState:
    name: str
    mean_brier: float
    fold_scores: dict[int, float]
    iteration_38_bundle: MatchupBundle
    iteration_49_bundle: MatchupBundle
    active_family_names: tuple[str, ...]


@dataclass(frozen=True)
class FamilyEvaluation:
    spec: FamilySpec
    validation_summary: dict[str, object]
    accepted: bool
    improved_seasons: tuple[int, ...]
    candidate_tables: OfficialMatchupTables
    iteration_38_added_columns: tuple[str, ...]
    iteration_49_added_columns: tuple[str, ...]
    next_benchmark_state: BenchmarkState | None
    artifact_dir: Path


def load_base_benchmark_summary() -> dict[str, object]:
    return cast(
        dict[str, object],
        orjson.loads((BASE_BENCHMARK_ARTIFACT_DIR / "validation_summary.json").read_bytes()),
    )


def base_benchmark_fold_scores() -> dict[int, float]:
    validation = cast(dict[str, object], load_base_benchmark_summary()["validation"])
    folds = cast(list[dict[str, float | int]], validation["folds"])
    return {int(fold["season"]): float(fold["combined_brier"]) for fold in folds}


def base_benchmark_mean_brier() -> float:
    validation = cast(dict[str, object], load_base_benchmark_summary()["validation"])
    return cast(float, validation["mean_brier"])


def deduplicated_feature_columns(feature_columns: list[str]) -> list[str]:
    return list(dict.fromkeys(feature_columns))


def build_base_benchmark_state() -> BenchmarkState:
    iteration_38_matchups, iteration_38_feature_columns = iteration_38.build_pooled_margin_matchups()
    iteration_49_matchups, iteration_49_feature_columns = iteration_49.build_pooled_margin_matchups()
    return BenchmarkState(
        name="iteration_111_official_benchmark",
        mean_brier=base_benchmark_mean_brier(),
        fold_scores=base_benchmark_fold_scores(),
        iteration_38_bundle=MatchupBundle(
            matchups=iteration_38_matchups,
            feature_columns=iteration_38_feature_columns,
        ),
        iteration_49_bundle=MatchupBundle(
            matchups=iteration_49_matchups,
            feature_columns=iteration_49_feature_columns,
        ),
        active_family_names=(),
    )


def build_official_matchup_tables(
    iteration_38_bundle: MatchupBundle,
    iteration_49_bundle: MatchupBundle,
) -> OfficialMatchupTables:
    return OfficialMatchupTables(
        benchmark_benchmark_bundle=iteration_38_bundle,
        benchmark_alternative_bundle=iteration_49_bundle,
        pooled_bundle=iteration_38_bundle,
        alternative_bundle=iteration_49_bundle,
    )


def extend_iteration_38_bundle(
    benchmark_state: BenchmarkState,
    family_spec: FamilySpec,
) -> tuple[MatchupBundle, tuple[str, ...]]:
    family_matchups, family_feature_columns = family_spec.extend_iteration_38(
        benchmark_state.iteration_38_bundle.matchups,
        benchmark_state.iteration_38_bundle.feature_columns,
    )
    added_feature_columns = tuple(
        column_name
        for column_name in deduplicated_feature_columns(family_feature_columns)
        if column_name not in benchmark_state.iteration_38_bundle.feature_columns
    )
    return (
        MatchupBundle(
            matchups=family_matchups,
            feature_columns=deduplicated_feature_columns(
                benchmark_state.iteration_38_bundle.feature_columns + list(added_feature_columns)
            ),
        ),
        added_feature_columns,
    )


def extend_iteration_49_bundle(
    benchmark_state: BenchmarkState,
    family_spec: FamilySpec,
) -> tuple[MatchupBundle, tuple[str, ...]]:
    iteration_49_builder = family_spec.extend_iteration_49 or family_spec.extend_iteration_38
    family_matchups, family_feature_columns = iteration_49_builder(
        benchmark_state.iteration_49_bundle.matchups,
        benchmark_state.iteration_49_bundle.feature_columns,
    )
    added_feature_columns = tuple(
        column_name
        for column_name in deduplicated_feature_columns(family_feature_columns)
        if column_name not in benchmark_state.iteration_49_bundle.feature_columns
    )
    return (
        MatchupBundle(
            matchups=family_matchups,
            feature_columns=deduplicated_feature_columns(
                benchmark_state.iteration_49_bundle.feature_columns + list(added_feature_columns)
            ),
        ),
        added_feature_columns,
    )


def build_family_matchup_tables(
    benchmark_state: BenchmarkState,
    family_spec: FamilySpec,
) -> tuple[OfficialMatchupTables, tuple[str, ...], tuple[str, ...]]:
    iteration_38_bundle, iteration_38_added_columns = extend_iteration_38_bundle(benchmark_state, family_spec)
    iteration_49_bundle, iteration_49_added_columns = extend_iteration_49_bundle(benchmark_state, family_spec)
    return (
        build_official_matchup_tables(
            iteration_38_bundle=iteration_38_bundle,
            iteration_49_bundle=iteration_49_bundle,
        ),
        iteration_38_added_columns,
        iteration_49_added_columns,
    )


def build_cached_fold_inputs(
    matchup_tables: OfficialMatchupTables,
) -> list[dict[str, object]]:
    return division_specific_search.build_cached_fold_inputs(
        benchmark_benchmark_matchups=matchup_tables.benchmark_benchmark_bundle.matchups,
        benchmark_benchmark_feature_columns=matchup_tables.benchmark_benchmark_bundle.feature_columns,
        benchmark_alternative_matchups=matchup_tables.benchmark_alternative_bundle.matchups,
        benchmark_alternative_feature_columns=matchup_tables.benchmark_alternative_bundle.feature_columns,
        pooled_matchups=matchup_tables.pooled_bundle.matchups,
        pooled_feature_columns=matchup_tables.pooled_bundle.feature_columns,
        alternative_matchups=matchup_tables.alternative_bundle.matchups,
        alternative_feature_columns=matchup_tables.alternative_bundle.feature_columns,
    )


def improved_seasons(
    validation_summary: dict[str, object],
    benchmark_state: BenchmarkState,
) -> tuple[int, ...]:
    folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    return tuple(
        int(fold["season"])
        for fold in folds
        if float(fold["combined_brier"]) < benchmark_state.fold_scores[int(fold["season"])]
    )


def is_accepted_candidate(
    validation_summary: dict[str, object],
    benchmark_state: BenchmarkState,
) -> tuple[bool, tuple[int, ...]]:
    candidate_mean_brier = cast(float, validation_summary["mean_brier"])
    accepted_improved_seasons = improved_seasons(validation_summary, benchmark_state)
    return (
        candidate_mean_brier < benchmark_state.mean_brier and len(accepted_improved_seasons) >= 4,
        accepted_improved_seasons,
    )


def updated_benchmark_state(
    benchmark_state: BenchmarkState,
    family_spec: FamilySpec,
    validation_summary: dict[str, object],
    iteration_38_bundle: MatchupBundle,
    iteration_49_bundle: MatchupBundle,
) -> BenchmarkState:
    folds = cast(list[dict[str, float | int]], validation_summary["folds"])
    return BenchmarkState(
        name=family_spec.name,
        mean_brier=cast(float, validation_summary["mean_brier"]),
        fold_scores={int(fold["season"]): float(fold["combined_brier"]) for fold in folds},
        iteration_38_bundle=iteration_38_bundle,
        iteration_49_bundle=iteration_49_bundle,
        active_family_names=benchmark_state.active_family_names + (family_spec.name,),
    )


def write_family_artifact(
    artifact_dir: Path,
    benchmark_state: BenchmarkState,
    evaluation: FamilyEvaluation,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_family": "LightGBM",
        "feature_family": evaluation.spec.name,
        "hypothesis": evaluation.spec.hypothesis,
        "validation_scope": "official_stage2_excluding_first_four",
        "information_cutoff": "Selection Sunday",
        "probability_mapping": "fixed_iteration_111_official_router",
        "benchmark_name": benchmark_state.name,
        "benchmark_mean_brier": benchmark_state.mean_brier,
        "benchmark_fold_scores": {str(season): score for season, score in benchmark_state.fold_scores.items()},
        "acceptance_rule": {
            "required_lower_mean_brier": True,
            "required_improved_fold_count": 4,
            "validation_fold_count": 5,
        },
        "active_family_names_before_run": list(benchmark_state.active_family_names),
        "iteration_38_added_columns": list(evaluation.iteration_38_added_columns),
        "iteration_49_added_columns": list(evaluation.iteration_49_added_columns),
        "iteration_38_feature_columns": evaluation.candidate_tables.pooled_bundle.feature_columns,
        "iteration_49_feature_columns": evaluation.candidate_tables.alternative_bundle.feature_columns,
        "benchmark_benchmark_feature_columns": evaluation.candidate_tables.benchmark_benchmark_bundle.feature_columns,
        "benchmark_alternative_feature_columns": evaluation.candidate_tables.benchmark_alternative_bundle.feature_columns,
        "pooled_feature_columns": evaluation.candidate_tables.pooled_bundle.feature_columns,
        "alternative_feature_columns": evaluation.candidate_tables.alternative_bundle.feature_columns,
        "accepted": evaluation.accepted,
        "improved_seasons": list(evaluation.improved_seasons),
        "active_family_names_after_run": list(
            evaluation.next_benchmark_state.active_family_names
            if evaluation.next_benchmark_state is not None
            else benchmark_state.active_family_names
        ),
        "next_benchmark_name": evaluation.next_benchmark_state.name if evaluation.next_benchmark_state is not None else None,
        "validation": evaluation.validation_summary,
        "official_router_config": official_benchmark.CONFIG,
    }
    (artifact_dir / "validation_summary.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def evaluate_family(
    family_spec: FamilySpec,
    benchmark_state: BenchmarkState,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> FamilyEvaluation:
    candidate_tables, iteration_38_added_columns, iteration_49_added_columns = build_family_matchup_tables(
        benchmark_state=benchmark_state,
        family_spec=family_spec,
    )
    cached_fold_inputs = build_cached_fold_inputs(candidate_tables)
    validation_summary = official_router.run_validation(
        cached_fold_inputs=cached_fold_inputs,
        config=official_benchmark.CONFIG,
    )
    accepted, accepted_improved_seasons = is_accepted_candidate(validation_summary, benchmark_state)
    next_benchmark_state = (
        updated_benchmark_state(
            benchmark_state=benchmark_state,
            family_spec=family_spec,
            validation_summary=validation_summary,
            iteration_38_bundle=candidate_tables.pooled_bundle,
            iteration_49_bundle=candidate_tables.alternative_bundle,
        )
        if accepted
        else None
    )
    evaluation = FamilyEvaluation(
        spec=family_spec,
        validation_summary=validation_summary,
        accepted=accepted,
        improved_seasons=accepted_improved_seasons,
        candidate_tables=candidate_tables,
        iteration_38_added_columns=iteration_38_added_columns,
        iteration_49_added_columns=iteration_49_added_columns,
        next_benchmark_state=next_benchmark_state,
        artifact_dir=artifact_root / family_spec.artifact_slug,
    )
    write_family_artifact(evaluation.artifact_dir, benchmark_state, evaluation)
    return evaluation


def run_family_sequence(
    family_specs: list[FamilySpec],
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    initial_benchmark_state: BenchmarkState | None = None,
) -> tuple[list[FamilyEvaluation], BenchmarkState]:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
    benchmark_state = initial_benchmark_state or build_base_benchmark_state()
    evaluations: list[FamilyEvaluation] = []

    for family_spec in family_specs:
        evaluation = evaluate_family(
            family_spec=family_spec,
            benchmark_state=benchmark_state,
            artifact_root=artifact_root,
        )
        evaluations.append(evaluation)
        if evaluation.next_benchmark_state is not None:
            benchmark_state = evaluation.next_benchmark_state

    return evaluations, benchmark_state
