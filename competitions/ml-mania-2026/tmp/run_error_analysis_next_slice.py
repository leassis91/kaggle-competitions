from __future__ import annotations

from pathlib import Path
import sys

import orjson
import polars as pl


REPO_ROOT = Path("/Users/mariofilho/ml/random/ml-mania-2026")
ROWS_PATH = REPO_ROOT / "tmp" / "error-analysis-inner-rows.parquet"


def base_lazy_frame() -> pl.LazyFrame:
    return (
        pl.scan_parquet(ROWS_PATH)
        .filter(pl.col("official_row_flag") == 1)
        .with_columns(
            pl.when(pl.col("iteration_71_probability") > pl.col("iteration_69_probability"))
            .then(pl.lit("iteration_71_higher"))
            .when(pl.col("iteration_71_probability") < pl.col("iteration_69_probability"))
            .then(pl.lit("iteration_69_higher"))
            .otherwise(pl.lit("equal"))
            .alias("disagreement_direction"),
            pl.format(
                "{}_vs_{}",
                pl.min_horizontal("low_seed", "high_seed").cast(pl.Int64),
                pl.max_horizontal("low_seed", "high_seed").cast(pl.Int64),
            ).alias("sorted_seed_pair"),
            pl.when(pl.col("low_seed") < pl.col("high_seed"))
            .then(pl.lit("lower_team_better_seed"))
            .when(pl.col("low_seed") > pl.col("high_seed"))
            .then(pl.lit("lower_team_worse_seed"))
            .otherwise(pl.lit("same_seed"))
            .alias("lower_team_seed_status"),
            pl.when(pl.col("delta_block_feature_late") > 0.0)
            .then(pl.lit("positive"))
            .when(pl.col("delta_block_feature_late") < 0.0)
            .then(pl.lit("negative"))
            .otherwise(pl.lit("zero"))
            .alias("delta_block_feature_late_sign"),
            pl.when(pl.col("opening_slot_gap") >= 3)
            .then(pl.lit("3_plus"))
            .otherwise(pl.col("opening_slot_gap").cast(pl.Int64).cast(pl.Utf8))
            .alias("opening_slot_gap_bucket"),
        )
    )


def summarize(
    lazy_frame: pl.LazyFrame,
    group_columns: list[str],
    *,
    min_rows: int = 1,
    min_outer_seasons: int = 1,
    sort_columns: list[str] | None = None,
    descending: list[bool] | None = None,
) -> list[dict[str, object]]:
    summary = (
        lazy_frame.group_by(group_columns)
        .agg(
            pl.len().alias("row_count"),
            pl.col("outer_season").n_unique().alias("outer_season_count"),
            pl.col("error_sq").mean().alias("mean_slice_brier"),
            pl.col("iteration_69_error_sq").mean().alias("iteration_69_brier"),
            pl.col("iteration_71_error_sq").mean().alias("iteration_71_brier"),
            pl.col("prediction").mean().alias("mean_prediction"),
            pl.col("target").mean().alias("positive_rate"),
            pl.col("probability_gap").mean().alias("mean_probability_gap"),
            pl.col("residual").mean().alias("mean_residual"),
            pl.when(pl.col("low_seed") != pl.col("high_seed"))
            .then(
                pl.when((pl.col("low_seed") > pl.col("high_seed")) & (pl.col("target") == 1))
                .then(pl.lit(1.0))
                .when((pl.col("high_seed") > pl.col("low_seed")) & (pl.col("target") == 0))
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
            )
            .otherwise(None)
            .mean()
            .alias("seed_upset_rate"),
        )
        .with_columns(
            (pl.col("mean_slice_brier") - pl.col("iteration_69_brier")).alias("router_regret_vs_69"),
            (pl.col("mean_slice_brier") - pl.col("iteration_71_brier")).alias("router_regret_vs_71"),
        )
        .filter(pl.col("row_count") >= min_rows)
        .filter(pl.col("outer_season_count") >= min_outer_seasons)
    )
    if sort_columns is not None:
        summary = summary.sort(sort_columns, descending=descending or False)
    return pl.DataFrame(summary.collect()).to_dicts()


def slice_one() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("router_regime") == "men_low"))
    return {
        "slice_key": "men_low_x_direction_x_gap_bucket",
        "groups": summarize(
            frame,
            ["disagreement_direction", "probability_gap_bucket"],
            min_rows=8,
            min_outer_seasons=3,
            sort_columns=["router_regret_vs_71", "mean_slice_brier"],
            descending=[True, True],
        ),
    }


def slice_two() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("router_regime") == "men_low"))
    return {
        "slice_key": "men_low_x_exact_seed_pair",
        "groups": summarize(
            frame,
            ["sorted_seed_pair"],
            min_rows=4,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
    }


def slice_three() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("router_regime") == "men_low"))
    return {
        "slice_key": "men_low_x_round_x_seed_gap_bucket",
        "groups": summarize(
            frame,
            ["possible_round", "abs_seed_gap_bucket"],
            min_rows=6,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
    }


def slice_four() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("confidence_decile").is_between(3, 6)))
    return {
        "slice_key": "men_mid_confidence_x_router_regime",
        "groups": summarize(
            frame,
            ["confidence_decile", "router_regime"],
            min_rows=8,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
    }


def slice_five() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("router_regime") == "men_low"))
    return {
        "slice_key": "men_low_x_block_late_x_seed_gap",
        "sign_groups": summarize(
            frame,
            ["delta_block_feature_late_sign", "abs_seed_gap_bucket"],
            min_rows=6,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
        "quantile_groups": summarize(
            frame,
            ["delta_block_feature_late_bucket", "abs_seed_gap_bucket"],
            min_rows=6,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
    }


def slice_six() -> dict[str, object]:
    frame = base_lazy_frame().filter((pl.col("men_women") == 1) & (pl.col("router_regime") == "men_low"))
    return {
        "slice_key": "men_low_x_same_region_x_opening_slot_gap",
        "groups": summarize(
            frame,
            ["same_region_flag", "opening_slot_gap_bucket"],
            min_rows=6,
            min_outer_seasons=3,
            sort_columns=["mean_slice_brier", "router_regret_vs_71"],
            descending=[True, True],
        ),
    }


SLICE_FUNCTIONS = {
    "1": slice_one,
    "2": slice_two,
    "3": slice_three,
    "4": slice_four,
    "5": slice_five,
    "6": slice_six,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SLICE_FUNCTIONS:
        raise SystemExit("Usage: python tmp/run_error_analysis_next_slice.py <1|2|3|4|5|6>")
    result = SLICE_FUNCTIONS[sys.argv[1]]()
    print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
