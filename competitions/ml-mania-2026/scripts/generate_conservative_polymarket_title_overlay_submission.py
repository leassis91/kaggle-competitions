from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import orjson
import polars as pl

import analyze_polymarket_title_bias as title_bias


BASE_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_quintile_blend_2026-03-17.csv"
)
OUTPUT_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_quintile_blend_title_conservative_2026-03-17.csv"
)
ARTIFACT_DIR = Path("artifacts/polymarket_title_overlay_conservative_2026-03-17")
ADJUSTED_ROWS_PATH = ARTIFACT_DIR / "adjusted_rows.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"

FIXED_ALPHA = 0.10
TEAM_OFFSET_CAP = 0.10
PER_GAME_MOVE_CAP = 0.03


def clip_value(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(lower_bound, min(upper_bound, value))


def build_team_offsets(
    title_rows: list[dict[str, Any]],
    model_title_probabilities: dict[int, float],
) -> dict[int, float]:
    offsets: dict[int, float] = {}
    for row in title_rows:
        team_id = int(row["TeamID"])
        raw_gap = title_bias.logit(float(row["market_probability_fair"]), title_bias.TITLE_PROBABILITY_CLIP) - title_bias.logit(
            float(model_title_probabilities[team_id]),
            title_bias.TITLE_PROBABILITY_CLIP,
        )
        capped_gap = clip_value(raw_gap, -title_bias.MAX_ABS_TITLE_GAP, title_bias.MAX_ABS_TITLE_GAP)
        offsets[team_id] = clip_value(FIXED_ALPHA * capped_gap, -TEAM_OFFSET_CAP, TEAM_OFFSET_CAP)
    return offsets


def build_adjusted_rows_frame(
    team_names: dict[int, str],
    team_id_to_seed: dict[int, str],
    pairwise_probabilities: dict[tuple[int, int], float],
    offsets: dict[int, float],
    slot_children: dict[str, tuple[str, str]],
) -> pl.DataFrame:
    pair_rows = title_bias.build_field_pair_rows(
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
        slot_children=slot_children,
    )

    adjusted_rows = []
    for row in pair_rows:
        base_probability = float(row["base_probability_lower_team"])
        adjusted_probability_raw = float(row["adjusted_probability_lower_team"])
        delta_probability_raw = adjusted_probability_raw - base_probability
        delta_probability_capped = clip_value(delta_probability_raw, -PER_GAME_MOVE_CAP, PER_GAME_MOVE_CAP)
        adjusted_probability_capped = base_probability + delta_probability_capped
        adjusted_rows.append(
            {
                **row,
                "ID": f"{title_bias.SEASON}_{int(row['LowerTeamID'])}_{int(row['HigherTeamID'])}",
                "adjusted_probability_lower_team_raw": adjusted_probability_raw,
                "delta_probability_lower_team_raw": delta_probability_raw,
                "adjusted_probability_lower_team_capped": adjusted_probability_capped,
                "delta_probability_lower_team_capped": delta_probability_capped,
            }
        )

    return pl.DataFrame(adjusted_rows)


def build_final_submission(base_submission: pl.DataFrame, adjusted_rows: pl.DataFrame) -> pl.DataFrame:
    return (
        base_submission.join(
            adjusted_rows.select(
                pl.col("ID"),
                pl.col("adjusted_probability_lower_team_capped").alias("overlay_pred"),
            ),
            on="ID",
            how="left",
        )
        .with_columns(pl.coalesce(["overlay_pred", "Pred"]).alias("Pred"))
        .select("ID", "Pred")
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    team_names = title_bias.load_team_names()
    seed_to_team_id, team_id_to_seed, seeded_team_ids = title_bias.load_seed_data()
    seeded_spelling_rows = title_bias.build_seeded_spelling_rows(seeded_team_ids)
    title_rows = title_bias.fetch_title_market_rows(
        seeded_spelling_rows=seeded_spelling_rows,
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
    )
    slot_children, root_slot = title_bias.load_slot_children()
    pairwise_probabilities = title_bias.load_pairwise_probabilities(BASE_SUBMISSION_PATH)
    model_title_probabilities = title_bias.compute_title_probabilities(
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
    )
    offsets = build_team_offsets(
        title_rows=title_rows,
        model_title_probabilities=model_title_probabilities,
    )
    adjusted_title_probabilities = title_bias.compute_title_probabilities(
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
    )

    adjusted_rows = build_adjusted_rows_frame(
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
        slot_children=slot_children,
    )
    adjusted_rows.write_csv(ADJUSTED_ROWS_PATH)

    base_submission = pl.read_csv(BASE_SUBMISSION_PATH)
    final_submission = build_final_submission(base_submission=base_submission, adjusted_rows=adjusted_rows)
    final_submission.write_csv(OUTPUT_SUBMISSION_PATH)

    if final_submission.height != base_submission.height:
        raise ValueError("Final submission row count changed.")
    if final_submission.get_column("ID").n_unique() != final_submission.height:
        raise ValueError("Final submission contains duplicate IDs.")
    if final_submission.get_column("Pred").null_count() != 0:
        raise ValueError("Final submission contains null predictions.")
    if not bool(final_submission.get_column("Pred").is_finite().all()):
        raise ValueError("Final submission contains non-finite predictions.")
    if final_submission.get_column("ID").to_list() != base_submission.get_column("ID").to_list():
        raise ValueError("Final submission IDs do not match the base submission exactly.")

    team_summary_rows = []
    for row in title_rows:
        team_id = int(row["TeamID"])
        team_summary_rows.append(
            {
                "TeamID": team_id,
                "TeamName": row["TeamName"],
                "Seed": row["Seed"],
                "market_probability_fair": float(row["market_probability_fair"]),
                "model_title_probability": float(model_title_probabilities[team_id]),
                "adjusted_title_probability": float(adjusted_title_probabilities[team_id]),
                "team_offset": float(offsets[team_id]),
            }
        )

    mean_abs_delta = float(adjusted_rows.select(pl.col("delta_probability_lower_team_capped").abs().mean()).item())
    max_abs_delta = float(adjusted_rows.select(pl.col("delta_probability_lower_team_capped").abs().max()).item())
    capped_move_count = int(
        adjusted_rows.select(
            (pl.col("delta_probability_lower_team_raw") != pl.col("delta_probability_lower_team_capped"))
            .sum()
            .cast(pl.Int64)
        ).item()
    )

    summary = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_submission_path": str(BASE_SUBMISSION_PATH),
        "output_submission_path": str(OUTPUT_SUBMISSION_PATH),
        "adjusted_rows_path": str(ADJUSTED_ROWS_PATH),
        "season": title_bias.SEASON,
        "overlay_parameters": {
            "fixed_alpha": FIXED_ALPHA,
            "team_offset_cap": TEAM_OFFSET_CAP,
            "per_game_move_cap": PER_GAME_MOVE_CAP,
            "title_gap_cap": title_bias.MAX_ABS_TITLE_GAP,
            "applies_to": "men seeded-vs-seeded tournament field rows only",
        },
        "row_counts": {
            "base_submission": base_submission.height,
            "adjusted_seeded_field_rows": adjusted_rows.height,
        },
        "prediction_summary": {
            "base_mean": float(base_submission.select(pl.col("Pred").mean()).item()),
            "final_mean": float(final_submission.select(pl.col("Pred").mean()).item()),
            "mean_abs_delta_on_adjusted_rows": mean_abs_delta,
            "max_abs_delta_on_adjusted_rows": max_abs_delta,
            "capped_move_count": capped_move_count,
        },
        "top_positive_team_offsets": sorted(team_summary_rows, key=lambda row: row["team_offset"], reverse=True)[:10],
        "top_negative_team_offsets": sorted(team_summary_rows, key=lambda row: row["team_offset"])[:10],
        "largest_pair_moves": adjusted_rows.sort(
            pl.col("delta_probability_lower_team_capped").abs(),
            descending=True,
        )
        .head(20)
        .to_dicts(),
    }
    SUMMARY_PATH.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    print(f"Wrote conservative title-overlay submission: {OUTPUT_SUBMISSION_PATH}")
    print(f"Wrote adjusted rows audit: {ADJUSTED_ROWS_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(f"Adjusted rows: {adjusted_rows.height}")
    print(f"Mean absolute delta on adjusted rows: {mean_abs_delta:.6f}")
    print(f"Max absolute delta on adjusted rows: {max_abs_delta:.6f}")
    print(f"Capped move count: {capped_move_count}")


if __name__ == "__main__":
    main()
