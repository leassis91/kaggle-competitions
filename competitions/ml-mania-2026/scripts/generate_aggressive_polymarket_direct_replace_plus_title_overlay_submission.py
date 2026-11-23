from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import httpx
import orjson
import polars as pl

import analyze_polymarket_title_bias as title_bias
import generate_aggressive_polymarket_direct_replace_submission as direct_overlay
import generate_conservative_polymarket_title_overlay_submission as title_overlay


BASE_SUBMISSION_PATH = direct_overlay.OUTPUT_SUBMISSION_PATH
DIRECT_ADJUSTED_ROWS_PATH = direct_overlay.ADJUSTED_ROWS_PATH
OUTPUT_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k_plus_title_conservative.csv"
)
ARTIFACT_DIR = Path("artifacts/polymarket_direct_replace_volume_gt_10k_plus_title_overlay_conservative")
ADJUSTED_ROWS_PATH = ARTIFACT_DIR / "adjusted_rows.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"


def fetch_current_title_rows_allow_missing(
    seeded_spelling_rows: list[tuple[str, int]],
    team_names: dict[int, str],
    team_id_to_seed: dict[int, str],
    seeded_team_ids: set[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved_rows_by_team_id: dict[int, dict[str, Any]] = {}
    skipped_market_questions: list[str] = []

    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"https://gamma-api.polymarket.com/events?slug={title_bias.TITLE_EVENT_SLUG}")
        response.raise_for_status()
        event_rows = response.json()

    if not event_rows:
        raise ValueError(f"No Polymarket event found for slug {title_bias.TITLE_EVENT_SLUG!r}.")

    event = event_rows[0]
    for market in event["markets"]:
        question = str(market.get("question", ""))
        if not question.startswith("Will ") or not question.endswith(f" win the {title_bias.SEASON} NCAA Tournament?"):
            continue
        if bool(market.get("closed", False)) or not bool(market.get("active", True)):
            continue

        team_name = question.removeprefix("Will ").removesuffix(f" win the {title_bias.SEASON} NCAA Tournament?")
        try:
            team_id = title_bias.resolve_title_market_team_id(team_name, seeded_spelling_rows)
        except ValueError:
            skipped_market_questions.append(question)
            continue

        if team_id not in seeded_team_ids:
            skipped_market_questions.append(question)
            continue

        row = {
            "market_slug": str(market["slug"]),
            "team_name_market": team_name,
            "TeamID": team_id,
            "TeamName": team_names[team_id],
            "Seed": team_id_to_seed[team_id],
            "market_probability_raw": float(title_bias.choose_market_probability(market)),
            "market_probability_outcome_price": float(title_bias.parse_probability_list(market["outcomePrices"])[0]),
            "best_bid": title_bias.parse_optional_float(market.get("bestBid")),
            "best_ask": title_bias.parse_optional_float(market.get("bestAsk")),
            "last_trade_price": title_bias.parse_optional_float(market.get("lastTradePrice")),
            "liquidity": float(market.get("liquidity") or 0.0),
            "volume": float(market.get("volume") or 0.0),
        }

        existing_row = resolved_rows_by_team_id.get(team_id)
        if existing_row is None or float(row["liquidity"]) > float(existing_row["liquidity"]):
            resolved_rows_by_team_id[team_id] = row

    raw_probability_sum = sum(float(row["market_probability_raw"]) for row in resolved_rows_by_team_id.values())
    if raw_probability_sum <= 0.0:
        raise ValueError("Resolved current title market probability sum was non-positive.")

    title_rows: list[dict[str, Any]] = []
    for team_id in sorted(seeded_team_ids):
        resolved_row = resolved_rows_by_team_id.get(team_id)
        if resolved_row is None:
            title_rows.append(
                {
                    "market_slug": None,
                    "team_name_market": team_names[team_id],
                    "TeamID": team_id,
                    "TeamName": team_names[team_id],
                    "Seed": team_id_to_seed[team_id],
                    "market_probability_raw": 0.0,
                    "market_probability_outcome_price": 0.0,
                    "best_bid": None,
                    "best_ask": None,
                    "last_trade_price": None,
                    "liquidity": 0.0,
                    "volume": 0.0,
                    "market_probability_fair": 0.0,
                }
            )
            continue

        title_rows.append(
            {
                **resolved_row,
                "market_probability_fair": float(resolved_row["market_probability_raw"]) / raw_probability_sum,
            }
        )

    return sorted(title_rows, key=lambda row: float(row["market_probability_fair"]), reverse=True), skipped_market_questions


def build_team_summary_rows(
    title_rows: list[dict[str, Any]],
    model_title_probabilities: dict[int, float],
    adjusted_title_probabilities: dict[int, float],
    offsets: dict[int, float],
) -> list[dict[str, Any]]:
    summary_rows = []
    for row in title_rows:
        team_id = int(row["TeamID"])
        summary_rows.append(
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
    return summary_rows


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_submission = pl.read_csv(BASE_SUBMISSION_PATH)
    submission_template = direct_overlay.base_submission_generator.parse_submission_template()
    direct_overlay.validate_submission_frame(base_submission, submission_template)

    direct_adjusted_rows = pl.read_csv(DIRECT_ADJUSTED_ROWS_PATH)
    direct_replaced_ids = set(direct_adjusted_rows.get_column("ID").to_list())

    team_names = title_bias.load_team_names()
    seed_to_team_id, team_id_to_seed, seeded_team_ids = title_bias.load_seed_data()
    seeded_spelling_rows = title_bias.build_seeded_spelling_rows(seeded_team_ids)
    title_rows, skipped_market_questions = fetch_current_title_rows_allow_missing(
        seeded_spelling_rows=seeded_spelling_rows,
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
        seeded_team_ids=seeded_team_ids,
    )

    slot_children, root_slot = title_bias.load_slot_children()
    pairwise_probabilities = title_bias.load_pairwise_probabilities(BASE_SUBMISSION_PATH)
    model_title_probabilities = title_bias.compute_title_probabilities(
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
    )
    offsets = title_overlay.build_team_offsets(
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

    raw_title_adjusted_rows = title_overlay.build_adjusted_rows_frame(
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
        slot_children=slot_children,
    )
    filtered_title_adjusted_rows = raw_title_adjusted_rows.filter(~pl.col("ID").is_in(direct_replaced_ids))
    filtered_title_adjusted_rows.write_csv(ADJUSTED_ROWS_PATH)

    final_submission = title_overlay.build_final_submission(
        base_submission=base_submission,
        adjusted_rows=filtered_title_adjusted_rows,
    )
    direct_overlay.validate_submission_frame(final_submission, submission_template)
    final_submission.write_csv(OUTPUT_SUBMISSION_PATH)

    team_summary_rows = build_team_summary_rows(
        title_rows=title_rows,
        model_title_probabilities=model_title_probabilities,
        adjusted_title_probabilities=adjusted_title_probabilities,
        offsets=offsets,
    )

    mean_abs_delta = (
        float(filtered_title_adjusted_rows.select(pl.col("delta_probability_lower_team_capped").abs().mean()).item())
        if not filtered_title_adjusted_rows.is_empty()
        else 0.0
    )
    max_abs_delta = (
        float(filtered_title_adjusted_rows.select(pl.col("delta_probability_lower_team_capped").abs().max()).item())
        if not filtered_title_adjusted_rows.is_empty()
        else 0.0
    )
    direct_preserved_count = int(
        final_submission.join(
            direct_adjusted_rows.select("ID", pl.col("final_pred").alias("direct_pred")),
            on="ID",
            how="inner",
        ).select((pl.col("Pred") == pl.col("direct_pred")).sum().cast(pl.Int64)).item()
    )

    summary = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_submission_path": str(BASE_SUBMISSION_PATH),
        "direct_adjusted_rows_path": str(DIRECT_ADJUSTED_ROWS_PATH),
        "output_submission_path": str(OUTPUT_SUBMISSION_PATH),
        "adjusted_rows_path": str(ADJUSTED_ROWS_PATH),
        "season": title_bias.SEASON,
        "overlay_parameters": {
            "direct_overlay_kept_fixed": True,
            "title_overlay_applied_only_to_non_direct_replaced_rows": True,
            "fixed_alpha": title_overlay.FIXED_ALPHA,
            "team_offset_cap": title_overlay.TEAM_OFFSET_CAP,
            "per_game_move_cap": title_overlay.PER_GAME_MOVE_CAP,
            "title_gap_cap": title_bias.MAX_ABS_TITLE_GAP,
            "applies_to": "men seeded-vs-seeded tournament field rows excluding direct-replaced rows",
        },
        "row_counts": {
            "base_submission": base_submission.height,
            "direct_replaced_rows": direct_adjusted_rows.height,
            "raw_title_adjusted_rows": raw_title_adjusted_rows.height,
            "title_adjusted_rows_after_direct_exclusion": filtered_title_adjusted_rows.height,
            "direct_and_title_overlap_rows": raw_title_adjusted_rows.height - filtered_title_adjusted_rows.height,
        },
        "prediction_summary": {
            "base_mean": float(base_submission.select(pl.col("Pred").mean()).item()),
            "final_mean": float(final_submission.select(pl.col("Pred").mean()).item()),
            "mean_abs_delta_on_title_adjusted_rows": mean_abs_delta,
            "max_abs_delta_on_title_adjusted_rows": max_abs_delta,
            "direct_rows_preserved_exactly": direct_preserved_count,
        },
        "current_title_market_summary": {
            "resolved_seeded_markets": sum(1 for row in title_rows if row["market_slug"] is not None),
            "missing_seeded_markets": sum(1 for row in title_rows if row["market_slug"] is None),
            "skipped_unresolved_market_questions": skipped_market_questions,
        },
        "top_positive_team_offsets": sorted(team_summary_rows, key=lambda row: row["team_offset"], reverse=True)[:10],
        "top_negative_team_offsets": sorted(team_summary_rows, key=lambda row: row["team_offset"])[:10],
        "largest_pair_moves": filtered_title_adjusted_rows.sort(
            pl.col("delta_probability_lower_team_capped").abs(),
            descending=True,
        )
        .head(20)
        .to_dicts(),
    }
    SUMMARY_PATH.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    print(f"Wrote combined submission: {OUTPUT_SUBMISSION_PATH}")
    print(f"Wrote filtered title-adjusted rows audit: {ADJUSTED_ROWS_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(f"Direct rows preserved exactly: {direct_preserved_count}")
    print(f"Filtered title-adjusted rows: {filtered_title_adjusted_rows.height}")


if __name__ == "__main__":
    main()
