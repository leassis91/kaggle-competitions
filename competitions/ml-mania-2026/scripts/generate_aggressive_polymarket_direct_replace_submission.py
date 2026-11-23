from __future__ import annotations

import collections
import datetime
import warnings
from pathlib import Path
from typing import Any, cast

import httpx
import orjson
import polars as pl

import analyze_polymarket_title_bias as polymarket_utils
import generate_final_stage2_current_snapshot_block_rate_submission as base_submission_generator


VOLUME_THRESHOLD = 10_000.0
DISCOVERY_URL = "https://gamma-api.polymarket.com/events?tag_slug=march-madness&closed=false&limit=500"
BASE_SUBMISSION_PATH = base_submission_generator.SUBMISSION_PATH
OUTPUT_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k.csv"
)
ARTIFACT_DIR = Path("artifacts/polymarket_direct_replace_volume_gt_10k")
ADJUSTED_ROWS_PATH = ARTIFACT_DIR / "adjusted_rows.csv"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

SPECIAL_DIRECT_MARKET_NAME_TO_TEAM_ID = {
    "miami hurricanes": 1274,
}
EMPTY_ADJUSTED_ROWS_SCHEMA = {
    "ID": pl.String,
    "game_title": pl.String,
    "market_slug": pl.String,
    "market_id": pl.String,
    "team0_market_name": pl.String,
    "team1_market_name": pl.String,
    "team0_id": pl.Int64,
    "team1_id": pl.Int64,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "volume": pl.Float64,
    "base_pred": pl.Float64,
    "market_pred": pl.Float64,
    "final_pred": pl.Float64,
    "delta_pred": pl.Float64,
}


def validate_submission_frame(
    submission_frame: pl.DataFrame,
    submission_template: pl.DataFrame,
) -> None:
    if submission_frame.height != base_submission_generator.EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            "Submission row count mismatch: "
            f"expected {base_submission_generator.EXPECTED_SUBMISSION_ROWS}, found {submission_frame.height}."
        )
    if submission_frame.get_column("ID").n_unique() != submission_frame.height:
        raise ValueError("Submission contains duplicate IDs.")
    if submission_frame.get_column("Pred").null_count() != 0:
        raise ValueError("Submission contains null predictions.")
    if not bool(submission_frame.get_column("Pred").is_finite().all()):
        raise ValueError("Submission contains non-finite predictions.")
    if not bool(
        submission_frame.select(
            ((pl.col("Pred") >= 0.0) & (pl.col("Pred") <= 1.0)).all()
        ).item()
    ):
        raise ValueError("Submission contains predictions outside [0, 1].")
    if submission_frame.get_column("ID").to_list() != submission_template.get_column("ID").to_list():
        raise ValueError("Submission IDs do not match the sample submission exactly.")


def load_or_generate_base_submission(
    submission_template: pl.DataFrame,
) -> tuple[pl.DataFrame, str]:
    if BASE_SUBMISSION_PATH.exists():
        base_submission = pl.read_csv(BASE_SUBMISSION_PATH)
        validate_submission_frame(base_submission, submission_template)
        return base_submission, "loaded_existing_base_submission_csv"

    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    augmented_tables, _ = base_submission_generator.build_augmented_matchup_tables(submission_template)
    base_submission = base_submission_generator.generate_submission(submission_template, augmented_tables)
    validate_submission_frame(base_submission, submission_template)
    return base_submission, "generated_in_memory_from_base_generator"


def build_team_lookup_rows() -> tuple[list[tuple[str, int, int]], dict[int, int]]:
    division_rows: list[tuple[int, int]] = []
    spelling_rows: list[tuple[str, int, int]] = []

    men_team_rows = pl.read_csv(base_submission_generator.pooled_base.DATA_DIR / "MTeams.csv").select("TeamID", "TeamName")
    women_team_rows = pl.read_csv(base_submission_generator.pooled_base.DATA_DIR / "WTeams.csv").select("TeamID", "TeamName")
    men_spelling_rows = pl.read_csv(base_submission_generator.pooled_base.DATA_DIR / "MTeamSpellings.csv").select(
        "TeamNameSpelling",
        "TeamID",
    )
    women_spelling_rows = pl.read_csv(base_submission_generator.pooled_base.DATA_DIR / "WTeamSpellings.csv").select(
        "TeamNameSpelling",
        "TeamID",
    )

    for team_id, team_name in men_team_rows.iter_rows():
        division_rows.append((int(team_id), 1))
        spelling_rows.append((polymarket_utils.normalize_name(str(team_name)), int(team_id), 1))
    for team_id, team_name in women_team_rows.iter_rows():
        division_rows.append((int(team_id), 0))
        spelling_rows.append((polymarket_utils.normalize_name(str(team_name)), int(team_id), 0))
    for team_name_spelling, team_id in men_spelling_rows.iter_rows():
        spelling_rows.append((polymarket_utils.normalize_name(str(team_name_spelling)), int(team_id), 1))
    for team_name_spelling, team_id in women_spelling_rows.iter_rows():
        spelling_rows.append((polymarket_utils.normalize_name(str(team_name_spelling)), int(team_id), 0))

    unique_spelling_rows = sorted(
        {(spelling, team_id, men_women) for spelling, team_id, men_women in spelling_rows if spelling},
        key=lambda item: (len(item[0]), item[0], item[1], item[2]),
        reverse=True,
    )
    division_by_team_id = {team_id: men_women for team_id, men_women in division_rows}
    return unique_spelling_rows, division_by_team_id


def resolve_market_team_id(
    market_team_name: str,
    team_lookup_rows: list[tuple[str, int, int]],
    expected_men_women: int | None,
) -> int:
    normalized_market_team_name = polymarket_utils.normalize_name(market_team_name)
    special_team_id = SPECIAL_DIRECT_MARKET_NAME_TO_TEAM_ID.get(normalized_market_team_name)
    if special_team_id is not None:
        return special_team_id

    filtered_lookup_rows = [
        (spelling, team_id)
        for spelling, team_id, men_women in team_lookup_rows
        if expected_men_women is None or men_women == expected_men_women
    ]

    exact_matches = {team_id for spelling, team_id in filtered_lookup_rows if spelling == normalized_market_team_name}
    if len(exact_matches) == 1:
        return next(iter(exact_matches))

    contained_rows = [
        (spelling, team_id)
        for spelling, team_id in filtered_lookup_rows
        if f" {spelling} " in f" {normalized_market_team_name} "
        or f" {normalized_market_team_name} " in f" {spelling} "
    ]
    if contained_rows:
        longest_spelling_length = max(len(spelling) for spelling, _team_id in contained_rows)
        longest_spelling_matches = {
            team_id for spelling, team_id in contained_rows if len(spelling) == longest_spelling_length
        }
        if len(longest_spelling_matches) == 1:
            return next(iter(longest_spelling_matches))

    raise ValueError(f"Could not resolve direct market team name {market_team_name!r}.")


def choose_midpoint_probability(market: dict[str, Any]) -> float:
    best_bid = polymarket_utils.parse_optional_float(market.get("bestBid"))
    best_ask = polymarket_utils.parse_optional_float(market.get("bestAsk"))
    if best_bid is None or best_ask is None:
        raise ValueError("Missing bid/ask midpoint.")
    if not (0.0 <= best_bid <= best_ask <= 1.0):
        raise ValueError("Invalid bid/ask midpoint bounds.")
    return (best_bid + best_ask) / 2.0


def parse_outcome_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [str(item) for item in orjson.loads(value)]
    return [str(item) for item in value]


def build_submission_id_lookup(submission_template: pl.DataFrame) -> dict[tuple[int, int], str]:
    return {
        (int(lower_team_id), int(higher_team_id)): str(submission_id)
        for submission_id, lower_team_id, higher_team_id in submission_template.select(
            "ID",
            "LowerTeamID",
            "HigherTeamID",
        ).iter_rows()
    }


def fetch_live_march_madness_events() -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(DISCOVERY_URL)
        response.raise_for_status()
        return cast(list[dict[str, Any]], response.json())


def select_direct_moneyline_market(event: dict[str, Any]) -> dict[str, Any]:
    event_title = str(event.get("title", ""))
    matching_markets = [market for market in event.get("markets", []) if str(market.get("question", "")) == event_title]
    if len(matching_markets) != 1:
        raise ValueError(f"Expected exactly one title-matching moneyline market, found {len(matching_markets)}.")

    selected_market = matching_markets[0]
    outcome_names = parse_outcome_names(selected_market["outcomes"])
    if len(outcome_names) != 2:
        raise ValueError("Selected moneyline market was not binary.")
    return selected_market


def discover_direct_market_rows(
    team_lookup_rows: list[tuple[str, int, int]],
    division_by_team_id: dict[int, int],
    submission_id_by_pair: dict[tuple[int, int], str],
    base_prediction_by_id: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    pending_eligible_rows: list[dict[str, Any]] = []
    live_events = fetch_live_march_madness_events()

    for event in live_events:
        event_title = str(event.get("title", ""))
        audit_row: dict[str, Any] = {
            "event_slug": str(event.get("slug", "")),
            "game_title": event_title,
            "status": "skipped",
            "skip_reason": None,
            "market_id": None,
            "market_slug": None,
            "market_question": None,
            "team0_market_name": None,
            "team1_market_name": None,
            "team0_id": None,
            "team1_id": None,
            "submission_id": None,
            "market_probability_lower_team": None,
            "best_bid": None,
            "best_ask": None,
            "volume": None,
        }

        try:
            event_slug = str(event.get("slug", ""))
            expected_men_women = 1 if event_slug.startswith("cbb-") else 0 if event_slug.startswith("cwbb-") else None
            selected_market = select_direct_moneyline_market(event)
            outcome_names = parse_outcome_names(selected_market["outcomes"])
            if len(outcome_names) != 2:
                raise ValueError("Direct moneyline market outcomes were not binary.")

            team0_market_name = str(outcome_names[0])
            team1_market_name = str(outcome_names[1])
            team0_id = resolve_market_team_id(team0_market_name, team_lookup_rows, expected_men_women)
            team1_id = resolve_market_team_id(team1_market_name, team_lookup_rows, expected_men_women)
            if team0_id == team1_id:
                raise ValueError("Direct moneyline market resolved to the same team twice.")
            if division_by_team_id[team0_id] != division_by_team_id[team1_id]:
                raise ValueError("Direct moneyline market teams resolved across mixed divisions.")

            lower_team_id, higher_team_id = sorted((team0_id, team1_id))
            submission_id = submission_id_by_pair.get((lower_team_id, higher_team_id))
            if submission_id is None:
                raise ValueError("Resolved market pair did not exist in the Stage 2 submission universe.")

            midpoint_probability_team0 = choose_midpoint_probability(selected_market)
            volume = float(selected_market.get("volume") or 0.0)
            best_bid = polymarket_utils.parse_optional_float(selected_market.get("bestBid"))
            best_ask = polymarket_utils.parse_optional_float(selected_market.get("bestAsk"))
            market_probability_lower_team = midpoint_probability_team0 if team0_id == lower_team_id else 1.0 - midpoint_probability_team0

            audit_row.update(
                {
                    "market_id": str(selected_market.get("id", "")),
                    "market_slug": str(selected_market.get("slug", "")),
                    "market_question": str(selected_market.get("question", "")),
                    "team0_market_name": team0_market_name,
                    "team1_market_name": team1_market_name,
                    "team0_id": team0_id,
                    "team1_id": team1_id,
                    "submission_id": submission_id,
                    "market_probability_lower_team": market_probability_lower_team,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "volume": volume,
                }
            )

            if volume <= VOLUME_THRESHOLD:
                audit_row["skip_reason"] = "volume_not_above_threshold"
                audit_rows.append(audit_row)
                continue

            audit_row["status"] = "eligible_pending_deduplication"
            audit_row["skip_reason"] = None
            audit_rows.append(audit_row)
            pending_eligible_rows.append(
                {
                    **audit_row,
                    "base_pred": base_prediction_by_id[submission_id],
                    "final_pred": market_probability_lower_team,
                    "delta_pred": market_probability_lower_team - base_prediction_by_id[submission_id],
                }
            )
        except Exception as error:
            audit_row["skip_reason"] = str(error)
            audit_rows.append(audit_row)

    pending_rows_by_submission_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for pending_row in pending_eligible_rows:
        pending_rows_by_submission_id[str(pending_row["submission_id"])].append(pending_row)

    applied_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for submission_id, matching_rows in pending_rows_by_submission_id.items():
        sorted_rows = sorted(
            matching_rows,
            key=lambda row: (
                float(row["volume"]),
                float(row["market_probability_lower_team"]),
                str(row["market_id"]),
            ),
            reverse=True,
        )
        applied_rows.append({**sorted_rows[0], "status": "applied"})
        for duplicate_row in sorted_rows[1:]:
            duplicate_rows.append(
                {
                    **duplicate_row,
                    "status": "skipped_duplicate_lower_priority_market",
                    "skip_reason": f"duplicate_submission_id:{submission_id}",
                }
            )

    if duplicate_rows:
        duplicate_rows_by_market_id = {
            str(duplicate_row["market_id"]): duplicate_row for duplicate_row in duplicate_rows
        }
        applied_market_ids = {str(applied_row["market_id"]) for applied_row in applied_rows}
        refreshed_audit_rows = []
        for audit_row in audit_rows:
            market_id = str(audit_row["market_id"]) if audit_row["market_id"] is not None else None
            if market_id is not None and market_id in duplicate_rows_by_market_id:
                refreshed_audit_rows.append(duplicate_rows_by_market_id[market_id])
                continue
            if market_id is not None and market_id in applied_market_ids:
                refreshed_audit_rows.append({**audit_row, "status": "applied"})
                continue
            refreshed_audit_rows.append(audit_row)
        audit_rows = refreshed_audit_rows
    else:
        applied_market_ids = {str(applied_row["market_id"]) for applied_row in applied_rows}
        audit_rows = [
            {**audit_row, "status": "applied"}
            if audit_row["market_id"] is not None and str(audit_row["market_id"]) in applied_market_ids
            else audit_row
            for audit_row in audit_rows
        ]

    return audit_rows, applied_rows


def apply_replacements(
    base_submission: pl.DataFrame,
    applied_rows: list[dict[str, Any]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    applied_rows_frame = pl.DataFrame(applied_rows)
    if applied_rows_frame.is_empty():
        final_submission = base_submission.clone()
        adjusted_rows = pl.DataFrame(schema=EMPTY_ADJUSTED_ROWS_SCHEMA)
    else:
        adjusted_rows = applied_rows_frame.select(
            "submission_id",
            "game_title",
            "market_slug",
            "market_id",
            "team0_market_name",
            "team1_market_name",
            "team0_id",
            "team1_id",
            "best_bid",
            "best_ask",
            "volume",
            "base_pred",
            pl.col("market_probability_lower_team").alias("market_pred"),
            "final_pred",
            "delta_pred",
        ).rename({"submission_id": "ID"})
        final_submission = (
            base_submission.join(
                adjusted_rows.select(pl.col("ID"), pl.col("final_pred").alias("override_pred")),
                on="ID",
                how="left",
            )
            .with_columns(pl.coalesce(["override_pred", "Pred"]).alias("Pred"))
            .select("ID", "Pred")
        )

    return final_submission, adjusted_rows


def write_manifest(
    submission_template: pl.DataFrame,
    base_submission_source: str,
    base_submission: pl.DataFrame,
    final_submission: pl.DataFrame,
    audit_rows: list[dict[str, Any]],
    adjusted_rows: pl.DataFrame,
) -> None:
    skip_reason_counter = collections.Counter()
    for audit_row in audit_rows:
        status = str(audit_row.get("status", ""))
        if status == "applied":
            continue
        skip_reason = audit_row.get("skip_reason")
        if skip_reason is None:
            continue
        skip_reason_counter[str(skip_reason)] += 1

    manifest = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_script": "scripts/generate_aggressive_polymarket_direct_replace_submission.py",
        "discovery_url": DISCOVERY_URL,
        "base_submission_path": str(BASE_SUBMISSION_PATH),
        "output_submission_path": str(OUTPUT_SUBMISSION_PATH),
        "adjusted_rows_path": str(ADJUSTED_ROWS_PATH),
        "volume_threshold_rule": "volume > 10000",
        "price_rule": "strict_bid_ask_midpoint_only",
        "overlay_rule": "replace_base_prediction_with_market_probability_for_eligible_rows",
        "base_submission_source": base_submission_source,
        "row_counts": {
            "submission_template": submission_template.height,
            "base_submission": base_submission.height,
            "final_submission": final_submission.height,
            "discovered_events": len(audit_rows),
            "applied_replacements": adjusted_rows.height,
        },
        "prediction_summary": {
            "base_mean": float(base_submission.select(pl.col("Pred").mean()).item()),
            "final_mean": float(final_submission.select(pl.col("Pred").mean()).item()),
            "mean_abs_delta_on_adjusted_rows": (
                float(adjusted_rows.select(pl.col("delta_pred").abs().mean()).item())
                if not adjusted_rows.is_empty()
                else 0.0
            ),
            "max_abs_delta_on_adjusted_rows": (
                float(adjusted_rows.select(pl.col("delta_pred").abs().max()).item())
                if not adjusted_rows.is_empty()
                else 0.0
            ),
        },
        "skip_reason_counts": dict(sorted(skip_reason_counter.items())),
        "applied_rows": adjusted_rows.sort(pl.col("delta_pred").abs(), descending=True).head(50).to_dicts(),
    }
    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    submission_template = base_submission_generator.parse_submission_template()
    base_submission, base_submission_source = load_or_generate_base_submission(submission_template)
    validate_submission_frame(base_submission, submission_template)

    team_lookup_rows, division_by_team_id = build_team_lookup_rows()
    submission_id_by_pair = build_submission_id_lookup(submission_template)
    base_prediction_by_id = {
        str(submission_id): float(prediction)
        for submission_id, prediction in base_submission.iter_rows()
    }

    audit_rows, applied_rows = discover_direct_market_rows(
        team_lookup_rows=team_lookup_rows,
        division_by_team_id=division_by_team_id,
        submission_id_by_pair=submission_id_by_pair,
        base_prediction_by_id=base_prediction_by_id,
    )
    final_submission, adjusted_rows = apply_replacements(base_submission, applied_rows)
    validate_submission_frame(final_submission, submission_template)

    final_submission.write_csv(OUTPUT_SUBMISSION_PATH)
    adjusted_rows.write_csv(ADJUSTED_ROWS_PATH)
    write_manifest(
        submission_template=submission_template,
        base_submission_source=base_submission_source,
        base_submission=base_submission,
        final_submission=final_submission,
        audit_rows=audit_rows,
        adjusted_rows=adjusted_rows,
    )

    print(f"Wrote submission: {OUTPUT_SUBMISSION_PATH}")
    print(f"Wrote adjusted rows audit: {ADJUSTED_ROWS_PATH}")
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Applied replacements: {adjusted_rows.height}")


if __name__ == "__main__":
    main()
