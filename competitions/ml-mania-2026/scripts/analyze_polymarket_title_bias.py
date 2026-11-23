from __future__ import annotations

import datetime
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import orjson
import polars as pl


SEASON = 2026
TITLE_EVENT_SLUG = "2026-ncaa-tournament-winner"
DEFAULT_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_quintile_blend_2026-03-17.csv"
)
ARTIFACT_DIR = Path("artifacts/polymarket_title_bias_2026-03-17")
TEAM_REPORT_PATH = ARTIFACT_DIR / "team_title_bias.csv"
FIELD_PAIR_REPORT_PATH = ARTIFACT_DIR / "bracket_field_pair_adjustments.csv"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"

DATA_DIR = Path("data")
TEAMS_PATH = DATA_DIR / "MTeams.csv"
TEAM_SPELLINGS_PATH = DATA_DIR / "MTeamSpellings.csv"
SEEDS_PATH = DATA_DIR / "MNCAATourneySeeds.csv"
SLOTS_PATH = DATA_DIR / "MNCAATourneySlots.csv"

MATCHUP_PROBABILITY_CLIP = 1e-4
TITLE_PROBABILITY_CLIP = 1e-3
MAX_ABS_TITLE_GAP = 1.25
ALPHA_GRID = [step / 100 for step in range(0, 41)]

SPECIAL_TITLE_NAME_TO_TEAM_ID = {
    "miami fl": 1274,
    "nc state": 1301,
    "queens": 1474,
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def clip_probability(probability: float, floor: float) -> float:
    return max(floor, min(1.0 - floor, probability))


def logit(probability: float, floor: float) -> float:
    clipped_probability = clip_probability(probability, floor)
    return math.log(clipped_probability / (1.0 - clipped_probability))


def logistic(logit_value: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit_value))


def parse_probability_list(value: Any) -> list[float]:
    if isinstance(value, str):
        parsed_value = orjson.loads(value)
    else:
        parsed_value = value
    return [float(item) for item in parsed_value]


def parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def choose_market_probability(market: dict[str, Any]) -> float:
    best_bid = parse_optional_float(market.get("bestBid"))
    best_ask = parse_optional_float(market.get("bestAsk"))
    if best_bid is not None and best_ask is not None and 0.0 <= best_bid <= best_ask <= 1.0:
        return (best_bid + best_ask) / 2.0

    outcome_prices = market.get("outcomePrices")
    if outcome_prices is not None:
        return parse_probability_list(outcome_prices)[0]

    last_trade_price = parse_optional_float(market.get("lastTradePrice"))
    if last_trade_price is not None:
        return last_trade_price

    raise ValueError(f"Could not determine market probability for {market['question']!r}.")


def load_team_names() -> dict[int, str]:
    return {
        int(team_id): team_name
        for team_id, team_name in pl.read_csv(TEAMS_PATH).select("TeamID", "TeamName").iter_rows()
    }


def load_seed_data() -> tuple[dict[str, int], dict[int, str], set[int]]:
    season_seeds = pl.read_csv(SEEDS_PATH).filter(pl.col("Season") == SEASON).select("Seed", "TeamID")
    seed_to_team_id = {str(seed): int(team_id) for seed, team_id in season_seeds.iter_rows()}
    team_id_to_seed = {team_id: seed for seed, team_id in seed_to_team_id.items()}
    return seed_to_team_id, team_id_to_seed, set(team_id_to_seed)


def build_seeded_spelling_rows(seeded_team_ids: set[int]) -> list[tuple[str, int]]:
    spelling_rows: list[tuple[str, int]] = []

    canonical_rows = pl.read_csv(TEAMS_PATH).select("TeamID", "TeamName")
    for team_id, team_name in canonical_rows.iter_rows():
        team_id_int = int(team_id)
        if team_id_int in seeded_team_ids:
            spelling_rows.append((normalize_name(str(team_name)), team_id_int))

    spelling_frame = pl.read_csv(TEAM_SPELLINGS_PATH).select("TeamNameSpelling", "TeamID")
    for team_name_spelling, team_id in spelling_frame.iter_rows():
        team_id_int = int(team_id)
        if team_id_int in seeded_team_ids:
            spelling_rows.append((normalize_name(str(team_name_spelling)), team_id_int))

    unique_rows = {(spelling, team_id) for spelling, team_id in spelling_rows if spelling}
    return sorted(unique_rows, key=lambda item: (len(item[0]), item[0], item[1]), reverse=True)


def resolve_title_market_team_id(team_name: str, seeded_spelling_rows: list[tuple[str, int]]) -> int:
    normalized_team_name = normalize_name(team_name)
    special_team_id = SPECIAL_TITLE_NAME_TO_TEAM_ID.get(normalized_team_name)
    if special_team_id is not None:
        return special_team_id

    exact_matches = {team_id for spelling, team_id in seeded_spelling_rows if spelling == normalized_team_name}
    if len(exact_matches) == 1:
        return next(iter(exact_matches))

    contains_matches = {
        team_id
        for spelling, team_id in seeded_spelling_rows
        if f" {spelling} " in f" {normalized_team_name} " or f" {normalized_team_name} " in f" {spelling} "
    }
    if len(contains_matches) == 1:
        return next(iter(contains_matches))

    raise ValueError(f"Could not resolve title market team name {team_name!r}.")


def fetch_title_market_rows(
    seeded_spelling_rows: list[tuple[str, int]],
    team_names: dict[int, str],
    team_id_to_seed: dict[int, str],
) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"https://gamma-api.polymarket.com/events?slug={TITLE_EVENT_SLUG}")
        response.raise_for_status()
        event_rows = response.json()

    if not event_rows:
        raise ValueError(f"No Polymarket event found for slug {TITLE_EVENT_SLUG!r}.")

    event = event_rows[0]
    title_rows: list[dict[str, Any]] = []
    for market in event["markets"]:
        question = str(market.get("question", ""))
        if not question.startswith("Will ") or not question.endswith(f" win the {SEASON} NCAA Tournament?"):
            continue
        if bool(market.get("closed", False)) or not bool(market.get("active", True)):
            continue

        team_name = question.removeprefix("Will ").removesuffix(f" win the {SEASON} NCAA Tournament?")
        team_id = resolve_title_market_team_id(team_name, seeded_spelling_rows)
        title_rows.append(
            {
                "market_slug": market["slug"],
                "team_name_market": team_name,
                "TeamID": team_id,
                "TeamName": team_names[team_id],
                "Seed": team_id_to_seed[team_id],
                "market_probability_raw": choose_market_probability(market),
                "market_probability_outcome_price": parse_probability_list(market["outcomePrices"])[0],
                "best_bid": parse_optional_float(market.get("bestBid")),
                "best_ask": parse_optional_float(market.get("bestAsk")),
                "last_trade_price": parse_optional_float(market.get("lastTradePrice")),
                "liquidity": float(market.get("liquidity") or 0.0),
                "volume": float(market.get("volume") or 0.0),
            }
        )

    unique_team_ids = {row["TeamID"] for row in title_rows}
    if len(unique_team_ids) != len(title_rows):
        raise ValueError("Duplicate seeded teams found in title market mapping.")

    missing_team_ids = set(team_id_to_seed) - unique_team_ids
    if missing_team_ids:
        missing_team_names = sorted(team_names[team_id] for team_id in missing_team_ids)
        raise ValueError(f"Missing title markets for seeded teams: {missing_team_names}")

    raw_probability_sum = sum(row["market_probability_raw"] for row in title_rows)
    if raw_probability_sum <= 0.0:
        raise ValueError("Title market probability sum was non-positive.")

    for row in title_rows:
        row["market_probability_fair"] = row["market_probability_raw"] / raw_probability_sum

    return sorted(title_rows, key=lambda row: row["market_probability_fair"], reverse=True)


def load_slot_children() -> tuple[dict[str, tuple[str, str]], str]:
    slot_rows = (
        pl.read_csv(SLOTS_PATH)
        .filter(pl.col("Season") == SEASON)
        .select("Slot", "StrongSeed", "WeakSeed")
    )
    slot_children = {
        str(slot): (str(strong_seed), str(weak_seed))
        for slot, strong_seed, weak_seed in slot_rows.iter_rows()
    }
    root_slots = set(slot_children) - {child for children in slot_children.values() for child in children}
    if len(root_slots) != 1:
        raise ValueError(f"Expected exactly one bracket root, found {sorted(root_slots)}")
    return slot_children, next(iter(root_slots))


def load_pairwise_probabilities(submission_path: Path) -> dict[tuple[int, int], float]:
    submission_frame = pl.read_csv(submission_path)
    pairwise_probabilities: dict[tuple[int, int], float] = {}
    for submission_id, prediction in submission_frame.iter_rows():
        _season, lower_team_id, higher_team_id = str(submission_id).split("_", 2)
        pairwise_probabilities[(int(lower_team_id), int(higher_team_id))] = float(prediction)
    return pairwise_probabilities


def matchup_probability(
    left_team_id: int,
    right_team_id: int,
    pairwise_probabilities: dict[tuple[int, int], float],
    offsets: dict[int, float],
) -> float:
    lower_team_id, higher_team_id = sorted((left_team_id, right_team_id))
    base_probability = pairwise_probabilities[(lower_team_id, higher_team_id)]
    if left_team_id != lower_team_id:
        base_probability = 1.0 - base_probability

    offset_difference = offsets.get(left_team_id, 0.0) - offsets.get(right_team_id, 0.0)
    adjusted_probability = logistic(logit(base_probability, MATCHUP_PROBABILITY_CLIP) + offset_difference)
    return clip_probability(adjusted_probability, MATCHUP_PROBABILITY_CLIP)


def compute_title_probabilities(
    slot_children: dict[str, tuple[str, str]],
    root_slot: str,
    seed_to_team_id: dict[str, int],
    pairwise_probabilities: dict[tuple[int, int], float],
    offsets: dict[int, float] | None = None,
) -> dict[int, float]:
    active_offsets = offsets or {}

    @lru_cache(maxsize=None)
    def slot_distribution(node: str) -> tuple[tuple[int, float], ...]:
        if node not in slot_children:
            if node not in seed_to_team_id:
                raise ValueError(f"Unknown bracket node {node!r}.")
            return ((seed_to_team_id[node], 1.0),)

        strong_node, weak_node = slot_children[node]
        strong_distribution = dict(slot_distribution(strong_node))
        weak_distribution = dict(slot_distribution(weak_node))
        winner_distribution: defaultdict[int, float] = defaultdict(float)

        for strong_team_id, strong_probability in strong_distribution.items():
            for weak_team_id, weak_probability in weak_distribution.items():
                game_probability = matchup_probability(
                    strong_team_id,
                    weak_team_id,
                    pairwise_probabilities,
                    active_offsets,
                )
                path_probability = strong_probability * weak_probability
                winner_distribution[strong_team_id] += path_probability * game_probability
                winner_distribution[weak_team_id] += path_probability * (1.0 - game_probability)

        return tuple(sorted(winner_distribution.items()))

    return dict(slot_distribution(root_slot))


def compute_possible_round_by_pair(
    slot_children: dict[str, tuple[str, str]],
) -> dict[tuple[str, str], int]:
    def parse_slot_round(slot: str) -> int:
        if slot.startswith("R"):
            return int(slot[1]) + 1
        return 1

    @lru_cache(maxsize=None)
    def descendants(node: str) -> tuple[str, ...]:
        if node not in slot_children:
            return (node,)
        strong_node, weak_node = slot_children[node]
        return descendants(strong_node) + descendants(weak_node)

    possible_round_by_pair: dict[tuple[str, str], int] = {}
    for slot in slot_children:
        round_number = parse_slot_round(slot)
        descendant_list = sorted(set(descendants(slot)))
        for left_index, left_seed in enumerate(descendant_list):
            for right_seed in descendant_list[left_index + 1 :]:
                seed_pair = (left_seed, right_seed)
                current_round = possible_round_by_pair.get(seed_pair)
                if current_round is None or round_number < current_round:
                    possible_round_by_pair[seed_pair] = round_number
    return possible_round_by_pair


def build_team_title_rows(
    title_rows: list[dict[str, Any]],
    model_title_probabilities: dict[int, float],
    adjusted_title_probabilities: dict[int, float],
    offsets: dict[int, float],
) -> list[dict[str, Any]]:
    team_title_rows: list[dict[str, Any]] = []
    for row in title_rows:
        team_id = int(row["TeamID"])
        model_title_probability = float(model_title_probabilities[team_id])
        market_title_probability_fair = float(row["market_probability_fair"])
        raw_title_gap = logit(market_title_probability_fair, TITLE_PROBABILITY_CLIP) - logit(
            model_title_probability,
            TITLE_PROBABILITY_CLIP,
        )
        team_title_rows.append(
            {
                **row,
                "model_title_probability": model_title_probability,
                "adjusted_title_probability": float(adjusted_title_probabilities[team_id]),
                "title_logit_gap_raw": raw_title_gap,
                "title_logit_gap_capped": max(-MAX_ABS_TITLE_GAP, min(MAX_ABS_TITLE_GAP, raw_title_gap)),
                "team_offset": float(offsets.get(team_id, 0.0)),
                "market_minus_model_title_probability": market_title_probability_fair - model_title_probability,
                "market_minus_adjusted_title_probability": market_title_probability_fair
                - float(adjusted_title_probabilities[team_id]),
            }
        )
    return sorted(team_title_rows, key=lambda team_row: abs(team_row["title_logit_gap_raw"]), reverse=True)


def alpha_objective(
    alpha: float,
    title_rows: list[dict[str, Any]],
    capped_title_gaps: dict[int, float],
    slot_children: dict[str, tuple[str, str]],
    root_slot: str,
    seed_to_team_id: dict[str, int],
    pairwise_probabilities: dict[tuple[int, int], float],
) -> tuple[float, dict[int, float], dict[int, float]]:
    offsets = {team_id: alpha * capped_gap for team_id, capped_gap in capped_title_gaps.items()}
    adjusted_title_probabilities = compute_title_probabilities(
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
    )

    weight_sum = 0.0
    objective = 0.0
    for row in title_rows:
        team_id = int(row["TeamID"])
        market_probability = float(row["market_probability_fair"])
        adjusted_probability = float(adjusted_title_probabilities[team_id])
        weight = math.sqrt(market_probability)
        objective += weight * (
            logit(adjusted_probability, TITLE_PROBABILITY_CLIP) - logit(market_probability, TITLE_PROBABILITY_CLIP)
        ) ** 2
        weight_sum += weight

    return objective / weight_sum, offsets, adjusted_title_probabilities


def choose_best_alpha(
    title_rows: list[dict[str, Any]],
    model_title_probabilities: dict[int, float],
    slot_children: dict[str, tuple[str, str]],
    root_slot: str,
    seed_to_team_id: dict[str, int],
    pairwise_probabilities: dict[tuple[int, int], float],
) -> tuple[float, dict[int, float], dict[int, float], list[dict[str, float]]]:
    capped_title_gaps = {
        int(row["TeamID"]): max(
            -MAX_ABS_TITLE_GAP,
            min(
                MAX_ABS_TITLE_GAP,
                logit(float(row["market_probability_fair"]), TITLE_PROBABILITY_CLIP)
                - logit(float(model_title_probabilities[int(row["TeamID"])]), TITLE_PROBABILITY_CLIP),
            ),
        )
        for row in title_rows
    }

    best_alpha = 0.0
    best_objective = float("inf")
    best_offsets: dict[int, float] = {}
    best_adjusted_title_probabilities = model_title_probabilities
    alpha_rows: list[dict[str, float]] = []

    for alpha in ALPHA_GRID:
        objective, offsets, adjusted_title_probabilities = alpha_objective(
            alpha=alpha,
            title_rows=title_rows,
            capped_title_gaps=capped_title_gaps,
            slot_children=slot_children,
            root_slot=root_slot,
            seed_to_team_id=seed_to_team_id,
            pairwise_probabilities=pairwise_probabilities,
        )
        alpha_rows.append({"alpha": alpha, "objective": objective})
        if objective < best_objective:
            best_alpha = alpha
            best_objective = objective
            best_offsets = offsets
            best_adjusted_title_probabilities = adjusted_title_probabilities

    return best_alpha, best_offsets, best_adjusted_title_probabilities, alpha_rows


def build_field_pair_rows(
    team_names: dict[int, str],
    team_id_to_seed: dict[int, str],
    pairwise_probabilities: dict[tuple[int, int], float],
    offsets: dict[int, float],
    slot_children: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    possible_round_by_pair = compute_possible_round_by_pair(slot_children)
    seeded_team_ids = sorted(team_id_to_seed)
    field_pair_rows: list[dict[str, Any]] = []

    for lower_index, lower_team_id in enumerate(seeded_team_ids):
        for higher_team_id in seeded_team_ids[lower_index + 1 :]:
            lower_seed = team_id_to_seed[lower_team_id]
            higher_seed = team_id_to_seed[higher_team_id]
            sorted_seed_pair = (lower_seed, higher_seed) if lower_seed <= higher_seed else (higher_seed, lower_seed)
            base_probability = pairwise_probabilities[(lower_team_id, higher_team_id)]
            adjusted_probability = matchup_probability(
                lower_team_id,
                higher_team_id,
                pairwise_probabilities,
                offsets,
            )
            field_pair_rows.append(
                {
                    "LowerTeamID": lower_team_id,
                    "LowerTeamName": team_names[lower_team_id],
                    "LowerSeed": lower_seed,
                    "HigherTeamID": higher_team_id,
                    "HigherTeamName": team_names[higher_team_id],
                    "HigherSeed": higher_seed,
                    "possible_round": possible_round_by_pair.get(sorted_seed_pair, 0),
                    "base_probability_lower_team": base_probability,
                    "adjusted_probability_lower_team": adjusted_probability,
                    "delta_probability_lower_team": adjusted_probability - base_probability,
                    "lower_team_offset": offsets.get(lower_team_id, 0.0),
                    "higher_team_offset": offsets.get(higher_team_id, 0.0),
                }
            )

    return sorted(field_pair_rows, key=lambda row: abs(row["delta_probability_lower_team"]), reverse=True)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    team_names = load_team_names()
    seed_to_team_id, team_id_to_seed, seeded_team_ids = load_seed_data()
    seeded_spelling_rows = build_seeded_spelling_rows(seeded_team_ids)
    title_rows = fetch_title_market_rows(
        seeded_spelling_rows=seeded_spelling_rows,
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
    )

    slot_children, root_slot = load_slot_children()
    pairwise_probabilities = load_pairwise_probabilities(DEFAULT_SUBMISSION_PATH)
    model_title_probabilities = compute_title_probabilities(
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
    )
    best_alpha, offsets, adjusted_title_probabilities, alpha_rows = choose_best_alpha(
        title_rows=title_rows,
        model_title_probabilities=model_title_probabilities,
        slot_children=slot_children,
        root_slot=root_slot,
        seed_to_team_id=seed_to_team_id,
        pairwise_probabilities=pairwise_probabilities,
    )

    team_title_rows = build_team_title_rows(
        title_rows=title_rows,
        model_title_probabilities=model_title_probabilities,
        adjusted_title_probabilities=adjusted_title_probabilities,
        offsets=offsets,
    )
    field_pair_rows = build_field_pair_rows(
        team_names=team_names,
        team_id_to_seed=team_id_to_seed,
        pairwise_probabilities=pairwise_probabilities,
        offsets=offsets,
        slot_children=slot_children,
    )

    team_title_frame = pl.DataFrame(team_title_rows)
    field_pair_frame = pl.DataFrame(field_pair_rows)
    team_title_frame.write_csv(TEAM_REPORT_PATH)
    field_pair_frame.write_csv(FIELD_PAIR_REPORT_PATH)

    summary = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "submission_path": str(DEFAULT_SUBMISSION_PATH),
        "title_event_slug": TITLE_EVENT_SLUG,
        "season": SEASON,
        "raw_title_probability_sum": float(sum(row["market_probability_raw"] for row in title_rows)),
        "fair_title_probability_sum": float(sum(row["market_probability_fair"] for row in title_rows)),
        "selected_alpha": best_alpha,
        "max_abs_title_gap": MAX_ABS_TITLE_GAP,
        "title_probability_clip": TITLE_PROBABILITY_CLIP,
        "matchup_probability_clip": MATCHUP_PROBABILITY_CLIP,
        "alpha_grid": alpha_rows,
        "top_positive_title_bias": team_title_frame.sort("title_logit_gap_raw", descending=True)
        .head(10)
        .to_dicts(),
        "top_negative_title_bias": team_title_frame.sort("title_logit_gap_raw").head(10).to_dicts(),
        "largest_field_pair_moves": field_pair_frame.head(20).to_dicts(),
    }
    SUMMARY_PATH.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    print(f"Wrote team title bias report: {TEAM_REPORT_PATH}")
    print(f"Wrote bracket field pair adjustments: {FIELD_PAIR_REPORT_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(f"Selected alpha: {best_alpha:.2f}")
    print(
        "Top title-market underrates: "
        + ", ".join(
            f"{row['TeamName']} ({row['title_logit_gap_raw']:+.3f})"
            for row in team_title_rows[:5]
            if row["title_logit_gap_raw"] > 0.0
        )
    )
    print(
        "Top title-market overrates: "
        + ", ".join(
            f"{row['TeamName']} ({row['title_logit_gap_raw']:+.3f})"
            for row in sorted(team_title_rows, key=lambda row: row["title_logit_gap_raw"])[:5]
        )
    )


if __name__ == "__main__":
    main()
