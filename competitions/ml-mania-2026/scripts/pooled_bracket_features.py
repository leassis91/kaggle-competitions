from __future__ import annotations

import itertools
import re
from functools import lru_cache

import polars as pl

import train_iteration_16_pooled_selection_sunday_baseline as baseline


REGION_CODE = {
    "W": 1,
    "X": 2,
    "Y": 3,
    "Z": 4,
}


def parse_seed_number(seed: str) -> int:
    return int(seed[1:3])


def parse_seed_region(seed: str) -> str:
    return seed[0]


def parse_play_in_flag(seed: str) -> int:
    return int(len(seed) > 3 and seed[-1].isalpha())


def parse_slot_round(slot: str) -> int:
    if slot.startswith("R"):
        return int(slot[1]) + 1
    return 1


def parse_opening_slot_index(slot: str) -> int:
    slot_match = re.search(r"(\d+)$", slot)
    if slot_match is None:
        return 0
    return int(slot_match.group(1))


def build_seed_team_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    return (
        pl.read_csv(config.seeds_path)
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select(
            pl.col("Season").cast(pl.Int64),
            pl.col("TeamID").cast(pl.Int64),
            pl.col("Seed"),
            pl.col("Seed").map_elements(parse_seed_number, return_dtype=pl.Int64).alias("seed_number"),
            pl.col("Seed").str.slice(0, 1).alias("seed_region"),
            pl.col("Seed")
            .str.slice(0, 1)
            .replace_strict(REGION_CODE, default=0)
            .cast(pl.Int8)
            .alias("seed_region_code"),
            pl.col("Seed")
            .str.slice(0, 1)
            .is_in(["W", "X"])
            .cast(pl.Int8)
            .alias("seed_final_four_side_flag"),
            pl.col("Seed").map_elements(parse_play_in_flag, return_dtype=pl.Int8).alias("play_in_flag"),
        )
    )


def build_pair_bracket_features(config: baseline.DivisionConfig) -> pl.DataFrame:
    seed_features = build_seed_team_features(config)
    slots = (
        pl.read_csv(baseline.DATA_DIR / f"{config.prefix}NCAATourneySlots.csv")
        .filter(pl.col("Season") >= baseline.MIN_POOLED_SEASON)
        .select("Season", "Slot", "StrongSeed", "WeakSeed")
    )

    bracket_rows: list[dict[str, int | float]] = []

    for season in sorted(seed_features["Season"].unique().to_list()):
        season_seed_features = seed_features.filter(pl.col("Season") == season)
        season_slots = slots.filter(pl.col("Season") == season)
        slot_children = {
            slot: (strong_seed, weak_seed)
            for slot, strong_seed, weak_seed in season_slots.select("Slot", "StrongSeed", "WeakSeed").iter_rows()
        }

        @lru_cache(maxsize=None)
        def slot_descendants(node: str) -> tuple[str, ...]:
            if node not in slot_children:
                return (node,)
            strong_seed, weak_seed = slot_children[node]
            return slot_descendants(strong_seed) + slot_descendants(weak_seed)

        opening_slot_by_seed: dict[str, int] = {}
        possible_round_by_seed_pair: dict[tuple[str, str], int] = {}

        for slot in season_slots["Slot"].to_list():
            slot_round = parse_slot_round(slot)
            descendants = set(slot_descendants(slot))

            if slot_round == 2:
                for seed in descendants:
                    opening_slot_by_seed[seed] = parse_opening_slot_index(slot)

            descendant_list = sorted(descendants)
            for left_seed, right_seed in itertools.combinations(descendant_list, 2):
                seed_pair = (left_seed, right_seed)
                current_round = possible_round_by_seed_pair.get(seed_pair)
                if current_round is None or slot_round < current_round:
                    possible_round_by_seed_pair[seed_pair] = slot_round

        season_rows = season_seed_features.rows(named=True)
        for low_seed_row, high_seed_row in itertools.combinations(season_rows, 2):
            low_team_id = int(low_seed_row["TeamID"])
            high_team_id = int(high_seed_row["TeamID"])
            if low_team_id > high_team_id:
                low_seed_row, high_seed_row = high_seed_row, low_seed_row
                low_team_id, high_team_id = high_team_id, low_team_id

            low_seed = str(low_seed_row["Seed"])
            high_seed = str(high_seed_row["Seed"])
            seed_pair = (low_seed, high_seed) if low_seed <= high_seed else (high_seed, low_seed)
            low_opening_slot_index = opening_slot_by_seed.get(low_seed, 0)
            high_opening_slot_index = opening_slot_by_seed.get(high_seed, 0)
            same_region_flag = int(low_seed_row["seed_region"] == high_seed_row["seed_region"])
            possible_round = possible_round_by_seed_pair.get(seed_pair, 0)

            bracket_rows.append(
                {
                    "Season": season,
                    "men_women": int(config.men_women_flag),
                    "LowerTeamID": low_team_id,
                    "HigherTeamID": high_team_id,
                    "low_seed_number": int(low_seed_row["seed_number"]),
                    "high_seed_number": int(high_seed_row["seed_number"]),
                    "delta_seed_number": int(high_seed_row["seed_number"]) - int(low_seed_row["seed_number"]),
                    "low_seed_region_code": int(low_seed_row["seed_region_code"]),
                    "high_seed_region_code": int(high_seed_row["seed_region_code"]),
                    "same_region_flag": same_region_flag,
                    "same_final_four_side_flag": int(
                        low_seed_row["seed_final_four_side_flag"] == high_seed_row["seed_final_four_side_flag"]
                    ),
                    "low_play_in_flag": int(low_seed_row["play_in_flag"]),
                    "high_play_in_flag": int(high_seed_row["play_in_flag"]),
                    "any_play_in_flag": int(low_seed_row["play_in_flag"] or high_seed_row["play_in_flag"]),
                    "both_play_in_flag": int(low_seed_row["play_in_flag"] and high_seed_row["play_in_flag"]),
                    "shared_seed_line_flag": int(
                        low_seed_row["seed_region"] == high_seed_row["seed_region"]
                        and low_seed_row["seed_number"] == high_seed_row["seed_number"]
                    ),
                    "low_opening_slot_index": low_opening_slot_index,
                    "high_opening_slot_index": high_opening_slot_index,
                    "opening_slot_gap": abs(high_opening_slot_index - low_opening_slot_index),
                    "same_first_weekend_pod_flag": int(
                        same_region_flag
                        and low_opening_slot_index > 0
                        and high_opening_slot_index > 0
                        and (low_opening_slot_index - 1) // 2 == (high_opening_slot_index - 1) // 2
                    ),
                    "same_regional_quadrant_flag": int(
                        same_region_flag
                        and low_opening_slot_index > 0
                        and high_opening_slot_index > 0
                        and (low_opening_slot_index - 1) // 4 == (high_opening_slot_index - 1) // 4
                    ),
                    "possible_round": possible_round,
                    "first_four_pair_flag": int(possible_round == 1),
                    "round64_pair_flag": int(possible_round == 2),
                    "round32_pair_flag": int(possible_round == 3),
                    "sweet16_pair_flag": int(possible_round == 4),
                    "elite8_pair_flag": int(possible_round == 5),
                    "final_four_pair_flag": int(possible_round == 6),
                    "title_game_pair_flag": int(possible_round == 7),
                }
            )

    return pl.DataFrame(bracket_rows)


def build_pooled_pair_bracket_features() -> pl.DataFrame:
    return pl.concat(
        [build_pair_bracket_features(config) for config in baseline.DIVISION_CONFIGS],
        how="vertical_relaxed",
    )
