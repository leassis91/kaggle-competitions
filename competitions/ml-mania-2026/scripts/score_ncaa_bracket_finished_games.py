from __future__ import annotations

import argparse
import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
import orjson
import polars as pl
from bs4 import BeautifulSoup, Tag


SEASON = 2026
DEFAULT_BRACKET_URL_BY_DIVISION = {
    "men": "https://www.ncaa.com/march-madness-live/bracket",
    "women": "https://www.ncaa.com/brackets/basketball-women/d1/2026",
}
DEFAULT_DIVISION = "combined"
DEFAULT_SUBMISSION_PATH = Path(
    "submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k_plus_title_conservative.csv"
)
ARTIFACT_DIR = Path("artifacts/live_ncaa_bracket_brier")
DIVISION_PREFIX_BY_NAME = {
    "men": "M",
    "women": "W",
}


@dataclass(frozen=True)
class ParsedTeamResult:
    name: str
    score: int
    seed: int | None
    is_loser: bool


@dataclass(frozen=True)
class ParsedGameResult:
    game_id: int
    is_play_in: bool
    teams: tuple[ParsedTeamResult, ParsedTeamResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a Stage 2 submission against finished NCAA bracket games from the NCAA live bracket page."
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="Path to the Stage 2 submission CSV to score.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Bracket page URL to fetch. Only valid for single-division runs.",
    )
    parser.add_argument(
        "--division",
        choices=["combined", *sorted(DIVISION_PREFIX_BY_NAME)],
        default=DEFAULT_DIVISION,
        help="Division lookup to use when resolving team names.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def fetch_bracket_html(url: str) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return response.text


def parse_optional_int(value: str) -> int | None:
    stripped_value = value.strip()
    if not stripped_value:
        return None
    return int(stripped_value)


def parse_team_result(team_node: Tag) -> ParsedTeamResult | None:
    seed_node = team_node.select_one("span.overline")
    body_nodes = team_node.select("p.body")
    if seed_node is None or len(body_nodes) < 2:
        raise ValueError("Unexpected team markup while parsing finished bracket results.")

    score_text = body_nodes[1].get_text(strip=True)
    if not score_text:
        return None

    class_attribute = team_node.get("class")
    if class_attribute is None:
        classes: set[str] = set()
    elif isinstance(class_attribute, str):
        classes = {class_attribute}
    else:
        classes = {str(class_name) for class_name in class_attribute}
    return ParsedTeamResult(
        name=body_nodes[0].get_text(strip=True),
        score=int(score_text),
        seed=parse_optional_int(seed_node.get_text(strip=True)),
        is_loser="team-loser" in classes,
    )


def extract_finished_games_from_dom(html: str) -> list[ParsedGameResult]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_games: list[ParsedGameResult] = []

    for anchor_node in cast(list[Tag], soup.select("a.game-pod.final")):
        game_id_value = anchor_node.get("id")
        if not isinstance(game_id_value, str):
            raise ValueError("Finished game anchor was missing a string id attribute.")

        game_id = int(game_id_value)
        team_nodes = cast(list[Tag], anchor_node.select("div.team"))
        if len(team_nodes) != 2:
            raise ValueError(f"Expected 2 teams for finished game {game_id}, found {len(team_nodes)}.")

        first_team = parse_team_result(team_nodes[0])
        second_team = parse_team_result(team_nodes[1])
        if first_team is None or second_team is None:
            continue

        parsed_games.append(
            ParsedGameResult(
                game_id=game_id,
                is_play_in=anchor_node.find_parent(class_="first-four") is not None,
                teams=(first_team, second_team),
            )
        )

    return parsed_games


def decode_nuxt_reference_payload(payload_text: str) -> object:
    payload = cast(list[object], orjson.loads(payload_text))
    decoded_cache: dict[int, object | None] = {}

    def decode_reference(value: object) -> object:
        if isinstance(value, int):
            return decode_entry(value)
        if isinstance(value, dict):
            return {key: decode_reference(item) for key, item in value.items()}
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                tag = value[0]
                if tag in {"Ref", "Reactive", "ShallowReactive"}:
                    return decode_entry(cast(int, value[1]))
                if tag == "Set":
                    decoded_set_items = decode_entry(cast(int, value[1]))
                    if isinstance(decoded_set_items, list):
                        return set(decoded_set_items)
                    return decoded_set_items
            return [decode_reference(item) for item in value]
        return value

    def decode_entry(index: int) -> object:
        if index in decoded_cache:
            return decoded_cache[index]

        raw_value = payload[index]
        decoded_cache[index] = None
        if isinstance(raw_value, (dict, list)):
            decoded_value = decode_reference(raw_value)
        else:
            decoded_value = raw_value
        decoded_cache[index] = decoded_value
        return decoded_value

    main_state_index = next(
        (index for index, value in enumerate(payload) if isinstance(value, dict) and "main" in value),
        None,
    )
    if main_state_index is None:
        raise ValueError("Could not find the Nuxt main state in the bracket payload.")

    return decode_entry(main_state_index)


def parse_nuxt_team_result(team_payload: dict[str, object]) -> ParsedTeamResult:
    score = cast(int | None, team_payload.get("score"))
    if score is None:
        raise ValueError("Nuxt team payload was missing a final score.")

    seed = cast(int | None, team_payload.get("seed"))
    is_winner = bool(team_payload.get("isWinner", False))
    return ParsedTeamResult(
        name=str(team_payload["nameShort"]),
        score=int(score),
        seed=int(seed) if seed is not None else None,
        is_loser=not is_winner,
    )


def extract_finished_games_from_nuxt(html: str) -> list[ParsedGameResult]:
    soup = BeautifulSoup(html, "html.parser")
    script_node = soup.select_one("script#__NUXT_DATA__")
    if script_node is None:
        return []

    decoded_root = decode_nuxt_reference_payload(script_node.get_text())
    if not isinstance(decoded_root, dict):
        raise ValueError("Unexpected Nuxt root payload type while parsing bracket results.")
    decoded_root_dict = cast(dict[str, object], decoded_root)

    main_state = decoded_root_dict.get("main")
    if not isinstance(main_state, dict):
        raise ValueError("Unexpected Nuxt main state while parsing bracket results.")
    main_state_dict = cast(dict[str, object], main_state)

    championship = main_state_dict.get("championship")
    if not isinstance(championship, dict):
        raise ValueError("Unexpected Nuxt championship payload while parsing bracket results.")
    championship_dict = cast(dict[str, object], championship)

    regions = championship_dict.get("regions")
    games = championship_dict.get("games")
    if not isinstance(regions, list) or not isinstance(games, list):
        raise ValueError("Nuxt championship payload was missing regions or games.")

    play_in_section_ids: set[int] = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_dict = cast(dict[str, object], region)
        if str(region_dict.get("regionCode", "")) == "TT":
            play_in_section_ids.add(int(cast(int, region_dict["sectionId"])))

    parsed_games: list[ParsedGameResult] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        game_dict = cast(dict[str, object], game)
        if str(game_dict.get("gameState", "")) != "F":
            continue

        teams = game_dict.get("teams")
        if not isinstance(teams, list) or len(teams) != 2:
            raise ValueError("Nuxt finished game payload did not contain exactly 2 teams.")
        if not all(isinstance(team, dict) for team in teams):
            raise ValueError("Nuxt finished game teams payload had an unexpected type.")

        game_id = int(cast(int, game_dict["bracketId"]))
        section_id = int(cast(int, game_dict["sectionId"]))
        parsed_games.append(
            ParsedGameResult(
                game_id=game_id,
                is_play_in=section_id in play_in_section_ids,
                teams=(
                    parse_nuxt_team_result(cast(dict[str, object], teams[0])),
                    parse_nuxt_team_result(cast(dict[str, object], teams[1])),
                ),
            )
        )

    return parsed_games


def extract_finished_games(html: str) -> list[ParsedGameResult]:
    dom_games = extract_finished_games_from_dom(html)
    if dom_games:
        return dom_games

    nuxt_games = extract_finished_games_from_nuxt(html)
    if nuxt_games:
        return nuxt_games

    raise ValueError("No finished games were found on the bracket page.")


def build_team_spelling_rows(division: str) -> list[tuple[str, int]]:
    division_prefix = DIVISION_PREFIX_BY_NAME[division]
    team_rows = pl.read_csv(Path("data") / f"{division_prefix}Teams.csv").select("TeamID", "TeamName")
    spelling_rows = pl.read_csv(Path("data") / f"{division_prefix}TeamSpellings.csv").select(
        "TeamID",
        "TeamNameSpelling",
    )

    normalized_rows: set[tuple[str, int]] = set()
    for team_id, team_name in team_rows.iter_rows():
        normalized_rows.add((normalize_name(str(team_name)), int(team_id)))
    for team_id, team_name_spelling in spelling_rows.iter_rows():
        normalized_rows.add((normalize_name(str(team_name_spelling)), int(team_id)))

    return sorted(
        [row for row in normalized_rows if row[0]],
        key=lambda row: (len(row[0]), row[0], row[1]),
        reverse=True,
    )


def build_team_name_by_id(division: str) -> dict[int, str]:
    division_prefix = DIVISION_PREFIX_BY_NAME[division]
    team_rows = pl.read_csv(Path("data") / f"{division_prefix}Teams.csv").select("TeamID", "TeamName")
    return {int(team_id): str(team_name) for team_id, team_name in team_rows.iter_rows()}


def resolve_team_id(team_name: str, team_spelling_rows: list[tuple[str, int]]) -> int:
    normalized_team_name = normalize_name(team_name)

    exact_matches = {team_id for spelling, team_id in team_spelling_rows if spelling == normalized_team_name}
    if len(exact_matches) == 1:
        return next(iter(exact_matches))
    if len(exact_matches) > 1:
        raise ValueError(f"Ambiguous exact match while resolving team name {team_name!r}.")

    contained_rows = [
        (spelling, team_id)
        for spelling, team_id in team_spelling_rows
        if f" {spelling} " in f" {normalized_team_name} " or f" {normalized_team_name} " in f" {spelling} "
    ]
    if contained_rows:
        longest_spelling_length = max(len(spelling) for spelling, _team_id in contained_rows)
        longest_spelling_matches = {
            team_id for spelling, team_id in contained_rows if len(spelling) == longest_spelling_length
        }
        if len(longest_spelling_matches) == 1:
            return next(iter(longest_spelling_matches))
        raise ValueError(f"Ambiguous contained match while resolving team name {team_name!r}.")

    compact_normalized_team_name = normalized_team_name.replace(" ", "")
    compact_matches = {
        team_id for spelling, team_id in team_spelling_rows if spelling.replace(" ", "") == compact_normalized_team_name
    }
    if len(compact_matches) == 1:
        return next(iter(compact_matches))
    if len(compact_matches) > 1:
        raise ValueError(f"Ambiguous compact match while resolving team name {team_name!r}.")

    raise ValueError(f"Could not resolve bracket team name {team_name!r}.")


def load_submission_predictions(submission_path: Path) -> dict[tuple[int, int], float]:
    submission_frame = pl.read_csv(submission_path).with_columns(pl.col("ID").str.split_exact("_", 2).alias("id_parts"))
    parsed_submission_frame = submission_frame.select(
        pl.col("Pred"),
        pl.col("id_parts").struct.field("field_1").cast(pl.Int64).alias("LowerTeamID"),
        pl.col("id_parts").struct.field("field_2").cast(pl.Int64).alias("HigherTeamID"),
    )

    return {
        (int(lower_team_id), int(higher_team_id)): float(prediction)
        for prediction, lower_team_id, higher_team_id in parsed_submission_frame.iter_rows()
    }


def validate_game_outcome(parsed_game: ParsedGameResult) -> None:
    first_team, second_team = parsed_game.teams
    if first_team.score == second_team.score:
        raise ValueError(f"Finished game {parsed_game.game_id} ended in a tie.")

    loser_count = int(first_team.is_loser) + int(second_team.is_loser)
    if loser_count == 1:
        if first_team.is_loser and first_team.score > second_team.score:
            raise ValueError(f"Loser class conflicted with score for game {parsed_game.game_id}.")
        if second_team.is_loser and second_team.score > first_team.score:
            raise ValueError(f"Loser class conflicted with score for game {parsed_game.game_id}.")


def build_finished_game_report(
    parsed_games: list[ParsedGameResult],
    division: str,
    prediction_by_pair: dict[tuple[int, int], float],
) -> pl.DataFrame:
    team_spelling_rows = build_team_spelling_rows(division)
    team_name_by_id = build_team_name_by_id(division)
    report_rows: list[dict[str, object]] = []

    for parsed_game in parsed_games:
        validate_game_outcome(parsed_game)
        first_team, second_team = parsed_game.teams
        first_team_id = resolve_team_id(first_team.name, team_spelling_rows)
        second_team_id = resolve_team_id(second_team.name, team_spelling_rows)
        if first_team_id == second_team_id:
            raise ValueError(f"Resolved the same team twice for finished game {parsed_game.game_id}.")

        lower_team_id, higher_team_id = sorted((first_team_id, second_team_id))
        lower_team_result = first_team if first_team_id == lower_team_id else second_team
        higher_team_result = second_team if first_team_id == lower_team_id else first_team
        lower_team_won = lower_team_result.score > higher_team_result.score

        prediction = prediction_by_pair.get((lower_team_id, higher_team_id))
        if prediction is None:
            raise ValueError(
                f"Finished game {parsed_game.game_id} resolved to pair {(lower_team_id, higher_team_id)} "
                "that was not present in the submission."
            )

        report_rows.append(
            {
                "Season": SEASON,
                "game_id": parsed_game.game_id,
                "is_play_in": parsed_game.is_play_in,
                "LowerTeamID": lower_team_id,
                "HigherTeamID": higher_team_id,
                "lower_team_name": team_name_by_id[lower_team_id],
                "higher_team_name": team_name_by_id[higher_team_id],
                "lower_team_score": lower_team_result.score,
                "higher_team_score": higher_team_result.score,
                "lower_team_seed": lower_team_result.seed,
                "higher_team_seed": higher_team_result.seed,
                "Target": float(lower_team_won),
                "Pred": prediction,
                "error_sq": (prediction - float(lower_team_won)) ** 2,
            }
        )

    report_frame = pl.DataFrame(report_rows).sort(["is_play_in", "game_id"])
    if report_frame.is_empty():
        raise ValueError("No finished games were available after team resolution.")
    return report_frame


def build_summary(
    report_frame: pl.DataFrame,
    submission_path: Path,
    url: str,
    division: str,
) -> dict[str, object]:
    official_frame = report_frame.filter(~pl.col("is_play_in"))

    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "submission_path": str(submission_path.resolve()),
        "source_url": url,
        "division": division,
        "finished_game_count": report_frame.height,
        "play_in_finished_game_count": int(report_frame.get_column("is_play_in").sum()),
        "official_finished_game_count": official_frame.height,
        "all_finished_brier": float(report_frame.select(pl.col("error_sq").mean()).item()),
        "official_brier_excluding_play_in": float(official_frame.select(pl.col("error_sq").mean()).item())
        if official_frame.height
        else None,
    }


def write_outputs(report_frame: pl.DataFrame, summary: dict[str, object], division: str) -> tuple[Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{division}_finished_games_report.csv"
    summary_path = ARTIFACT_DIR / f"{division}_finished_games_summary.json"
    report_frame.write_csv(report_path)
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return report_path, summary_path


def score_single_division(
    submission_path: Path,
    division: str,
    url: str,
) -> tuple[pl.DataFrame, dict[str, object], Path, Path]:
    bracket_html = fetch_bracket_html(url)
    parsed_games = extract_finished_games(bracket_html)
    prediction_by_pair = load_submission_predictions(submission_path)
    report_frame = build_finished_game_report(parsed_games, division, prediction_by_pair)
    summary = build_summary(report_frame, submission_path, url, division)
    report_path, summary_path = write_outputs(report_frame, summary, division)
    return report_frame, summary, report_path, summary_path


def print_summary(summary: dict[str, object], report_path: Path, summary_path: Path) -> None:
    print(f"Submission: {summary['submission_path']}")
    print(f"Source URL: {summary['source_url']}")
    print(f"Division: {summary['division']}")
    print(f"Finished games found: {summary['finished_game_count']}")
    print(f"Official finished games excluding play-ins: {summary['official_finished_game_count']}")
    print(f"All finished games Brier: {summary['all_finished_brier']:.6f}")
    official_brier = summary["official_brier_excluding_play_in"]
    if official_brier is None:
        print("Official finished games Brier excluding play-ins: unavailable")
    else:
        print(f"Official finished games Brier excluding play-ins: {official_brier:.6f}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")


def build_combined_summary(
    combined_report_frame: pl.DataFrame,
    submission_path: Path,
    division_summaries: dict[str, dict[str, object]],
) -> dict[str, object]:
    official_frame = combined_report_frame.filter(~pl.col("is_play_in"))
    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "submission_path": str(submission_path.resolve()),
        "source_url": {
            division: summary["source_url"] for division, summary in division_summaries.items()
        },
        "division": "combined",
        "finished_game_count": combined_report_frame.height,
        "play_in_finished_game_count": int(combined_report_frame.get_column("is_play_in").sum()),
        "official_finished_game_count": official_frame.height,
        "all_finished_brier": float(combined_report_frame.select(pl.col("error_sq").mean()).item()),
        "official_brier_excluding_play_in": float(official_frame.select(pl.col("error_sq").mean()).item())
        if official_frame.height
        else None,
        "division_summaries": division_summaries,
    }


def main() -> None:
    args = parse_args()
    if args.division == "combined":
        if args.url is not None:
            raise ValueError("--url is only supported with --division men or --division women.")

        division_report_frames: list[pl.DataFrame] = []
        division_summaries: dict[str, dict[str, object]] = {}
        for division in sorted(DIVISION_PREFIX_BY_NAME):
            report_frame, summary, report_path, summary_path = score_single_division(
                args.submission_path,
                division,
                DEFAULT_BRACKET_URL_BY_DIVISION[division],
            )
            division_report_frames.append(report_frame.with_columns(pl.lit(division).alias("division")))
            division_summaries[division] = summary
            print_summary(summary, report_path, summary_path)
            print()

        combined_report_frame = pl.concat(division_report_frames, how="vertical_relaxed").sort(
            ["is_play_in", "division", "game_id"]
        )
        combined_summary = build_combined_summary(combined_report_frame, args.submission_path, division_summaries)
        combined_report_path, combined_summary_path = write_outputs(combined_report_frame, combined_summary, "combined")
        print_summary(combined_summary, combined_report_path, combined_summary_path)
        return

    division = str(args.division)
    url = str(args.url) if args.url is not None else DEFAULT_BRACKET_URL_BY_DIVISION[division]
    report_frame, summary, report_path, summary_path = score_single_division(args.submission_path, division, url)
    print_summary(summary, report_path, summary_path)


if __name__ == "__main__":
    main()
