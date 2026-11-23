from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Final

import polars as pl

MPLCONFIGDIR: Final[str] = "/tmp/ml-mania-2026-mplconfig"

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
JOURNAL_PATH: Final[Path] = REPO_ROOT / "JOURNAL.md"
ARTIFACT_DIR: Final[Path] = REPO_ROOT / "artifacts" / "journal_experiment_history"
CSV_PATH: Final[Path] = ARTIFACT_DIR / "journal_experiment_timeline.csv"
PLOT_PATH: Final[Path] = ARTIFACT_DIR / "journal_validation_brier_timeline.png"

ENTRY_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^### (\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC) - (.+)$",
    re.MULTILINE,
)
MEAN_BRIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^- Mean Brier:\s*(.+)$", re.MULTILINE)
NOTES_PATTERN: Final[re.Pattern[str]] = re.compile(r"^- Notes:\s*(.+)$", re.MULTILINE)
FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d+\.\d+")

KEPT_NOTE_PHRASES: Final[tuple[str, ...]] = (
    "notes: accepted.",
    " it is accepted as the new current best",
    " it is accepted as the new current best model",
    " it is accepted as the current best",
    " becomes the new current best",
    " becomes the new current best model",
    " becomes the new current best official benchmark",
    " becomes the new active benchmark",
)

TITLE_PREFIX_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^Iteration \d+\s*-\s*"),
    re.compile(r"^Pending Feature Family \d+\s*-\s*"),
)
TITLE_SUFFIX_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\s+[Oo]n [Ii]teration \d+\s*$"),
    re.compile(r"\s+[Ff]rom [Ii]teration \d+\s*$"),
)
IMPROVEMENT_LABEL_OVERRIDES: Final[dict[str, str]] = {
    "Baseline Probability Shrink": "baseline shrink",
    "Men Two-Model Ensemble With Calibrated Shrink": "2-model shrink",
    "Pooled Margin Regression With Learned Probability Map": "learned prob map",
    "Bracket Geometry and Seed Structure": "bracket geometry",
    "Bracket-Aware Calibration on the Bracket Feature Model": "bracket calib",
    "Mike Seeded-Team-Focused Quality": "Mike quality",
    "Vilnius Filtered Quality": "Vilnius quality",
    "Add Capped-Margin Quality To Iteration 32": "capped-margin quality",
    "Division-Specific Margin Models": "div-specific models",
    "Conservative Raw-Margin Blend Of Iteration 44 And Iteration 50": "44/50 blend",
    "Fixed Probability Blend Of Iteration 69 And Iteration 71": "69/71 blend",
    "Fixed Division-Specific Probability Blend": "div blend",
    "Fixed Women Probability-Gap Gate": "women gate",
    "Fixed Extended Women Probability-Gap Gate": "women gate+",
    "Fixed Official Joint Men-Women Blend": "joint blend",
    "Fixed Official Joint Men Post-Threshold And Women Tail Blend": "router",
    "Current-season block-rate features": "block-rate features",
}
IMPROVEMENT_LABEL_Y_OFFSETS: Final[tuple[int, ...]] = (8, 14, 20, 10, 16)
PLOT_LABEL_MAX_LENGTH: Final[int] = 28


def parse_entry_timestamp(timestamp_text: str) -> datetime.datetime:
    return datetime.datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M UTC").replace(
        tzinfo=datetime.timezone.utc
    )


def short_change_label(title: str) -> str:
    short_label = title.strip()
    for prefix_pattern in TITLE_PREFIX_PATTERNS:
        short_label = prefix_pattern.sub("", short_label)
    for suffix_pattern in TITLE_SUFFIX_PATTERNS:
        short_label = suffix_pattern.sub("", short_label)
    return short_label


def extract_numeric_brier(mean_brier_text: str) -> float | None:
    numeric_matches = FLOAT_PATTERN.findall(mean_brier_text)
    if not numeric_matches:
        return None
    return float(numeric_matches[0])


def plot_label(short_change_label: str) -> str:
    if short_change_label in IMPROVEMENT_LABEL_OVERRIDES:
        return IMPROVEMENT_LABEL_OVERRIDES[short_change_label]
    if len(short_change_label) <= PLOT_LABEL_MAX_LENGTH:
        return short_change_label
    return short_change_label[: PLOT_LABEL_MAX_LENGTH - 3].rstrip() + "..."


def is_kept_experiment(notes_text: str) -> bool:
    normalized_notes = f"notes: {notes_text.strip().lower()}"
    return any(phrase in normalized_notes for phrase in KEPT_NOTE_PHRASES)


def journal_entries() -> list[dict[str, object]]:
    journal_text = JOURNAL_PATH.read_text()
    heading_matches = list(ENTRY_HEADING_PATTERN.finditer(journal_text))
    extracted_entries: list[dict[str, object]] = []

    for source_order, heading_match in enumerate(heading_matches, start=1):
        entry_timestamp_text = heading_match.group(1)
        entry_title = heading_match.group(2).strip()
        entry_body_start = heading_match.end()
        entry_body_end = (
            heading_matches[source_order].start()
            if source_order < len(heading_matches)
            else len(journal_text)
        )
        entry_body = journal_text[entry_body_start:entry_body_end]

        mean_brier_match = MEAN_BRIER_PATTERN.search(entry_body)
        if mean_brier_match is None:
            continue

        mean_brier_text = mean_brier_match.group(1).strip()
        validation_brier = extract_numeric_brier(mean_brier_text)
        if validation_brier is None:
            continue

        notes_match = NOTES_PATTERN.search(entry_body)
        notes_text = notes_match.group(1).strip() if notes_match is not None else ""

        extracted_entries.append(
            {
                "source_order": source_order,
                "timestamp_text": entry_timestamp_text,
                "timestamp_utc": parse_entry_timestamp(entry_timestamp_text),
                "title": entry_title,
                "short_change_label": short_change_label(entry_title),
                "validation_brier": validation_brier,
                "mean_brier_text": mean_brier_text,
                "notes_text": notes_text,
                "kept": is_kept_experiment(notes_text),
            }
        )

    extracted_entries.sort(key=lambda row: (row["timestamp_utc"], row["source_order"]))

    for experiment_index, row in enumerate(extracted_entries, start=1):
        row["experiment_index"] = experiment_index

    return extracted_entries


def build_experiment_frame(entries: list[dict[str, object]]) -> pl.DataFrame:
    experiment_frame = pl.DataFrame(entries).sort(["timestamp_utc", "source_order"]).with_columns(
        pl.col("experiment_index").cast(pl.Int64),
        pl.col("validation_brier").cast(pl.Float64),
        pl.col("kept").cast(pl.Boolean),
    )
    kept_running_best_brier: list[float | None] = []
    current_kept_best: float | None = None
    for row in experiment_frame.iter_rows(named=True):
        validation_brier = float(row["validation_brier"])
        if bool(row["kept"]):
            if current_kept_best is None or validation_brier < current_kept_best:
                current_kept_best = validation_brier
        kept_running_best_brier.append(current_kept_best)

    return experiment_frame.with_columns(
        pl.col("validation_brier").cum_min().alias("raw_running_best_brier"),
        pl.Series("kept_running_best_brier", kept_running_best_brier, dtype=pl.Float64),
        pl.when(pl.col("kept"))
        .then(pl.lit("kept"))
        .otherwise(pl.lit("discarded"))
        .alias("status"),
    ).select(
        "experiment_index",
        "timestamp_text",
        "title",
        "short_change_label",
        "validation_brier",
        "raw_running_best_brier",
        "kept_running_best_brier",
        "status",
        "kept",
        "mean_brier_text",
        "notes_text",
    )


def save_plot(experiment_frame: pl.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(18, 8.5))
    figure.patch.set_facecolor("#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    axis.grid(axis="y", color="#d7d7d7", linewidth=1.0, alpha=0.8)
    axis.grid(axis="x", color="#e7e7e7", linewidth=0.8, alpha=0.7)

    discarded_frame = experiment_frame.filter(~pl.col("kept"))
    kept_frame = experiment_frame.filter(pl.col("kept"))

    axis.scatter(
        discarded_frame["experiment_index"].to_list(),
        discarded_frame["validation_brier"].to_list(),
        s=18,
        color="#d3d3d3",
        alpha=0.8,
        edgecolors="none",
        label="Discarded experiments",
        zorder=2,
    )
    axis.scatter(
        kept_frame["experiment_index"].to_list(),
        kept_frame["validation_brier"].to_list(),
        s=52,
        color="#2f9e57",
        alpha=0.95,
        edgecolors="white",
        linewidths=0.8,
        label="Accepted experiments",
        zorder=3,
    )
    axis.step(
        experiment_frame["experiment_index"].to_list(),
        experiment_frame["kept_running_best_brier"].to_list(),
        where="post",
        color="#23894b",
        linewidth=2.4,
        label="Running best (accepted)",
        zorder=1,
    )

    for label_index, row in enumerate(kept_frame.iter_rows(named=True)):
        axis.annotate(
            plot_label(str(row["short_change_label"])),
            (float(row["experiment_index"]), float(row["validation_brier"])),
            xytext=(4, IMPROVEMENT_LABEL_Y_OFFSETS[label_index % len(IMPROVEMENT_LABEL_Y_OFFSETS)]),
            textcoords="offset points",
            rotation=28,
            fontsize=8,
            color="#2c8f50",
            alpha=0.95,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=4,
        )

    accepted_count = int(kept_frame.height)
    axis.set_title("Journal Experiment Progress", pad=14, fontsize=18, weight="semibold")
    figure.text(
        0.5,
        0.94,
        (
            f"{experiment_frame.height} experiments, {accepted_count} accepted. "
            "Labels mark accepted experiments; some gray points sit below the green line because they "
            "lowered mean Brier but were rejected by the fold-robustness rule."
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#5a5a5a",
    )
    axis.set_xlabel("Experiment #")
    axis.set_ylabel("Validation Brier")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3f}"))
    axis.margins(x=0.02, y=0.02)
    legend = axis.legend(
        frameon=True,
        loc="upper right",
        fontsize=10,
        title_fontsize=10,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    sns.despine()
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(PLOT_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    entries = journal_entries()
    experiment_frame = build_experiment_frame(entries)
    experiment_frame.write_csv(CSV_PATH)
    save_plot(experiment_frame)

    kept_count = int(experiment_frame.select(pl.col("kept").sum()).item())
    discarded_count = int(experiment_frame.height - kept_count)
    best_brier = float(experiment_frame.select(pl.col("validation_brier").min()).item())

    print(f"Parsed experiments with numeric Brier scores: {experiment_frame.height}")
    print(f"Kept experiments: {kept_count}")
    print(f"Discarded experiments: {discarded_count}")
    print(f"Best validation Brier in chronological series: {best_brier:.6f}")
    print(f"Wrote timeline CSV: {CSV_PATH}")
    print(f"Wrote plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
