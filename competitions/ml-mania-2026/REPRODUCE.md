# Reproducing on a New Machine

`artifacts/` and `data/` are gitignored. Everything below regenerates them.
Run all commands from this folder (`competitions/ml-mania-2026/`); scripts use paths relative to it.

## 1. Install dependencies

The repo-root `pyproject.toml` does not list everything these scripts import. Install the extras:

```bash
uv pip install polars orjson lightgbm optuna scipy httpx beautifulsoup4 joblib
```

Python 3.13 (see repo-root `.python-version`).

## 2. Download the Kaggle data

Needs `~/.kaggle/kaggle.json` (never commit it).

```bash
kaggle competitions download -c march-machine-learning-mania-2026 -p data
unzip data/march-machine-learning-mania-2026.zip -d data
```

Caveat: the final model was trained on the "refreshed current snapshot" of Kaggle data from mid-March 2026. Kaggle's current download may include later updates, so scores can drift slightly from the numbers in `README.md` / `JOURNAL.md`.

## 3. Rebuild the final base submission

The final script checks itself against a saved benchmark JSON and aborts if missing. Generate that benchmark first:

```bash
python scripts/train_iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search.py
python scripts/generate_final_stage2_current_snapshot_block_rate_submission.py
```

Outputs:
- `artifacts/iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search/validation_summary.json`
- `artifacts/final_stage2_current_snapshot_block_rate/validation_summary.json` (README "Results" table links here)
- `submissions/2026_stage2_current_snapshot_block_rate.csv`

Expected refreshed-data mean Brier: `0.159773` (see README "Results").

## 4. Polymarket overlays: not reproducible

The `generate_*polymarket*` scripts fetch live markets from `gamma-api.polymarket.com` (`closed=false`). Those markets closed after the 2026 tournament, so the scripts cannot reproduce the March 17, 2026 snapshot. The resulting submissions are already committed in `submissions/` and are the source of truth.

## 5. Any other experiment

Each `scripts/train_iteration_*.py` writes its own `artifacts/<script_name>/` folder. Rerun the script to regenerate it. `JOURNAL.md` records the result of each run.
