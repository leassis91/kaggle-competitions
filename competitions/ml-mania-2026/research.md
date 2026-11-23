# research.md — Deep-Dive Into `ml-mania-2026`

A study of the repository at `competitions/ml-mania-2026/`. The repo is a single-author entry into Kaggle's **March Machine Learning Mania 2026** Stage 2 competition (predict every possible 2026 same-division NCAA matchup, Brier score on the games that actually get played). It is also, deliberately, an experiment in *running an LLM coding agent (Codex) as a semi-autonomous ML researcher*. The README frames the meta-experiment as the main artifact: the modeling solution is secondary; the research loop is the point.

This document captures what the repo is, how it is organized, and how the agent-driven research loop is engineered to behave under strict validation rules. It is the result of reading the top-level docs (`README.md`, `AGENTS.md`, `CHECKLIST.md`, `checklist-prompt.md`, `POLY-ODDS-BLEND.md`, `data-description.md`), the full structure of `scripts/`, key training-pipeline modules, the experiment journal (`JOURNAL.md`, 197 entries / 2566 lines), error-analysis tmp artifacts, and the saved validation artifacts.

---

## 1. What this repository is, at a glance

- **Competition:** Kaggle [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/), Stage 2 only. Predict `Pred = P(LowerTeamID wins)` for every same-division matchup row in `data/SampleSubmissionStage2.csv` (132,133 rows: men + women combined). Only games that actually get played count toward Brier score, but the model must produce probabilities for the full matchup universe.
- **Hard self-imposed rule:** LightGBM only (single or LightGBM-only ensembles). See [AGENTS.md:8](AGENTS.md).
- **Final Kaggle finish:** **249th**, beating ~92% of participants ([docs/final-pos.png](docs/final-pos.png)). Mean refreshed-snapshot validation Brier `0.159773`; final leaderboard Brier `0.1261605`. The Brier gap is mostly competition noise (tournament upsets), not modeling improvement.
- **Author:** [Mario Filho](https://www.kaggle.com/mariofilho), Kaggle Grandmaster. The framing in `README.md` is explicit: this is "A Kaggle Grandmaster Tries to Semi-Automate Himself."
- **Agent involved:** Codex (the file even mentions "GPT 5.4" in the closing notes). All code, journal entries, and most submission-time decisions were produced via prompts to Codex; the human's interventions were almost entirely at the level of prompts, validation rules, and high-level direction.

---

## 2. Repository layout

```
ml-mania-2026/
├── AGENTS.md                       — agent operating contract (constraints, validation rules, style)
├── README.md                       — public-facing writeup + post-mortems for top-3 finishers
├── CHECKLIST.md                    — feature-family backlog (~100 items, all checked off)
├── checklist-prompt.md             — the autonomous prompt that drives the checklist sweep
├── POLY-ODDS-BLEND.md              — design doc for the submission-time Polymarket overlay
├── JOURNAL.md                      — 197 append-only experiment entries (2566 lines)
├── data-description.md             — local copy of Kaggle's data dictionary
├── data/                           — raw Kaggle CSVs (gitignored)
├── artifacts/                      — validation_summary.json + caches per iteration (gitignored, except 1)
├── submissions/                    — final CSVs, including the 5 polymarket overlay variants
├── resources/                      — 6 winning-solution notebooks from previous years (NCAA 2025 1st/3rd/4th, Vilnius, GOTO)
├── scripts/                        — 141 Python files: training iterations + feature builders + drivers
├── docs/                           — final leaderboard screenshot + Brier-over-time plot
└── tmp/                            — error-analysis scratch (markdown plan + parquet + summary)
```

`/data/` and `/artifacts/` are both in [.gitignore](.gitignore). One exception: `artifacts/first_lightgbm_baseline/` (the original baseline) is present, containing two `.joblib` model files, a submission CSV, and `validation_summary.json`. Everything else under `artifacts/` is regenerable from scripts.

There is **no `pyproject.toml`, no `requirements.txt`, no test suite, no Makefile**. The repo is deliberately flat — `AGENTS.md` says "do not invent extra project structure unless the user asks for it." Scripts are run directly. Dependencies inferred from imports: `polars`, `numpy`, `scipy.sparse`, `scikit-learn`, `lightgbm`, `joblib`, `orjson`, `httpx`. The author uses `uvx ruff check` and `uvx ty check` for linting (see early journal entries).

---

## 3. The competition task (concrete shape)

From `data-description.md` and `AGENTS.md`:

- **Submission row:** `ID = Season_LowerTeamID_HigherTeamID`, `Pred ∈ [0, 1]` = P(lower TeamID wins). Men's TeamIDs are 1000–1999, women's 3000–3999, so same-division is enforced by ID range.
- **Cutoff:** Local CSVs are complete through **DayNum 132 (Selection Sunday)**. The competition data is refreshed by Kaggle during submission week; the repo treats the latest pre-deadline refresh as the official data snapshot. There is an explicit late-week `DayNum=132` Massey refresh exception.
- **Scoring:** Brier score = MSE on binary outcomes. Stage 2 rows score 0 until the tournament begins, so the Kaggle leaderboard is useless for model selection.
- **Excluded from scoring surface:** First Four / play-in games. The repo's local validation excludes them too, to mirror the official metric.

---

## 4. The validation protocol (the most important design decision)

This is the load-bearing piece of the whole repo. The agent's freedom is bounded by it, and the agent will absolutely game any weakness in it. From [AGENTS.md:17–31](AGENTS.md) and [README.md:90–98](README.md):

1. **Walk-forward by season.** Validation folds are the latest 5 completed seasons: **2021, 2022, 2023, 2024, 2025**.
2. **Train on prior seasons only.** Each fold trains strictly on seasons before its validation year.
3. **Selection-Sunday-style information cutoff.** Even within a validation season, features can only see what would have been available by that season's Selection Sunday.
4. **Official scoring surface.** First Four rows are excluded to match Kaggle's actual scoring set.
5. **Acceptance rule: 4-of-5 + mean.** A candidate is only accepted if it (a) lowers mean Brier across the 5 folds AND (b) improves at least 4 of the 5 individual folds. Improving 3/5 with a better average is **rejected** as fragile.
6. **No random row splits. No skipping folds.** Speed comes from caching, not from cutting folds.

The README explicitly calls out that this strictness is the human contribution: "Agentic systems are very good at exploiting weaknesses in the objective you give them. If the validation setup is weak, they will eventually find ways to look better without actually becoming more robust."

The 4-of-5 rule is what blocks the agent from accumulating gains that are really one-year flukes. The journal contains many entries where mean Brier improved but the candidate was rejected for not clearing 4-of-5 — and a few where the agent had to be reminded (via the rule) that a beautiful average is not enough.

---

## 5. How the agent loop is engineered

Three documents form the "operating contract" for the agent:

### 5.1 `AGENTS.md` — the constraints layer

Structured as **Task / Constraints / Goal**, with very explicit redundancy on validation. Key prompt patterns the author calls out in `README.md`:

- `"As we get deeper into research, it's expected that we will have many iterations without progress in our metric. This is completely normal. Don't get discouraged..."` — counters the agent's tendency to declare a problem unsolvable after a few failed tries.
- `"Default experiment runs should be validation-only."` and `"Only create a full Stage 2 submission file and persist trained model artifacts when the user explicitly asks for them."` — prevents the agent from saving full model artifacts and submissions per experiment (cost discipline).
- `"Use JOURNAL.md at the repository root to log experiments."` — gives the agent a cross-session memory layer that survives compaction.
- Polars-first, `pathlib`/`orjson`/`httpx`/`joblib`/`seaborn`, no pandas, lazy-frame-preferring style guide.

### 5.2 `checklist-prompt.md` — the autonomous loop

Tiny (18 lines) but carefully tuned. It instructs the agent to walk `CHECKLIST.md` from top to bottom, running each unchecked feature family in turn. The author explains in `README.md` which sentences matter and why:

- `"For every pending feature family (unchecked) test adding at least 10 new features of that family"` and `"It's better to test more features and get no improvement than not testing ideas"` — forces breadth at each item without letting the agent inflate iteration count with trivial variations.
- `"Complete one item/family at a time. Work autonomously and only stop when you have finished the list."` — gives a hard, observable stop condition. The author found that "stop after N iterations" prompts caused the agent to fake iterations.
- `"Update the checklist and journal as you complete each item, before moving to the next one"` — write-as-you-go memory, not at-the-end memory.
- `"Treat partial progress as non-terminal. Never end the turn just to report status."` — added after the author asked the agent why it stopped early; the agent's own vocabulary ("partial progress", "non-terminal") was reused to bypass that failure mode through the embedding space.

### 5.3 `CHECKLIST.md` — the work backlog

100 feature families, every one checked off ([CHECKLIST.md:1-99](CHECKLIST.md)). Roughly: prior-season summaries → momentum/Elo → calibration → pooled Selection Sunday → split-context family (neutral-site, home-road, rest, fatigue, OT, close-game, blowout, volatility, common-opponent…) → conference tournament features → coach/program history → detailed-stat (4-factors per shot type) → opponent-tier records → bracket priors (slot upset rate, path chalkiness, regional structure) → Massey rank-trend.

In the journal, almost all of these end up **rejected** by the 4-of-5 rule. The accepted families are a small minority — and `README.md` is explicit that this is the desired outcome.

### 5.4 What "memory" looks like for the agent

`JOURNAL.md` is the durable cross-session memory. Each entry uses the same template:

```
### YYYY-MM-DD HH:MM UTC — short experiment name
- Hypothesis:
- Data window:
- Validation folds:
- Feature set:
- LightGBM config:
- Mean Brier:
- Fold scores:
- Artifacts:
- Notes:
- Next step:
```

That `Next step:` line is doing real work — it functions as the prompt the *next* session/agent reads on resume. The journal becomes both a postmortem record and a stateful instruction queue. With 197 entries spanning ~10 days of work, this is the substrate that lets the agent recover after Kaggle data refreshes, after context compaction, and after the human steps out.

---

## 6. The final modeling solution

Even though the README declares it secondary, the architecture is interesting because it shows what survives a strict 4-of-5 validation regime. From `README.md` and verified against [scripts/generate_final_stage2_current_snapshot_block_rate_submission.py](scripts/generate_final_stage2_current_snapshot_block_rate_submission.py):

### 6.1 Features

Two parallel feature stacks, both built off a pooled (men+women) Selection-Sunday-cutoff matchup table where each row is `LowerTeamID` vs `HigherTeamID`:

- **`iteration_38` stack** — pooled bracket-aware Selection Sunday model: current-season box-score averages (Score/FGA/OR/DR/Blk/PF + opp FGA/Blk/PF + avg point diff), seed values + delta_seed + fallback default seed for unseeded teams, bracket geometry (`possible_round`, `same_region_flag`, `same_final_four_side_flag`, `any_play_in_flag`, `opening_slot_gap`, regional quadrant), bracket × seed interactions, `mike_quality`, `vilnius_quality`, `capped_margin_quality`.
- **`iteration_49` stack** — extends iteration_38 with `seed_closure_quality` (a two-hop seeded-closure variant of the latent quality solve).
- Both stacks are then layered with the final accepted family: **current-season block-rate features** (`block_feature_overall`, `block_feature_late`, `block_feature_neutral`, `block_feature_home_road_gap`, each as low/high/delta).

The `*_quality` features are season-by-season ridge-regularized team strength estimates solved by sparse `lsqr` on a sparse design matrix where each game contributes a `+1/-1` row for winning/losing TeamID with the score margin as target (see [scripts/pooled_quality_variants.py](scripts/pooled_quality_variants.py)). Variants differ in:
- margin cap (`mike_quality`: none, `capped_margin_quality`: clip ±15, etc.)
- relevance hops from seeded teams (1 vs 2)
- whether unseeded teams are dumped into a shared `QUALITY_BUCKET_TEAM_ID=0` aggregate
- ridge_lambda
- optional `DayNum`-recency weighting from 0.5 → 1.5

### 6.2 Model families (LightGBM-only, as required)

- **`iteration_38` regressor:** pooled `LGBMRegressor` on the iteration_38 feature stack, predicting margin (lower – higher score), not probability.
- **`iteration_41` regressor:** same data, Optuna-tuned params (`learning_rate=0.0117`, `num_leaves=104`, `min_child_samples=102`, ridge `reg_lambda=3.64`, `min_split_gain=0.25`, `n_estimators=400` frozen).
- **`iteration_44` composition:** division-specific — men use iteration_38, women use iteration_41. This was the key acceptance: it captured the women-side gains from tuning without giving back the 2023 men regression.
- **`iteration_49` branch:** iteration_44 architecture on the iteration_49 (seed-closure) feature stack.
- **`iteration_53` benchmark blend:** raw-margin blend of iteration_44 (0.75) + iteration_49 (0.25) before calibration.

### 6.3 Calibration — nested out-of-fold logistic on predicted margin

All source models predict margin first. Margin → probability is mapped by a **nested OOF logistic calibrator**: for each outer validation season, the calibrator is fit only on rows from seasons strictly before that season, and within those seasons via inner walk-forward folds. Two calibrated probability views are built:

- `iteration_69`: raw margin = `0.955*benchmark + 0.020*iteration_38 + 0.015*iteration_49 + 0.010*iteration_41`, then bracket-aware calibration.
- `iteration_71`: raw margin = `0.965*benchmark + 0.020*iteration_38 + 0.015*iteration_49`, then division-aware calibration.

### 6.4 Final ensemble — probability-gap router

Per-row, define `gap = |p_71 − p_69|`, then route based on division and gap regime (concrete weights in [scripts/official_current_block_rate_benchmark.py](scripts/official_current_block_rate_benchmark.py:121-164)):

**Men:**
- `gap ≤ 0.040`: 44% iter71 + 56% iter69
- `gap ≤ 0.0425`: 34% iter71 + 66% iter69
- `gap > 0.0425`: 38% iter71 + 62% iter69

**Women:**
- `gap ≤ 0.054`: 100% iter71
- `gap ≤ 0.065`: 100% iter69
- `gap ≤ 0.080`: 100% iter69
- `gap > 0.080`: 24% iter71 + 76% iter69

That router was the accepted benchmark before the final feature sweep; the last accepted improvement was layering the block-rate family on top of the fixed router. The final archived submission is best described as **block-rate-enhanced + fixed routed blend over nested-calibrated probabilities over division-specific Optuna-tuned margin regressors on bracket-aware pooled Selection-Sunday features**.

### 6.5 Final-submission reproducibility guarantee

[scripts/generate_final_stage2_current_snapshot_block_rate_submission.py](scripts/generate_final_stage2_current_snapshot_block_rate_submission.py) reproduces validation first and aborts if the saved benchmark Brier values don't match to `1e-12`. This is critical when the agent runs the final pipeline after a Kaggle data refresh.

---

## 7. The Polymarket overlay (submission-time, unvalidated)

`POLY-ODDS-BLEND.md` is the design doc for this; it sits outside the validated LightGBM stack because there is no historical Polymarket data to validate against.

Three overlays were applied or considered in the final hours:

1. **Direct-game logit blend (conservative).** For rows that map cleanly to a Polymarket head-to-head moneyline market, blend `logit(p_blend) = 0.90*logit(p_model) + 0.10*logit(p_market)`, then cap the move at ±0.05 from the model and clip the result to `[0.02, 0.98]`. Quality gates require: head-to-head, unambiguous team mapping, oriented to the lower-TeamID side, open & liquid market. Touched ~28 rows.
2. **Title-futures overlay (conservative).** Pull Polymarket's `2026-ncaa-tournament-winner` board, normalize to fair title odds, run an exact bracket DP on men's seeded-field matchups to infer model-implied title odds, convert team-level logit gaps to small offsets with `alpha=0.10`, cap team-offset ±0.10 and per-game move ±0.03. Applied to ~2278 seeded men rows.
3. **Direct-game hard replacement (aggressive).** A late-game decision: where Polymarket had a live, uniquely mappable head-to-head with `volume > 10000`, **replace the model probability outright with the market midpoint**. Touched 31 rows, mean absolute move 0.055, max 0.177. Combined with overlay (2) on the remaining seeded men rows, this is the candidate submission that was actually shipped (or at least preserved as the most aggressive variant).

Five submission CSVs are preserved in [submissions/](submissions/):
- `2026_stage2_current_snapshot_block_rate.csv` — pure model base
- `..._polymarket_quintile_blend_2026-03-17.csv` — conservative direct blend
- `..._polymarket_quintile_blend_title_conservative_2026-03-17.csv` — adds title overlay
- `..._polymarket_direct_replace_volume_gt_10k.csv` — aggressive replace
- `..._polymarket_direct_replace_volume_gt_10k_plus_title_conservative.csv` — aggressive replace + title overlay

The README is candid: "One or two games with a slightly better prediction can be the difference between 100s of places... I don't believe our model is better than the market-aggregated info."

---

## 8. The journal as a research dataset

`JOURNAL.md` is the single most interesting artifact for understanding how the loop actually behaved. 197 entries, broadly grouped:

| Phase | Iterations | What happened |
|---|---|---|
| Prior-season baseline & calibration | iter 1–15 | Established the original baseline; momentum/Elo/efficiency families mostly rejected; probability shrink (`0.70` men, `0.80` women) accepted; men 2-model ensemble accepted; nested OOF isotonic rejected (fixed shrink wins). |
| Pooled Selection-Sunday pivot | iter 16–22 | Moved from prior-season-only to same-season Selection-Sunday data; first pooled baselines failed; the breakthrough was **iter 21**: pooled margin regression + learned probability map, first to clear 4-of-5. |
| Notebook ports | iter 23–40 | Translations from the resources/ notebooks (GOTO, Mike Kim, Vilnius). Most failed; iters 24, 26 (bracket features/calibration), 28 (Mike filtered quality), 32 (Vilnius unregularized quality), 38 (capped-margin quality on top of Vilnius) were the accepted ones. |
| Optuna + composition | iter 41–53 | Optuna search → tuned regressor (iter 41); division-specific composition (iter 44 = men_baseline + women_tuned) accepted; alternative seed-closure-quality branch (iter 49) explored; raw-margin blend of iter 44 + iter 50 = iter 53 accepted. |
| Calibrated probability blends & routers | iter 54–115 | Search-heavy. Probability-gap-routed blends of iter 69 / iter 71 → iter 105 (joint men/women gap); iter 110–111 fixed the official router config used in the shipped submission. |
| Pending-feature-family sweep | iter 116 family sweep | Walked the remaining ~50–80 items in `CHECKLIST.md`. Only **current-season block-rate features** survived. Long tail of rejected ideas (coach tenure, conference tournament path difficulty, secondary tournament participation streaks, Massey bucket shares, etc.). |
| Error analysis + late stabilization | iter 117–120 | Tried to use error analysis to find missing structure (men upset bands, low-gap mid-confidence games). The nested rule search at iter 120 chose `no_rule` on every fold — meaning even with diagnostic guidance, no learned override beat the existing router under walk-forward validation. |
| Sanity check + Polymarket overlays | 2026-03-16 → 2026-03-19 | Seed-baseline correlation QA (sub sits comfortably inside historical validation range); three Polymarket overlay submissions produced. |

The plot at [docs/journal_validation_brier_timeline.png](docs/journal_validation_brier_timeline.png) is generated from this journal by [scripts/plot_journal_experiment_history.py](scripts/plot_journal_experiment_history.py), which regex-parses entries and tags accepted ones via phrases like `"it is accepted as the new current best"`. The journal entries thus serve as both human-readable log *and* structured data.

### 8.1 Recurring patterns in the journal

- **Most ideas fail.** Of ~197 logged experiments, only ~15–20 were accepted. The 4-of-5 rule is doing the filtering.
- **Year 2023 is a recurring blocker.** Many candidates regress 2023 (a high-upset year). Several accepted iterations explicitly trade 2024 for 2023 stability or vice versa.
- **Pooled > separate for the *backbone*; division-specific for the *tuning*.** Pooled training + division-aware calibration + division-specific routing won.
- **Composition > raw features after iter 40.** Once the feature stack stabilized, gains came from blending source models and routing by probability-gap, not from new features. The pending-feature-family sweep at iter 116 confirms this — 80+ families tested, ~1 accepted.
- **Calibration discipline is consistent.** All learned calibrators (logistic, isotonic) are fit nested-OOF on prior-season-only data. The Stage 2 final-train and final-predict pass uses the exact same calibrator architecture, just with the target season swapped in.

---

## 9. Specificities and gotchas worth knowing

- **`men_women` flag convention:** men = 1, women = 0 (see [scripts/train_iteration_16_pooled_selection_sunday_baseline.py:55-57](scripts/train_iteration_16_pooled_selection_sunday_baseline.py)). Used as both a grouping key and a feature.
- **Overtime adjustment:** All current-season detailed averages divide raw counts by `(40 + 5*NumOT)/40` to normalize per-40 minutes before averaging.
- **`LOW_HISTORY_DEFAULT_SEED = 17.0`** is the fallback seed used for any unseeded team in submission rows — submission must cover *every* matchup, not just the tournament field, so unseeded teams need a sensible default.
- **`MIN_POOLED_SEASON = 2010`** — pooled training restricts to seasons ≥ 2010 because women's detailed box scores start in 2010.
- **Two parallel feature stacks (iter_38 + iter_49)** are maintained in parallel through the entire pipeline because both feed into the ensemble. The driver in [scripts/official_feature_family_driver.py](scripts/official_feature_family_driver.py) builds both bundles in one pass.
- **Fold-input caching:** [scripts/official_current_block_rate_benchmark.py](scripts/official_current_block_rate_benchmark.py) caches the inner fold inputs to `joblib`. Re-running the validation hits the cache; rebuilding takes meaningfully longer.
- **The `1e-12` reproducibility check** in the submission script is a real guard: it caught a data-refresh discrepancy mid-week, leading to the explicit benchmark reset noted in the journal at 2026-03-16 18:58 UTC.
- **`AGENTS.md` deliberately blocks shortcuts:** "Do not skip folds to iterate faster. Speed should come from caching and efficient feature generation, not from partial validation." The author treats this as a top-line rule.
- **Resources are inputs, not blueprints:** Six previous-year winning notebooks in `resources/`. Most direct ports failed (`team_quality`, isotonic, recency-weighting); the notebooks were useful for *ideas* like bracket geometry, filtered seeded-team quality, capped margins.

---

## 10. Honest read on what this is

This repo is **two artifacts in one shell**:

1. **A modeling solution** that finished 249th — respectable, beaten by 1st place mostly because of men's-side current-state information (injuries, reranking) and by 3rd place because of heavy market-blended overlays, not because of inferior offline Brier (winner CV `0.1590`, this repo's CV `0.159473–0.159773`).
2. **A demonstration of a disciplined LLM-driven research loop:** explicit `Task / Constraints / Goal` prompt structure, an append-only journal as cross-session memory, an autonomous checklist sweep with a clean stop condition, a strict statistical acceptance rule that prevents the agent from exploiting validation weaknesses, and an explicit separation between validated model selection and submission-time overlays.

`README.md` is unambiguous about which one matters going forward: the loop, not the model. The most reusable assets in this repo are not the LightGBM scripts — they are `AGENTS.md`, `checklist-prompt.md`, and the journal template. Together they constitute a small but tested template for letting a coding agent run a multi-day ML competition autonomously without (a) drifting into mass-iteration overfitting, (b) accepting fragile gains, (c) losing state across sessions, or (d) confusing submission-time tricks with validated modeling progress.

The post-mortems in `README.md` (vs. 1st/2nd/3rd place) are themselves LLM-written ("by GPT 5.4 XHigh"), which is a clean final detail: the agent that ran the experiment also writes the retrospective on its own work. The conclusion the agent draws — that the biggest missing piece was not model complexity but a cleaner three-layer architecture (validated benchmark + men's current-state reranker + auditable market overlay) — is a coherent and self-aware diagnosis, and worth taking forward as the prompt-engineering lesson rather than just an ML one.
