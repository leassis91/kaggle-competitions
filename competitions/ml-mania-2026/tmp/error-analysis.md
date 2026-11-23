# Error Analysis Plan

## Goal

Find the data slices where the current active benchmark underperforms, using an inner train/validation split that stays fully inside each outer walk-forward fold.

## Active Benchmark To Analyze

- Treat the active model as:
  - the fixed official router from `scripts/train_iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend.py`
  - plus the accepted `Current-season block-rate features` family from `scripts/train_iteration_116_pending_feature_family_sweep.py`
- Do not analyze bare `iteration_111` alone, because the journal says the accepted block-rate family is the current active benchmark.

## Step 1: Rebuild The True Active Feature Tables

1. Start from the official benchmark tables exposed through `official_feature_family_driver.py`.
2. Rebuild the `iteration_38` and `iteration_49` matchup bundles.
3. Extend both bundles with the accepted block-rate family:
   - `block_feature_overall`
   - `block_feature_late`
   - `block_feature_neutral`
   - `block_feature_home_road_gap`
4. Keep the existing official router config fixed.

## Step 2: Preserve The Outer Validation Protocol

1. Keep the outer validation folds exactly as they are now: `2021, 2022, 2023, 2024, 2025`.
2. Do not use the inner analysis to score or select the main benchmark.
3. Use the inner split only for diagnostics.

## Step 3: Define The Inner Train/Val Split

For each outer validation season `Y`:

1. Restrict inner analysis to seasons strictly earlier than `Y`.
2. Choose an inner validation season from that training window.
3. Preferred simple rule:
   - use the latest eligible tournament season before `Y`
   - avoid assuming `Y - 1` always works because 2020 has no tournament targets
4. Preferred stronger rule:
   - run an inner walk-forward over all eligible pre-`Y` seasons
   - pool the resulting row-level errors for that outer fold

## Step 4: Reuse The Existing Fold Cache Flow

1. Use `build_cached_fold_inputs` from `scripts/train_iteration_76_iteration_73_division_specific_probability_blend_search.py`.
2. Keep the existing fold-safe objects:
   - `validation_rows`
   - `targets`
   - `iteration_69_probabilities`
   - `iteration_71_probabilities`
3. Reuse the current raw-margin and calibrator path from `scripts/train_iteration_72_iteration_69_iteration_71_probability_blend_search.py`.

## Step 5: Reapply The Current Official Router Exactly

For each inner validation split:

1. Compute `probability_gap = abs(iteration_71_probability - iteration_69_probability)`.
2. Apply the same men and women routing thresholds and weights used by the fixed official benchmark.
3. Apply the official Stage 2 mask that excludes First Four rows.
4. Do not search any new router settings during error analysis.

## Step 6: Persist Row-Level Diagnostics

Write one row per scored inner-validation matchup to a diagnostics artifact.

Store at least:

1. `outer_season`
2. `inner_validation_season`
3. `Season`
4. `LowerTeamID`
5. `HigherTeamID`
6. `target`
7. `prediction`
8. `squared_error`
9. `residual = prediction - target`
10. `iteration_69_probability`
11. `iteration_71_probability`
12. `probability_gap`
13. router regime label:
    - `men_low`
    - `men_mid`
    - `men_high`
    - `women_low`
    - `women_mid`
    - `women_high`
    - `women_extreme`

## Step 7: Keep Existing Context Columns For Slice Analysis

Retain the fields already present in `validation_rows` whenever available:

1. `men_women`
2. `possible_round`
3. `delta_seed_number`
4. `same_region_flag`
5. `opening_slot_gap`
6. `first_four_pair_flag`
7. bracket pair flags when useful

Also retain the block-rate matchup views:

1. `low_block_feature_overall`
2. `high_block_feature_overall`
3. `delta_block_feature_overall`
4. `low_block_feature_late`
5. `high_block_feature_late`
6. `delta_block_feature_late`
7. `low_block_feature_neutral`
8. `high_block_feature_neutral`
9. `delta_block_feature_neutral`
10. `low_block_feature_home_road_gap`
11. `high_block_feature_home_road_gap`
12. `delta_block_feature_home_road_gap`

## Step 8: Score Candidate Slices In Polars

Build grouped diagnostics in Polars for each slice definition.

For every slice, compute:

1. `row_count`
2. `positive_rate`
3. `mean_prediction`
4. `slice_brier`
5. `excess_brier = slice_brier - overall_brier`
6. `mean_residual`
7. `mean_abs_residual`
8. `iteration_69_slice_brier`
9. `iteration_71_slice_brier`
10. `router_regret_vs_69 = slice_brier - iteration_69_slice_brier`
11. `router_regret_vs_71 = slice_brier - iteration_71_slice_brier`

## Step 9: Start With The Highest-Signal Slice Families

Inspect these first:

1. `men_women`
2. router regime
3. `possible_round`
4. buckets of `abs(delta_seed_number)`
5. prediction confidence deciles
6. `probability_gap` deciles
7. block-rate delta buckets
8. `delta_block_feature_late` buckets
9. `delta_block_feature_neutral` buckets
10. interactions:
    - `men_women x router regime`
    - `men_women x possible_round`
    - `possible_round x abs(delta_seed_number) bucket`
    - `router regime x probability_gap decile`
    - `block-rate bucket x seed-gap bucket`

## Step 10: Rank Slices Conservatively

Avoid chasing tiny noisy groups.

1. Require a minimum row count before reporting a slice.
2. Rank slices by excess Brier first.
3. Break ties with row count and consistency across outer folds.
4. Prefer slices that recur across multiple outer folds over one-off anomalies.

## Step 11: Interpret The Failure Mode

Use the branch comparison to identify what is broken.

1. If the final routed prediction is worse than `iteration_69` and `iteration_71` on a slice, the routing logic is likely wrong there.
2. If the final routed prediction is worse than one branch but not the other, the routing threshold or regime assignment is likely wrong there.
3. If both branches are bad on the slice, the issue is more likely feature quality or model capacity.
4. If ranking looks fine but `mean_prediction` is far from `positive_rate`, the issue is more likely calibration.

## Step 12: Produce Two Outputs

Write:

1. a row-level parquet or CSV with the full inner-validation diagnostics
2. a compact markdown summary with:
   - worst slices overall
   - worst men slices
   - worst women slices
   - worst block-rate-related slices
   - slices where routing hurts most versus branch-only alternatives

## Step 13: Keep This Separate From The Benchmark Scripts

Implement the analysis as a standalone script instead of modifying the production benchmark path.

Reasons:

1. it avoids changing the accepted training pipeline
2. it keeps the main benchmark artifacts clean
3. it lets diagnostics evolve without touching model-selection code

## Recommended Implementation Order

1. Build a standalone script that reconstructs the active benchmark with the accepted block-rate family.
2. Add inner split generation.
3. Save row-level diagnostics for one outer fold first.
4. Validate the row counts and Brier math against the existing outer-fold summaries.
5. Expand to all outer folds.
6. Add grouped slice reports in Polars.
7. Review the worst recurring slices before making any modeling changes.

## First Iteration Execution

- Standalone runner: `tmp/run_error_analysis_first_pass.py`
- Row artifact: `tmp/error-analysis-inner-rows.parquet`
- Summary artifact: `tmp/error-analysis-summary.json`
- Outer seasons analyzed: `2021, 2022, 2023, 2024, 2025`
- Inner validation seasons used: `2019, 2021, 2022, 2023, 2024`
- Official rows scored: `629`
- This first pass used one inner validation season per outer fold, not the stronger full inner walk-forward variant.

## First Iteration Findings

### Overall

- Pooled inner Brier for the routed active benchmark: `0.163476`
- Pooled inner Brier for `iteration_69`: `0.164679`
- Pooled inner Brier for `iteration_71`: `0.164806`
- The current routed benchmark still beats both branches overall on this inner diagnostic pass.

### Division Split

- Men are the clear drag:
  - final routed benchmark: `0.192784`
  - `iteration_69`: `0.194105`
  - `iteration_71`: `0.191773`
- Women are materially stronger:
  - final routed benchmark: `0.134261`
  - `iteration_69`: `0.135347`
  - `iteration_71`: `0.137924`
- Interpretation:
  - the women-side routing still looks directionally right
  - the men side still appears slightly too conservative toward `iteration_69`

### Router Regimes

- `women_low` is the main success regime:
  - `263` rows
  - Brier `0.113519`
  - final equals `iteration_71` here, which looks correct
- `men_low` is the largest weak regime:
  - `189` rows
  - Brier `0.202214`
  - `iteration_71` is better at `0.200675`
- `men_mid` and `men_high` are smaller and better than `men_low`, but `iteration_71` is still marginally strongest in both.
- `women_mid`, `women_high`, and `women_extreme` are weak, but they are small samples:
  - `women_mid`: `30` rows, Brier `0.221228`
  - `women_high`: `14` rows, Brier `0.289341`
  - `women_extreme`: `8` rows, Brier `0.218620`
- Interpretation:
  - `women_mid` and `women_high` are not obvious blend failures because the router is already effectively on `iteration_69` there
  - these look more like branch-level or calibration issues than simple weight problems

### Worst Recurring Slices

- Mid-confidence games are the biggest recurring error cluster.
- Worst recurring slice: `confidence_decile=3`
  - `55` rows across all `5` outer folds
  - mean slice Brier `0.269739`
  - mean outer excess Brier `0.106240`
  - mean prediction `0.346484`
  - actual lower-team win rate `0.484900`
- The same pattern continues in `confidence_decile=4`, `5`, and `6`.
- Interpretation:
  - the model is missing in the middle-confidence band, especially where it leans moderately against the lower TeamID side
  - this looks more like calibration or matchup-translation weakness than a single global routing error

- Men with `possible_round=4` are another strong recurring problem:
  - `40` rows
  - mean slice Brier `0.268787`
  - mean outer excess Brier `0.105288`
  - `iteration_71` is better than the final blend here
- Interpretation:
  - this slice looks more routing-sensitive than the mid-confidence slices
  - it is a candidate for a men-side context gate or recalibration keyed to round context

- One-seed-gap games are consistently difficult:
  - slice `abs_seed_gap_bucket=seed_gap_1`
  - `108` rows
  - mean slice Brier `0.234052`
  - mean outer excess Brier `0.070552`
  - direct row-level check showed mean prediction around `0.482` versus actual lower-team win rate around `0.565`
  - `iteration_69` was slightly better than the final blend on this slice
- Interpretation:
  - the benchmark is underestimating the lower TeamID side in many near-adjacent-seed games
  - this looks partly calibration-driven and not purely a `71`-weight issue

- The tightest men disagreement region also stands out:
  - slice `router_regime=men_low | probability_gap_bucket=gap_le_0p02`
  - `61` rows
  - mean slice Brier `0.237046`
  - mean outer excess Brier `0.073546`
  - direct row-level check showed mean prediction `0.523` versus actual lower-team win rate `0.623`
  - `iteration_71` was best on this slice
- Interpretation:
  - the current men low-gap weighting still appears too conservative

### Men Low-Gap Slice Interpretation

- The tight `men_low | gap_le_0p02` slice is not concentrated in one odd corner of the bracket:
  - `61` games
  - `69` unique teams
  - rows in every inner validation season from `2019` through `2024`
- The teams that appear most often are mostly established tournament programs:
  - Baylor and Gonzaga: `5` appearances each
  - Clemson, Connecticut, Kansas, North Carolina, and USC: `4` each
  - Duke, Michigan State, St Mary's CA, and Utah St: `3` each
- Seed-based upsets are common in this slice:
  - `26` of `61` games were upsets, about `42.6%`
  - the heaviest concentration is in near-coin-flip seed bands rather than extreme Cinderella cases
  - `8` vs `9` matchups account for `20` games and produced upset rates between `60%` and `70%`, depending on orientation
  - `5` vs `4` matchups had a `50%` upset rate
  - `3` vs `6` matchups had a `50%` upset rate
- The bigger upset examples in the slice are still real:
  - `2022`: North Carolina `8` over Duke `2`
  - `2022`: Houston `5` over Arizona `1`
  - `2023`: San Diego St `5` over Alabama `1`
  - `2021`: USC `6` over Kansas `3`
  - `2023`: Creighton `6` over Baylor `3`
  - `2024`: Clemson `6` over Baylor `3`
- Higher-level interpretation:
  - this is not mainly a slice of wild shocks or data oddities
  - it is mostly competitive men's tournament games where the favorite is real but not secure
  - the model looks more like it is flattening credible upset risk into overly smooth near-coin-flip probabilities than failing on obvious mismatches

### Block-Rate-Specific Findings

- Most broad block-rate buckets are not catastrophic by themselves.
- The clearest block-related failure slice was:
  - `delta_block_feature_late_bucket=delta_block_feature_late_q3 | abs_seed_gap_bucket=seed_gap_1`
  - `25` rows across all `5` outer folds
  - mean slice Brier `0.259101`
  - mean outer excess Brier `0.095602`
- Interpretation:
  - this is a real interaction slice, not a broad “block-rate is bad” result
  - the final routed benchmark was slightly worse than `iteration_69` and slightly better than `iteration_71` here
  - that suggests the issue is not the block-rate family globally, but how late block-rate interacts with close seed-gap games

- Broader block-rate bucket summaries were mixed rather than monotonic:
  - some neutral-block buckets leaned toward `iteration_69`
  - others leaned toward `iteration_71`
- Interpretation:
  - the block-rate feature family is heterogeneous
  - it should probably be handled through conditional routing or interaction-aware calibration rather than a global rollback

## Practical Readout From This First Pass

- The current active benchmark is still better than either source branch overall.
- The next place to look is the men side, not the women side.
- The strongest men-side candidate problem is the low-gap regime, especially the very tight disagreement rows.
- The most obvious calibration problem is the middle-confidence band, where the model is too pessimistic on the lower TeamID side.
- The accepted block-rate family does not look broadly broken, but one late-block-rate plus one-seed-gap interaction slice is clearly weak.

## Recommended Next Iteration

1. Rerun this analysis with the stronger full inner walk-forward instead of one inner split per outer fold.
2. Add a targeted men-only slice report for:
   - `men_low`
   - `men_low | gap_le_0p02`
   - `possible_round`
   - `abs_seed_gap_bucket`
3. Add branch-choice diagnostics that directly ask:
   - where `iteration_71` beats final on men rows
   - where `iteration_69` beats final on women rows
4. Test whether the weak mid-confidence slices are primarily calibration failures or whether they survive when examined on raw margins instead of probabilities.

## Second-Pass Slice Queue

The next six slice checks are designed to separate routing mistakes from calibration mistakes and block-rate interactions. The row-level artifact for this pass remains `tmp/error-analysis-inner-rows.parquet`, so these are still diagnostics on the same inner-split scaffold as the first pass.

1. `men_low x sign(iteration_71_probability - iteration_69_probability) x probability_gap_bucket`
2. `men_low x exact seed pair`
3. `men_low x possible_round x abs_seed_gap_bucket`
4. `men confidence_decile 3-6 x router_regime`
5. `men_low x delta_block_feature_late sign/quantile x abs_seed_gap_bucket`
6. `men_low x same_region_flag x opening_slot_gap`

### Slice 1 - `men_low x sign(iteration_71_probability - iteration_69_probability) x probability_gap_bucket`

- The directional split matters more than the gap bucket by itself.
- The clearest weak cell is:
  - `iteration_71_higher | gap_le_0p02`
  - `28` rows across all `5` outer folds
  - mean slice Brier `0.229495`
  - final versus `iteration_71`: `+0.002232` regret
  - mean prediction `0.463951`
  - actual lower-team win rate `0.714286`
- Interpretation:
  - the worst ultra-tight men-low rows are not symmetric
  - the main problem is the subset where `iteration_71` is more bullish on the lower TeamID side and the final blend smooths that edge down too far
  - this is a stronger routing clue than the undirected `gap_le_0p02` slice by itself
- The opposite ultra-tight direction is not the same problem:
  - `iteration_69_higher | gap_le_0p02`
  - `33` rows
  - final and `iteration_69` are effectively tied there
- Practical read:
  - if we revisit routing later, the right follow-up is probably a directional men-low rule rather than a symmetric low-gap weight increase

### Slice 2 - `men_low x exact seed pair`

- The recurring weak seed-pair cells are concentrated in classic opening-weekend upset bands, not across every near-peer seed pairing.
- The worst recurring pairs with at least `5` rows across all `5` folds are:
  - `6_vs_11`
    - `13` rows
    - mean slice Brier `0.302277`
    - upset rate `0.692308`
    - final versus `iteration_71`: `+0.011903` regret
  - `5_vs_12`
    - `6` rows across `3` folds
    - mean slice Brier `0.280414`
    - upset rate `0.500000`
    - final versus `iteration_71`: `+0.007950` regret
  - `8_vs_9`
    - `20` rows
    - mean slice Brier `0.275433`
    - upset rate `0.650000`
    - final versus `iteration_71`: `+0.002169` regret
  - `7_vs_10`
    - `18` rows
    - mean slice Brier `0.249149`
    - upset rate `0.444444`
    - final versus `iteration_71`: `+0.001950` regret
- Better-behaved comparison cells:
  - `3_vs_6` and `2_vs_7` lean slightly toward `iteration_69`, not `iteration_71`
  - that suggests the men-low weakness is not just “all seed-gap tournament games”
- Interpretation:
  - the model looks most vulnerable in the seed-pair bands that already have meaningful historical upset frequency
  - this reinforces the view that the problem is concentrated in credible upset-threat matchups rather than all close games equally

### Slice 3 - `men_low x possible_round x abs_seed_gap_bucket`

- The weak men-low behavior is still most visible in round-2 opening-weekend cells, not uniformly across the bracket.
- The clearest recurring cells are:
  - `possible_round=2 | seed_gap_1`
    - `20` rows across all `5` folds
    - mean slice Brier `0.275433`
    - upset rate `0.650000`
    - final versus `iteration_71`: `+0.002169` regret
  - `possible_round=2 | seed_gap_3_4`
    - `18` rows
    - mean slice Brier `0.249149`
    - upset rate `0.444444`
    - final versus `iteration_71`: `+0.001950` regret
- There is also a smaller late-round problem:
  - `possible_round=4 | seed_gap_3_4`
    - `13` rows
    - mean slice Brier `0.295804`
    - final versus `iteration_71`: `+0.007273` regret
  - `possible_round=4 | seed_gap_5_plus`
    - `6` rows across `4` folds
    - mean slice Brier `0.334294`
    - final versus `iteration_71`: `+0.006241` regret
- Important contrast:
  - `possible_round=3 | seed_gap_1` and `possible_round=4 | seed_gap_1` actually lean toward `iteration_69`
  - so the original “late-round near-peer men games” story is too broad
- Interpretation:
  - the largest repeatable failure mode is still opening-weekend upset-prone men matchups
  - there is a second, smaller late-round issue, but it shows up more in wider seed-gap cells than in pure near-peer seed bands

### Slice 4 - `men confidence_decile 3-6 x router_regime`

- The middle-confidence problem is mostly `men_low`, but not exclusively.
- The single worst cell is:
  - `confidence_decile=3 | men_high`
  - `12` rows across `4` folds
  - mean slice Brier `0.265739`
  - final versus `iteration_71`: `+0.008877` regret
  - mean prediction `0.347023`
  - actual lower-team win rate `0.500000`
- The larger recurring middle-confidence cells are still `men_low`:
  - `confidence_decile=4 | men_low`
    - `27` rows
    - mean slice Brier `0.264908`
    - final versus `iteration_71`: `+0.001718` regret
  - `confidence_decile=5 | men_low`
    - `29` rows
    - mean slice Brier `0.243432`
    - final versus `iteration_71`: `+0.003485` regret
  - `confidence_decile=6 | men_low`
    - `35` rows
    - mean slice Brier `0.233822`
    - final versus `iteration_71`: `+0.001802` regret
- Interpretation:
  - the messy-middle weakness is not purely a `men_low` story
  - most of the volume still sits in `men_low`, but there is also a smaller high-gap moderate-underdog pocket where `iteration_71` is materially better
  - that points to a broader men-side tendency to smooth away useful lower-team upset signal, not just a single low-gap threshold problem

### Slice 5 - `men_low x delta_block_feature_late sign/quantile x abs_seed_gap_bucket`

- The sign split does not isolate one clean directional block-rate problem.
- Sign-based view:
  - `negative | seed_gap_3_4`
    - `22` rows
    - mean slice Brier `0.247399`
    - final versus `iteration_71`: `+0.001498` regret
  - `positive | seed_gap_3_4`
    - `20` rows
    - mean slice Brier `0.244413`
    - final versus `iteration_71`: `+0.002897` regret
  - `negative | seed_gap_1`
    - `25` rows
    - final is slightly better than `iteration_71`
  - `positive | seed_gap_1`
    - `18` rows
    - final versus `iteration_71`: `+0.002893` regret
- Quantile-based view is more specific but still non-monotonic:
  - worst recurring bucket:
    - `delta_block_feature_late_q4 | seed_gap_3_4`
    - `14` rows
    - mean slice Brier `0.291612`
    - final versus `iteration_71`: `+0.006397` regret
  - another notable bucket:
    - `delta_block_feature_late_q2 | seed_gap_5_plus`
    - `23` rows
    - mean slice Brier `0.210939`
    - final versus `iteration_71`: `+0.004594` regret
  - counterexample:
    - `delta_block_feature_late_q3 | seed_gap_1`
    - `10` rows across `4` folds
    - final actually leans toward `iteration_69`
- Interpretation:
  - late block-rate still behaves like an interaction modifier, not a clean monotonic routing variable
  - there is no strong evidence for a simple “positive late block delta should route harder to `iteration_71`” rule
  - if we use block-rate again, it should probably stay as a secondary interaction check rather than the primary routing trigger

### Slice 6 - `men_low x same_region_flag x opening_slot_gap`

- Bracket structure does not open a strong new recurring lever.
- Only same-region cells met the recurrence filter; no cross-region cell cleared the row-count and fold-count thresholds.
- The recurring same-region cells are:
  - `same_region=1 | opening_slot_gap=1`
    - `25` rows across all `5` folds
    - mean slice Brier `0.224119`
    - final actually leans slightly toward `iteration_69`
  - `same_region=1 | opening_slot_gap=0`
    - `98` rows
    - mean slice Brier `0.187648`
    - final versus `iteration_71`: `+0.002396` regret
  - `same_region=1 | opening_slot_gap=3_plus`
    - `52` rows
    - mean slice Brier `0.208935`
    - final versus `iteration_71`: `+0.000448` regret
- Interpretation:
  - same-region structure is common, but it does not isolate a clean fix direction
  - one same-region opening-slot cell leans toward `iteration_69`, while the larger same-region cells lean only mildly toward `iteration_71`
  - this looks more like weak context shading than a primary routing handle

## Second-Pass Synthesis

- The strongest recurring story is now more specific than the first pass:
  - the benchmark is most vulnerable in men opening-weekend upset-threat bands
  - the clearest directional failure is the ultra-tight `men_low` subset where `iteration_71` is more bullish on the lower TeamID side
- Exact seed-pair and round interactions matter more than broad seed-gap labels:
  - `6/11`, `5/12`, `8/9`, and `7/10` are the main recurring weak pair types
  - round-2 men-low cells carry more of the repeatable weakness than a blanket “late-round near-peer” story
- The middle-confidence issue is broader than `men_low`, but still men-specific:
  - most of the volume is in `men_low`
  - a smaller `men_high` moderate-underdog pocket also points toward lost lower-team upset signal
- Block-rate late-delta and bracket-structure slices do not look strong enough to be primary routing levers.
- Practical read:
  - if we take one more model-actionable step from this analysis, it should probably be a directional men upset-band check
  - the cleanest candidate is a men-only rule or expert that activates when `iteration_71 > iteration_69` inside ultra-tight men disagreement rows, especially in round-2 upset-prone seed bands

## Nested Constrained Validation Of Men Upset-Band Rules

- I ran a proper nested season-based search that treated the upset-band idea as a model family instead of a hand-picked post-hoc rule.
- Validation design:
  - outer folds: `2021` to `2025`
  - inner selection seasons available before each outer fold: `2011` to `2019`, then `2021+` as eligible
  - candidate family:
    - `no_rule`
    - men-only upset-band overrides over:
      - direction filter: `{none, iteration_71_higher}`
      - gap threshold: `{0.015, 0.020, 0.025}`
      - round scope: `{eq_2, lte_2, lte_3}`
      - seed band: `{none, upset_core, top_mid_upset_core, near_peer_upset_core, seed_gap_1, seed_gap_3_4}`
      - `iteration_71` weight: `{0.60, 0.70, 0.75, 0.80}`
  - support guards:
    - at least `30` activated rows
    - at least `3` activated seasons
    - both classes present on activated rows
- Important caveat:
  - this nested validation used the current local data snapshot
  - it does **not** exactly reproduce the accepted block-rate artifact from earlier on `2026-03-16`, because local `data/` files were refreshed after that accepted run
  - the correct comparison baseline for this experiment is therefore the rebuilt current-snapshot block-rate benchmark, not the earlier saved accepted artifact
- Current-snapshot baseline:
  - mean validation Brier `0.159773`
  - folds:
    - `2021=0.178380`
    - `2022=0.185079`
    - `2023=0.171144`
    - `2024=0.142189`
    - `2025=0.122073`
- Nested result:
  - mean validation Brier `0.159773`
  - improved folds: `0/5`
  - selected candidate on every outer fold: `no_rule`
  - accepted: `false`
- What the inner searches actually preferred:
  - the best non-`no_rule` candidates were not the sharp directional round-2 rule from the slice analysis
  - the inner search repeatedly drifted toward milder, broader overrides such as:
    - `gap_threshold=0.015`
    - `iteration_71_weight=0.60`
    - usually `direction_filter=none`
    - seed bands like `upset_core`, `near_peer_upset_core`, or `seed_gap_1`
  - even those milder candidates were still slightly *worse* than `no_rule` on inner mean Brier for every outer fold
- Best eligible non-`no_rule` candidate by outer fold:
  - outer `2021`:
    - `direction=none | gap<=0.015 | round<=3 | upset_core | w71=0.60`
    - inner delta versus `no_rule`: `+0.0000095`
  - outer `2022`:
    - `direction=none | gap<=0.015 | round<=3 | seed_gap_1 | w71=0.60`
    - inner delta versus `no_rule`: `+0.0000070`
  - outer `2023`:
    - `direction=none | gap<=0.015 | round<=3 | near_peer_upset_core | w71=0.60`
    - inner delta versus `no_rule`: `+0.0000033`
  - outer `2024`:
    - `direction=none | gap<=0.015 | round<=3 | near_peer_upset_core | w71=0.60`
    - inner delta versus `no_rule`: `+0.0000028`
  - outer `2025`:
    - `direction=none | gap<=0.015 | round==2 | seed_gap_1 | w71=0.60`
    - inner delta versus `no_rule`: `+0.0000035`
- Interpretation:
  - the descriptive slice analysis found a real-looking pattern, but the training history would not have justified turning it into a rule
  - once the rule is forced through the same outer/inner constraint we use for any learned component, the data prefers to do nothing
  - the directional round-2 upset-band story is therefore not robust enough, at least in this hand-coded override family
