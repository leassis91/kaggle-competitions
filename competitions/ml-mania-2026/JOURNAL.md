# JOURNAL.md

## Purpose

- Track meaningful experiments for March Machine Learning Mania 2026.
- Keep a lightweight record of what was tried, how it scored on the mandatory 5-fold walk-forward validation, and what to do next.
- Append entries rather than rewriting history.

## Entry Template

### YYYY-MM-DD HH:MM UTC - Short Experiment Name

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

## Entries

### 2026-03-07 23:41 UTC - Prior-Season Compact Baseline

- Hypothesis: A pre-season LightGBM baseline built from prior-season compact results, prior tournament outcomes, prior seeds, conference strength, and rolling team history should produce a usable first Stage 2 submission without current-season leakage.
- Data window: Men 1985-2025 inputs with 1986-2025 feature seasons, women 1998-2025 inputs with 1999-2025 feature seasons, 2026 submission target from `data/SampleSubmissionStage2.csv`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Separate men and women models using prior-season regular-season compact aggregates, opponent-strength aggregates, conference-strength aggregates, prior NCAA tournament wins/appearance/margin, prior seed features, rolling 3-season team summaries, and men-only prior-season Massey summary features.
- LightGBM config: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.197565.
- Fold scores: 2021=0.189475, 2022=0.210162, 2023=0.232854, 2024=0.174545, 2025=0.180786.
- Artifacts: `artifacts/first_lightgbm_baseline/submission.csv`, `artifacts/first_lightgbm_baseline/validation_summary.json`, `artifacts/first_lightgbm_baseline/men_lightgbm.joblib`, `artifacts/first_lightgbm_baseline/women_lightgbm.joblib`.
- Notes: Submission row count matches Stage 2 sample row count at 132133 with no null predictions or duplicate IDs. Men underperformed women in every validation fold, with 2023 men the weakest fold. `uvx ruff check` passed. `uvx ty check` reported Polars/LightGBM typing issues that did not block execution.
- Next step: Improve the men model first with stronger prior-season team-strength features, especially ratings derived from prior full-season game history and better conference or schedule-strength signals.

### 2026-03-07 23:56 UTC - Prior-Season Last-10 Momentum

- Hypothesis: Adding prior-season late-season momentum from each team’s last 10 regular-season games should improve tournament prediction quality, especially for men.
- Data window: Same as baseline, with prior-season last-10 regular-season aggregates added before shifting to the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Baseline features plus `prev_last10_win_pct`, `prev_last10_margin`, and `prev_last10_games`, with matchup deltas derived from those team-level features.
- LightGBM config: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.196799.
- Fold scores: 2021=0.191195, 2022=0.207599, 2023=0.229680, 2024=0.175106, 2025=0.180413.
- Artifacts: `artifacts/iteration_01_prior_last10/submission.csv`, `artifacts/iteration_01_prior_last10/validation_summary.json`, `artifacts/iteration_01_prior_last10/men_lightgbm.joblib`, `artifacts/iteration_01_prior_last10/women_lightgbm.joblib`.
- Notes: Candidate beat the baseline average Brier, but it improved only 3 of 5 folds: 2022, 2023, and 2025. It regressed on 2021 and 2024, so it fails the robustness rule and is rejected as not improved. Submission format checks passed and `uvx ruff check` passed.
- Next step: Try a strength feature that is less season-phase specific and more stable across years, such as prior-season Elo-style ratings or better prior-season schedule-strength summaries.

### 2026-03-08 00:15 UTC - Prior-Season Elo Features

- Hypothesis: Prior-season regular-season Elo summaries should provide a more stable team-strength signal than raw aggregate margins and improve fold robustness.
- Data window: Same as baseline, with prior-season Elo features derived from previous regular-season compact results only.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Baseline features plus `prev_elo_end`, `prev_elo_max`, `prev_elo_min`, `prev_elo_range`, and `rolling_3_elo_end`.
- LightGBM config: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.198873.
- Fold scores: 2021=0.194543, 2022=0.212224, 2023=0.241834, 2024=0.178335, 2025=0.167427.
- Artifacts: `artifacts/iteration_02_prior_elo/submission.csv`, `artifacts/iteration_02_prior_elo/validation_summary.json`, `artifacts/iteration_02_prior_elo/men_lightgbm.joblib`, `artifacts/iteration_02_prior_elo/women_lightgbm.joblib`.
- Notes: Candidate regressed on 4 of 5 folds and only improved 2025, so it is clearly rejected. `uvx ruff check` passed.
- Next step: Try a lighter-weight composition change rather than another broad feature family.

### 2026-03-08 00:22 UTC - Baseline Probability Shrink

- Hypothesis: The baseline LightGBM outputs are too extreme for Brier score, so shrinking probabilities toward `0.5` by division should improve calibration and produce a more robust validation profile.
- Data window: Same as the baseline. No feature changes.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as the baseline.
- LightGBM config: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.189316.
- Fold scores: 2021=0.183365, 2022=0.197698, 2023=0.213655, 2024=0.174349, 2025=0.177515.
- Artifacts: `artifacts/iteration_03_probability_shrink/submission.csv`, `artifacts/iteration_03_probability_shrink/validation_summary.json`, `artifacts/iteration_03_probability_shrink/men_lightgbm.joblib`, `artifacts/iteration_03_probability_shrink/women_lightgbm.joblib`.
- Notes: Applied division-specific shrink factors of `0.70` for men and `0.80` for women. This candidate improves the mean Brier and all 5 folds versus the baseline, so it is accepted as the new current best model. Submission format checks passed and `uvx ruff check` passed.
- Next step: Build future feature iterations on top of this calibrated baseline rather than the original unshrunk predictions.

### 2026-03-08 01:06 UTC - Men Two-Model Ensemble With Calibrated Shrink

- Hypothesis: Men can benefit from blending the accepted baseline LightGBM with a smoother LightGBM variant, while keeping the accepted women model fixed.
- Data window: Same as the accepted probability-shrink model. No feature changes.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as the baseline.
- LightGBM config: Women keep the baseline LightGBM config. Men use a weighted two-model ensemble:
  `baseline`: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`
  `smooth`: `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`
- Mean Brier: 0.188035.
- Fold scores: 2021=0.182922, 2022=0.196307, 2023=0.210755, 2024=0.174183, 2025=0.176009.
- Artifacts: `artifacts/iteration_04_men_ensemble/submission.csv`, `artifacts/iteration_04_men_ensemble/validation_summary.json`, `artifacts/iteration_04_men_ensemble/men_lightgbm.joblib`, `artifacts/iteration_04_men_ensemble/women_lightgbm.joblib`.
- Notes: Men ensemble weights are `0.30` baseline and `0.70` smooth, followed by probability shrink `0.80` for men; women stay at shrink `0.80`. This candidate improves the mean Brier and all 5 folds over the prior accepted model, so it is accepted as the new current best. Submission format checks passed and `uvx ruff check` passed.
- Next step: Search for a women-side ensemble or a similarly stable composition change on top of this new best model.

### 2026-03-08 00:24 UTC - Prior-Season Detailed Net Efficiency

- Hypothesis: Adding prior-season detailed-stat net efficiency should improve preseason team-strength estimation beyond compact-only scoring features.
- Data window: Same as iteration 04, with prior-season detailed regular-season box score inputs shifted into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_net_eff_z` derived from prior-season detailed regular-season possessions and team net efficiency.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.189682.
- Fold scores: 2021=0.187679, 2022=0.195717, 2023=0.211381, 2024=0.178444, 2025=0.175190.
- Artifacts: `artifacts/iteration_05_detailed_efficiency/submission.csv`, `artifacts/iteration_05_detailed_efficiency/validation_summary.json`, `artifacts/iteration_05_detailed_efficiency/men_lightgbm.joblib`, `artifacts/iteration_05_detailed_efficiency/women_lightgbm.joblib`.
- Notes: The candidate improved 2022, 2023, and 2025, but regressed on 2021 and 2024 and also missed the current-best mean Brier. It fails both the mean and 4-of-5-fold improvement rules, so it is rejected.
- Next step: Try a more stable preseason feature family than raw detailed efficiency, with stronger emphasis on robust schedule-adjusted or historical-program signals.

### 2026-03-08 00:25 UTC - Prior-Season Detailed Four Factors

- Hypothesis: A broader prior-season detailed-stat block covering shooting, turnovers, rebounding, free-throw rate, and net efficiency will add stable structure beyond the compact baseline.
- Data window: Same as iteration 04, with prior-season detailed regular-season box score inputs shifted into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_net_eff_z`, `prev_efg_pct`, `prev_opp_efg_pct`, `prev_turnover_rate`, `prev_opp_turnover_rate`, `prev_orb_rate`, `prev_opp_orb_rate`, `prev_ft_rate`, and `prev_opp_ft_rate`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.188907.
- Fold scores: 2021=0.185412, 2022=0.198162, 2023=0.205181, 2024=0.184285, 2025=0.171492.
- Artifacts: `artifacts/iteration_06_four_factors/submission.csv`, `artifacts/iteration_06_four_factors/validation_summary.json`, `artifacts/iteration_06_four_factors/men_lightgbm.joblib`, `artifacts/iteration_06_four_factors/women_lightgbm.joblib`.
- Notes: The candidate improved only 2023 and 2025 versus the current best, and it materially regressed 2021, 2022, and especially 2024. It fails the robustness rule and is rejected.
- Next step: Avoid broad detailed-stat blocks for now and try a narrower preseason feature family with a better chance of improving at least 4 folds.

### 2026-03-08 00:45 UTC - Second-Order Schedule Strength

- Hypothesis: Opponent-of-opponent strength should give a more stable preseason schedule-context signal than the existing single-hop opponent averages.
- Data window: Same as iteration 04, with prior-season regular-season compact results used to build second-order opponent summaries before shifting into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_avg_opp_opp_win_pct` and `prev_avg_opp_opp_margin`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.187931.
- Fold scores: 2021=0.181774, 2022=0.203914, 2023=0.206916, 2024=0.175208, 2025=0.171844.
- Artifacts: `artifacts/iteration_07_second_order_schedule/submission.csv`, `artifacts/iteration_07_second_order_schedule/validation_summary.json`, `artifacts/iteration_07_second_order_schedule/men_lightgbm.joblib`, `artifacts/iteration_07_second_order_schedule/women_lightgbm.joblib`.
- Notes: This candidate beat the current-best mean Brier, but it improved only 2021, 2023, and 2025 while regressing on 2022 and 2024. It fails the 4-of-5-fold robustness rule and is rejected.
- Next step: Try a more explicitly multi-year feature family, such as trajectory and stability signals, instead of richer schedule context.

### 2026-03-08 00:58 UTC - Multi-Year Trajectory Features

- Hypothesis: Explicit one-year trend and prior-three-season deviation features will capture stable program direction that the raw lagged aggregates are not expressing well enough on their own.
- Data window: Same as iteration 04, with trajectory features computed only from prior completed seasons before shifting into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_win_pct_change_1y`, `prev_avg_margin_change_1y`, `prev_avg_opp_win_pct_change_1y`, `prev_tourney_wins_change_1y`, `prev_win_pct_vs_prior_3_mean`, `prev_avg_margin_vs_prior_3_mean`, and `prev_avg_opp_win_pct_vs_prior_3_mean`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.188026.
- Fold scores: 2021=0.186168, 2022=0.191346, 2023=0.206065, 2024=0.178487, 2025=0.178066.
- Artifacts: `artifacts/iteration_08_trajectory_features/submission.csv`, `artifacts/iteration_08_trajectory_features/validation_summary.json`, `artifacts/iteration_08_trajectory_features/men_lightgbm.joblib`, `artifacts/iteration_08_trajectory_features/women_lightgbm.joblib`.
- Notes: The candidate slightly improved the mean Brier versus the current best, but only improved 2022 and 2023 while regressing on 2021, 2024, and 2025. It fails the robustness rule and is rejected.
- Next step: Try compact-only normalization or relative-strength features rather than more history-shape features.

### 2026-03-08 01:11 UTC - Compact Feature Season Z-Scores

- Hypothesis: Season-normalizing the existing compact team-strength features will make them more comparable across years and improve robustness without adding new data sources.
- Data window: Same as iteration 04, with season-level z-scores computed from prior-season compact features before shifting into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_win_pct_z`, `prev_avg_points_for_z`, `prev_avg_points_against_z`, `prev_avg_margin_z`, and `prev_avg_opp_win_pct_z`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.187613.
- Fold scores: 2021=0.187470, 2022=0.199379, 2023=0.205762, 2024=0.174603, 2025=0.170851.
- Artifacts: `artifacts/iteration_09_compact_zscores/submission.csv`, `artifacts/iteration_09_compact_zscores/validation_summary.json`, `artifacts/iteration_09_compact_zscores/men_lightgbm.joblib`, `artifacts/iteration_09_compact_zscores/women_lightgbm.joblib`.
- Notes: This candidate improved the mean Brier versus the current best, but only improved 2023 and 2025 while regressing on 2021, 2022, and 2024. It fails the robustness rule and is rejected.
- Next step: Apply new compact features more selectively by division rather than forcing the same block onto both men and women.

### 2026-03-08 13:48 UTC - Prior-Season Team Quality

- Hypothesis: A prior-season schedule-adjusted team-quality estimate, inspired by the notebook's `team_quality` feature but shifted into the next season, can strengthen preseason team-strength estimates without leakage.
- Data window: Same as iteration 04, with compact regular-season point margins solved season-by-season into ridge-regularized team effects before shifting to the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus `prev_team_quality` for both divisions.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.191474.
- Fold scores: 2021=0.191427, 2022=0.204080, 2023=0.216133, 2024=0.173515, 2025=0.172215.
- Artifacts: `artifacts/iteration_12_prior_team_quality/submission.csv`, `artifacts/iteration_12_prior_team_quality/validation_summary.json`, `artifacts/iteration_12_prior_team_quality/men_lightgbm.joblib`, `artifacts/iteration_12_prior_team_quality/women_lightgbm.joblib`.
- Notes: The candidate regressed on all 5 folds relative to the current accepted model. The notebook's same-season `team_quality` signal did not transfer into a prior-season-only feature in this setup, so the idea is rejected.
- Next step: Keep the notebook translation narrow and test its selected-stat and calibration ideas separately.

### 2026-03-08 13:49 UTC - Men Selected Detailed Stat Subset

- Hypothesis: A small men-only subset of prior-season detailed box-score averages, mirroring the notebook's selected stat block, can outperform the previously rejected broad detailed-stat additions.
- Data window: Same as iteration 04, with prior-season men detailed regular-season means for FGA, OR, DR, Blk, PF, opponent FGA, opponent Blk, and opponent PF shifted into the next feature season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus men-only `prev_avg_fga`, `prev_avg_or`, `prev_avg_dr`, `prev_avg_blk`, `prev_avg_pf`, `prev_avg_opp_fga`, `prev_avg_opp_blk`, and `prev_avg_opp_pf`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.189105.
- Fold scores: 2021=0.188925, 2022=0.197261, 2023=0.207568, 2024=0.176909, 2025=0.174861.
- Artifacts: `artifacts/iteration_13_men_selected_detailed/submission.csv`, `artifacts/iteration_13_men_selected_detailed/validation_summary.json`, `artifacts/iteration_13_men_selected_detailed/men_lightgbm.joblib`, `artifacts/iteration_13_men_selected_detailed/women_lightgbm.joblib`.
- Notes: This narrower detailed block beat the earlier broad detailed-stat experiments on average, but it still improved only 2023 and 2025 while regressing on 2021, 2022, and 2024. It fails the 4-of-5-fold rule and is rejected.
- Next step: Test the notebook's calibration idea without changing the accepted feature set.

### 2026-03-08 13:58 UTC - Nested OOF Isotonic Calibration

- Hypothesis: The notebook's learned prediction-to-probability mapping can outperform fixed shrink when translated into a fold-safe nested out-of-fold isotonic calibrator by division.
- Data window: Same as iteration 04. No feature changes.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 04.
- LightGBM config: Same as iteration 04, but raw model probabilities are calibrated with division-specific isotonic regression fit only on nested walk-forward out-of-fold predictions from seasons before each validation season.
- Mean Brier: 0.189099.
- Fold scores: 2021=0.185774, 2022=0.196290, 2023=0.209424, 2024=0.178122, 2025=0.175884.
- Artifacts: `artifacts/iteration_14_oof_isotonic_calibration/validation_summary.json`.
- Notes: The learned calibrator improved 2022 and 2023, but regressed on 2021, 2024, and 2025 and also missed the current-best mean Brier. Fixed shrink remains the better calibration choice here, so this idea is rejected.
- Next step: Finish the notebook-inspired queue with a richer men-only Massey summary block.

### 2026-03-08 13:59 UTC - Men Massey Dispersion Features

- Hypothesis: Richer men-only Massey consensus and dispersion signals can recover some of the notebook's latent-strength benefit without leaving the repo's data sources.
- Data window: Same as iteration 04, with men-only `MMasseyOrdinals.csv` day-133 summaries extended using worst rank, median rank, rank standard deviation, rank range, and top-25 share.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 04 features plus men-only `prev_massey_worst_rank`, `prev_massey_median_rank`, `prev_massey_rank_std`, `prev_massey_rank_range`, and `prev_massey_top25_share`.
- LightGBM config: Same as iteration 04. Men use the accepted two-model ensemble with shrink `0.80`; women use the baseline model with shrink `0.80`.
- Mean Brier: 0.189306.
- Fold scores: 2021=0.190184, 2022=0.196430, 2023=0.210030, 2024=0.174858, 2025=0.175026.
- Artifacts: `artifacts/iteration_15_men_massey_dispersion/validation_summary.json`.
- Notes: The candidate improved only 2023 and 2025 versus the current accepted model and missed the current-best mean Brier. The richer Massey block is rejected.
- Next step: Close out the notebook-inspired queue and return to non-notebook feature searches that better match the repo's preseason constraints.

### 2026-03-08 14:18 UTC - Pooled Selection Sunday Baseline

- Hypothesis: A single pooled LightGBM trained across men and women on Selection Sunday-safe current-season selected box-score averages plus current-season seeds can beat the separate-division preseason baseline once the information cutoff is relaxed.
- Data window: Men and women pooled from 2010-2025 for validation, using same-season regular-season detailed averages and same-season NCAA tournament seeds for each historical tournament season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: One pooled training table with `men_women`, same-season `low/high` selected stat averages (`Score`, `FGA`, `OR`, `DR`, `Blk`, `PF`, opponent `FGA`, opponent `Blk`, opponent `PF`, `PointDiff`), and same-season `low_seed`, `high_seed`, and `delta_seed`.
- LightGBM config: Single pooled LightGBM classifier with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.196802.
- Fold scores: 2021=0.220878, 2022=0.216726, 2023=0.184873, 2024=0.188766, 2025=0.172765.
- Artifacts: `artifacts/iteration_16_pooled_selection_sunday_baseline/validation_summary.json`.
- Notes: This first pooled Selection Sunday baseline improved only 2023 and 2025 versus the current accepted model and materially regressed 2021, 2022, and 2024. The pooled structure plus current-season seeds and selected stat averages are now implemented, but this exact baseline is rejected.
- Next step: Add current-season Massey features to the pooled baseline before revisiting pooled LightGBM tuning or calibration.

### 2026-03-08 14:22 UTC - Pooled Selection Sunday Baseline With Current-Season Massey

- Hypothesis: Current-season Massey summaries added to the pooled Selection Sunday baseline can recover enough team-strength signal on the men side to offset the baseline's weak pooled calibration.
- Data window: Same as iteration 16, with current-season same-season regular-season detailed averages and same-season NCAA tournament seeds for each historical tournament season, plus current-season men Massey summaries from the latest available ranking day in each season.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 16 pooled features plus `low/high/delta` current-season Massey mean rank, best rank, and system count.
- LightGBM config: Same single pooled LightGBM classifier as iteration 16 with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.197086.
- Fold scores: 2021=0.223002, 2022=0.221643, 2023=0.190997, 2024=0.185760, 2025=0.164026.
- Artifacts: `artifacts/iteration_17_pooled_selection_sunday_massey/validation_summary.json`.
- Notes: Adding current-season Massey made the pooled baseline slightly worse on average and still improved only 2023 and 2025 versus the current accepted model. The pooled current-season Massey idea is rejected in this form.
- Next step: Move to pooled current-season Elo or pooled current-season team-quality before spending time on broader pooled LightGBM tuning.

### 2026-03-08 14:25 UTC - Pooled Selection Sunday Baseline With Current-Season Elo

- Hypothesis: Adding current-season end-of-regular-season Elo ratings to the pooled Selection Sunday baseline can recover team-strength signal in the notebook's original pooled setup better than the earlier prior-season Elo attempt.
- Data window: Same as iteration 16, with same-season regular-season detailed averages and same-season NCAA tournament seeds for each historical tournament season, plus same-season regular-season compact Elo ratings for men and women.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 16 pooled features plus `low/high/delta` current-season Elo end ratings.
- LightGBM config: Same single pooled LightGBM classifier as iteration 16 with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.194569.
- Fold scores: 2021=0.213940, 2022=0.217600, 2023=0.183377, 2024=0.185894, 2025=0.172035.
- Artifacts: `artifacts/iteration_18_pooled_selection_sunday_elo/validation_summary.json`.
- Notes: Current-season Elo helped relative to the pooled baseline, but the candidate still improved only 2023 and 2025 versus the accepted benchmark and remained much worse on 2021, 2022, and 2024. This pooled Elo translation is rejected.
- Next step: Test current-season pooled team-quality before deciding whether the problem is mostly pooled feature choice or pooled model calibration.

### 2026-03-08 14:25 UTC - Pooled Selection Sunday Baseline With Current-Season Team Quality

- Hypothesis: Same-season ridge-solved team-quality, closer to the notebook's intended timing than the earlier prior-season version, can improve pooled tournament matchup modeling beyond current-season seeds and selected stat averages.
- Data window: Same as iteration 16, with same-season regular-season detailed averages and same-season NCAA tournament seeds for each historical tournament season, plus same-season regular-season compact-results team-quality estimates for men and women.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 16 pooled features plus `low/high/delta` current-season team-quality.
- LightGBM config: Same single pooled LightGBM classifier as iteration 16 with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Mean Brier: 0.192945.
- Fold scores: 2021=0.214445, 2022=0.213452, 2023=0.178439, 2024=0.191098, 2025=0.167293.
- Artifacts: `artifacts/iteration_19_pooled_selection_sunday_team_quality/validation_summary.json`.
- Notes: Current-season team-quality beat the pooled baseline, pooled Massey, and pooled Elo on average, but it still improved only 2023 and 2025 versus the accepted benchmark. The feature family is rejected under the project robustness rule, though it is the strongest direct notebook-style pooled feature addition so far.
- Next step: Separate model-quality effects from feature effects by testing a smoother pooled LightGBM on the shared pooled baseline table.

### 2026-03-08 14:25 UTC - Smoother Pooled Selection Sunday LightGBM

- Hypothesis: The pooled Selection Sunday baseline is too aggressive for Brier score, so a smoother pooled LightGBM with larger `min_child_samples` and stronger `reg_lambda` can recover robustness without changing the feature table.
- Data window: Same as iteration 16, using the shared pooled current-season selected-stat and current-season seed feature table.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 16 pooled baseline.
- LightGBM config: Single pooled LightGBM classifier with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`.
- Mean Brier: 0.181490.
- Fold scores: 2021=0.199045, 2022=0.196556, 2023=0.183532, 2024=0.176490, 2025=0.151829.
- Artifacts: `artifacts/iteration_20_pooled_selection_sunday_smooth/validation_summary.json`.
- Notes: This is the best pooled mean Brier so far and it beats the accepted benchmark average decisively, but it still improves only 2023 and 2025 versus the benchmark and narrowly misses 2022 while regressing clearly on 2021 and 2024. The pooled architecture now looks calibration-sensitive rather than feature-starved, but this candidate is still rejected by the 4-of-5-fold rule.
- Next step: Test pooled calibration or pooled LightGBM blends on top of the smoother pooled baseline instead of adding more direct notebook feature families.

### 2026-03-08 14:25 UTC - Pooled Margin Regression With Learned Probability Map

- Hypothesis: The notebook's regression-on-margin framing may fit tournament matchups better than direct classification in the pooled setup if predicted margins are mapped back to win probability with a fold-safe learned calibrator.
- Data window: Same as iteration 16, using the pooled current-season selected-stat and current-season seed feature table, plus tournament score margin aligned to lower TeamID minus higher TeamID as the regression target.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 16 pooled baseline.
- LightGBM config: Single pooled `LGBMRegressor` with `objective=regression`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`, followed by nested OOF logistic calibration from predicted margin to win probability.
- Mean Brier: 0.180056.
- Fold scores: 2021=0.195022, 2022=0.189301, 2023=0.191744, 2024=0.170484, 2025=0.153728.
- Artifacts: `artifacts/iteration_21_pooled_selection_sunday_margin_regression/validation_summary.json`.
- Notes: This candidate improves the mean Brier and 4 of 5 folds versus the prior accepted model from iteration 04, missing only 2021. It is accepted as the new current best model and is the first pooled Selection Sunday setup to clear the project robustness gate.
- Next step: Treat iteration 21 as the active benchmark and test whether pooled learned calibration or pooled blends can beat it under the same acceptance rule.

### 2026-03-08 14:25 UTC - Pooled Smooth LightGBM With Nested OOF Isotonic

- Hypothesis: The best pooled classifier candidate from iteration 20 can recover the remaining fold stability through a learned pooled probability map fit on nested walk-forward out-of-fold predictions.
- Data window: Same as iteration 16, using the pooled current-season selected-stat and current-season seed feature table.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 16 pooled baseline.
- LightGBM config: Same single pooled smooth `LGBMClassifier` as iteration 20 with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`, followed by nested OOF isotonic calibration.
- Mean Brier: 0.183863.
- Fold scores: 2021=0.199107, 2022=0.198238, 2023=0.190041, 2024=0.177405, 2025=0.154525.
- Artifacts: `artifacts/iteration_22_pooled_selection_sunday_smooth_isotonic/validation_summary.json`.
- Notes: The isotonic map improves only 2023 versus the new accepted benchmark from iteration 21 and loses the average comparison as well. Learned pooled isotonic calibration is rejected in this setup.
- Next step: Test a small pooled LightGBM blend across complementary feature views rather than another calibration layer.

### 2026-03-08 14:25 UTC - Pooled Smooth Baseline and Team-Quality Blend

- Hypothesis: A small pooled LightGBM blend across the strong smooth baseline feature table and the smoother team-quality feature view can improve average Brier without giving up too much fold stability.
- Data window: Same as iterations 16 and 19, using the pooled current-season selected-stat and current-season seed feature table for one model and the same table plus current-season team-quality for the second model.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Blend of iteration 20 pooled baseline features and iteration 19 pooled team-quality features.
- LightGBM config: Two pooled smooth `LGBMClassifier` models, each with `objective=binary`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`, blended with weights `0.75` baseline smooth and `0.25` team-quality smooth.
- Mean Brier: 0.179937.
- Fold scores: 2021=0.197957, 2022=0.195165, 2023=0.182829, 2024=0.175304, 2025=0.148432.
- Artifacts: `artifacts/iteration_23_pooled_selection_sunday_smooth_blend/validation_summary.json`.
- Notes: The blend slightly beats iteration 21 on average, but it improves only 2023 and 2025 versus the new accepted benchmark while regressing on 2021, 2022, and 2024. It is rejected under the 4-of-5-fold rule even though it has the best pooled average Brier so far.
- Next step: Keep iteration 21 as the active benchmark and revisit only feature families or blends that have a plausible path to repairing the weak 2021 fold instead of merely lowering the mean.

### 2026-03-08 14:39 UTC - Bracket Geometry and Seed Structure on Iteration 21

- Hypothesis: The third-place and GOTO notebooks may be contributing mostly bracket geometry rather than raw team priors, so adding seed-path and slot-derived structure to the accepted pooled margin-regression model can improve robustness across folds.
- Data window: Same as iteration 21, using pooled current-season selected-stat and current-season seed features, plus same-season slot-graph-derived bracket structure from `MNCAATourneySlots.csv` and `WNCAATourneySlots.csv`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 21 features plus `seed_number`, `seed_region_code`, `same_region_flag`, `same_final_four_side_flag`, play-in flags, opening-slot indices, opening-slot gap, shared seed-line flag, first-weekend and regional-quadrant flags, and `possible_round` with per-round indicator flags.
- LightGBM config: Same pooled `LGBMRegressor` as iteration 21 with `objective=regression`, `learning_rate=0.03`, `n_estimators=400`, `num_leaves=31`, `min_child_samples=100`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `random_state=7`, followed by the same nested OOF logistic calibration from predicted margin to win probability.
- Mean Brier: 0.179351.
- Fold scores: 2021=0.197517, 2022=0.189263, 2023=0.188750, 2024=0.170246, 2025=0.150981.
- Artifacts: `artifacts/iteration_24_pooled_selection_sunday_bracket_features/validation_summary.json`.
- Notes: This candidate improves the mean Brier and 4 of 5 folds versus iteration 21, missing only 2021. Bracket geometry by itself is enough to clear the project acceptance gate and becomes the new current best model.
- Next step: Test whether explicit bracket interaction terms can beat this stronger bracket-aware benchmark.

### 2026-03-08 14:39 UTC - Bracket Interaction Features on Top of Iteration 24

- Hypothesis: Once the bracket geometry is present, explicit round-by-seed interaction terms can expose additional signal that LightGBM may not recover cleanly from the raw bracket features alone.
- Data window: Same as iteration 24.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 24 features plus `possible_round x delta_seed_number`, `same_region_flag x delta_seed_number`, `same_final_four_side_flag x delta_seed_number`, `opening_slot_gap x delta_seed_number`, and related round-interaction terms.
- LightGBM config: Same pooled `LGBMRegressor` and same nested OOF logistic margin-to-probability calibration as iteration 24.
- Mean Brier: 0.179691.
- Fold scores: 2021=0.198191, 2022=0.188326, 2023=0.189396, 2024=0.171869, 2025=0.150672.
- Artifacts: `artifacts/iteration_25_pooled_selection_sunday_bracket_interactions/validation_summary.json`.
- Notes: The interaction block lowers the average slightly versus iteration 21, but it does not beat the stronger iteration 24 benchmark and improves only 2 of 5 folds relative to iteration 24. The extra interaction terms are rejected as unnecessary noise in this setup.
- Next step: Move the bracket logic into the calibration layer instead of adding more direct interaction features to the regressor.

### 2026-03-08 14:39 UTC - Bracket-Aware Calibration on the Bracket Feature Model

- Hypothesis: The GOTO and third-place notebooks are fundamentally tournament-structured, so the remaining gain may come from a probability map that conditions on bracket context rather than from more raw model features.
- Data window: Same as iteration 24, using the same pooled bracket-aware feature table.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 25 for the regressor, but the learned probability map uses predicted margin plus bracket context from `possible_round`, `delta_seed_number`, `same_region_flag`, `any_play_in_flag`, and their interactions with predicted margin.
- LightGBM config: Same pooled `LGBMRegressor` as iterations 24 and 25. Probability calibration changes from one-dimensional nested OOF logistic on predicted margin alone to nested OOF logistic on predicted margin plus bracket context.
- Mean Brier: 0.173684.
- Fold scores: 2021=0.190831, 2022=0.195243, 2023=0.182202, 2024=0.161519, 2025=0.138626.
- Artifacts: `artifacts/iteration_26_pooled_selection_sunday_bracket_calibration/validation_summary.json`.
- Notes: This candidate improves the mean Brier and 4 of 5 folds versus iteration 24, missing only 2022. It is accepted as the new current best model and strongly suggests the missing value from these previous-winner notebooks is bracket-aware probability mapping more than extra raw team features.
- Next step: Keep iteration 26 as the active benchmark and only revisit the GOTO path after adding external progression-prior data locally.

### 2026-03-08 14:58 UTC - Mike Form and Matchup Priors on Top of Iteration 26

- Hypothesis: Mike Kim's short-horizon form block and head-to-head Laplace matchup priors can improve the current bracket-aware model by adding late-season and opponent-specific information not captured by the current bracket calibration.
- Data window: Same as iteration 26, using same-season regular-season compact results to build `WinRatio14d`, away win rate, and Laplace-smoothed head-to-head matchup priors.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low/high/delta` late-season win ratio, `low/high/delta` away win rate, `h2h_meetings`, and `h2h_laplace_win_pct`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 26.
- Mean Brier: 0.174070.
- Fold scores: 2021=0.191289, 2022=0.195768, 2023=0.180154, 2024=0.161926, 2025=0.141214.
- Artifacts: `artifacts/iteration_27_mike_form_matchup_priors/validation_summary.json`.
- Notes: This block only improves 2023 versus the accepted benchmark from iteration 26 and regresses the other four folds. The late-season and matchup-prior features do not transfer cleanly on top of the current bracket-aware model, so this candidate is rejected.
- Next step: Test Mike's seeded-team-focused quality construction separately, since that is the most structurally different feature from our current setup.

### 2026-03-08 14:58 UTC - Mike Seeded-Team-Focused Quality

- Hypothesis: Mike Kim's seeded-team-focused same-season quality estimate, which concentrates regular-season quality fitting on seeded teams plus teams that beat them, can outperform the current bracket-aware model by denoising team-strength estimation near tournament-relevant opponents.
- Data window: Same as iteration 26, using same-season regular-season compact results and same-season tournament seeds to define the quality-fit population for each season and division.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low_mike_quality`, `high_mike_quality`, and `delta_mike_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 26.
- Mean Brier: 0.170581.
- Fold scores: 2021=0.186010, 2022=0.193032, 2023=0.176028, 2024=0.159258, 2025=0.138575.
- Artifacts: `artifacts/iteration_28_mike_filtered_quality/validation_summary.json`.
- Notes: This candidate improves the mean Brier and all 5 folds versus iteration 26. It is accepted as the new current best model and indicates that Mike's filtered same-season quality construction transfers much better than his late-season form and head-to-head prior block.
- Next step: Test whether combining the winning filtered-quality feature with the rejected short-horizon features helps or merely reintroduces noise.

### 2026-03-08 14:58 UTC - Combined Portable Mike Features

- Hypothesis: The full portable Mike Kim feature set may work best jointly, with filtered quality capturing tournament-relevant strength while late-season form and matchup priors provide useful local adjustments.
- Data window: Same as iterations 27 and 28.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus the iteration 27 form and matchup-prior block and the iteration 28 filtered-quality block.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iterations 26 through 28.
- Mean Brier: 0.170950.
- Fold scores: 2021=0.187584, 2022=0.191036, 2023=0.177799, 2024=0.160381, 2025=0.137948.
- Artifacts: `artifacts/iteration_29_mike_combined/validation_summary.json`.
- Notes: The combined candidate remains better than iteration 26 on average, but it is slightly worse than the new accepted benchmark from iteration 28 and improves only 2022 and 2025 relative to that benchmark. The notebook's transferable value here is concentrated in filtered quality, not in the full Mike feature block, so the combined candidate is rejected.
- Next step: Keep iteration 28 as the active benchmark and mine other winner notebooks for similarly targeted strength-estimation ideas rather than broad feature dumps.

### 2026-03-08 15:11 UTC - Vilnius Spline Calibration on Iteration 28

- Hypothesis: The Vilnius notebook's one-dimensional spline from predicted point spread to win probability may outperform the current bracket-aware logistic calibration even when the regressor and feature stack stay fixed.
- Data window: Same as iteration 28, using the current pooled Selection Sunday feature table with bracket features and Mike filtered quality.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 28.
- LightGBM config: Same pooled `LGBMRegressor` as iteration 28. Probability calibration changes from nested OOF logistic on predicted margin plus bracket context to nested OOF spline calibration on predicted margin only with clipping at `[-25, 25]`.
- Mean Brier: 0.173795.
- Fold scores: 2021=0.193149, 2022=0.188645, 2023=0.174147, 2024=0.164345, 2025=0.148687.
- Artifacts: `artifacts/iteration_30_vilnius_spline_calibration/validation_summary.json`.
- Notes: The spline map improves only 2022 and 2023 versus the accepted benchmark from iteration 28 and loses clearly on average. The Vilnius calibration idea does not beat the stronger bracket-aware logistic map in this repo, so this candidate is rejected.
- Next step: Test the notebook's exact Elo variant separately, since the remaining transferable value may be in same-season team-strength features rather than the probability map.

### 2026-03-08 15:11 UTC - Vilnius High-K Elo on Iteration 28

- Hypothesis: The Vilnius notebook's same-season Elo recipe with `base_elo=1000`, `k_factor=100`, and no location adjustment may add useful volatility-aware strength signal beyond the current bracket-aware model.
- Data window: Same as iteration 28, using same-season regular-season compact results to build end-of-season Elo ratings for each division.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 28 features plus `low_vilnius_elo`, `high_vilnius_elo`, and `delta_vilnius_elo`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 28.
- Mean Brier: 0.170745.
- Fold scores: 2021=0.187647, 2022=0.191219, 2023=0.176929, 2024=0.161085, 2025=0.136846.
- Artifacts: `artifacts/iteration_31_vilnius_elo/validation_summary.json`.
- Notes: This Elo block improves only 2022 and 2025 versus iteration 28 and is still slightly worse on average. The notebook's more aggressive Elo update rule is not enough to clear the acceptance rule here, so this candidate is rejected.
- Next step: Test the Vilnius filtered-quality analogue, which is the notebook's strongest remaining same-season strength-estimation idea.

### 2026-03-08 15:11 UTC - Vilnius Filtered Quality

- Hypothesis: The Vilnius notebook's filtered same-season quality fit on seeded teams plus upset winners, without the Mike ridge penalty, may estimate tournament-relevant strength more cleanly than the accepted Mike filtered quality feature.
- Data window: Same as iteration 28, using same-season regular-season compact results and same-season tournament seeds to define the filtered fit population for each season and division.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low_vilnius_quality`, `high_vilnius_quality`, and `delta_vilnius_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 28.
- Mean Brier: 0.169282.
- Fold scores: 2021=0.187514, 2022=0.190833, 2023=0.174253, 2024=0.157493, 2025=0.136317.
- Artifacts: `artifacts/iteration_32_vilnius_filtered_quality/validation_summary.json`.
- Notes: This candidate lowers the mean Brier and improves 4 of 5 folds versus the accepted benchmark from iteration 28, missing only 2021. It is accepted as the new current best model and suggests the Vilnius notebook's main transferable value here is its unregularized filtered quality fit rather than its calibration or Elo choices.
- Next step: Test whether the notebook's narrower selected-stat subset can improve the new accepted benchmark or whether it just removes useful signal.

### 2026-03-08 15:11 UTC - Vilnius Selected-Stat Pruning

- Hypothesis: The Vilnius notebook's narrower detailed-stat subset may denoise the current accepted bracket-aware model by dropping the offensive- and defensive-rebound averages from the broader pooled stat block.
- Data window: Same as iteration 32.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 28 features with `low/high/delta avg_or` and `low/high/delta avg_dr` removed.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 32.
- Mean Brier: 0.170658.
- Fold scores: 2021=0.186425, 2022=0.193090, 2023=0.179365, 2024=0.160595, 2025=0.133814.
- Artifacts: `artifacts/iteration_33_vilnius_selected_stat_subset/validation_summary.json`.
- Notes: The stat-pruned candidate improves only 2021 and 2025 versus the new benchmark from iteration 32 and loses on average. The Vilnius stat subset is too aggressive once bracket features and filtered quality are already present, so this candidate is rejected.
- Next step: Keep iteration 32 as the active benchmark and move on to the next previous-winner notebook or to final-submission wiring when requested.

### 2026-03-08 15:30 UTC - Optuna LightGBM Tuning On Iteration 32

- Hypothesis: The current best pooled margin-regression pipeline may still have untapped LightGBM headroom, and a small Optuna search over regressor hyperparameters can beat the hand-carried smooth parameter family without changing the feature stack or calibrator design.
- Data window: Same as iteration 32, using the current pooled Selection Sunday feature table with Vilnius filtered quality and the same bracket-aware nested OOF logistic calibration.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 32.
- LightGBM config: Optuna TPE search with 20 trials over pooled `LGBMRegressor` hyperparameters, freezing `n_estimators=400` and using no early stopping. Search space covered `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`, and `min_split_gain`.
- Mean Brier: No accepted trial. Best raw trial mean Brier `0.168698`.
- Fold scores: Best raw trial 2021=0.182792, 2022=0.188018, 2023=0.178656, 2024=0.159643, 2025=0.134380.
- Artifacts: `artifacts/iteration_34_optuna_tune_iteration_32/validation_summary.json`.
- Notes: The best raw trial lowered the average Brier versus iteration 32 but improved only 3 of 5 folds: 2021, 2022, and 2025. Under the project rule, it is rejected. The search therefore suggests there is still average-score headroom in LightGBM tuning, but the current best remains more robust across folds than the best 20-trial Optuna candidate.
- Next step: If we revisit tuning, bias the search toward recovering 2023 and 2024 rather than chasing average Brier alone, or run a larger search only after deciding that the 4-of-5 gate should remain unchanged.

### 2026-03-08 15:42 UTC - Recency-Weighted Filtered Quality

- Hypothesis: The current best filtered-quality feature may underweight late-season information, so solving the same Vilnius-style quality system with heavier weights on later regular-season games can improve tournament fit without changing the downstream model or calibrator.
- Data window: Same as iteration 32, using same-season regular-season compact results and same-season tournament seeds, with game weights increasing linearly from `0.5` to `1.5` by regular-season `DayNum`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low_recency_weighted_quality`, `high_recency_weighted_quality`, and `delta_recency_weighted_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 32.
- Mean Brier: 0.168745.
- Fold scores: 2021=0.185636, 2022=0.188307, 2023=0.174950, 2024=0.158388, 2025=0.136445.
- Artifacts: `artifacts/iteration_35_recency_weighted_quality/validation_summary.json`.
- Notes: This candidate lowers the average Brier versus iteration 32, but it improves only 2021 and 2022 while regressing on 2023, 2024, and 2025. The recency weighting is promising but not robust enough under the 4-of-5 rule, so it is rejected.
- Next step: Test whether capping blowout margins produces a cleaner quality estimate than pure recency weighting.

### 2026-03-08 15:45 UTC - Capped-Margin Filtered Quality

- Hypothesis: The current best filtered-quality feature may be too sensitive to regular-season blowouts, so capping the point-margin target inside the Vilnius-style quality solve can reduce noise and improve robustness.
- Data window: Same as iteration 32, using same-season regular-season compact results and same-season tournament seeds, with regular-season score margins clipped to `[-15, 15]` before solving the filtered quality system.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low_capped_margin_quality`, `high_capped_margin_quality`, and `delta_capped_margin_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 32.
- Mean Brier: 0.168692.
- Fold scores: 2021=0.184963, 2022=0.188442, 2023=0.177700, 2024=0.155115, 2025=0.137240.
- Artifacts: `artifacts/iteration_36_capped_margin_quality/validation_summary.json`.
- Notes: This candidate lowers the average Brier versus iteration 32 and improves 2021, 2022, and 2024, but it regresses on 2023 and 2025. It still fails the 4-of-5-fold rule, so it is rejected.
- Next step: Combine recency weighting and margin capping to test whether the two partial gains are complementary.

### 2026-03-08 15:47 UTC - Recency-Weighted And Capped Filtered Quality

- Hypothesis: Recency weighting and margin capping may be complementary inside the filtered quality solve, producing a stronger tournament-oriented strength estimate than either modification alone.
- Data window: Same as iteration 32, using same-season regular-season compact results and same-season tournament seeds, with game weights increasing linearly from `0.5` to `1.5` by `DayNum` and score margins clipped to `[-15, 15]`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 26 features plus `low_recency_capped_quality`, `high_recency_capped_quality`, and `delta_recency_capped_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 32.
- Mean Brier: 0.169482.
- Fold scores: 2021=0.185385, 2022=0.189119, 2023=0.178748, 2024=0.159206, 2025=0.134954.
- Artifacts: `artifacts/iteration_37_recency_capped_quality/validation_summary.json`.
- Notes: This combined replacement improves 2021, 2022, and 2025 versus iteration 32, but it regresses on 2023 and 2024 and also misses the benchmark average. The recency and capping signals appear complementary to the current best, but not strong enough to replace it directly, so this candidate is rejected.
- Next step: Keep the current Vilnius quality and test alternative quality views as additive features instead of replacement features.

### 2026-03-08 15:52 UTC - Add Capped-Margin Quality To Iteration 32

- Hypothesis: The capped-margin quality view may be useful as a second same-season strength estimate even though it is not strong enough to replace the current Vilnius quality on its own, because the two quality views improve different folds and may be complementary.
- Data window: Same as iteration 32, using same-season regular-season compact results and same-season tournament seeds, with the added quality view built from score margins clipped to `[-15, 15]`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 32 features plus `low_capped_margin_quality`, `high_capped_margin_quality`, and `delta_capped_margin_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 32.
- Mean Brier: 0.168094.
- Fold scores: 2021=0.186586, 2022=0.190231, 2023=0.173654, 2024=0.154136, 2025=0.135862.
- Artifacts: `artifacts/iteration_38_vilnius_plus_capped_quality/validation_summary.json`.
- Notes: This candidate lowers the mean Brier and improves all 5 folds versus iteration 32. It is accepted as the new current best model. The most useful follow-up from the failed replacement runs was not to swap out the existing quality feature, but to keep the baseline Vilnius quality and add the capped-margin quality as a complementary second view.
- Next step: Stop the current search here and carry this new accepted benchmark forward before considering any further tuning or additive quality views.

### 2026-03-08 15:56 UTC - Add Recency-Weighted Quality To Iteration 38

- Hypothesis: The recency-weighted quality view may still help once paired with the stronger iteration 38 benchmark, even though it was not a robust standalone replacement for the Vilnius quality feature.
- Data window: Same as iteration 38, using same-season regular-season compact results and same-season tournament seeds, with the added quality view weighted from `0.5` to `1.5` by regular-season `DayNum`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 38 features plus `low_recency_weighted_quality`, `high_recency_weighted_quality`, and `delta_recency_weighted_quality`.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 38.
- Mean Brier: 0.169634.
- Fold scores: 2021=0.187124, 2022=0.189512, 2023=0.175996, 2024=0.158652, 2025=0.136888.
- Artifacts: `artifacts/iteration_39_vilnius_plus_recency_quality/validation_summary.json`.
- Notes: This candidate improves only 2022 versus the new accepted benchmark from iteration 38 and regresses the other four folds while also missing the benchmark mean. The recency-weighted quality view is rejected as an additive feature.
- Next step: Test the combined additive quality-view variant in case the recency signal only becomes useful when paired with the capped-margin quality view.

### 2026-03-08 16:00 UTC - Add Both Extra Quality Views To Iteration 38

- Hypothesis: The capped-margin and recency-weighted quality views may only become useful together, giving the model multiple tournament-focused strength views that are individually weak but jointly complementary to the accepted iteration 38 stack.
- Data window: Same as iteration 38, using same-season regular-season compact results and same-season tournament seeds, with one added quality view clipped to `[-15, 15]` and the other weighted from `0.5` to `1.5` by `DayNum`.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 38 features plus the recency-weighted quality block and the capped-margin quality block.
- LightGBM config: Same pooled `LGBMRegressor` and same bracket-aware nested OOF logistic calibration as iteration 38.
- Mean Brier: 0.168543.
- Fold scores: 2021=0.186548, 2022=0.190135, 2023=0.175093, 2024=0.155350, 2025=0.135591.
- Artifacts: `artifacts/iteration_40_vilnius_plus_quality_views/validation_summary.json`.
- Notes: This combined additive candidate improves 2021, 2022, and 2025 versus iteration 38, but it regresses on 2023 and 2024 and still misses the benchmark average. The quality-variant path now looks exhausted as a direct next step, so this candidate is rejected.
- Next step: Move to model-composition ideas on top of iteration 38, starting with the strongest complementary parameterization we already observed in the Optuna search.

### 2026-03-08 16:05 UTC - Optuna Parameterization On Iteration 38

- Hypothesis: The best raw Optuna parameterization found on iteration 32 may become robust enough once paired with the stronger iteration 38 feature stack, even if it was not accepted on the earlier benchmark.
- Data window: Same as iteration 38.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 38.
- LightGBM config: Single pooled `LGBMRegressor` with the best raw Optuna parameters from iteration 34: `learning_rate=0.011719612493324991`, `n_estimators=400`, `num_leaves=104`, `min_child_samples=102`, `subsample=0.8893860711323764`, `colsample_bytree=0.9911958047986411`, `reg_lambda=3.644791535514301`, `min_split_gain=0.25056023182996895`, followed by the same bracket-aware nested OOF logistic calibration.
- Mean Brier: 0.167566.
- Fold scores: 2021=0.180643, 2022=0.186684, 2023=0.178996, 2024=0.157920, 2025=0.133585.
- Artifacts: `artifacts/iteration_41_iteration_38_optuna_params/validation_summary.json`.
- Notes: This candidate lowers the average Brier materially versus iteration 38, but it improves only 2021, 2022, and 2025 while regressing on 2023 and 2024. The Optuna parameterization remains complementary but not robust enough as a replacement, so it is rejected.
- Next step: Blend the iteration 38 baseline regressor and this Optuna-parameterized regressor at the raw-margin level before calibration.

### 2026-03-08 16:10 UTC - Baseline-Heavy Margin Blend With Tuned Regressor

- Hypothesis: The tuned Optuna regressor may be useful mainly as a complementary component rather than as a replacement, so blending its raw margin predictions with the accepted iteration 38 regressor before calibration can preserve the strong 2023 and 2024 folds while recovering the tuned model's gains in 2021, 2022, and 2025.
- Data window: Same as iteration 38.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 38.
- LightGBM config: Two pooled `LGBMRegressor` models on the iteration 38 feature table. Raw margins are blended `0.65` accepted baseline and `0.35` tuned Optuna regressor, then mapped to probabilities with the same bracket-aware nested OOF logistic calibrator.
- Mean Brier: 0.167751.
- Fold scores: 2021=0.184716, 2022=0.188875, 2023=0.175112, 2024=0.155185, 2025=0.134865.
- Artifacts: `artifacts/iteration_42_iteration_38_optuna_blend/validation_summary.json`.
- Notes: This blend lowers the average Brier versus iteration 38, but it still improves only 2021, 2022, and 2025 while regressing on 2023 and 2024. The composition idea remains promising, but this weight is still too aggressive toward the tuned model, so the candidate is rejected.
- Next step: Try a more conservative blend weight that keeps the accepted baseline closer to the center of the final raw-margin prediction.

### 2026-03-08 16:14 UTC - Conservative Margin Blend With Tuned Regressor

- Hypothesis: A smaller tuned-model contribution may keep the accepted iteration 38 fold stability while still capturing part of the Optuna model's upside in 2021, 2022, and 2025.
- Data window: Same as iteration 38.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 38.
- LightGBM config: Two pooled `LGBMRegressor` models on the iteration 38 feature table. Raw margins are blended `0.80` accepted baseline and `0.20` tuned Optuna regressor, then mapped to probabilities with the same bracket-aware nested OOF logistic calibrator.
- Mean Brier: 0.167883.
- Fold scores: 2021=0.185551, 2022=0.189449, 2023=0.174438, 2024=0.154701, 2025=0.135274.
- Artifacts: `artifacts/iteration_43_iteration_38_optuna_blend_conservative/validation_summary.json`.
- Notes: The more conservative blend still improves only 2021, 2022, and 2025 versus iteration 38 while regressing on 2023 and 2024. The global blend path is not fixing the same fold pair, so this candidate is rejected.
- Next step: Move to division-specific composition, since the tuned model appears materially more useful for women than for men on the current feature stack.

### 2026-03-08 16:18 UTC - Division-Specific Margin Models On Iteration 38

- Hypothesis: The Optuna-tuned regressor appears complementary by division rather than globally, so using the accepted baseline regressor for men and the tuned regressor for women on the same iteration 38 feature stack can capture the tuned model's women-side gains without reintroducing the men-side regression seen in 2023.
- Data window: Same as iteration 38.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 38.
- LightGBM config: Division-specific pooled-margin composition on a shared feature table. Men use the accepted iteration 38 `LGBMRegressor`; women use the Optuna-tuned `LGBMRegressor` from iteration 41. The resulting raw margins are passed through the same bracket-aware nested OOF logistic calibrator.
- Mean Brier: 0.167112.
- Fold scores: 2021=0.183228, 2022=0.189536, 2023=0.173191, 2024=0.154749, 2025=0.134857.
- Artifacts: `artifacts/iteration_44_division_specific_margin_models/validation_summary.json`.
- Notes: This candidate lowers the mean Brier and improves 4 of 5 folds versus iteration 38, missing only 2024. It is accepted as the new current best model. The key improvement came from letting the tuned regressor influence women only, instead of forcing its tradeoffs onto both divisions.
- Next step: Carry iteration 44 forward as the active benchmark before revisiting any additional tuning or calibration changes.

### 2026-03-08 20:06 UTC - Division-Specific Calibration On Iteration 44

- Hypothesis: The division-specific raw-margin composition from iteration 44 may need separate men and women logistic calibrators, even if the underlying bracket-context matrix stays unchanged.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 44.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 44. The probability mapping is changed to nested OOF division-specific logistic calibration, fitting one calibrator on men's inner OOF rows and one on women's inner OOF rows.
- Mean Brier: 0.166899.
- Fold scores: 2021=0.178954, 2022=0.189935, 2023=0.179881, 2024=0.149173, 2025=0.136553.
- Artifacts: `artifacts/iteration_45_division_specific_calibration/validation_summary.json`.
- Notes: This candidate lowers the average Brier versus iteration 44 but improves only 2021 and 2024, while regressing on 2022, 2023, and 2025. It fails the 4-of-5 robustness rule and is rejected.
- Next step: Keep the accepted division-specific margin composition and test whether a single pooled calibrator with explicit division-aware calibration features can capture the same upside more robustly.

### 2026-03-08 20:18 UTC - Division-Aware Calibration Matrix On Iteration 44

- Hypothesis: The accepted division-specific raw-margin composition may need only a richer pooled calibration matrix, not separate calibrators, with division-awareness injected directly through calibration features and interactions.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 44.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 44. The probability mapping is changed to a nested OOF pooled logistic calibrator with extra division-aware calibration terms: `men_women`, `predicted_margin x men_women`, `possible_round x men_women`, and `delta_seed_number x men_women`.
- Mean Brier: 0.166514.
- Fold scores: 2021=0.179715, 2022=0.190517, 2023=0.176952, 2024=0.150380, 2025=0.135005.
- Artifacts: `artifacts/iteration_46_division_aware_calibration_matrix/validation_summary.json`.
- Notes: This candidate lowers the average Brier versus iteration 44 but improves only 2021 and 2024, while regressing on 2022, 2023, and 2025. It fails the 4-of-5 robustness rule and is rejected.
- Next step: Shift away from calibration-only changes and test raw-margin feature or model-composition ideas that can improve the weaker men's folds without giving back the women-side gains from iteration 44.

### 2026-03-08 20:16 UTC - Optuna Search For Women Params On Iteration 44

- Hypothesis: Iteration 44's only fold-level miss versus iteration 38 is 2024, and that regression comes mostly from the women-side tuned regressor, so a 20-trial LightGBM search on the women branch alone may recover 2024 without sacrificing the other four accepted gains.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 44.
- LightGBM config: Men are fixed to the accepted iteration 38 regressor. Women use a 20-trial Optuna search over pooled `LGBMRegressor` params with `n_estimators=400` frozen and no early stopping, followed by the same nested OOF bracket-aware logistic calibrator.
- Mean Brier: No accepted trial. Best raw mean Brier 0.167112.
- Fold scores: Best raw trial reproduces the iteration 44 benchmark: 2021=0.183228, 2022=0.189536, 2023=0.173191, 2024=0.154749, 2025=0.134857.
- Artifacts: `artifacts/iteration_47_optuna_women_params_on_iteration_44/validation_summary.json`.
- Notes: The search did not find any accepted trial, and the best raw trial exactly matched the existing women-side parameterization from iteration 44. That strongly suggests the current women branch is already near the best point within this search space, so the women-side tuning path is exhausted for now.
- Next step: Move the same direct search logic to the men branch, where the current accepted model is still using the older baseline parameterization rather than a composition-specific optimum.

### 2026-03-08 20:23 UTC - Optuna Search For Men Params On Iteration 44

- Hypothesis: The iteration 44 men branch still uses the older baseline parameterization, so a 20-trial LightGBM search on the men branch alone may improve the combined model while holding the accepted women-side tuned regressor fixed.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 44.
- LightGBM config: Women are fixed to the accepted tuned regressor from iteration 44. Men use a 20-trial Optuna search over pooled `LGBMRegressor` params with `n_estimators=400` frozen and no early stopping, followed by the same nested OOF bracket-aware logistic calibrator.
- Mean Brier: No accepted trial. Best raw mean Brier 0.167159.
- Fold scores: Best raw trial 2021=0.185361, 2022=0.185693, 2023=0.174391, 2024=0.154278, 2025=0.136072.
- Artifacts: `artifacts/iteration_48_optuna_men_params_on_iteration_44/validation_summary.json`.
- Notes: The search found a modestly better raw mean than iteration 44, but the best trial improved only 2022 and 2024 while regressing on 2021, 2023, and 2025. This rules out simple men-side parameter tuning as the next robust upgrade.
- Next step: Return to quality-feature exploration, where the prior accepted improvements actually came from, and test a materially new additive quality view rather than more parameter-only changes.

### 2026-03-08 20:31 UTC - Add Two-Hop Seed-Closure Quality To Iteration 44

- Hypothesis: The Vilnius quality block may still have headroom if we broaden the latent graph from `seeded teams + direct upset winners` to a two-hop seeded-closure graph, then expose that as a second additive quality view on top of the accepted iteration 44 feature stack.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 44 features plus `low_seed_closure_quality`, `high_seed_closure_quality`, and `delta_seed_closure_quality`, where the quality solve includes seeded teams and teams connected by up to two upset hops.
- LightGBM config: Same division-specific `LGBMRegressor` setup and same nested OOF bracket-aware logistic calibration as iteration 44.
- Mean Brier: 0.167464.
- Fold scores: 2021=0.180844, 2022=0.191450, 2023=0.176379, 2024=0.155306, 2025=0.133339.
- Artifacts: `artifacts/iteration_49_seed_closure_quality/validation_summary.json`.
- Notes: This candidate improves 2021 and 2025 versus iteration 44 but regresses on 2022, 2023, and 2024. The broader seeded-closure graph is directionally interesting, but the uncapped version is not robust enough and is rejected.
- Next step: Keep the expanded relevance graph idea and combine it with the capped-margin stabilization that already helped in iteration 38.

### 2026-03-08 20:34 UTC - Add Capped Two-Hop Seed-Closure Quality To Iteration 44

- Hypothesis: The broader seeded-closure graph from iteration 49 may need the same margin-cap stabilization that already proved complementary in iteration 38, so the right test is the expanded relevance graph with capped margins inside the latent-quality solve.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 44 features plus `low_capped_seed_closure_quality`, `high_capped_seed_closure_quality`, and `delta_capped_seed_closure_quality`, where the quality solve uses two-hop seeded closure and clips regular-season margins to `[-15, 15]`.
- LightGBM config: Same division-specific `LGBMRegressor` setup and same nested OOF bracket-aware logistic calibration as iteration 44.
- Mean Brier: 0.167455.
- Fold scores: 2021=0.181991, 2022=0.189030, 2023=0.176805, 2024=0.154935, 2025=0.134516.
- Artifacts: `artifacts/iteration_50_capped_seed_closure_quality/validation_summary.json`.
- Notes: This is the strongest structural-quality variant after iteration 44 so far, improving 2021, 2022, and 2025, but it still regresses on 2023 and 2024 and therefore fails the 4-of-5 rule. The graph-expansion idea remains plausible, but it now looks like the remaining issue is stability of the latent solve rather than relevance coverage alone.
- Next step: Test a stabilized latent-quality variant, likely with ridge-style shrinkage, instead of continuing to broaden the graph.

### 2026-03-08 20:37 UTC - Add Ridge-Stabilized Capped Two-Hop Seed-Closure Quality To Iteration 44

- Hypothesis: The capped seeded-closure variant from iteration 50 may still be viable if the latent-quality solve is stabilized with ridge-style shrinkage toward zero for non-bucket teams.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 44 features plus `low_ridge_capped_seed_closure_quality`, `high_ridge_capped_seed_closure_quality`, and `delta_ridge_capped_seed_closure_quality`, using two-hop seeded closure, capped margins, and `ridge_lambda=1.0`.
- LightGBM config: Same division-specific `LGBMRegressor` setup and same nested OOF bracket-aware logistic calibration as iteration 44.
- Mean Brier: 0.167878.
- Fold scores: 2021=0.182595, 2022=0.190396, 2023=0.176238, 2024=0.157376, 2025=0.132783.
- Artifacts: `artifacts/iteration_51_ridge_capped_seed_closure_quality/validation_summary.json`.
- Notes: This candidate improves 2021 and 2025 but regresses on 2022, 2023, and 2024, landing further from acceptance than iteration 50. Ridge shrinkage did not salvage the expanded seeded-closure graph, so that branch is rejected for now.
- Next step: Test ridge stabilization on the original one-hop capped quality view instead of on the broader seeded-closure graph.

### 2026-03-08 20:41 UTC - Add Ridge-Stabilized Capped Quality To Iteration 44

- Hypothesis: The accepted capped-quality add-on from iteration 38 may still have a useful complementary variant if the same one-hop filtered graph is solved with ridge shrinkage, producing a smoother second strength view without changing the underlying tournament-focused relevance set.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Iteration 44 features plus `low_ridge_capped_quality`, `high_ridge_capped_quality`, and `delta_ridge_capped_quality`, using the original one-hop filtered graph, capped margins, and `ridge_lambda=1.0`.
- LightGBM config: Same division-specific `LGBMRegressor` setup and same nested OOF bracket-aware logistic calibration as iteration 44.
- Mean Brier: 0.167805.
- Fold scores: 2021=0.182972, 2022=0.189224, 2023=0.175073, 2024=0.156781, 2025=0.134974.
- Artifacts: `artifacts/iteration_52_ridge_capped_quality/validation_summary.json`.
- Notes: This candidate improves 2021 and 2022 versus iteration 44 but regresses on 2023, 2024, and 2025. The result confirms that additive quality variants are no longer a strong enough standalone branch, so further progress is more likely to come from model composition than from another isolated quality view.
- Next step: Treat the strongest near-miss, iteration 50, as a complementary model and test a conservative raw-margin blend with iteration 44.

### 2026-03-08 20:48 UTC - Conservative Raw-Margin Blend Of Iteration 44 And Iteration 50

- Hypothesis: Iteration 50 is the strongest near-miss after iteration 44, so a conservative raw-margin blend can keep its gains in 2021, 2022, and 2025 while smoothing away the small fold regressions that blocked it from acceptance as a standalone replacement.
- Data window: Same as iteration 44.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Two synchronized feature stacks. The benchmark branch uses the accepted iteration 44 feature stack. The alternative branch uses the iteration 50 feature stack with capped two-hop seeded-closure quality. Raw margins are blended `0.75` benchmark and `0.25` alternative before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 44 on both branches, with only the feature stack differing on the alternative side.
- Mean Brier: 0.167003.
- Fold scores: 2021=0.182754, 2022=0.189200, 2023=0.173824, 2024=0.154629, 2025=0.134609.
- Artifacts: `artifacts/iteration_53_iteration_44_iteration_50_blend/validation_summary.json`.
- Notes: This candidate lowers the mean Brier versus iteration 44 and improves 4 of 5 folds, missing only 2023 by a small amount. It is accepted as the new current best model. The key win came from using iteration 50 as a complementary margin source instead of forcing it to replace the accepted baseline.
- Next step: Carry iteration 53 forward as the new benchmark before testing any further compositions or calibrator changes.

### 2026-03-09 00:06 UTC - More Conservative Raw-Margin Blend Of Iteration 44 And Iteration 50

- Hypothesis: Iteration 53 misses only 2023, so a lighter alternative-model weight may preserve the accepted gains in 2021, 2022, 2024, and 2025 while smoothing away the remaining 2023 regression.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53, but with raw margins blended `0.80` benchmark and `0.20` alternative before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the blend weight changing.
- Mean Brier: 0.167016.
- Fold scores: 2021=0.182842, 2022=0.189254, 2023=0.173680, 2024=0.154640, 2025=0.134664.
- Artifacts: `artifacts/iteration_54_iteration_44_iteration_50_blend_more_conservative/validation_summary.json`.
- Notes: This candidate does improve 2023 versus iteration 53, but it regresses on 2021, 2022, 2024, and 2025, so it fails the 4-of-5 rule. The remaining issue is not the global weight alone. The next step is a division-specific blend because iteration 50's 2023 regression is concentrated on the women side rather than shared equally across both divisions.
- Next step: Keep iteration 53 as the benchmark and test division-specific alternative weights for the iteration 44 / iteration 50 blend.

### 2026-03-09 00:14 UTC - Division-Specific Raw-Margin Blend Of Iteration 44 And Iteration 50

- Hypothesis: Iteration 50's remaining weakness is concentrated on the women side in 2023, so a division-specific blend should keep the stronger benchmark-heavy weight for women while preserving the accepted iteration 53 weight on men.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53, but raw margins are blended by division before the shared nested OOF bracket-aware logistic calibrator: men `0.75/0.25`, women `0.85/0.15`.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the division-specific blend weights changing.
- Mean Brier: 0.167022.
- Fold scores: 2021=0.182930, 2022=0.189214, 2023=0.173484, 2024=0.154715, 2025=0.134769.
- Artifacts: `artifacts/iteration_55_iteration_44_iteration_50_division_specific_blend/validation_summary.json`.
- Notes: This candidate does improve 2023 versus iteration 53, but it regresses on 2021, 2022, 2024, and 2025, so it fails the 4-of-5 rule. The raw-weight asymmetry alone is not enough to fix the remaining miss.
- Next step: Move away from raw-weight adjustments and test a targeted calibration change on top of iteration 53 that acts specifically on the women-side 2023 miss.

### 2026-03-09 00:22 UTC - Men-Only Raw-Margin Blend Of Iteration 44 And Iteration 50

- Hypothesis: The alternative branch is helping 2023 on the men side while hurting it on the women side, so the cleanest composition test is to keep the iteration 53 alternative weight on men and turn the alternative branch off entirely on women.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53, but raw margins are blended only for men's rows (`0.75/0.25`) while women's rows use the benchmark branch only before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the men-side blend active.
- Mean Brier: 0.167067.
- Fold scores: 2021=0.183201, 2022=0.189263, 2023=0.173002, 2024=0.154849, 2025=0.135017.
- Artifacts: `artifacts/iteration_57_iteration_44_iteration_50_men_only_blend/validation_summary.json`.
- Notes: This candidate does fix 2023 versus iteration 53, but it regresses on 2021, 2022, 2024, and 2025, so it still fails the 4-of-5 rule. That confirms the remaining issue is no longer best attacked with raw-weight edits.
- Next step: Run the targeted iteration 53 calibration variant instead of further blend experiments.

### 2026-03-09 00:31 UTC - Division-Specific Calibration On Iteration 53

- Hypothesis: Iteration 53's only remaining miss is concentrated on the women side, so the next best calibration test is to keep the raw-margin blend fixed and fit separate division calibrators, using the full bracket-aware matrix for men and a simpler women matrix to reduce overfit.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup and raw-margin blend as iteration 53.
- LightGBM config: Same division-specific `LGBMRegressor` setup and same `0.75/0.25` raw-margin blend as iteration 53. The probability mapping changes to nested OOF division-specific logistic calibration, with a simpler women calibration matrix built from predicted margin, `possible_round`, `delta_seed_number`, and `same_region_flag`.
- Mean Brier: 0.166563.
- Fold scores: 2021=0.178943, 2022=0.189578, 2023=0.179197, 2024=0.149015, 2025=0.136082.
- Artifacts: `artifacts/iteration_56_iteration_53_division_specific_calibration/validation_summary.json`.
- Notes: This candidate lowers the average Brier materially versus iteration 53, but it improves only 2021 and 2024 while regressing on 2022, 2023, and 2025. It fails the 4-of-5 rule and is rejected. That confirms the remaining gap is not most effectively addressed in the calibration layer.
- Next step: Reassess for a new complementary raw-margin source or composition that can improve 2023 without giving back the accepted gains from iteration 53.

### 2026-03-09 00:40 UTC - Dual-Margin Calibrator On Iteration 53

- Hypothesis: Instead of forcing iteration 44 and iteration 50 into a fixed pre-calibration blend, the calibrator may perform better if it can see both branch margins and their gap directly, learning when the alternative branch should be trusted or discounted.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same benchmark and alternative feature stacks as iteration 53. The raw-margin blend stays `0.75/0.25`, but the nested OOF logistic calibrator receives a branch-aware matrix containing benchmark margin, alternative margin, blended margin, prediction gap, and bracket context.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the probability mapping changing.
- Mean Brier: 0.168721.
- Fold scores: 2021=0.178367, 2022=0.192312, 2023=0.182475, 2024=0.152671, 2025=0.137778.
- Artifacts: `artifacts/iteration_58_iteration_53_dual_margin_calibration/validation_summary.json`.
- Notes: This candidate improves 2021 and 2024 but regresses sharply on 2022, 2023, and 2025, making it clearly worse than iteration 53 under the 4-of-5 rule. Giving the calibrator direct access to both branch margins is not enough on its own and appears to overfit the mapping layer.
- Next step: Keep iteration 53 raw probabilities as the stable anchor and test a meta-level probability stack that uses iteration 56 only as a complementary probability source.

### 2026-03-09 00:48 UTC - Probability Stack Of Iteration 53 And Iteration 56

- Hypothesis: Iteration 56 contains useful complementary probability signal even though it fails the 4-of-5 rule as a standalone replacement, so a nested OOF meta-level logistic stack on top of iteration 53 and iteration 56 probabilities may recover that signal without hard-coding a global tradeoff.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same benchmark and alternative feature stacks as iteration 53. The meta-calibrator receives `p53`, `p56`, division flag, round context, seed context, and the probability gap.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 and iteration 56; only the final probability mapping changes to a nested OOF logistic stack over the two probability sources.
- Mean Brier: 0.173092.
- Fold scores: 2021=0.184554, 2022=0.196259, 2023=0.184906, 2024=0.161142, 2025=0.138599.
- Artifacts: `artifacts/iteration_59_iteration_53_56_probability_stack/validation_summary.json`.
- Notes: This candidate is substantially worse than iteration 53 on both mean and fold stability. The meta-stack is rejected. That rules out the current calibration/stacking branch as a useful next path.
- Next step: Return to raw-margin composition, specifically a narrow women-only context gate that can turn off the alternative branch only where it appears to be harmful.

### 2026-03-09 01:03 UTC - Women Context Gate On Iteration 53

- Hypothesis: The remaining weakness in iteration 53 is a localized women-side failure, so turning off the alternative branch only for early-round women matchups with small seed gaps can keep the accepted gains while repairing the bad context.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53. Men keep the accepted `0.75/0.25` raw-margin blend. Women use the same blend by default, but rows with `possible_round <= 2` and `abs(delta_seed_number) <= 2` switch to the benchmark branch only before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the women-side context gate changing the raw-margin composition.
- Mean Brier: 0.167014.
- Fold scores: 2021=0.182760, 2022=0.189373, 2023=0.173474, 2024=0.154955, 2025=0.134507.
- Artifacts: `artifacts/iteration_60_iteration_53_women_context_gate/validation_summary.json`.
- Notes: This candidate improves 2023 and 2025 and is nearly unchanged on 2021, but it still regresses on 2022 and 2024, so it fails the 4-of-5 rule. The gate direction is plausible, but this version is too broad.
- Next step: Tighten the women-side context gate so fewer rows fall back to the benchmark branch.

### 2026-03-09 01:08 UTC - Tighter Women Context Gate On Iteration 53

- Hypothesis: The women-side gate from iteration 60 is directionally useful but too broad, so tightening the allowable seed-gap threshold should keep more of iteration 53's accepted behavior while preserving the 2023 repair.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 60, but the women fallback gate is tightened to `possible_round <= 2` and `abs(delta_seed_number) <= 1`.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the women-side gate threshold changing.
- Mean Brier: 0.167014.
- Fold scores: 2021=0.182760, 2022=0.189373, 2023=0.173474, 2024=0.154955, 2025=0.134507.
- Artifacts: `artifacts/iteration_61_iteration_53_women_context_gate_tighter/validation_summary.json`.
- Notes: This candidate reproduces the same fold scores as iteration 60, so tightening the seed-gap threshold alone did not materially change the effective women fallback set. It is rejected.
- Next step: Stop manual gate tweaking and search a small discrete grid of women-gate parameters instead.

### 2026-03-09 01:18 UTC - Women Gate Grid Search On Iteration 53

- Hypothesis: The women-side gate idea still has potential, but the exact cut points are unclear, so a small discrete search over round, seed-gap, and women gate weights may find an accepted local configuration near iteration 60.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53, with men fixed at the accepted `0.75/0.25` blend and women using a searched gate over round and seed-gap contexts.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the women-side gate parameters searched.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167007.
- Fold scores: Best raw configuration (`possible_round_max=2`, `seed_gap_max=1`, `women_gate_alternative_weight=0.1`) yields 2021=0.182753, 2022=0.189301, 2023=0.173614, 2024=0.154823, 2025=0.134547.
- Artifacts: `artifacts/iteration_62_iteration_53_women_gate_grid_search/validation_summary.json`.
- Notes: No searched women gate clears the 4-of-5 rule against iteration 53. The best raw configuration improves 2021, 2023, and 2025 but still regresses on 2022 and 2024. That exhausts the current seed/round gate family.
- Next step: Test a women-side confidence gate based on the blended raw margin itself instead of seed/round proxies.

### 2026-03-09 01:52 UTC - Women Confidence Gate Grid Search On Iteration 53

- Hypothesis: The remaining weakness in iteration 53 may be concentrated in low-confidence women rows, so a women-side gate driven by the blended raw margin magnitude should be a cleaner way to switch back toward the benchmark than seed and round proxies.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same synchronized branch setup as iteration 53, with men fixed at the accepted `0.75/0.25` blend and women using a searched low-confidence gate based on `abs(default blended raw margin)`.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on both branches, with only the women-side gate parameters searched.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.166954.
- Fold scores: Best raw configuration (`margin_abs_max=4.0`, `women_gate_alternative_weight=0.0`) yields 2021=0.182893, 2022=0.188996, 2023=0.173245, 2024=0.154701, 2025=0.134933.
- Artifacts: `artifacts/iteration_63_iteration_53_women_confidence_gate_grid_search/validation_summary.json`.
- Notes: No searched women confidence gate clears the 4-of-5 rule against iteration 53. The best raw configuration improves only 2022 and 2023 while regressing on 2021, 2024, and 2025. That exhausts the women-side gating branch.
- Next step: Move away from gating and test a genuinely new complementary raw-margin source rather than further edits to the iteration 44 and iteration 50 composition.

### 2026-03-09 02:07 UTC - Small Third-Source Blend Of Iteration 53 And Iteration 38

- Hypothesis: Iteration 38 is the only prior model that improves both 2023 and 2024 versus iteration 53, so adding it back as a very small third raw-margin source before calibration may repair the benchmark's remaining weak folds without giving back the accepted gains elsewhere.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: The benchmark branch is the accepted iteration 53 raw-margin blend of iteration 44 and iteration 50. An additional pooled branch from iteration 38 is mixed in at small searched weights before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on the benchmark side, plus the pooled iteration 38 `LGBMRegressor` on its own feature stack. The searched `iteration_38` weight grid is `0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167017.
- Fold scores: Best raw configuration (`iteration_38_weight=0.02`) yields 2021=0.182831, 2022=0.189210, 2023=0.173808, 2024=0.154608, 2025=0.134626.
- Artifacts: `artifacts/iteration_64_iteration_53_iteration_38_blend_search/validation_summary.json`.
- Notes: The small third-source blend does exactly what the fold diagnostics predicted, improving only 2023 and 2024 versus iteration 53 while regressing slightly on 2021, 2022, and 2025. That confirms iteration 38 is a useful corrective source, but not sufficient on its own to clear the 4-of-5 rule.
- Next step: Add a second small complementary source that historically helps 2021, 2022, and 2025, specifically the Optuna-parameterized iteration 41 branch, and test a conservative three-source search around iteration 53.

### 2026-03-09 02:24 UTC - Conservative Three-Source Blend Of Iteration 53, Iteration 38, And Iteration 41

- Hypothesis: Iteration 38 and iteration 41 have nearly complementary fold strengths against iteration 53, so adding both as very small pooled corrections around the accepted benchmark may recover the missing weak folds without destabilizing the benchmark's accepted gains.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: The benchmark branch is the accepted iteration 53 raw-margin blend of iteration 44 and iteration 50. Two additional pooled branches are mixed in before the shared nested OOF bracket-aware logistic calibrator: iteration 38 and the Optuna-parameterized iteration 41 on the same pooled feature stack.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 on the benchmark side, plus the pooled iteration 38 and pooled iteration 41 `LGBMRegressor` branches. The searched grid uses `iteration_38_weight in {0.01, 0.02, 0.03}` and `iteration_41_weight in {0.02, 0.04, 0.06}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167009.
- Fold scores: Best raw configuration (`iteration_38_weight=0.01`, `iteration_41_weight=0.02`) yields 2021=0.182744, 2022=0.189135, 2023=0.173906, 2024=0.154669, 2025=0.134591.
- Artifacts: `artifacts/iteration_65_iteration_53_iteration_38_iteration_41_blend_search/validation_summary.json`.
- Notes: This coarse grid lands on the opposite side of the tradeoff from iteration 64, improving 2021, 2022, and 2025 while losing the 2023 and 2024 repairs. The direction remains promising because the searched weights bracket the complementary fold patterns tightly, but the coarse search is still too aggressive on iteration 41.
- Next step: Refine the three-source search around `iteration_38_weight ~= 0.02` with smaller iteration 41 weights to see whether the tradeoff boundary contains a 4-of-5 configuration.

### 2026-03-09 02:37 UTC - Refined Three-Source Blend Of Iteration 53, Iteration 38, And Iteration 41

- Hypothesis: The coarse iteration 38 plus iteration 41 search bracketed a narrow tradeoff boundary, so a refined search with smaller iteration 41 weights around the `iteration_38 ~= 0.02` region may still contain a 4-of-5 improvement that the coarse grid skipped over.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same four-source raw-margin composition family as iteration 65: the accepted iteration 53 benchmark plus pooled iteration 38 and pooled iteration 41 corrections before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same model setup as iteration 65, with refined weights `iteration_38_weight in {0.015, 0.020, 0.025}` and `iteration_41_weight in {0.005, 0.010, 0.015, 0.020}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167013.
- Fold scores: Best raw configuration (`iteration_38_weight=0.015`, `iteration_41_weight=0.005`) yields 2021=0.182785, 2022=0.189193, 2023=0.173853, 2024=0.154596, 2025=0.134637.
- Artifacts: `artifacts/iteration_66_iteration_53_iteration_38_iteration_41_blend_refined_search/validation_summary.json`.
- Notes: The refined search confirms the branch is close but still not enough. It produces two different near-miss regions: one that improves `2022/2024`, and another that improves `2022/2023/2024`. None of the searched settings recover either 2021 or 2025 at the same time, so this four-source geometry is rejected.
- Next step: Shift to a different 2023 corrective source that stays much closer to iteration 53 on the other folds, starting with the men-only iteration 57 branch rather than continuing to push pooled iteration 38 and iteration 41 weights around.

### 2026-03-09 02:53 UTC - Three-Source Blend Of Iteration 53, Iteration 57, And Iteration 41

- Hypothesis: The men-only iteration 57 branch is the cleanest existing 2023 repair relative to iteration 53, so blending it with the 2021/2022/2025-leaning iteration 41 branch may be a better way to clear the 4-of-5 rule than continuing to use pooled iteration 38 as the corrective source.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: The accepted iteration 53 benchmark is blended with the men-only iteration 57 raw-margin branch and the pooled iteration 41 raw-margin branch before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same division-specific `LGBMRegressor` setup as iteration 53 and iteration 57 on the 44/50 feature stacks, plus the pooled iteration 41 `LGBMRegressor`. The searched grid uses `iteration_57_weight in {0.10, 0.20, 0.30}` and `iteration_41_weight in {0.01, 0.02, 0.03}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167005.
- Fold scores: Best raw configuration (`iteration_57_weight=0.10`, `iteration_41_weight=0.02`) yields 2021=0.182757, 2022=0.189189, 2023=0.173918, 2024=0.154636, 2025=0.134526.
- Artifacts: `artifacts/iteration_67_iteration_53_iteration_57_iteration_41_blend_search/validation_summary.json`.
- Notes: This family stays too close to iteration 53. The branch either improves `2022/2023` or `2021/2022/2025`, but never both at once, and it does not provide a path to a fourth improved fold. The local 2023 fix from iteration 57 is not strong enough to justify further tuning in this composition.
- Next step: Return to the stronger pooled 2023/2024 corrective source from iteration 38, but pair it with a different complementary source that leans toward 2021 and 2025 rather than toward 2022.

### 2026-03-09 03:12 UTC - Three-Source Blend Of Iteration 53, Iteration 38, And Iteration 49

- Hypothesis: Iteration 38 is still the strongest 2023/2024 corrective source, while iteration 49 is the strongest unused 2021/2025 source, so a conservative three-source blend around iteration 53 may finally clear the 4-of-5 rule by covering those complementary fold groups directly.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: The accepted iteration 53 benchmark is blended with pooled iteration 38 raw margins and the division-specific iteration 49 raw margins before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same model setup as the source runs, with `iteration_38_weight in {0.02, 0.03, 0.04}` and `iteration_49_weight in {0.005, 0.010, 0.015, 0.020}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167004.
- Fold scores: Best raw configuration (`iteration_38_weight=0.02`, `iteration_49_weight=0.015`) yields 2021=0.182817, 2022=0.189204, 2023=0.173774, 2024=0.154552, 2025=0.134677.
- Artifacts: `artifacts/iteration_68_iteration_53_iteration_38_iteration_49_blend_search/validation_summary.json`.
- Notes: This is the closest raw-margin composition yet. It improves `2023/2024/2025` and nearly matches the benchmark mean, missing acceptance by a very small amount while leaving only 2021 and 2022 unresolved. That makes it the right anchor for one more tightly targeted experiment with a very small iteration 41 add-on.
- Next step: Keep the best iteration 68 setting fixed and test whether a tiny iteration 41 contribution can recover 2021 and 2022 without giving back the new 2023/2024/2025 gains.

### 2026-03-09 03:26 UTC - Tiny Iteration 41 Add-On To The Best Iteration 68 Blend

- Hypothesis: The best iteration 68 setting already fixes `2023/2024/2025`, so a very small iteration 41 contribution may be enough to recover 2021 and 2022 without destabilizing the new gains.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same four-source raw-margin stack as the best iteration 68 setting, with fixed `iteration_38_weight=0.02` and `iteration_49_weight=0.015`, and only a tiny iteration 41 add-on searched before the shared nested OOF bracket-aware logistic calibrator.
- LightGBM config: Same source models as iteration 68, with `iteration_41_weight in {0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.167005.
- Fold scores: Best raw configuration (`iteration_41_weight=0.0100`) yields 2021=0.182823, 2022=0.189180, 2023=0.173853, 2024=0.154560, 2025=0.134615.
- Artifacts: `artifacts/iteration_69_iteration_68_plus_tiny_iteration_41_search/validation_summary.json`.
- Notes: The tiny iteration 41 add-on shifts the best region from improving `2023/2024/2025` to improving `2022/2024/2025`, but it still cannot recover a fourth improved fold. That exhausts the fixed-weight blend family around the iteration 68 anchor.
- Next step: Stop hand-searching global weights and move to a nested meta-level raw-margin combiner that can choose among the near-miss source models by matchup context instead of enforcing one global mixture.

### 2026-03-09 03:39 UTC - Nested Meta-Level Raw-Margin Blender

- Hypothesis: The recent near misses show that the source models are complementary but not under one global weight vector, so a nested LightGBM meta-model over raw margins and bracket context may learn when to trust each source better than manual weight searches.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Nested OOF raw margins from iteration 53, iteration 38, iteration 41, and iteration 49, plus source-vs-benchmark margin gaps and matchup context (`men_women`, `possible_round`, `delta_seed_number`, `same_region_flag`, `any_play_in_flag`).
- LightGBM config: Source models use their existing configs. The meta-layer is a small `LGBMClassifier` with `learning_rate=0.03`, `n_estimators=300`, `num_leaves=15`, `min_child_samples=100`, `subsample=0.9`, `colsample_bytree=0.9`, and `reg_lambda=5.0`, trained only on nested OOF source predictions from prior seasons.
- Mean Brier: 0.175344.
- Fold scores: 2021=0.199585, 2022=0.188573, 2023=0.189865, 2024=0.146859, 2025=0.151837.
- Artifacts: `artifacts/iteration_70_nested_meta_margin_blender/validation_summary.json`.
- Notes: This is not a near miss. The meta-blender badly overfits the small training history and is substantially worse than the accepted benchmark. The context-aware stacking family is rejected.
- Next step: Return to the strongest raw-margin anchor from iteration 68 and test whether a different calibrator, rather than another source model, can recover the missing folds.

### 2026-03-09 03:49 UTC - Division-Aware Recalibration On The Best Iteration 68 Raw Blend

- Hypothesis: The best fixed raw blend from iteration 68 already covers `2023/2024/2025`, so the remaining gap may be in the probability mapping rather than in the source weights. A richer division-aware calibrator on top of that raw anchor may recover 2021 and 2022.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Fixed raw-margin blend from the best iteration 68 setting: `0.965` benchmark iteration 53, `0.020` iteration 38, and `0.015` iteration 49.
- LightGBM config: Same source models as iteration 68. Only the probability mapping changes, using the division-aware calibration matrix from iteration 46 on top of the fixed raw blend.
- Mean Brier: 0.166510.
- Fold scores: 2021=0.179646, 2022=0.190334, 2023=0.177628, 2024=0.150256, 2025=0.134684.
- Artifacts: `artifacts/iteration_71_iteration_68_division_aware_calibration/validation_summary.json`.
- Notes: This candidate improves 2021 and 2024 but regresses on 2022, 2023, and 2025, so it still fails the 4-of-5 rule. Even so, it provides the missing 2021 recovery needed to complement the best tiny-iteration-41 branch.
- Next step: Blend the fold-safe probabilities from the best tiny-iteration-41 branch and this division-aware recalibration to see whether their complementary fold strengths can clear the 4-of-5 rule.

### 2026-03-09 04:03 UTC - Probability Blend Search Of Iteration 69 And Iteration 71

- Hypothesis: The best tiny-iteration-41 branch and the best division-aware recalibration branch are complementary at the fold level, so a modest probability blend between those complete, fold-safe pipelines may recover 2021 without giving back the accepted gains in 2022, 2024, and 2025.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same source-model stack as iterations 69 and 71. The only change is the final probability-level blend between the two completed pipelines.
- LightGBM config: Same source models and calibrators as iterations 69 and 71. The searched final blend weight is `iteration_71_probability_weight in {0.10, 0.15, 0.20, 0.25, 0.30}`.
- Mean Brier: Accepted. Best mean Brier 0.166629.
- Fold scores: Best configuration (`iteration_71_probability_weight=0.25`) yields 2021=0.181747, 2022=0.189185, 2023=0.174511, 2024=0.153322, 2025=0.134381.
- Artifacts: `artifacts/iteration_72_iteration_69_iteration_71_probability_blend_search/validation_summary.json`.
- Notes: This candidate lowers the mean Brier and improves 4 of 5 folds versus iteration 53, missing only 2023. It is accepted. The useful lesson was that the complementary signals lived in the calibrators, not in yet another raw-margin source.
- Next step: Freeze the accepted configuration into a dedicated non-search script and make it the new active benchmark.

### 2026-03-09 04:11 UTC - Fixed Probability Blend Of Iteration 69 And Iteration 71

- Hypothesis: The accepted search result from iteration 72 should reproduce exactly when moved into a dedicated fixed script, giving us a submission-capable benchmark instead of leaving the accepted model inside a search artifact.
- Data window: Same as iteration 53.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 72, with fixed source-model weights `iteration_38=0.020`, `iteration_49=0.015`, `iteration_41=0.010`, and a final probability blend of `0.75` from the iteration 69 branch plus `0.25` from the iteration 71 branch.
- LightGBM config: Same source models and calibrators as iteration 72. Only the search loop is removed.
- Mean Brier: 0.166629.
- Fold scores: 2021=0.181747, 2022=0.189185, 2023=0.174511, 2024=0.153322, 2025=0.134381.
- Artifacts: `artifacts/iteration_73_iteration_69_iteration_71_probability_blend/validation_summary.json`.
- Notes: This reproduces the accepted search result exactly and becomes the new current best model. It improves 2021, 2022, 2024, and 2025 versus iteration 53, with only 2023 still slightly worse.
- Next step: Carry iteration 73 forward as the active benchmark for any further work.

### 2026-03-10 03:27 UTC - Refined Probability Blend Search Around Iteration 73

- Hypothesis: The accepted `0.25` probability blend from iteration 73 may not be the true local optimum, so a narrow search around that weight may yield another accepted improvement without changing the underlying model family.
- Data window: Same as iteration 73.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 73. Only the final probability-blend weight is searched.
- LightGBM config: Same source models and calibrators as iteration 73. The searched final blend weight is `iteration_71_probability_weight in {0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.166585.
- Fold scores: Best raw configuration (`iteration_71_probability_weight=0.29`) yields 2021=0.181598, 2022=0.189202, 2023=0.174633, 2024=0.153126, 2025=0.134364.
- Artifacts: `artifacts/iteration_74_iteration_73_probability_blend_refined_search/validation_summary.json`.
- Notes: This is a mean improvement versus iteration 73, but it improves only `2021/2024/2025` and therefore fails the 4-of-5 rule. The weight search reveals a sharp boundary: lower weights improve `2022/2023`, higher weights improve `2021/2024/2025`, and the accepted `0.25` point sits between those regimes.
- Next step: Probe the narrow boundary between `0.25` and `0.26` directly to see whether a finer search can recover both 2021 and 2022 without giving back the accepted gains in 2024 and 2025.

### 2026-03-13 13:25 UTC - Division-Specific Probability Blend Search Around Iteration 73

- Hypothesis: The global iteration 73 blend is sitting on a fold-level boundary because men and women want different amounts of the iteration 71 recalibration, so a division-specific probability blend can clear the current 4-of-5 rule without changing the underlying source-model family.
- Data window: Same as iteration 73.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 73, but the final probability blend is allowed to use different iteration 71 weights for men and women after the fixed source-model stack from iterations 69 and 71.
- LightGBM config: Same source models and calibrators as iteration 73. The search grid uses `men_iteration_71_probability_weight in {0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29}` and `women_iteration_71_probability_weight in {0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29}` with cached fold-level iteration 69 and iteration 71 probabilities.
- Mean Brier: Accepted. Best mean Brier 0.166594.
- Fold scores: Best configuration (`men_iteration_71_probability_weight=0.29`, `women_iteration_71_probability_weight=0.24`) yields 2021=0.181618, 2022=0.189056, 2023=0.174500, 2024=0.153307, 2025=0.134488.
- Artifacts: `artifacts/iteration_76_iteration_73_division_specific_probability_blend_search/validation_summary.json`.
- Notes: This candidate lowers the mean Brier versus iteration 73 and improves 4 of 5 folds, missing only 2025. The important correction was asymmetry: men benefit from a higher iteration 71 probability weight while women need a slightly lower one. `uvx ruff check` and `uvx ty check` passed on the new search script.
- Next step: Freeze the accepted men/women split into a dedicated non-search script and promote it to the new active benchmark.

### 2026-03-13 13:26 UTC - Fixed Division-Specific Probability Blend From Iteration 76

- Hypothesis: The accepted division-specific search result from iteration 76 should reproduce exactly when moved into a dedicated fixed script, so the new best model is available as a stable benchmark instead of living only inside a search artifact.
- Data window: Same as iteration 73.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 76, with fixed source-model weights `iteration_38=0.020`, `iteration_49=0.015`, `iteration_41=0.010`, plus a division-specific final probability blend using `men_iteration_71_probability_weight=0.29` and `women_iteration_71_probability_weight=0.24`.
- LightGBM config: Same source models and calibrators as iteration 76. Only the search loop is removed.
- Mean Brier: 0.166594.
- Fold scores: 2021=0.181618, 2022=0.189056, 2023=0.174500, 2024=0.153307, 2025=0.134488.
- Artifacts: `artifacts/iteration_77_iteration_76_division_specific_probability_blend/validation_summary.json`.
- Notes: This reproduces the accepted search result exactly and becomes the new current best model. It improves 2021, 2022, 2023, and 2024 versus iteration 73, with only 2025 slightly worse. `uvx ruff check` and `uvx ty check` passed on the fixed benchmark script.
- Next step: Carry iteration 77 forward as the active benchmark for any further work.

### 2026-03-13 13:37 UTC - Men Probability-Gap Gate On Iteration 77

- Hypothesis: The remaining 2025 regression in iteration 77 may be concentrated in low-information men rows where the iteration 69 and iteration 71 probabilities are already close, so lowering the men-side iteration 71 weight only inside a probability-gap gate could recover 2025 without giving back the new accepted gains.
- Data window: Same as iteration 77.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 77, with men defaulting to the accepted `0.29` iteration 71 weight and women fixed at `0.24`, plus a men-only probability-gap gate that switches selected rows to a lower iteration 71 weight.
- LightGBM config: Same source models and calibrators as iteration 77. The search grid uses `men_gate_mode in {low_gap, high_gap}`, `men_gap_threshold in {0.01, 0.02, 0.03, 0.04, 0.05}`, and `men_gate_iteration_71_probability_weight in {0.24, 0.25, 0.26, 0.27, 0.28}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.166594.
- Fold scores: Best raw configuration (`men_gate_mode=low_gap`, `men_gap_threshold=0.01`, `men_gate_iteration_71_probability_weight=0.28`) yields 2021=0.181617, 2022=0.189058, 2023=0.174499, 2024=0.153310, 2025=0.134486.
- Artifacts: `artifacts/iteration_78_iteration_77_men_gap_gate_search/validation_summary.json`.
- Notes: The direction was plausible and the best point improved 2021, 2023, and 2025, but it still missed both the mean and the 4-of-5 rule versus iteration 77. The gap-based men rollback is too localized to clear acceptance on its own. `uvx ty check` passed on the search script.
- Next step: Try a broader men-only localization based on round context instead of probability-gap magnitude.

### 2026-03-13 13:39 UTC - Men Round Gate On Iteration 77

- Hypothesis: The men-side tradeoff in iteration 77 may be tied to bracket location rather than just branch disagreement, so rolling back the men iteration 71 weight only within a round-defined context could recover the missing fold more effectively than the probability-gap gate.
- Data window: Same as iteration 77.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 77, with men defaulting to the accepted `0.29` iteration 71 weight and women fixed at `0.24`, plus a men-only round gate that switches selected rows to a lower iteration 71 weight.
- LightGBM config: Same source models and calibrators as iteration 77. The search grid uses `men_round_gate_mode in {round_lte, round_gte}`, `men_round_threshold in {1, 2, 3, 4, 5}`, and `men_gate_iteration_71_probability_weight in {0.24, 0.25, 0.26, 0.27, 0.28}`.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.166593.
- Fold scores: Best raw configuration (`men_round_gate_mode=round_lte`, `men_round_threshold=1`, `men_gate_iteration_71_probability_weight=0.24`) yields 2021=0.181608, 2022=0.189056, 2023=0.174500, 2024=0.153312, 2025=0.134489.
- Artifacts: `artifacts/iteration_79_iteration_77_men_round_gate_search/validation_summary.json`.
- Notes: This was an even closer mean near miss, but it improved only 2021 and 2023 versus iteration 77. The men-only rollback still cannot clear the project robustness rule, which points to the remaining gain being more likely on the women side or in a cross-division composition. `uvx ty check` passed on the search script.
- Next step: Search a women-only localization instead of continuing to force the fix through the men branch.

### 2026-03-13 13:42 UTC - Women Probability-Gap Gate On Iteration 77

- Hypothesis: Iteration 77's remaining gap may live on the women side, where a higher iteration 71 weight could still be valuable on low-disagreement rows without giving back the accepted men-side gains, so a women-only probability-gap gate can outperform the fixed global women weight.
- Data window: Same as iteration 77.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 77, with men fixed at the accepted `0.29` iteration 71 weight, women defaulting to `0.24`, and a women-only probability-gap gate that raises the iteration 71 weight on selected rows.
- LightGBM config: Same source models and calibrators as iteration 77. The search grid uses `women_gate_mode in {low_gap, high_gap}`, `women_gap_threshold in {0.01, 0.02, 0.03, 0.04, 0.05}`, and `women_gate_iteration_71_probability_weight in {0.25, 0.26, 0.27, 0.28, 0.29}`.
- Mean Brier: Accepted. Best mean Brier 0.166546.
- Fold scores: Best configuration (`women_gate_mode=low_gap`, `women_gap_threshold=0.05`, `women_gate_iteration_71_probability_weight=0.29`) yields 2021=0.181609, 2022=0.189030, 2023=0.174476, 2024=0.153185, 2025=0.134430.
- Artifacts: `artifacts/iteration_80_iteration_77_women_gap_gate_search/validation_summary.json`.
- Notes: This candidate lowers the mean Brier versus iteration 77 and improves all 5 folds. The winning change is asymmetric and selective: keep the accepted men setup intact, but restore more iteration 71 influence for women on low-gap rows. `uvx ty check` passed on the search script.
- Next step: Freeze the accepted women-gap gate into a dedicated non-search script and promote it to the new active benchmark.

### 2026-03-13 13:44 UTC - Fixed Women Probability-Gap Gate From Iteration 80

- Hypothesis: The accepted women-gap search result from iteration 80 should reproduce exactly when moved into a dedicated fixed script, so the new best model is available as a stable benchmark rather than living only in a search artifact.
- Data window: Same as iteration 77.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 80, with men fixed at `iteration_71_probability_weight=0.29`, women defaulting to `0.24`, and women rows with `abs(iteration_71_probability - iteration_69_probability) <= 0.05` using `women_gate_iteration_71_probability_weight=0.29`.
- LightGBM config: Same source models and calibrators as iteration 80. Only the search loop is removed.
- Mean Brier: 0.166546.
- Fold scores: 2021=0.181609, 2022=0.189030, 2023=0.174476, 2024=0.153185, 2025=0.134430.
- Artifacts: `artifacts/iteration_81_iteration_80_women_gap_gate/validation_summary.json`.
- Notes: This reproduces the accepted search result exactly and becomes the new current best model. It improves all 5 folds versus iteration 77. `uvx ruff check` and `uvx ty check` passed on the fixed benchmark script.
- Next step: Carry iteration 81 forward as the active benchmark for any further work.

### 2026-03-13 13:53 UTC - Extended Women Probability-Gap Gate Search On Iteration 81

- Hypothesis: The accepted women-gap gate from iteration 81 finished on the search boundary for both threshold and gate weight, so extending the same low-gap women search farther in that direction may yield another accepted improvement without changing the model family.
- Data window: Same as iteration 81.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 81, with men fixed at `iteration_71_probability_weight=0.29`, women defaulting to `0.24`, and a women-only low-gap gate that now searches larger thresholds and larger gated iteration 71 weights.
- LightGBM config: Same source models and calibrators as iteration 81. The search grid uses `women_gap_threshold in {0.055, 0.06, 0.07, 0.08, 0.09, 0.10}` and `women_gate_iteration_71_probability_weight in {0.30, 0.31, 0.32, 0.33, 0.34, 0.35}` with `women_gate_mode=low_gap`.
- Mean Brier: Accepted. Best mean Brier 0.166476.
- Fold scores: Best configuration (`women_gap_mode=low_gap`, `women_gap_threshold=0.055`, `women_gate_iteration_71_probability_weight=0.35`) yields 2021=0.181577, 2022=0.188945, 2023=0.174584, 2024=0.152931, 2025=0.134343.
- Artifacts: `artifacts/iteration_82_iteration_81_women_gap_gate_extended_search/validation_summary.json`.
- Notes: This candidate lowers the mean Brier versus iteration 81 and improves 4 of 5 folds, missing only 2023. The previous accepted women-gap family was still under-tuned, and the winning extension raises the gated women iteration 71 weight materially while only expanding the low-gap region slightly. `uvx ruff check` and `uvx ty check` passed on the new search script.
- Next step: Freeze the accepted extended women-gap gate into a dedicated non-search script and promote it to the new active benchmark.

### 2026-03-13 13:54 UTC - Fixed Extended Women Probability-Gap Gate From Iteration 82

- Hypothesis: The accepted extended women-gap search result from iteration 82 should reproduce exactly when moved into a dedicated fixed script, so the new best model is available as a stable benchmark rather than living only in a search artifact.
- Data window: Same as iteration 81.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 82, with men fixed at `iteration_71_probability_weight=0.29`, women defaulting to `0.24`, and women rows with `abs(iteration_71_probability - iteration_69_probability) <= 0.055` using `women_gate_iteration_71_probability_weight=0.35`.
- LightGBM config: Same source models and calibrators as iteration 82. Only the search loop is removed.
- Mean Brier: 0.166476.
- Fold scores: 2021=0.181577, 2022=0.188945, 2023=0.174584, 2024=0.152931, 2025=0.134343.
- Artifacts: `artifacts/iteration_83_iteration_82_women_gap_gate/validation_summary.json`.
- Notes: This reproduces the accepted search result exactly and becomes the new current best model. It improves 2021, 2022, 2024, and 2025 versus iteration 81, with only 2023 slightly worse. `uvx ruff check` and `uvx ty check` passed on the fixed benchmark script.
- Next step: Carry iteration 83 forward as the active benchmark for any further work.

### 2026-03-13 14:03 UTC - Men Round Gate Search On Iteration 83

- Hypothesis: The remaining miss after iteration 83 may be concentrated in a men-only bracket segment, so lowering the men iteration 71 probability weight only for selected rounds could improve the mean without disturbing the accepted women gate.
- Data window: Same as iteration 83.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 83, with men fixed at `0.29`, women defaulting to `0.24`, women low-gap rows using `0.35`, and a new men-only round gate on the final probability blend.
- LightGBM config: Same source models and calibrators as iteration 83; searched men round-gate mode, round cutoff, and gated men iteration 71 probability weight.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.166474.
- Fold scores: Best raw configuration (`men_round_gate_mode=round_lte`, `men_round_threshold=1`, `men_gate_iteration_71_probability_weight=0.18`) yields 2021=0.181555, 2022=0.188946, 2023=0.174583, 2024=0.152942, 2025=0.134345.
- Artifacts: `artifacts/iteration_84_iteration_83_men_round_gate_search/validation_summary.json`.
- Notes: This is a tiny mean improvement versus iteration 83, but it improves only 2021 and 2023, so it fails the 4-of-5 rule.
- Next step: Shift the next search back to the women branch instead of further localizing the men side.

### 2026-03-13 14:06 UTC - Women Gap-Gate Weight Extension On Iteration 83

- Hypothesis: The accepted women low-gap gate from iteration 83 may still be underweighting iteration 71, so extending the gated women weight range farther upward could produce another accepted improvement.
- Data window: Same as iteration 83.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 83, keeping men fixed at `0.29`, women defaulting to `0.24`, and searching a stronger women-only low-gap override.
- LightGBM config: Same source models and calibrators as iteration 83; searched the women low-gap threshold and the gated women iteration 71 probability weight beyond the iteration 82 boundary.
- Mean Brier: Accepted. Best mean Brier 0.166247.
- Fold scores: Best configuration (`women_gate_mode=low_gap`, `women_gap_threshold=0.055`, `women_gate_iteration_71_probability_weight=0.6`) yields 2021=0.181531, 2022=0.188722, 2023=0.174816, 2024=0.152116, 2025=0.134051.
- Artifacts: `artifacts/iteration_85_iteration_83_women_gap_gate_weight_extension_search/validation_summary.json`.
- Notes: This lowers the mean Brier versus iteration 83 and improves 2021, 2022, 2024, and 2025, missing only 2023. The women gate still had material headroom.
- Next step: Keep pushing the same women low-gap family while the accepted point remains on the search boundary.

### 2026-03-13 14:09 UTC - Second Women Gap-Gate Weight Extension On Iteration 85

- Hypothesis: The accepted iteration 85 women gate still finished on the weight ceiling, so another extension in the same direction may unlock more gain without changing the model family.
- Data window: Same as iteration 85.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 85, with men fixed at `0.29`, women defaulting to `0.24`, and a stronger women low-gap override.
- LightGBM config: Same source models and calibrators as iteration 85; searched a tighter low-gap threshold range with larger women gate weights.
- Mean Brier: Accepted. Best mean Brier 0.165992.
- Fold scores: Selected configuration (`women_gate_mode=low_gap`, `women_gap_threshold=0.0525`, `women_gate_iteration_71_probability_weight=1.0`) yields 2021=0.181077, 2022=0.188729, 2023=0.174733, 2024=0.151733, 2025=0.133690.
- Artifacts: `artifacts/iteration_86_iteration_85_women_gap_gate_weight_extension_search/validation_summary.json`.
- Notes: This lowers the mean Brier versus iteration 85 and improves 2021, 2023, 2024, and 2025, missing only 2022. The best accepted point fully trusts iteration 71 on low-gap women rows. The raw best mean (`0.165992`) improved fewer folds and was not selected.
- Next step: Hold the low-gap women override fixed and search the default women weight outside the gate.

### 2026-03-13 14:14 UTC - Women Default Weight Search On Iteration 86

- Hypothesis: Once low-gap women rows fully trust iteration 71, the remaining miss may come from the default women weight outside that gate, so a small rollback there could improve the rest of the women surface.
- Data window: Same as iteration 86.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 86, with men fixed at `0.29`, women low-gap rows using `1.0`, and a searched default women iteration 71 probability weight for rows above the gap threshold.
- LightGBM config: Same source models and calibrators as iteration 86; searched `women_gap_threshold` and `women_default_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best mean Brier 0.165738.
- Fold scores: Best configuration (`women_gap_threshold=0.054`, `women_default_iteration_71_probability_weight=0.12`) yields 2021=0.181003, 2022=0.187986, 2023=0.174669, 2024=0.151202, 2025=0.133829.
- Artifacts: `artifacts/iteration_87_iteration_86_women_default_weight_search/validation_summary.json`.
- Notes: This lowers the mean Brier versus iteration 86 and improves 2021, 2022, 2023, and 2024, missing only 2025. A small women default rollback outside the low-gap gate is better than the previous all-or-nothing extension.
- Next step: Probe whether the women default should be pushed even lower.

### 2026-03-13 14:17 UTC - Women Default Weight Extension On Iteration 87

- Hypothesis: The accepted iteration 87 women default weight may still be too high, so extending the default women search toward zero could recover more average Brier.
- Data window: Same as iteration 87.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 87, with the iteration 86 low-gap women override preserved and only the default women probability blend outside the gate retuned.
- LightGBM config: Same source models and calibrators as iteration 87; extended the search toward lower default women iteration 71 weights.
- Mean Brier: No accepted configuration. Best raw mean Brier 0.165625.
- Fold scores: Best raw configuration (`women_gap_threshold=0.054`, `women_default_iteration_71_probability_weight=0.0`) yields 2021=0.180936, 2022=0.187543, 2023=0.174408, 2024=0.151268, 2025=0.133972.
- Artifacts: `artifacts/iteration_88_iteration_87_women_default_weight_extension_search/validation_summary.json`.
- Notes: This is a mean improvement versus iteration 87, but it improves only 2021, 2022, and 2023, so it fails the 4-of-5 rule.
- Next step: Re-evaluate the current benchmark on the official scoring surface before spending more cycles inside the old full-row metric.

### 2026-03-13 14:31 UTC - Official-Score Women Default Weight Search On Iteration 87

- Hypothesis: Once validation excludes actual First Four rows to match official Stage 2 scoring, the accepted iteration 87 women default blend may no longer be optimal, and a small official-score retune may produce a cleaner benchmark for the late-stage search.
- Data window: Same as iteration 87.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 87, but fold scoring excludes rows with `first_four_pair_flag == 1` to mirror the official competition surface.
- LightGBM config: Same source models and calibrators as iteration 87; searched `women_gap_threshold` and `women_default_iteration_71_probability_weight` under the official-score mask.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160588 versus official benchmark 0.160712.
- Fold scores: Best raw configuration (`women_gap_threshold=0.054`, `women_default_iteration_71_probability_weight=0.0`) yields 2021=0.181030, 2022=0.184365, 2023=0.170654, 2024=0.141606, 2025=0.125284.
- Artifacts: `artifacts/iteration_90_iteration_87_official_play_in_excluded_women_default_weight_search/validation_summary.json`.
- Notes: This establishes the official-score branch and confirms that the official 2021-2025 mean is materially lower than the old full-row metric, but the best raw point improves only 2021, 2022, and 2023, so it is rejected.
- Next step: Test whether a probability recalibration on the official branch is stronger than another direct weight sweep.

### 2026-03-13 14:36 UTC - Official Calibration Women Default Weight Search

- Hypothesis: A calibration-aware branch may want a different women default weight under official scoring than the direct iteration 87 probability-routing branch.
- Data window: Same as iteration 87.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same source-model family as the official iteration 87 branch, but benchmarked through the calibration-specific path used in this search script.
- LightGBM config: Same underlying LightGBM source models; searched `women_gap_threshold` and `women_default_iteration_71_probability_weight` against the calibration-specific benchmark.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.161175 versus benchmark 0.161233.
- Fold scores: Best raw configuration (`women_gap_threshold=0.054`, `women_default_iteration_71_probability_weight=0.0`) yields 2021=0.183975, 2022=0.183693, 2023=0.170088, 2024=0.141923, 2025=0.126199.
- Artifacts: `artifacts/iteration_91_iteration_87_official_calibration_women_default_weight_search/validation_summary.json`.
- Notes: This branch is worse than the simpler official iteration 90 routing benchmark even before applying the acceptance rule, so it is not a promising direction.
- Next step: Return to direct probability routing and test whether the women adjustment should vary by round.

### 2026-03-13 14:43 UTC - Official Women Round-Segment Search On Iteration 90

- Hypothesis: The official-score women default rollback may be useful only in early rounds, so splitting the women correction by round could rescue the missed folds from iteration 90.
- Data window: Same as iteration 90.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 90, with an early/late women round split layered onto the official probability routing.
- LightGBM config: Same source models and calibrators as iteration 90; searched `women_gap_threshold`, `women_round_cutoff`, `women_early_iteration_71_probability_weight`, and `women_late_iteration_71_probability_weight`.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160657.
- Fold scores: Best raw configuration (`women_gap_threshold=0.054`, `women_round_cutoff=3`, `women_early_iteration_71_probability_weight=0.0`, `women_late_iteration_71_probability_weight=0.12`) yields 2021=0.181188, 2022=0.184497, 2023=0.170712, 2024=0.141606, 2025=0.125284.
- Artifacts: `artifacts/iteration_92_iteration_90_women_round_segment_search/validation_summary.json`.
- Notes: The round segmentation does not solve the fold tradeoff and improves only 2022 and 2023 versus the official benchmark.
- Next step: Try a constrained probability recalibrator instead of more hand-routed women-only gates.

### 2026-03-13 14:46 UTC - Official Constrained Probability Calibrator Search On Iteration 90

- Hypothesis: A tiny constrained meta-calibrator on the benchmark probabilities, division, and disagreement features may beat manual gate tables under official scoring.
- Data window: Same as iteration 90.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Official iteration 90 benchmark probabilities plus compact calibration features derived from logits, disagreement magnitude, and division context.
- LightGBM config: The source LightGBM models are unchanged; searched small logistic-calibration feature sets and regularization strengths on top of the fixed benchmark probabilities.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.163092.
- Fold scores: Best raw configuration (`feature_set=logit_gap_division`, `regularization_c=0.1`) yields 2021=0.181099, 2022=0.184603, 2023=0.177473, 2024=0.144614, 2025=0.127671.
- Artifacts: `artifacts/iteration_93_iteration_90_constrained_probability_calibrator_search/validation_summary.json`.
- Notes: Even the best constrained calibrator badly overfits and is materially worse than the official benchmark.
- Next step: Drop the calibrator detour and test whether the official scoring surface prefers a different training-side women default branch.

### 2026-03-13 14:49 UTC - Official Training-Side Women Default Weight Search

- Hypothesis: The official score may prefer a different women default probability weight when the branch is rebuilt from the training-side variant rather than the current routing benchmark.
- Data window: Same as iteration 87.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same training-side branch as the iteration 87 family, re-evaluated under official scoring with a searched women default weight.
- LightGBM config: Same source models and calibrators as the training-side branch; searched `women_gap_threshold` and `women_default_iteration_71_probability_weight`.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160934 versus branch benchmark 0.161079.
- Fold scores: Best raw configuration (`women_gap_threshold=0.055`, `women_default_iteration_71_probability_weight=0.0`) yields 2021=0.183605, 2022=0.182658, 2023=0.169781, 2024=0.143995, 2025=0.124629.
- Artifacts: `artifacts/iteration_94_iteration_87_official_training_women_default_weight_search/validation_summary.json`.
- Notes: This branch improves its own local benchmark but remains worse than the direct iteration 90 official-routing branch, so it is not carried forward.
- Next step: Test the team feedback idea of a women-only corrective blend using older accepted probability sources.

### 2026-03-13 14:54 UTC - Official Women Corrective Probability Blend Search On Iteration 90

- Hypothesis: Women rows may still need a small corrective blend from older accepted probability sources, especially for the 2023 miss highlighted in the external review.
- Data window: Same as iteration 90.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Official iteration 90 benchmark probabilities with an optional women-only corrective blend from the iteration 53 and iteration 44 probability branches.
- LightGBM config: Same source models and calibrators as iteration 90; searched women-only scope switches and small additive `iteration_53` and `iteration_44` probability weights.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160712.
- Fold scores: Best raw configuration (`women_scope=all_women`, `iteration_53_probability_weight=0.0`, `iteration_44_probability_weight=0.0`) yields 2021=0.181099, 2022=0.184863, 2023=0.170932, 2024=0.141536, 2025=0.125132.
- Artifacts: `artifacts/iteration_95_iteration_90_women_corrective_probability_blend_search/validation_summary.json`.
- Notes: The search collapses back to the benchmark with zero corrective weight, so the tested women-only additive blend is not useful on the official branch.
- Next step: Replace the two-regime women routing with a larger official-score gap table.

### 2026-03-13 15:05 UTC - Official Women Three-Regime Gap Search On Iteration 90

- Hypothesis: The official women surface likely wants more than a single gap cutoff, so a three-regime women gap table can absorb the remaining tradeoff more cleanly than one default weight.
- Data window: Same as iteration 90.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 90, but the women branch uses low-gap, mid-gap, and high-gap routing under official scoring.
- LightGBM config: Same source models and calibrators as iteration 90; searched `women_low_gap_threshold`, `women_mid_gap_threshold`, `women_mid_gap_iteration_71_probability_weight`, and `women_high_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best selected mean Brier 0.160663.
- Fold scores: Selected configuration (`women_low_gap_threshold=0.054`, `women_mid_gap_threshold=0.065`, `women_mid_gap_iteration_71_probability_weight=0.0`, `women_high_gap_iteration_71_probability_weight=0.2`) yields 2021=0.180852, 2022=0.185041, 2023=0.170809, 2024=0.141521, 2025=0.125091.
- Artifacts: `artifacts/iteration_96_iteration_90_official_women_three_regime_gap_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 90 and improves 2021, 2023, 2024, and 2025, missing only 2022. The raw best mean (`0.160616`) improved only 2021, 2022, and 2023 and was rejected.
- Next step: Extend the women table by splitting the high-gap tail again.

### 2026-03-13 15:10 UTC - Official Women Extreme-Gap Search On Iteration 96

- Hypothesis: The accepted three-regime women table from iteration 96 may still be too coarse in the far tail, so a separate extreme-gap women bucket could produce another accepted improvement.
- Data window: Same as iteration 96.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 96, with the women gap table extended to include a separate extreme-gap regime.
- LightGBM config: Same source models and calibrators as iteration 96; searched `women_high_gap_threshold`, `women_high_gap_iteration_71_probability_weight`, and `women_extreme_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best mean Brier 0.160556.
- Fold scores: Best configuration (`women_high_gap_threshold=0.08`, `women_high_gap_iteration_71_probability_weight=0.04`, `women_extreme_gap_iteration_71_probability_weight=0.28`) yields 2021=0.180838, 2022=0.184410, 2023=0.170799, 2024=0.141488, 2025=0.125244.
- Artifacts: `artifacts/iteration_97_iteration_96_official_women_extreme_gap_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 96 and improves 2021, 2022, 2023, and 2024, missing only 2025.
- Next step: Retune the men side against the stronger women table instead of only pushing the women tail farther.

### 2026-03-13 15:14 UTC - Official Men Weight Search On Iteration 97

- Hypothesis: After the accepted women-table improvements, the remaining miss may come from the global men weight, so a men-only retune could clear the next acceptance threshold.
- Data window: Same as iteration 97.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 97, with only the men iteration 71 probability weight searched.
- LightGBM config: Same source models and calibrators as iteration 97; searched the global men iteration 71 probability weight.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160457.
- Fold scores: Best raw configuration (`men_iteration_71_probability_weight=0.4`) yields 2021=0.180446, 2022=0.184128, 2023=0.170854, 2024=0.141360, 2025=0.125496.
- Artifacts: `artifacts/iteration_98_iteration_97_official_men_weight_search/validation_summary.json`.
- Notes: This is a strong mean improvement versus iteration 97, but it improves only 2021, 2022, and 2024, so it fails the 4-of-5 rule.
- Next step: Let men and women move together in a small joint search rather than forcing a single men-only change.

### 2026-03-13 15:19 UTC - Official Joint Men-Women Gap Search On Iteration 97

- Hypothesis: The next accepted improvement likely requires a coupled men/women move, with a modest men rollback and a slightly more aggressive women extreme-gap rule.
- Data window: Same as iteration 97.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 97, with the men global weight and the women high-gap and extreme-gap tail weights searched jointly.
- LightGBM config: Same source models and calibrators as iteration 97; searched `men_iteration_71_probability_weight`, `women_high_gap_threshold`, `women_high_gap_iteration_71_probability_weight`, and `women_extreme_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best selected mean Brier 0.160494.
- Fold scores: Selected configuration (`men_iteration_71_probability_weight=0.34`, `women_high_gap_threshold=0.08`, `women_high_gap_iteration_71_probability_weight=0.0`, `women_extreme_gap_iteration_71_probability_weight=0.24`) yields 2021=0.180693, 2022=0.184146, 2023=0.170790, 2024=0.141445, 2025=0.125396.
- Artifacts: `artifacts/iteration_99_iteration_97_official_joint_men_women_gap_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 97 and improves 2021, 2022, 2023, and 2024, missing only 2025. The raw best mean (`0.160405`) improved only 2021, 2022, and 2024 and was rejected.
- Next step: Try the external leaf-logistic calibrator idea before spending more budget on manual routing tables.

### 2026-03-13 15:25 UTC - Official Leaf-Logistic Calibrator Search On Iteration 99

- Hypothesis: The current stack may benefit from a LightGBM leaf-index logistic calibrator, matching the recent March Mania writeup pattern highlighted in the external review.
- Data window: Same as iteration 99.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Official iteration 99 benchmark probabilities plus nested OOF LightGBM leaf-index features and compact bracket context.
- LightGBM config: The source LightGBM margin models are unchanged; searched logistic regularization over the leaf-feature calibrator.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.171696.
- Fold scores: Best raw configuration (`regularization_c=0.01`) yields 2021=0.192388, 2022=0.195480, 2023=0.208424, 2024=0.140000, 2025=0.122190.
- Artifacts: `artifacts/iteration_100_iteration_99_official_leaf_logistic_calibrator_search/validation_summary.json`.
- Notes: The leaf calibrator overfits badly and is not competitive with the hand-routed benchmark despite helping 2024 and 2025.
- Next step: Return to the strong routing family and fine-search the joint men/women table instead.

### 2026-03-13 15:28 UTC - Official Joint Men-Women Gap Fine Search On Iteration 99

- Hypothesis: The accepted iteration 99 routing table is close enough that a tighter local joint search may uncover a cleaner accepted point without changing model structure.
- Data window: Same as iteration 99.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 99, with a finer local search over the men weight and women extreme-gap settings.
- LightGBM config: Same source models and calibrators as iteration 99; searched a narrower neighborhood around the accepted joint men/women routing table.
- Mean Brier: Accepted. Best selected mean Brier 0.160489.
- Fold scores: Selected configuration (`men_iteration_71_probability_weight=0.35`, `women_high_gap_threshold=0.08`, `women_high_gap_iteration_71_probability_weight=0.0`, `women_extreme_gap_iteration_71_probability_weight=0.22`) yields 2021=0.180670, 2022=0.184127, 2023=0.170785, 2024=0.141441, 2025=0.125419.
- Artifacts: `artifacts/iteration_101_iteration_99_official_joint_men_women_gap_fine_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 99 and improves 2021, 2022, 2023, and 2024, missing only 2025. The raw best mean (`0.160415`) again failed the 4-of-5 rule.
- Next step: Replace the single men weight with a two-regime men gap table.

### 2026-03-13 15:31 UTC - Official Men Gap-Weight Search On Iteration 101

- Hypothesis: The men side likely wants different iteration 71 weights in low-gap and high-gap rows, so a two-regime men table can beat the single-weight iteration 101 benchmark.
- Data window: Same as iteration 101.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 101, with a two-regime men gap split layered onto the accepted official women table.
- LightGBM config: Same source models and calibrators as iteration 101; searched `men_gap_threshold`, `men_low_gap_iteration_71_probability_weight`, and `men_high_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best selected mean Brier 0.160431.
- Fold scores: Selected configuration (`men_gap_threshold=0.04`, `men_low_gap_iteration_71_probability_weight=0.42`, `men_high_gap_iteration_71_probability_weight=0.38`) yields 2021=0.180473, 2022=0.184051, 2023=0.170767, 2024=0.141355, 2025=0.125507.
- Artifacts: `artifacts/iteration_102_iteration_101_official_men_gap_weight_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 101 and improves 2021, 2022, 2023, and 2024, missing only 2025. The raw best mean (`0.160427`) improved only 2021, 2022, and 2024 and was rejected.
- Next step: Check whether men need a third regime rather than another joint search immediately.

### 2026-03-13 15:34 UTC - Official Men Three-Regime Gap Search On Iteration 102

- Hypothesis: The accepted two-regime men table from iteration 102 may still be too coarse, so a mid-gap men bucket could recover more mean Brier.
- Data window: Same as iteration 102.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 102, with a three-regime men gap table replacing the accepted two-regime split.
- LightGBM config: Same source models and calibrators as iteration 102; searched `men_mid_gap_threshold`, `men_mid_gap_iteration_71_probability_weight`, and `men_high_gap_iteration_71_probability_weight`.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160432.
- Fold scores: Best raw configuration (`men_mid_gap_threshold=0.05`, `men_mid_gap_iteration_71_probability_weight=0.34`, `men_high_gap_iteration_71_probability_weight=0.42`) yields 2021=0.180510, 2022=0.184140, 2023=0.170715, 2024=0.141306, 2025=0.125491.
- Artifacts: `artifacts/iteration_103_iteration_102_official_men_three_regime_gap_search/validation_summary.json`.
- Notes: The mean is nearly flat, but the candidate improves only 2023, 2024, and 2025, so it fails the 4-of-5 rule.
- Next step: Hold the men split fixed and fine-search the women extreme-gap tail.

### 2026-03-13 15:37 UTC - Official Women Extreme-Gap Fine Search On Iteration 102

- Hypothesis: The remaining miss after iteration 102 may be in the women extreme-gap tail, so a finer women-only search around the accepted weights could clear another accepted step.
- Data window: Same as iteration 102.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 102, with the men two-regime table fixed and only the women high-gap and extreme-gap tail retuned.
- LightGBM config: Same source models and calibrators as iteration 102; searched `women_high_gap_threshold`, `women_high_gap_iteration_71_probability_weight`, and `women_extreme_gap_iteration_71_probability_weight` in a narrow local neighborhood.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160424.
- Fold scores: Best raw configuration (`women_high_gap_threshold=0.08`, `women_high_gap_iteration_71_probability_weight=0.0`, `women_extreme_gap_iteration_71_probability_weight=0.26`) yields 2021=0.180448, 2022=0.184037, 2023=0.170787, 2024=0.141339, 2025=0.125507.
- Artifacts: `artifacts/iteration_104_iteration_102_official_women_extreme_gap_fine_search/validation_summary.json`.
- Notes: This slightly beats the mean benchmark but improves only 2021, 2022, and 2024, so it is rejected.
- Next step: Run a final joint fine search over the strongest men and women knobs together.

### 2026-03-13 15:39 UTC - Official Joint Men-Women Fine Search On Iteration 102

- Hypothesis: The current official benchmark is close enough that one more joint fine search over the strongest men and women routing knobs may find the next accepted point.
- Data window: Same as iteration 102.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 102, with a joint search over the men two-regime gap table and the women extreme-gap tail while keeping the accepted women low/mid/high routing fixed.
- LightGBM config: Same source models and calibrators as iteration 102; searched `men_gap_threshold`, `men_low_gap_iteration_71_probability_weight`, `men_high_gap_iteration_71_probability_weight`, and `women_extreme_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best selected official-score mean Brier 0.160409.
- Fold scores: Best configuration (`men_gap_threshold=0.04`, `men_low_gap_iteration_71_probability_weight=0.44`, `men_high_gap_iteration_71_probability_weight=0.36`, `women_extreme_gap_iteration_71_probability_weight=0.26`) yields 2021=0.180428, 2022=0.184088, 2023=0.170742, 2024=0.141309, 2025=0.125480.
- Artifacts: `artifacts/iteration_105_iteration_102_official_joint_men_women_fine_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 102 and improves 2021, 2023, 2024, and 2025, missing only 2022. The raw best mean (`0.160406`) improved only 2021, 2022, and 2024 and was rejected.
- Next step: Freeze this routing table into a dedicated benchmark, then test a tiny benchmark-probability recalibrator and one more men-table refinement.

### 2026-03-13 15:43 UTC - Official Benchmark-Probability Calibrator Search On Iteration 105

- Hypothesis: A very small logistic recalibrator on the accepted iteration 105 benchmark probability plus context may beat the hand-tuned routing table without changing the underlying source models.
- Data window: Same as iteration 105.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Accepted iteration 105 benchmark probabilities plus compact calibration features from logits, disagreement, round, and division context.
- LightGBM config: Source LightGBM models unchanged; searched small logistic feature sets, regularization, and whether calibration was fit only on official rows.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.162495.
- Fold scores: Best raw configuration (`feature_set=logit_gap_context`, `regularization_c=0.1`, `official_only_calibration=false`) yields 2021=0.180428, 2022=0.179719, 2023=0.179099, 2024=0.145159, 2025=0.128072.
- Artifacts: `artifacts/iteration_106_iteration_105_official_benchmark_probability_calibrator_search/validation_summary.json`.
- Notes: Even the cheapest benchmark-probability recalibrator is materially worse than the accepted routing table.
- Next step: Return to direct routing and try one more men-side split after the accepted threshold.

### 2026-03-13 15:45 UTC - Official Men Post-Threshold Split Search On Iteration 105

- Hypothesis: The accepted iteration 105 men table may still hide one more gain if the post-threshold region is split again without touching the women settings.
- Data window: Same as iteration 105.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 105, with an extra men post-threshold split layered onto the accepted women routing.
- LightGBM config: Same source models and calibrators as iteration 105; searched `men_mid_gap_threshold`, `men_mid_gap_iteration_71_probability_weight`, and `men_high_gap_iteration_71_probability_weight`.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160402.
- Fold scores: Best raw configuration (`men_mid_gap_threshold=0.045`, `men_mid_gap_iteration_71_probability_weight=0.34`, `men_high_gap_iteration_71_probability_weight=0.42`) yields 2021=0.180335, 2022=0.184026, 2023=0.170824, 2024=0.141260, 2025=0.125563.
- Artifacts: `artifacts/iteration_107_iteration_105_official_men_post_threshold_split_search/validation_summary.json`.
- Notes: This is the best raw mean in the late official branch, but it improves only 2021, 2022, and 2024, so it is rejected by the 4-of-5 rule.
- Next step: Confirm the remaining fixed source-model blend weight is already at its local optimum before freezing the benchmark.

### 2026-03-13 15:55 UTC - Official Iteration 41 Weight Search On Iteration 105

- Hypothesis: The remaining fixed `iteration_41` source-model weight might still be slightly mistuned even though the routing table appears close to saturated.
- Data window: Same as iteration 105.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 105, with only the fixed `iteration_41` contribution to the upstream source-probability blend searched.
- LightGBM config: Same source models and calibrators as iteration 105; searched `iteration_41_weight` while rebuilding the cached upstream probabilities for each candidate.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160409.
- Fold scores: Best raw configuration (`iteration_41_weight=0.01`) yields 2021=0.180428, 2022=0.184088, 2023=0.170742, 2024=0.141309, 2025=0.125480.
- Artifacts: `artifacts/iteration_108_iteration_105_official_iteration41_weight_search/validation_summary.json`.
- Notes: The search lands exactly on the existing benchmark value, so the current `iteration_41` weight is already locally optimal in this family.
- Next step: Freeze iteration 105 into a dedicated fixed benchmark and carry it forward as the current official model.

### 2026-03-13 16:37 UTC - Fixed Official Joint Men-Women Blend From Iteration 105

- Hypothesis: The accepted iteration 105 official routing table should reproduce exactly when moved into a dedicated fixed script, so the best official benchmark no longer depends on the search loop or selection artifact.
- Data window: Same as iteration 105.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 105, with fixed source-model weights `iteration_38=0.020`, `iteration_49=0.015`, `iteration_41=0.010`, men using `men_gap_threshold=0.04`, `men_low_gap_iteration_71_probability_weight=0.44`, `men_high_gap_iteration_71_probability_weight=0.36`, and women keeping the accepted low/mid/high official table plus `women_extreme_gap_iteration_71_probability_weight=0.26`.
- LightGBM config: Same source models and calibrators as iteration 105. Only the search loop is removed.
- Mean Brier: 0.160409.
- Fold scores: 2021=0.180428, 2022=0.184088, 2023=0.170742, 2024=0.141309, 2025=0.125480.
- Artifacts: `artifacts/iteration_109_iteration_105_official_joint_men_women_blend/validation_summary.json`.
- Notes: This reproduces the accepted iteration 105 search result exactly and becomes the new current best official benchmark. `uvx ruff check` and `uvx ty check` passed on the fixed benchmark script.
- Next step: Carry iteration 109 forward as the active official benchmark for any further research.

### 2026-03-15 22:54 UTC - Official Men Post-Threshold Plus Women Tail Search On Iteration 109

- Hypothesis: The strongest late official near miss was the rejected men post-threshold split, so coupling a narrower men high-gap split with a small women tail retune may recover the missed folds without changing the accepted model family.
- Data window: Same as iteration 109.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 109, with the accepted men two-regime gap routing extended to a three-regime men table above the `0.04` low-gap cutoff and the women high/extreme tail searched locally.
- LightGBM config: Same source models and calibrators as iteration 109; searched `men_mid_gap_threshold`, `men_mid_gap_iteration_71_probability_weight`, `men_high_gap_iteration_71_probability_weight`, `women_high_gap_threshold`, `women_high_gap_iteration_71_probability_weight`, and `women_extreme_gap_iteration_71_probability_weight`.
- Mean Brier: Accepted. Best selected official-score mean Brier 0.160402.
- Fold scores: Selected configuration (`men_mid_gap_threshold=0.0425`, `men_mid_gap_iteration_71_probability_weight=0.34`, `men_high_gap_iteration_71_probability_weight=0.38`, `women_high_gap_threshold=0.08`, `women_high_gap_iteration_71_probability_weight=0.0`, `women_extreme_gap_iteration_71_probability_weight=0.24`) yields 2021=0.180402, 2022=0.184074, 2023=0.170738, 2024=0.141288, 2025=0.125507.
- Artifacts: `artifacts/iteration_110_iteration_109_official_men_post_threshold_women_tail_search/validation_summary.json`.
- Notes: This lowers the mean official Brier versus iteration 109 and improves 2021, 2022, 2023, and 2024, missing only 2025. The raw best mean (`0.160386`) again failed the 4-of-5 rule by improving only 2021, 2022, and 2024, so the accepted point came from the more conservative corner of the search. `uvx ruff check` and `uvx ty check` passed on the search script.
- Next step: Freeze this routing table into a dedicated benchmark and carry it forward as the new active official model.

### 2026-03-15 22:54 UTC - Fixed Official Joint Men Post-Threshold And Women Tail Blend From Iteration 110

- Hypothesis: The accepted iteration 110 routing table should reproduce exactly when moved into a dedicated fixed script, so the best official benchmark no longer depends on the search loop or selection artifact.
- Data window: Same as iteration 110.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same as iteration 110, with fixed source-model weights `iteration_38=0.020`, `iteration_49=0.015`, `iteration_41=0.010`, men using `men_low_gap_threshold=0.04`, `men_low_gap_iteration_71_probability_weight=0.44`, `men_mid_gap_threshold=0.0425`, `men_mid_gap_iteration_71_probability_weight=0.34`, `men_high_gap_iteration_71_probability_weight=0.38`, and women using `women_low_gap_threshold=0.054`, `women_mid_gap_threshold=0.065`, `women_high_gap_threshold=0.08`, `women_low_gap_iteration_71_probability_weight=1.0`, `women_mid_gap_iteration_71_probability_weight=0.0`, `women_high_gap_iteration_71_probability_weight=0.0`, and `women_extreme_gap_iteration_71_probability_weight=0.24`.
- LightGBM config: Same source models and calibrators as iteration 110. Only the search loop is removed.
- Mean Brier: 0.160402.
- Fold scores: 2021=0.180402, 2022=0.184074, 2023=0.170738, 2024=0.141288, 2025=0.125507.
- Artifacts: `artifacts/iteration_111_iteration_110_official_joint_men_post_threshold_women_tail_blend/validation_summary.json`.
- Notes: This reproduces the accepted iteration 110 search result exactly and becomes the new current best official benchmark. `uvx ruff check` and `uvx ty check` passed on the fixed benchmark script.
- Next step: Carry iteration 111 forward as the active official benchmark for any further research.

### 2026-03-15 23:12 UTC - Official Men Context High-Gap Gate Search On Iteration 111

- Hypothesis: The remaining miss after iteration 111 may live in early-round, tight-slot men rows, so raising the men high-gap `iteration_71` weight only inside that context can recover 2025 without paying the broad 2022 penalty from the more aggressive men family.
- Data window: Same as iteration 111.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 111, with the accepted routing preserved everywhere except selected men high-gap rows gated by `possible_round` and `opening_slot_gap`.
- LightGBM config: Same source models and calibrators as iteration 111; searched `men_round_cutoff`, `men_opening_gap_max`, and `men_context_high_gap_iteration_71_probability_weight`.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160396.
- Fold scores: Best raw configuration (`men_round_cutoff=2`, `men_opening_gap_max=1`, `men_context_high_gap_iteration_71_probability_weight=0.42`) yields 2021=0.180332, 2022=0.184064, 2023=0.170776, 2024=0.141259, 2025=0.125550.
- Artifacts: `artifacts/iteration_112_iteration_111_official_men_context_high_gap_gate_search/validation_summary.json`.
- Notes: The context gate improves only 2021, 2022, and 2024 versus iteration 111, so it misses the 4-of-5 rule even though the raw mean is better. The attempted bridge between the `2022`-friendly and `2025`-friendly men families is still too crude in this context. `uvx ruff check` and `uvx ty check` passed on the search script.
- Next step: Move the next search upstream and jointly retune the fixed source-model blend weights under the accepted iteration 111 routing.

### 2026-03-15 23:45 UTC - Official Men Expert Context Switch Search On Iteration 111

- Hypothesis: The late official branch now has two complementary men configurations, so selectively switching from the accepted men router to the `2025`-friendly men expert inside targeted contexts may recover the missed fold without touching women or the upstream source blend.
- Data window: Same as iteration 111.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Same completed pipelines as iteration 111, with women fixed to the accepted routing and men switching between the accepted router and a `2025`-friendly men expert based on `possible_round`, `opening_slot_gap`, and minimum absolute seed gap.
- LightGBM config: Same source models and calibrators as iteration 111; searched `men_round_cutoff`, `men_opening_gap_max`, and `men_seed_gap_min` for the context switch.
- Mean Brier: No accepted configuration. Best raw official-score mean Brier 0.160412.
- Fold scores: Best raw configuration (`men_round_cutoff=3`, `men_opening_gap_max=1`, `men_seed_gap_min=0`) yields 2021=0.180482, 2022=0.184087, 2023=0.170703, 2024=0.141313, 2025=0.125477.
- Artifacts: `artifacts/iteration_115_iteration_111_official_men_expert_context_switch_search/validation_summary.json`.
- Notes: This idea is weaker than the benchmark even before the 4-of-5 rule. The men expert switch improves only 2023 and 2025, while giving back 2021, 2022, and 2024. `uvx ruff check` and `uvx ty check` passed on the search script.
- Next step: Return to upstream blend retuning, but keep the search narrow enough to finish quickly and explicitly checkpoint progress.

### 2026-03-16 01:05 UTC - Pending Feature Family 01 - Current-season neutral-site-only strength features

- Hypothesis: Neutral-only current-season strength splits can add portable tournament signal.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Current-season neutral-site-only strength features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159561.
- Fold scores: 2021=0.180735, 2022=0.181525, 2023=0.174645, 2024=0.141769, 2025=0.119133.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_neutral_site_only_strength_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season home-road split gap features` against the active benchmark.

### 2026-03-16 01:07 UTC - Pending Feature Family 02 - Current-season home-road split gap features

- Hypothesis: Home-road split gaps may reveal teams whose strength travels better to the tournament.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season home-road split gap features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161096.
- Fold scores: 2021=0.179588, 2022=0.185561, 2023=0.171630, 2024=0.144712, 2025=0.123986.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_home_road_split_gap_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season conference-vs-nonconference split features` against the active benchmark.

### 2026-03-16 01:12 UTC - Pending Feature Family 03 - Current-season conference-vs-nonconference split features

- Hypothesis: Conference versus nonconference splits can expose schedule-context inflation.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season conference-vs-nonconference split features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161571.
- Fold scores: 2021=0.178357, 2022=0.188067, 2023=0.173104, 2024=0.142793, 2025=0.125535.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_conference_vs_nonconference_split_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season preconference-vs-conference-play trajectory features` against the active benchmark.

### 2026-03-16 01:15 UTC - Pending Feature Family 04 - Current-season preconference-vs-conference-play trajectory features

- Hypothesis: Preconference-to-conference changes may capture how teams scale once schedules harden.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season preconference-vs-conference-play trajectory features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161276.
- Fold scores: 2021=0.181232, 2022=0.185589, 2023=0.172245, 2024=0.142284, 2025=0.125031.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_preconference_vs_conference_play_trajectory_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season rest-day gap features` against the active benchmark.

### 2026-03-16 01:18 UTC - Pending Feature Family 05 - Current-season rest-day gap features

- Hypothesis: Rest-context performance may matter once the tournament compresses turnaround windows.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Current-season rest-day gap features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162008.
- Fold scores: 2021=0.183138, 2022=0.186276, 2023=0.174364, 2024=0.143164, 2025=0.123100.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_rest_day_gap_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season schedule-density and fatigue features` against the active benchmark.

### 2026-03-16 01:21 UTC - Pending Feature Family 06 - Current-season schedule-density and fatigue features

- Hypothesis: Schedule density may separate teams that sustain quality under repeated short turns.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Current-season schedule-density and fatigue features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162079.
- Fold scores: 2021=0.185269, 2022=0.185020, 2023=0.174155, 2024=0.142204, 2025=0.123750.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_schedule_density_and_fatigue_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season overtime propensity and performance features` against the active benchmark.

### 2026-03-16 01:24 UTC - Pending Feature Family 07 - Current-season overtime propensity and performance features

- Hypothesis: Overtime frequency and results may capture late-game resilience and variance appetite.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season overtime propensity and performance features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160848.
- Fold scores: 2021=0.180807, 2022=0.184451, 2023=0.172664, 2024=0.141045, 2025=0.125273.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_overtime_propensity_and_performance_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2024, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season close-game execution features` against the active benchmark.

### 2026-03-16 01:27 UTC - Pending Feature Family 08 - Current-season close-game execution features

- Hypothesis: Close-game execution can matter when tournament spreads compress late in the bracket.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season close-game execution features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161762.
- Fold scores: 2021=0.181451, 2022=0.184492, 2023=0.174503, 2024=0.143209, 2025=0.125156.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_close_game_execution_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season blowout tendency features` against the active benchmark.

### 2026-03-16 01:30 UTC - Pending Feature Family 09 - Current-season blowout tendency features

- Hypothesis: Blowout creation and avoidance may capture team ceiling and floor better than average margin alone.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season blowout tendency features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161014.
- Fold scores: 2021=0.181206, 2022=0.184280, 2023=0.171650, 2024=0.143143, 2025=0.124791.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_blowout_tendency_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season margin volatility and consistency features` against the active benchmark.

### 2026-03-16 01:33 UTC - Pending Feature Family 10 - Current-season margin volatility and consistency features

- Hypothesis: Margin volatility may help distinguish stable favorites from high-variance teams.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Current-season margin volatility and consistency features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161815.
- Fold scores: 2021=0.181910, 2022=0.183570, 2023=0.175519, 2024=0.142258, 2025=0.125818.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_margin_volatility_and_consistency_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season common-opponent comparison features` against the active benchmark.

### 2026-03-16 01:36 UTC - Pending Feature Family 11 - Current-season common-opponent comparison features

- Hypothesis: Common-opponent comparisons can anchor cross-schedule matchup translation.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 13 new feature-view columns to the iteration 38 stack and 13 to the iteration 49 stack for the `Current-season common-opponent comparison features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161727.
- Fold scores: 2021=0.182172, 2022=0.183374, 2023=0.171841, 2024=0.146708, 2025=0.124541.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_common_opponent_comparison_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season opponent-tier record features` against the active benchmark.

### 2026-03-16 01:39 UTC - Pending Feature Family 12 - Current-season opponent-tier record features

- Hypothesis: Opponent-tier splits may separate teams that beat good competition from teams that feast on weak schedules.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Current-season opponent-tier record features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160807.
- Fold scores: 2021=0.182128, 2022=0.179557, 2023=0.174088, 2024=0.141542, 2025=0.126719.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_opponent_tier_record_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season rematch familiarity features` against the active benchmark.

### 2026-03-16 01:42 UTC - Pending Feature Family 13 - Current-season rematch familiarity features

- Hypothesis: Regular-season familiarity can reshape tournament rematch outcomes.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Current-season rematch familiarity features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160607.
- Fold scores: 2021=0.179832, 2022=0.184522, 2023=0.171537, 2024=0.142201, 2025=0.124941.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_rematch_familiarity_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Venue familiarity from repeated game-city exposure` against the active benchmark.

### 2026-03-16 01:45 UTC - Pending Feature Family 14 - Venue familiarity from repeated game-city exposure

- Hypothesis: Repeated city exposure may proxy travel and venue comfort effects entering March.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Venue familiarity from repeated game-city exposure` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161613.
- Fold scores: 2021=0.183897, 2022=0.185734, 2023=0.174967, 2024=0.139660, 2025=0.123805.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/venue_familiarity_from_repeated_game_city_exposure/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2024, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Conference tournament participation and win-count features` against the active benchmark.

### 2026-03-16 01:46 UTC - Pending Feature Family 15 - Conference tournament participation and win-count features

- Hypothesis: Conference tournament participation and wins may capture bid-pressure form entering Selection Sunday.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament participation and win-count features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161085.
- Fold scores: 2021=0.179523, 2022=0.185109, 2023=0.172660, 2024=0.144152, 2025=0.123980.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_participation_and_win_count_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Conference tournament champion and finalist flags` against the active benchmark.

### 2026-03-16 01:48 UTC - Pending Feature Family 16 - Conference tournament champion and finalist flags

- Hypothesis: Conference tournament title and finalist signals may carry fresh information into the bracket.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament champion and finalist flags` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161461.
- Fold scores: 2021=0.177968, 2022=0.185200, 2023=0.173283, 2024=0.145488, 2025=0.125366.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_champion_and_finalist_flags/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Conference tournament margin and dominance features` against the active benchmark.

### 2026-03-16 01:50 UTC - Pending Feature Family 17 - Conference tournament margin and dominance features

- Hypothesis: Conference tournament dominance may matter more than raw participation counts.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament margin and dominance features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162168.
- Fold scores: 2021=0.179735, 2022=0.187625, 2023=0.171593, 2024=0.143710, 2025=0.128176.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_margin_and_dominance_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Conference-relative percentile and z-score features` against the active benchmark.

### 2026-03-16 01:52 UTC - Pending Feature Family 18 - Conference-relative percentile and z-score features

- Hypothesis: Conference-relative ranks may normalize across different league environments.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference-relative percentile and z-score features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159539.
- Fold scores: 2021=0.178061, 2022=0.179960, 2023=0.172215, 2024=0.139960, 2025=0.127498.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_relative_percentile_and_z_score_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022, 2024. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Interconference regular-season strength features` against the active benchmark.

### 2026-03-16 01:54 UTC - Pending Feature Family 19 - Interconference regular-season strength features

- Hypothesis: Interconference results may calibrate cross-league strength better than overall schedule summaries.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Interconference regular-season strength features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.163163.
- Fold scores: 2021=0.181253, 2022=0.185715, 2023=0.175120, 2024=0.145606, 2025=0.128124.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/interconference_regular_season_strength_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Same-conference matchup and conference-pair interaction features` against the active benchmark.

### 2026-03-16 01:56 UTC - Pending Feature Family 20 - Same-conference matchup and conference-pair interaction features

- Hypothesis: Conference-pair context may capture recurring stylistic edges and same-league familiarity.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Same-conference matchup and conference-pair interaction features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160827.
- Fold scores: 2021=0.179822, 2022=0.185217, 2023=0.173216, 2024=0.141986, 2025=0.123891.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/same_conference_matchup_and_conference_pair_interaction_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Recent NCAA conference-performance prior features` against the active benchmark.

### 2026-03-16 01:58 UTC - Pending Feature Family 21 - Recent NCAA conference-performance prior features

- Hypothesis: Recent conference tournament performance in the NCAA field may carry across seasons.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Recent NCAA conference-performance prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161173.
- Fold scores: 2021=0.179852, 2022=0.187545, 2023=0.171421, 2024=0.142353, 2025=0.124692.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/recent_ncaa_conference_performance_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men coach tenure and continuity features` against the active benchmark.

### 2026-03-16 02:00 UTC - Pending Feature Family 22 - Men coach tenure and continuity features

- Hypothesis: Men coach continuity may proxy system stability not captured by team results alone.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men coach tenure and continuity features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160599.
- Fold scores: 2021=0.180198, 2022=0.184474, 2023=0.172679, 2024=0.141156, 2025=0.124486.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_coach_tenure_and_continuity_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2024, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men midseason coaching-change features` against the active benchmark.

### 2026-03-16 02:02 UTC - Pending Feature Family 23 - Men midseason coaching-change features

- Hypothesis: Men midseason coaching changes may create structural breaks in season-long summaries.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men midseason coaching-change features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160570.
- Fold scores: 2021=0.178579, 2022=0.185232, 2023=0.171911, 2024=0.142322, 2025=0.124806.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_midseason_coaching_change_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men coach tournament-history prior features` against the active benchmark.

### 2026-03-16 02:04 UTC - Pending Feature Family 24 - Men coach tournament-history prior features

- Hypothesis: Men coaches bring reusable tournament priors that team-only models miss.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Men coach tournament-history prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160920.
- Fold scores: 2021=0.181244, 2022=0.185071, 2023=0.172262, 2024=0.141550, 2025=0.124472.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_coach_tournament_history_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men program Division-I tenure and transition-recency features` against the active benchmark.

### 2026-03-16 02:06 UTC - Pending Feature Family 25 - Men program Division-I tenure and transition-recency features

- Hypothesis: Program D-I tenure may matter for younger transitional programs on the men's side.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men program Division-I tenure and transition-recency features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161113.
- Fold scores: 2021=0.179979, 2022=0.186574, 2023=0.174368, 2024=0.142806, 2025=0.121838.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_program_division_i_tenure_and_transition_recency_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Previous-season secondary-tournament participation features` against the active benchmark.

### 2026-03-16 02:08 UTC - Pending Feature Family 26 - Previous-season secondary-tournament participation features

- Hypothesis: Prior secondary-tournament participation may differentiate fringe postseason programs.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Previous-season secondary-tournament participation features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161198.
- Fold scores: 2021=0.181622, 2022=0.184252, 2023=0.173666, 2024=0.140939, 2025=0.125513.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/previous_season_secondary_tournament_participation_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2024. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Previous-season secondary-tournament performance features` against the active benchmark.

### 2026-03-16 02:10 UTC - Pending Feature Family 27 - Previous-season secondary-tournament performance features

- Hypothesis: Secondary-tournament performance may capture latent team quality for fringe bids.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Previous-season secondary-tournament performance features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160615.
- Fold scores: 2021=0.179755, 2022=0.183230, 2023=0.173846, 2024=0.141691, 2025=0.124552.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/previous_season_secondary_tournament_performance_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men Massey rank-trend-over-time features` against the active benchmark.

### 2026-03-16 02:13 UTC - Pending Feature Family 28 - Men Massey rank-trend-over-time features

- Hypothesis: Men Massey trend features may capture late-season improvement the current blend misses.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men Massey rank-trend-over-time features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161425.
- Fold scores: 2021=0.180448, 2022=0.183953, 2023=0.174819, 2024=0.142600, 2025=0.125304.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_massey_rank_trend_over_time_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Men Massey late-season rank-volatility features` against the active benchmark.

### 2026-03-16 02:19 UTC - Pending Feature Family 29 - Men Massey late-season rank-volatility features

- Hypothesis: Men late-season Massey volatility may identify unstable profiles entering the field.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men Massey late-season rank-volatility features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160659.
- Fold scores: 2021=0.180452, 2022=0.181461, 2023=0.174388, 2024=0.142178, 2025=0.124818.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_massey_late_season_rank_volatility_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Seed-vs-Massey discrepancy features` against the active benchmark.

### 2026-03-16 02:22 UTC - Pending Feature Family 30 - Seed-vs-Massey discrepancy features

- Hypothesis: Seed versus Massey disagreement may expose bracket mispricing after committee seeding.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Seed-vs-Massey discrepancy features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160218.
- Fold scores: 2021=0.178570, 2022=0.180996, 2023=0.172689, 2024=0.143505, 2025=0.125329.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/seed_vs_massey_discrepancy_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season pace-estimate features` against the active benchmark.

### 2026-03-16 02:25 UTC - Pending Feature Family 31 - Current-season pace-estimate features

- Hypothesis: Current-season pace splits may sharpen matchup translation for tempo-sensitive teams.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season pace-estimate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161977.
- Fold scores: 2021=0.182696, 2022=0.183197, 2023=0.177121, 2024=0.140672, 2025=0.126200.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_pace_estimate_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2024. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season offensive-rating features` against the active benchmark.

### 2026-03-16 02:28 UTC - Pending Feature Family 32 - Current-season offensive-rating features

- Hypothesis: Current offensive rating splits may improve absolute scoring-strength estimates.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season offensive-rating features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159249.
- Fold scores: 2021=0.175885, 2022=0.183640, 2023=0.173652, 2024=0.142335, 2025=0.120732.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_offensive_rating_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season defensive-rating features` against the active benchmark.

### 2026-03-16 02:31 UTC - Pending Feature Family 33 - Current-season defensive-rating features

- Hypothesis: Current defensive rating splits may carry more stable tournament signal than raw points allowed.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season defensive-rating features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162266.
- Fold scores: 2021=0.181376, 2022=0.185012, 2023=0.174662, 2024=0.145246, 2025=0.125036.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_defensive_rating_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season effective-field-goal features` against the active benchmark.

### 2026-03-16 02:33 UTC - Pending Feature Family 34 - Current-season effective-field-goal features

- Hypothesis: Effective field-goal splits may capture shot-quality and spacing advantages in portable form.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season effective-field-goal features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160321.
- Fold scores: 2021=0.180934, 2022=0.184555, 2023=0.171109, 2024=0.143271, 2025=0.121734.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_effective_field_goal_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season true-shooting features` against the active benchmark.

### 2026-03-16 02:36 UTC - Pending Feature Family 35 - Current-season true-shooting features

- Hypothesis: True-shooting splits may encode shot-profile efficiency more robustly than raw percentages.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season true-shooting features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161825.
- Fold scores: 2021=0.180975, 2022=0.187470, 2023=0.172845, 2024=0.145417, 2025=0.122416.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_true_shooting_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season two-point efficiency features` against the active benchmark.

### 2026-03-16 02:38 UTC - Pending Feature Family 36 - Current-season two-point efficiency features

- Hypothesis: Two-point efficiency splits may help on teams that win through interior scoring.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season two-point efficiency features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160430.
- Fold scores: 2021=0.180389, 2022=0.186651, 2023=0.172472, 2024=0.141372, 2025=0.121268.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_two_point_efficiency_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season three-point efficiency features` against the active benchmark.

### 2026-03-16 02:40 UTC - Pending Feature Family 37 - Current-season three-point efficiency features

- Hypothesis: Three-point efficiency splits may matter disproportionately in upset-prone games.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season three-point efficiency features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.163107.
- Fold scores: 2021=0.181711, 2022=0.187403, 2023=0.177131, 2024=0.144995, 2025=0.124296.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_three_point_efficiency_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season free-throw accuracy features` against the active benchmark.

### 2026-03-16 02:43 UTC - Pending Feature Family 38 - Current-season free-throw accuracy features

- Hypothesis: Free-throw accuracy splits may help in late-game tournament contexts.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season free-throw accuracy features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161325.
- Fold scores: 2021=0.183137, 2022=0.183909, 2023=0.171898, 2024=0.142147, 2025=0.125536.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_free_throw_accuracy_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season shot-mix composition features` against the active benchmark.

### 2026-03-16 02:45 UTC - Pending Feature Family 39 - Current-season shot-mix composition features

- Hypothesis: Shot-mix composition may describe stylistic fit beyond pure efficiency numbers.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season shot-mix composition features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161397.
- Fold scores: 2021=0.181918, 2022=0.185023, 2023=0.172380, 2024=0.143233, 2025=0.124433.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_shot_mix_composition_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season assist-rate features` against the active benchmark.

### 2026-03-16 02:49 UTC - Pending Feature Family 40 - Current-season assist-rate features

- Hypothesis: Assist-rate splits may capture ball movement quality and shot creation style.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season assist-rate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160984.
- Fold scores: 2021=0.178976, 2022=0.183929, 2023=0.173003, 2024=0.144493, 2025=0.124517.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_assist_rate_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022, 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season assist-to-turnover ratio features` against the active benchmark.

### 2026-03-16 02:52 UTC - Pending Feature Family 41 - Current-season assist-to-turnover ratio features

- Hypothesis: Assist-to-turnover splits may capture offensive control that persists in neutral settings.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season assist-to-turnover ratio features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160960.
- Fold scores: 2021=0.181054, 2022=0.184301, 2023=0.172255, 2024=0.145077, 2025=0.122113.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_assist_to_turnover_ratio_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season turnover-rate features` against the active benchmark.

### 2026-03-16 14:58 UTC - Pending Feature Family 42 - Current-season turnover-rate features

- Hypothesis: Turnover-rate splits may matter more against postseason pressure defenses.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season turnover-rate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161904.
- Fold scores: 2021=0.183246, 2022=0.185294, 2023=0.173206, 2024=0.142247, 2025=0.125529.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_turnover_rate_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season steal-rate features` against the active benchmark.

### 2026-03-16 15:01 UTC - Pending Feature Family 43 - Current-season steal-rate features

- Hypothesis: Steal-rate splits may help identify defenses that create extra possessions in March.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season steal-rate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162369.
- Fold scores: 2021=0.183760, 2022=0.182007, 2023=0.175833, 2024=0.144721, 2025=0.125525.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_steal_rate_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. Benchmark remained `iteration_111_official_benchmark` at 0.160402 because this family did not satisfy the 4-of-5 acceptance rule.
- Next step: Run `Current-season block-rate features` against the active benchmark.

### 2026-03-16 15:04 UTC - Pending Feature Family 44 - Current-season block-rate features

- Hypothesis: Block-rate splits may matter for interior defenses with tournament portability.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season block-rate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159473.
- Fold scores: 2021=0.178115, 2022=0.185049, 2023=0.170476, 2024=0.141028, 2025=0.122699.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_block_rate_features/validation_summary.json`.
- Notes: Accepted. Improved seasons versus the active benchmark: 2021, 2023, 2024, 2025. This becomes the new active benchmark and carries the block-rate family forward for the remaining pending-family tests.
- Next step: Run `Current-season foul-differential features` against the updated active benchmark.

### 2026-03-16 15:10 UTC - Pending Feature Family 45 - Current-season foul-differential features

- Hypothesis: Foul differential may proxy whistle leverage in slower postseason games.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season foul-differential features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162086.
- Fold scores: 2021=0.181600, 2022=0.184304, 2023=0.177158, 2024=0.142435, 2025=0.124931.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_foul_differential_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Current-season offensive-rebound-share features` against the updated active benchmark.

### 2026-03-16 15:14 UTC - Pending Feature Family 46 - Current-season offensive-rebound-share features

- Hypothesis: Offensive rebounding splits may identify second-chance strength that survives bad shooting nights.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season offensive-rebound-share features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161822.
- Fold scores: 2021=0.178400, 2022=0.187951, 2023=0.177132, 2024=0.140910, 2025=0.124718.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_offensive_rebound_share_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2024. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Current-season defensive-rebound-share features` against the updated active benchmark.

### 2026-03-16 15:18 UTC - Pending Feature Family 47 - Current-season defensive-rebound-share features

- Hypothesis: Defensive rebounding splits may help on teams that close possessions reliably.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season defensive-rebound-share features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161162.
- Fold scores: 2021=0.182848, 2022=0.182510, 2023=0.173310, 2024=0.144145, 2025=0.122996.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_defensive_rebound_share_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Current-season possession-regime split features` against the updated active benchmark.

### 2026-03-16 15:22 UTC - Pending Feature Family 48 - Current-season possession-regime split features

- Hypothesis: Fast-versus-slow performance splits may describe portability across tempo regimes.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 18 new feature-view columns to the iteration 38 stack and 18 to the iteration 49 stack for the `Current-season possession-regime split features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161157.
- Fold scores: 2021=0.180842, 2022=0.187354, 2023=0.170548, 2024=0.143566, 2025=0.123472.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_possession_regime_split_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Current-season detailed-stat trend-slope features` against the updated active benchmark.

### 2026-03-16 15:31 UTC - Pending Feature Family 49 - Current-season detailed-stat trend-slope features

- Hypothesis: Detailed-stat trend slopes may capture momentum that simple late-season averages blur.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Current-season detailed-stat trend-slope features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159772.
- Fold scores: 2021=0.176205, 2022=0.184854, 2023=0.170831, 2024=0.143728, 2025=0.123245.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_detailed_stat_trend_slope_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021, 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Current-season style-compatibility matchup features` against the updated active benchmark.

### 2026-03-16 15:36 UTC - Pending Feature Family 50 - Current-season style-compatibility matchup features

- Hypothesis: Style-compatibility matchup features may capture interaction effects the base delta features miss.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Current-season style-compatibility matchup features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160961.
- Fold scores: 2021=0.182091, 2022=0.184933, 2023=0.173656, 2024=0.141821, 2025=0.122305.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/current_season_style_compatibility_matchup_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Record-versus-seeded-opponent features` against the updated active benchmark.

### 2026-03-16 15:41 UTC - Pending Feature Family 51 - Record-versus-seeded-opponent features

- Hypothesis: Results versus seeded opponents may be a more tournament-relevant schedule split.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Record-versus-seeded-opponent features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161137.
- Fold scores: 2021=0.177044, 2022=0.186645, 2023=0.173403, 2024=0.143792, 2025=0.124801.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/record_versus_seeded_opponent_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Margin-versus-seeded-opponent features` against the updated active benchmark.

### 2026-03-16 15:45 UTC - Pending Feature Family 52 - Margin-versus-seeded-opponent features

- Hypothesis: Margin versus seeded opponents may carry more resolution than seeded-opponent record alone.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Margin-versus-seeded-opponent features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161630.
- Fold scores: 2021=0.174719, 2022=0.188112, 2023=0.174133, 2024=0.145037, 2025=0.126149.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/margin_versus_seeded_opponent_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Seeded-opponent schedule-share features` against the updated active benchmark.

### 2026-03-16 15:48 UTC - Pending Feature Family 53 - Seeded-opponent schedule-share features

- Hypothesis: The mix of seeded opponents faced may improve schedule-quality calibration.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Seeded-opponent schedule-share features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160881.
- Fold scores: 2021=0.175818, 2022=0.187580, 2023=0.173866, 2024=0.142773, 2025=0.124369.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/seeded_opponent_schedule_share_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Record-versus-top-Massey-opponent features` against the updated active benchmark.

### 2026-03-16 15:52 UTC - Pending Feature Family 54 - Record-versus-top-Massey-opponent features

- Hypothesis: Record versus top-Massey opponents may distinguish inflated records from tested contenders.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Record-versus-top-Massey-opponent features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160599.
- Fold scores: 2021=0.180421, 2022=0.184942, 2023=0.172400, 2024=0.140601, 2025=0.124632.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/record_versus_top_massey_opponent_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2024. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Margin-versus-top-Massey-opponent features` against the updated active benchmark.

### 2026-03-16 15:55 UTC - Pending Feature Family 55 - Margin-versus-top-Massey-opponent features

- Hypothesis: Margins versus top-Massey opponents may provide a stronger strength proxy than win-loss splits alone.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Margin-versus-top-Massey-opponent features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160297.
- Fold scores: 2021=0.179344, 2022=0.184494, 2023=0.173612, 2024=0.140266, 2025=0.123770.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/margin_versus_top_massey_opponent_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2024. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Top-Massey-opponent schedule-share features` against the updated active benchmark.

### 2026-03-16 15:58 UTC - Pending Feature Family 56 - Top-Massey-opponent schedule-share features

- Hypothesis: The distribution of top-Massey opponents faced may better calibrate season difficulty.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Top-Massey-opponent schedule-share features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160163.
- Fold scores: 2021=0.178552, 2022=0.184746, 2023=0.173474, 2024=0.139756, 2025=0.124285.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/top_massey_opponent_schedule_share_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2024. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Quality-win-count and bad-loss-count features` against the updated active benchmark.

### 2026-03-16 16:01 UTC - Pending Feature Family 57 - Quality-win-count and bad-loss-count features

- Hypothesis: Quality-win and bad-loss counts may compress schedule-adjusted resume information into portable features.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Quality-win-count and bad-loss-count features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160598.
- Fold scores: 2021=0.179190, 2022=0.184087, 2023=0.173388, 2024=0.142566, 2025=0.123759.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/quality_win_count_and_bad_loss_count_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Short-rest and compressed-turnaround features` against the updated active benchmark.

### 2026-03-16 16:04 UTC - Pending Feature Family 58 - Short-rest and compressed-turnaround features

- Hypothesis: Short-rest and compressed-turnaround performance may matter in Thursday-Saturday and Saturday-Monday paths.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Short-rest and compressed-turnaround features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160636.
- Fold scores: 2021=0.181860, 2022=0.186814, 2023=0.173589, 2024=0.139507, 2025=0.121410.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/short_rest_and_compressed_turnaround_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2024, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Selection-Sunday conference-finals exposure features` against the updated active benchmark.

### 2026-03-16 16:07 UTC - Pending Feature Family 59 - Selection-Sunday conference-finals exposure features

- Hypothesis: Conference-finals timing may proxy fatigue, rust, or public overreaction entering the field.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Selection-Sunday conference-finals exposure features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161165.
- Fold scores: 2021=0.178682, 2022=0.186508, 2023=0.173785, 2024=0.143389, 2025=0.123459.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/selection_sunday_conference_finals_exposure_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Seed-line historical advancement prior features` against the updated active benchmark.

### 2026-03-16 16:10 UTC - Pending Feature Family 60 - Seed-line historical advancement prior features

- Hypothesis: Historical performance by seed line may add a stable prior on expected advancement.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Seed-line historical advancement prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159946.
- Fold scores: 2021=0.178276, 2022=0.182964, 2023=0.171932, 2024=0.142830, 2025=0.123726.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/seed_line_historical_advancement_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Seed-pair upset-rate prior features` against the updated active benchmark.

### 2026-03-16 16:13 UTC - Pending Feature Family 61 - Seed-pair upset-rate prior features

- Hypothesis: Historical upset behavior at the exact seed-pair level may improve bracket priors.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Seed-pair upset-rate prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161488.
- Fold scores: 2021=0.180377, 2022=0.187679, 2023=0.173182, 2024=0.142043, 2025=0.124160.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/seed_pair_upset_rate_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Slot-specific historical upset prior features` against the updated active benchmark.

### 2026-03-16 16:16 UTC - Pending Feature Family 62 - Slot-specific historical upset prior features

- Hypothesis: Slot-context upset priors may help beyond generic seed-pair history.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Slot-specific historical upset prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161369.
- Fold scores: 2021=0.179378, 2022=0.187733, 2023=0.173184, 2024=0.142095, 2025=0.124455.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/slot_specific_historical_upset_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Path-chalkiness prior features` against the updated active benchmark.

### 2026-03-16 16:19 UTC - Pending Feature Family 63 - Path-chalkiness prior features

- Hypothesis: Historical chalkiness by bracket context may act as a useful prior on upset frequency.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Path-chalkiness prior features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161278.
- Fold scores: 2021=0.179851, 2022=0.188193, 2023=0.173751, 2024=0.141478, 2025=0.123118.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/path_chalkiness_prior_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Potential same-conference bracket-collision features` against the updated active benchmark.

### 2026-03-16 16:25 UTC - Pending Feature Family 65 - Program NCAA appearance-streak features

- Hypothesis: Program appearance streaks may capture institutional tournament baseline.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program NCAA appearance-streak features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160574.
- Fold scores: 2021=0.179724, 2022=0.186182, 2023=0.173044, 2024=0.143132, 2025=0.120788.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_ncaa_appearance_streak_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program seed-stability features` against the updated active benchmark.

### 2026-03-16 16:38 UTC - Pending Feature Family 67 - Program historical seed outperformance features

- Hypothesis: Programs that historically outperform seed expectations may deserve a prior bump.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program historical seed outperformance features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160380.
- Fold scores: 2021=0.181692, 2022=0.183584, 2023=0.171960, 2024=0.142793, 2025=0.121872.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_historical_seed_outperformance_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program underdog upset propensity features` against the updated active benchmark.

### 2026-03-16 16:41 UTC - Pending Feature Family 68 - Program underdog upset propensity features

- Hypothesis: Program-specific upset propensity may matter when seeded as an underdog.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program underdog upset propensity features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160818.
- Fold scores: 2021=0.180975, 2022=0.186425, 2023=0.173088, 2024=0.141564, 2025=0.122040.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_underdog_upset_propensity_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program favorite hold-serve propensity features` against the updated active benchmark.

### 2026-03-16 16:44 UTC - Pending Feature Family 69 - Program favorite hold-serve propensity features

- Hypothesis: Programs that consistently hold serve as favorites may deserve extra prior credit.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program favorite hold-serve propensity features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160461.
- Fold scores: 2021=0.180012, 2022=0.186261, 2023=0.172969, 2024=0.141884, 2025=0.121176.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_favorite_hold_serve_propensity_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program recent tournament depth features` against the updated active benchmark.

### 2026-03-16 16:50 UTC - Pending Feature Family 71 - Conference auto-bid-versus-at-large proxy features

- Hypothesis: Conference bid-shape priors may help separate likely auto-bid profiles from at-large leagues.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 15 new feature-view columns to the iteration 38 stack and 15 to the iteration 49 stack for the `Conference auto-bid-versus-at-large proxy features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160918.
- Fold scores: 2021=0.179310, 2022=0.185226, 2023=0.173982, 2024=0.141683, 2025=0.124389.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_auto_bid_versus_at_large_proxy_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference seed-distribution strength features` against the updated active benchmark.

### 2026-03-16 16:54 UTC - Pending Feature Family 72 - Conference seed-distribution strength features

- Hypothesis: Conference seed distribution priors may summarize the field strength surrounding each team.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference seed-distribution strength features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160996.
- Fold scores: 2021=0.179381, 2022=0.184926, 2023=0.173336, 2024=0.142841, 2025=0.124495.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_seed_distribution_strength_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference parity and top-heaviness features` against the updated active benchmark.

### 2026-03-16 16:47 UTC - Pending Feature Family 70 - Program recent tournament depth features

- Hypothesis: Recent program depth may still matter after current-season form is accounted for.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program recent tournament depth features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160635.
- Fold scores: 2021=0.180208, 2022=0.183827, 2023=0.173182, 2024=0.144393, 2025=0.121566.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_recent_tournament_depth_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference auto-bid-versus-at-large proxy features` against the updated active benchmark.

### 2026-03-16 16:27 UTC - Pending Feature Family 66 - Program seed-stability features

- Hypothesis: Program seed stability may reflect durable program quality independent of this season's record.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Program seed-stability features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160822.
- Fold scores: 2021=0.180651, 2022=0.184556, 2023=0.173483, 2024=0.142858, 2025=0.122563.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/program_seed_stability_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program historical seed outperformance features` against the updated active benchmark.

### 2026-03-16 16:21 UTC - Pending Feature Family 64 - Potential same-conference bracket-collision features

- Hypothesis: Same-conference collision context may matter when teams know each other unusually well.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 10 new feature-view columns to the iteration 38 stack and 10 to the iteration 49 stack for the `Potential same-conference bracket-collision features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160394.
- Fold scores: 2021=0.178488, 2022=0.185703, 2023=0.171776, 2024=0.141244, 2025=0.124760.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/potential_same_conference_bracket_collision_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Program NCAA appearance-streak features` against the updated active benchmark.

### 2026-03-16 17:00 UTC - Pending Feature Family 73 - Conference parity and top-heaviness features

- Hypothesis: Conference parity and top-heaviness priors may influence how robust each resume really is.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference parity and top-heaviness features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.161240.
- Fold scores: 2021=0.178832, 2022=0.185656, 2023=0.173752, 2024=0.143037, 2025=0.124921.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_parity_and_top_heaviness_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference tournament upset-rate features` against the updated active benchmark.

### 2026-03-16 17:08 UTC - Pending Feature Family 74 - Conference tournament upset-rate features

- Hypothesis: Conference tournament upset rates may reflect how reliable a league hierarchy really is.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament upset-rate features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160128.
- Fold scores: 2021=0.174747, 2022=0.186072, 2023=0.172485, 2024=0.142887, 2025=0.124449.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_upset_rate_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The initial attempt exposed missing-column bugs in the conference-tournament conference-summary builder, which were fixed before the completed validation run. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference tournament close-game performance features` against the updated active benchmark.

### 2026-03-16 17:15 UTC - Pending Feature Family 75 - Conference tournament close-game performance features

- Hypothesis: Conference tournament close-game surfaces may add a league-level variance prior.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament close-game performance features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160908.
- Fold scores: 2021=0.176652, 2022=0.185951, 2023=0.172320, 2024=0.143302, 2025=0.126316.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_close_game_performance_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Conference tournament path-difficulty features` against the updated active benchmark.

### 2026-03-16 17:21 UTC - Pending Feature Family 76 - Conference tournament path-difficulty features

- Hypothesis: Conference tournament path difficulty may proxy the quality of late-week tests before Selection Sunday.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Conference tournament path-difficulty features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.162117.
- Fold scores: 2021=0.177113, 2022=0.189324, 2023=0.173486, 2024=0.143954, 2025=0.126706.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/conference_tournament_path_difficulty_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2021. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Secondary-tournament championship-depth features` against the updated active benchmark.

### 2026-03-16 17:26 UTC - Pending Feature Family 77 - Secondary-tournament championship-depth features

- Hypothesis: Secondary-tournament success may separate fringe teams with postseason competence from empty resumes.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Secondary-tournament championship-depth features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160824.
- Fold scores: 2021=0.180058, 2022=0.184784, 2023=0.174883, 2024=0.140944, 2025=0.123453.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/secondary_tournament_championship_depth_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2024. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Secondary-tournament participation-streak features` against the updated active benchmark.

### 2026-03-16 17:33 UTC - Pending Feature Family 78 - Secondary-tournament participation-streak features

- Hypothesis: Secondary-tournament streaks may proxy the stability of the fringe-postseason tier.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Secondary-tournament participation-streak features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160869.
- Fold scores: 2021=0.180247, 2022=0.185343, 2023=0.174253, 2024=0.141483, 2025=0.123020.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/secondary_tournament_participation_streak_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: none. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Men Massey elite-system subset consensus features` against the updated active benchmark.

### 2026-03-16 17:41 UTC - Pending Feature Family 79 - Men Massey elite-system subset consensus features

- Hypothesis: Consensus across elite Massey systems may be more informative than all-system averages.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men Massey elite-system subset consensus features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.159951.
- Fold scores: 2021=0.178403, 2022=0.183651, 2023=0.173162, 2024=0.142557, 2025=0.121984.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_massey_elite_system_subset_consensus_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022, 2025. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Run `Men Massey top-N bucket-share features` against the updated active benchmark.

### 2026-03-16 17:49 UTC - Pending Feature Family 80 - Men Massey top-N bucket-share features

- Hypothesis: Top-N bucket shares may capture how broadly supported a team is across ranking systems.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: 2021, 2022, 2023, 2024, 2025.
- Feature set: Added 12 new feature-view columns to the iteration 38 stack and 12 to the iteration 49 stack for the `Men Massey top-N bucket-share features` family.
- LightGBM config: Same fixed iteration 111 official router and source-model LightGBM configurations. No routing changes.
- Mean Brier: 0.160153.
- Fold scores: 2021=0.179317, 2022=0.184586, 2023=0.172143, 2024=0.142005, 2025=0.122715.
- Artifacts: `artifacts/iteration_116_pending_feature_family_sweep/full_run/men_massey_top_n_bucket_share_features/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the active benchmark: 2022. The active benchmark remains `Current-season block-rate features` at 0.159473.
- Next step: Pending-feature-family sweep complete. No unchecked families remain.

### 2026-03-16 18:56 UTC - Iteration 120 - Nested men upset-band rule search on current-snapshot block-rate benchmark

- Hypothesis: The men upset-band weakness from error analysis may survive a proper nested season-based rule search when treated like any other learned routing component.
- Data window: Selection Sunday cutoff. Official Stage 2 validation excluding First Four rows.
- Validation folds: Outer folds `2021, 2022, 2023, 2024, 2025`; inner rule-selection seasons drawn only from prior years, using cached fold inputs for `2011-2019` and `2021-2025`.
- Feature set: Same `Current-season block-rate features` source-model stacks and official router baseline as the current active family; only an extra men-only override family was searched on top of the rebuilt current-snapshot routed probabilities.
- LightGBM config: Same fixed source-model and official-routing LightGBM configuration as the current block-rate benchmark. The searched rule family included `no_rule` plus men-only override candidates over direction filter `{none, iteration_71_higher}`, gap threshold `{0.015, 0.020, 0.025}`, round scope `{eq_2, lte_2, lte_3}`, seed band `{none, upset_core, top_mid_upset_core, near_peer_upset_core, seed_gap_1, seed_gap_3_4}`, and `iteration_71` weight `{0.60, 0.70, 0.75, 0.80}`. Eligibility required at least `30` activated rows, at least `3` activated seasons, and both classes present.
- Mean Brier: `0.159773` for the nested selected model versus `0.159773` for the rebuilt current-snapshot no-rule baseline.
- Fold scores: `2021=0.178380`, `2022=0.185079`, `2023=0.171144`, `2024=0.142189`, `2025=0.122073`.
- Artifacts: `artifacts/iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search/validation_summary.json`.
- Notes: Rejected. Improved seasons versus the rebuilt current-snapshot no-rule baseline: none. The nested selection chose `no_rule` on every outer fold, so no override activated in held-out validation. The best eligible non-`no_rule` candidates were consistently mild broader overrides such as `gap<=0.015`, `iteration_71_weight=0.60`, and no direction filter, but each still lost to `no_rule` on inner mean Brier. Important caveat: local `data/` files were refreshed after the earlier accepted block-rate artifact, so this run is comparable to a rebuilt current-snapshot benchmark (`0.159773`), not directly to the saved accepted artifact (`0.159473`).
- Next step: Either reproduce the accepted block-rate artifact exactly on the original data snapshot, or move upstream and test a learned close-game men expert branch instead of more hand-coded router overrides.

### 2026-03-16 18:58 UTC - Benchmark reset to refreshed-data current-snapshot block-rate baseline

- Decision: Promote the rebuilt refreshed-data `Current-season block-rate features` benchmark to the working benchmark for all future experiments.
- Reason: Local `data/` files were refreshed after the earlier accepted `0.159473` artifact, so future validations will run on a different data snapshot. Comparisons against the older pre-refresh benchmark are no longer apples-to-apples.
- New working benchmark: `0.159773` mean Brier on `2021-2025`, with folds `2021=0.178380`, `2022=0.185079`, `2023=0.171144`, `2024=0.142189`, `2025=0.122073`.
- Benchmark artifact: `artifacts/iteration_120_iteration_116_block_rate_nested_men_upset_band_rule_search/validation_summary.json` under `current_snapshot_no_rule_validation`.
- Historical note: The earlier `0.159473` result remains the accepted benchmark on the older pre-refresh snapshot, but it should now be treated as historical context only.
- Going forward: All new experiments should be judged against the refreshed-data `0.159773` benchmark unless the older data snapshot is fully restored and revalidated.

### 2026-03-16 20:21 UTC - Seed-baseline unsupervised submission sanity check

- Hypothesis: If the refreshed-data final submission is wired correctly, its seeded 2026 matchup probabilities should show roughly the same correlation to a simple seed-only baseline as the official validation folds.
- Data window: Official validation folds `2021-2025` excluding First Four rows, plus the 2026 final submission restricted to seeded-seeded matchups. The official-like 2026 check excludes same-seed-line play-in pairs to mirror the research scoring scope.
- Method: Fit separate men and women logistic seed baselines on historical tournament outcomes using only `delta_seed_number`, always training on seasons strictly before the scored season. Compare Pearson correlation between the refreshed-data current-snapshot block-rate probabilities and the seed-baseline probabilities. The 2026 check is outcome-free and serves only as a structural sanity check.
- Validation correlation: `2021=0.9432`, `2022=0.9575`, `2023=0.9498`, `2024=0.9526`, `2025=0.9469`; mean `0.9500`, range `[0.9432, 0.9575]`.
- 2026 submission correlation: Official-like seeded subset `4548` rows overall `0.9575`, with men `0.9519` on `2274` rows and women `0.9653` on `2274` rows. Including the 8 same-seed-line play-in pairs changes the all-seeded correlation only slightly to `0.9574` on `4556` rows.
- Notes: No model, feature, calibration, or routing changes. The 2026 seeded submission sits inside the historical validation correlation range and near its upper end, which is consistent with a submission aligned to the accepted validation behavior rather than a wiring error.
- Next step: Keep this as a lightweight final-submission QA check; if needed, add a second sanity pass on seeded-vs-unseeded and prediction-distribution slices.

### 2026-03-17 14:06 UTC - Conservative Polymarket title-futures overlay candidate

- Hypothesis: Men’s Polymarket title futures may reveal mild team-level bias even when direct game markets are sparse, but the safest use is a tightly capped seeded-field overlay rather than a full-board replacement.
- Data window: Current 2026 men’s title-futures board from `2026-ncaa-tournament-winner`, normalized to fair probabilities, applied on top of `submissions/2026_stage2_current_snapshot_block_rate_polymarket_quintile_blend_2026-03-17.csv`.
- Method: Exact men’s bracket dynamic programming over `MNCAATourneySeeds.csv` and `MNCAATourneySlots.csv` to compare model-implied title odds to Polymarket fair title odds, then convert the team-level logit gaps into small team offsets with `alpha=0.10`, offset cap `±0.10`, and per-game probability cap `±0.03` on seeded men field pairs only.
- Artifacts: `artifacts/polymarket_title_bias_2026-03-17/team_title_bias.csv`, `artifacts/polymarket_title_overlay_conservative_2026-03-17/adjusted_rows.csv`, `artifacts/polymarket_title_overlay_conservative_2026-03-17/summary.json`.
- Output submission: `submissions/2026_stage2_current_snapshot_block_rate_polymarket_quintile_blend_title_conservative_2026-03-17.csv`.
- Notes: This is a submission-time exploratory overlay only, not a validated model change. It adjusted `2278` seeded men field rows with mean absolute move `0.00562`, max move `0.03`, and `14` rows hitting the move cap. The strongest positive team offsets favored St John’s, Wisconsin, UCLA, and BYU, while the strongest negative offsets pushed down Virginia, Iowa State, Duke, and Nebraska.
- Next step: If this candidate is worth considering for a live entry, compare it side-by-side against the already submitted direct-game Polymarket overlay and decide whether the futures signal is additive or too noisy.

### 2026-03-19 13:31 UTC - Aggressive direct Polymarket replacement overlay

- Hypothesis: For live March Madness games with sufficiently liquid direct Polymarket moneylines, a hard replacement with the market midpoint may outperform the validated model on those rows despite being much more aggressive than the earlier bounded overlay.
- Data window: Live Polymarket March Madness direct game markets from `https://gamma-api.polymarket.com/events?tag_slug=march-madness&closed=false&limit=500`, applied on top of `submissions/2026_stage2_current_snapshot_block_rate.csv`.
- Validation folds: None. Submission-time live overlay only.
- Feature set: No LightGBM feature or model changes. Replaced eligible submission rows with the oriented bid/ask midpoint from uniquely mapped direct game markets when `volume > 10000`.
- LightGBM config: Unchanged from the refreshed-data current-snapshot block-rate benchmark. Overlay only.
- Mean Brier: Not available. No historical validation for this live-only overlay.
- Fold scores: Not available.
- Artifacts: `artifacts/polymarket_direct_replace_volume_gt_10k/adjusted_rows.csv`, `artifacts/polymarket_direct_replace_volume_gt_10k/manifest.json`.
- Output submission: `submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k.csv`.
- Notes: Applied `31` direct-market replacements and skipped `1` discovered March Madness event because its live volume was not above `10000`. Submission sanity checks passed: exact `132133` rows, preserved ID order, no nulls, finite predictions, and all probabilities in `[0, 1]`. The aggressive override produced mean absolute move `0.05498` across changed rows, max move `0.17696`, `16` rows with move at least `0.05`, and `5` rows with move at least `0.10`.
- Next step: Compare this aggressive direct-replacement submission against the conservative direct-moneyline and title-futures overlays before deciding whether the loss of model smoothing is acceptable.

### 2026-03-19 13:37 UTC - Aggressive direct replacement plus conservative title overlay on remaining rows

- Hypothesis: Keep the high-confidence direct-game Polymarket overrides fixed on liquid live markets, then still harvest the weaker title-futures signal on the rest of the seeded men field rather than discarding it entirely.
- Data window: Base input `submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k.csv` plus the live 2026 men’s title-futures board from `2026-ncaa-tournament-winner`.
- Validation folds: None. Submission-time live overlay only.
- Feature set: No LightGBM feature or model changes. Kept the `31` direct-game midpoint replacements fixed and applied the conservative title-futures team-offset overlay only to title-eligible rows not already replaced by direct markets.
- LightGBM config: Unchanged from the refreshed-data current-snapshot block-rate benchmark. Overlay only.
- Mean Brier: Not available. No historical validation for this live-only overlay stack.
- Fold scores: Not available.
- Artifacts: `artifacts/polymarket_direct_replace_volume_gt_10k_plus_title_overlay_conservative/adjusted_rows.csv`, `artifacts/polymarket_direct_replace_volume_gt_10k_plus_title_overlay_conservative/summary.json`.
- Output submission: `submissions/2026_stage2_current_snapshot_block_rate_polymarket_direct_replace_volume_gt_10k_plus_title_conservative.csv`.
- Notes: The current title board resolved `64` seeded teams and left `4` seeded teams with missing live title markets, which were treated as zero title probability. Raw title overlay coverage was `2278` seeded men field rows; after excluding the `31` direct-replaced rows, `2247` title-overlay rows remained eligible. The written final submission preserved all `31` direct-replaced rows exactly, passed submission sanity checks, and produced title-only mean absolute move `0.00657` versus the direct-only submission with max move `0.03`. Because many filtered title rows had capped delta `0`, the title audit contains `2247` eligible rows while `1803` rows changed in the final file versus the direct-only submission.
- Next step: Decide whether the combined stack should be the live entry or whether the title overlay remains too noisy relative to the direct-only aggressive submission.
