# POLY-ODDS-BLEND

## Goal

Use Polymarket as a conservative, post-prediction adjustment on top of the best accepted model at submission time.

This is a submission-time overlay only. It is not part of the validated LightGBM training stack because we do not have the historical Polymarket data needed to train or validate it under the repo's 5-fold walk-forward rules.

## Decision

For the single final submission:

- Start from the calibrated probabilities produced by the best accepted model at submission time.
- Blend Polymarket only for rows with a direct, confidently mapped matchup market.
- Use a small logit-space blend weight.
- Cap the adjustment so Polymarket can only nudge the model, not replace it.
- Do not use hard `0.0` or `1.0` overrides.

## Why This Shape

- The GOTO notebook in [resources/updated-goto-conversion-winning-solution.ipynb](/Users/mariofilho/ml/random/ml-mania-2026/resources/updated-goto-conversion-winning-solution.ipynb) uses externally produced market probabilities as an ingredient, not as a trained model feature.
- Our accepted pipeline is already calibrated and validated. Replacing it with an unvalidated market signal would be too aggressive.
- With one submission and no market history, the safest reasonable use of Polymarket is a bounded post-calibration tilt.

## Exact Adjustment

For each submission row:

1. Compute `p_model`.
   Source: the final calibrated probability from the best accepted model available at submission time.

2. Attempt to compute `p_market`.
   `p_market` must mean `P(lower TeamID wins)`.

3. If no direct Polymarket matchup is available, keep:
   `p_final = p_model`

4. If a direct Polymarket matchup is available and passes the quality gates below, compute:

```text
logit(p_blend) = 0.90 * logit(p_model) + 0.10 * logit(p_market)
```

Then cap the move away from the model:

```text
p_final = clip(
    p_model + clip(p_blend - p_model, -0.05, 0.05),
    0.02,
    0.98
)
```

Where:

- `logit(p) = ln(p / (1 - p))`
- probabilities should be clipped to `[0.02, 0.98]` before applying `logit`

## Quality Gates

Only use Polymarket for a row if all of the following are true:

- The market is a direct head-to-head game market for that exact matchup.
- Team mapping to Kaggle `TeamID` is unambiguous.
- The market side can be mapped cleanly to `P(lower TeamID wins)`.
- The market is still open and recent enough to be actionable.
- The market is sufficiently liquid relative to the rest of the board.

If any gate fails:

- set `p_final = p_model`

## Mapping Rule

Submission rows use:

- `ID = Season_LowerTeamID_HigherTeamID`
- `Pred = P(lower TeamID wins)`

So the Polymarket probability must always be re-oriented to the lower TeamID side:

- if the market contract is "lower TeamID team wins", then `p_market = market_price`
- if the market contract is "higher TeamID team wins", then `p_market = 1 - market_price`

Never blend before this orientation step.

## Market Price Choice

Use one probability per eligible market.

Preferred order:

1. a clean midpoint from the current market
2. otherwise a recent traded probability if the market is active

In either case, clip the chosen `p_market` to `[0.02, 0.98]` before blending.

## What Not To Do

- Do not add Polymarket as a LightGBM feature.
- Do not retrain the calibrator on Polymarket inputs.
- Do not apply the GOTO notebook's hard upset-pick override.
- Do not use futures or title markets as a proxy for direct game odds.
- Do not let unmatched or weakly matched markets affect the submission.

## Recommended Operational Order

1. Generate the submission probabilities from the best accepted model available at submission time.
2. Build a clean mapping table from Polymarket game markets to Kaggle team IDs.
3. Compute `p_market` only for direct, eligible markets.
4. Apply the bounded logit blend row by row.
5. Run final sanity checks:
   - all `Pred` values in `[0, 1]`
   - no nulls
   - no duplicate IDs
   - unchanged row count
   - men and women mapping checked separately

## Honest Assessment

This adjustment is intentionally conservative because it is not historically validated in this repo. Its purpose is to inject current market information without surrendering the validated gains of the best accepted model available at submission time. If Polymarket is noisy, stale, or thin, this recipe should mostly fall back to the model. If Polymarket is informative, it can still move the submission in the right direction, but only in a controlled way.





https://polymarket.com/event/2026-ncaa-tournament-winner
