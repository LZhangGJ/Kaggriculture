# C01 round 1

Complete runnable source and compiled native checkpoint. Entry point: main.py:agent(observation, configuration). Offline build: python3 -B build.py --cxx g++.

## Actual terminal results

{
  "packaged_utc": "2026-09-13T18:12:00.160461+00:00",
  "development_selected": {
    "games": 48,
    "distinct_seeds": 24,
    "mean_terminal_cash": 209684.08333333334,
    "all_terminal_checks_pass": true
  },
  "untouched_validation": {
    "games": 128,
    "distinct_seeds": 64,
    "mean_terminal_cash": 201614.625,
    "all_terminal_checks_pass": true
  },
  "planned_validation_games": 128,
  "completed_policy_games": 930,
  "other_result_records": 194,
  "source_directory": "/mnt/data/c01_submission",
  "snapshot_delivery": true
}

Only completed 719-transition games are counted. A partial validation panel is not a full-panel claim. Full receipts, registered seeds, action traces, build timings, terminal audits where completed, and failures/cancellations are under reports/. No competitive acceptance claim is made.

The working environment restarted during the round. Earlier raw logs were lost; all detailed receipts here are actual post-recovery execution, not reconstructed original measurements.

## Strategy

The selected C01 checkpoint retains the supplied complete A06 crop, livestock, feeding, market, land, staffing and routing strategy. It adds remaining-game marginal cash curves, a guard requiring positive terminal payback, and a ranking penalty proportional to capital tied up over time relative to available liquidity. The rule applies to daily and intraday investment. Projected cash is not current spendable cash. The executor uses minimum legal crop maturity and clips finite crop projections to the remaining horizon, so viable short late crops can actually be planted. The selected configuration uses a four-day conditional rollout. Forecasts use permitted observations, not evaluation seeds, future random draws, opponent private inventory or replay actions.
