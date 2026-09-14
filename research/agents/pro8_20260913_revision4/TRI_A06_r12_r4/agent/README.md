# TRI_A06_r12_r3_fix1

**Status: complete, compiled candidate with bounded conditional validation. No new full match or opponent-pool win rate is claimed.** This is a late continuation of the original third round, whose deadline remains 2026-09-13 21:00:45 UTC. The feedback did not restart a two-hour budget.

## Exact lineage and scope

Immediate parent: TRI_A06_r12_r3, ZIP SHA256 `2e577b72aa047a7c9c36197d32f7730c87dcfca98daaf6a3bf95d82b98ef6e1b`, native SHA256 `7d7fd9540328b4d87df7b9970c03d63837831713efd781b86359a33de76836f5`. Its immutable production snapshot is `parent_r3/`; the byte-exact original archive is `provenance/TRI_A06_r12_r3.zip`. The underlying r2 ancestor is the central-tested `638626de...` native, not the alternate same-name `bd0c7dd...` r2.

Production native SHA256: `ed0d275ca9caf13fdaaf2c833e164c5b8c1ccf313dbf96296fb9f7896254ab90`.

Only three of the 50 production/build inputs change: `policy/ongoing_supply_admission.inc` (one admission mechanism), `policy/triad.hpp` (diagnostic fields), and the root `main.py` description. The production configuration and compiler flags are unchanged. Calendar optimization, positive-return investment, rolling execution, stock/labor/transport constraints, r2 survival-wage ordering, r1 late planting, and the earlier r3 recurring-fertilizer repair remain intact. See `SOURCE_CHANGE.diff` and `SOURCE_FREEZE.json`.

## Evidence and diagnosis

Central supplied real, independently generated r2/r3 own-visible trajectories. Their results are historical evidence, NOT fix1 results. In the target R2 805057947 seat0 game, own terminal cash rises from 94,240 to 100,669 under r3, but the public rival cash rises from 93,932 to 102,074: margin +308 becomes -1,405. It is incorrect to interpret the loss as simple fertilizer waste.

An own-unit replay and daily plant-refresh audit finds 159 successful fertilizer applications in r2 versus 172 in r3, with zero failed fertilizer applications in either. Actual strawberry production/harvest is 306/306 versus 309/309; tomato production/harvest is 147/145 versus 134/130. These are different complete trajectories, not an isolated fertilizer experiment. The late fertilizer arrival audit verifies 2 purchased units in r2 versus 18 units in r3 (steps 384/480/600/624/672 receive 4/9/3/1/1). It reconstructs the own unit phase, then compares actual next-observation warehouse fertilizer; it does not assume every requested order filled.

**Individual spending is not uniquely attributable from every saved own observation.** Simultaneous other sales and purchases and unknown opponent orders affect settlement. `logs/exact_fertilizer_arrivals.json` records actual whole-step cash deltas and confirmed quantities; `logs/own_audit_*.json.gz` labels own-only market quotations as quotations, with a whole-step cash reconciliation flag. Quoted costs on non-reconciling steps are not presented as actual fertilizer spending. Official conditional tests below do capture actual per-unit committed prices.

The structural issue is narrower and reproducible: r3 asks whether extra fertilizer reaches additional selected maintenance, but not whether the changed preparation order and execution route improve realized final cash. At target step672 (day28), one extra purchase increases the current-day coverage screen from 8 to 9, yet shifts the order/worker/harvest/delivery path. A terminal comparison can lose money even though fertilizer is used correctly. Its counterfactual cash is model evidence, not the recorded game's alternative future.

## Actual change

Keep r3's original current-day feasibility and coverage screen. Only when the true game terminal is at most 48 transitions away, compare the supplied and no-extra-purchase plans by executing each to the actual terminal in two existing legal public-flow scenarios: no new rival supply, and the unchanged configured public-supply forecast (scale 0.85).

Admit the extra procurement only when **both** scenarios show strictly positive own terminal cash gain and strictly positive competition-weighted cash-margin gain. The competition weight is the existing unchanged configuration (2); no new economic coefficients were tuned. Cash is the terminal objective: no unsold inventory, standing yield, unused fertilizer, or theoretical harvest is credited. Earlier r3 decisions are not subjected to this new terminal screen. Uncertain/error cases fail closed to the original no-extra-purchase preparation, not to a different agent. Normal resource recovery stays enabled.

The 48-step limit bounds computation and ensures a real terminal endpoint; it is not a seed/opponent-specific rule. Forecast construction uses current public farm/market data, the policy's existing public-flow model, and own private state only. Synthetic rival inventory exists only inside the explicitly conditional forecast. No hidden actual rival state, random script, or post-divergence saved future is read. Diagnostic counters are returned in `td_debug_json`.

## Completed validation

* The research-only diagnostic build reproduces all 2,876 supplied r3 actions across four histories. Its additional measurements are not production code or matches.
* Fix1 remains identical for all 719 actions of market_smart568422814, moon31485687, and R2 1827852203, including the earlier R2 loss-to-win case. The target R2 805057947 matches 672 actions, then diverges at step672; historical probing stops immediately. These are recorded-action checks, not new wins.
* Four C++ suites pass, 23,892 assertions total: 18,403 + 3,029 + 46 inherited assertions, plus 2,414 terminal-economic assertions. The latter includes both positive and nonpositive terminal outcomes, adversarial margin signs, nonfinite values, true endpoint checks, existing stock, and zero funding.
* Ten official-engine conditional trajectories (five parent/fix1 pairs), 47 steps each, total 470 steps, reach the real terminal with reconciled committed-cash ledgers. Every next observation is generated by the official engine after the fork; saved observations after the fork are never used. Capacity, nonnegative resources/cash, hand counts, and order limits are checked. These are not complete new matches.
* Independent default production rebuild produces a byte-identical native. Actual GCC14.2.0 build receipts and logs are included; central GCC13.3 verification supplied by the user concerned parent r3, not this fix1.
* Root/native/input checks pass: 50 input hashes, 384 prefix/seat-isolation calls, 8 resets, and 3 negative entry checks. See `logs/entry_result.json` and the final delivery receipt.

### Official terminal forks: actual committed cash within stated scenarios

All forks start at actual r3 observation672. Missing rival private inventory/actions are declared synthetic inputs, not reconstructions of the real opponent. Both variants are warmed on the matching actual prefix. Synthetic seed 2609134201 is recorded.

| Case / condition | r3 cash | fix1 cash | own gain | unweighted relative cash gain |
|---|---:|---:|---:|---:|
| R2 805057947, no new rival supply | 103,132 | 103,667 | +535 | +535 |
| R2 805057947, rival supply/input pressure | 102,661 | 103,136 | +475 | +542 |
| R2 1827852203, no new rival supply | 100,108 | 100,108 | 0 | 0 |
| R2 1827852203, rival supply/input pressure | 98,536 | 98,536 | 0 | 0 |
| R2 805057947, zero initial cash/shed/seeds | 4,789 | 4,789 | 0 | 0 |

In the target idle-supply fork, r3 actually buys one fertilizer for 15; fix1 buys none. Successful applications fall 23 to 19 and bonus-bearing production events fall 10 to 9, while sold tomatoes increase 40 to 46, wheat 196 to 201, and fertilizer 16 to 20. Both sell 18 strawberries and hire 24 workers. The result is actual conditional cash after purchases and wages, not a valuation of unsold produce. Under supply/input pressure the removed unit costs 19 and own cash improves 475. In both positive-control forks the original nine-unit purchase is retained (actual spending 295 / 331); all 47 own actions and terminal outcomes remain identical. Under zero funding there is no fertilizer purchase, no negative resources, and both agents recover the same 4,789 through subsequent actual sales.

## Limits and failure record

No actual R2 opponent source or future response is available; **this does not prove that the -1,405 recorded loss becomes a win**. The full fixed 12-opponent, 64-seed, two-seat panel is still central's task. The included 1,087/1,536 = 70.7682% panel is r2 historical data, not r3 or fix1. The >85% criterion (at least 1,306 strict wins; ties not wins) has not been demonstrated.

Only the terminal economic admission gap is addressed. Earlier bulk purchase quantity/frontier effects remain diagnostic findings, not silently implemented extra changes. Two conditional scenarios do not guarantee the true adversarial market response. Public-flow model cash is approximate (for example, one internal idle prediction differs from the official fork by 2); official fork results, not predictions, are used in the table. The two remaining days may still suffer other crop/market/planning errors.

The first price-floor negative fixture incorrectly used market inventory 10,000, which was not actually the official price floor and still earned +44. That assertion failed, its source/logs were retained, and the fixture was corrected to a genuine saturated 1,000,000-inventory case (zero terminal gain, rejected). No production behavior was changed to force that test. A packaging-script quoting SyntaxError, corrected before staging, and tool transport issues are also retained/described in `FAILURES_AND_LIMITS.md`. No failure was counted as a pass.

Maximum measured fix1 call in these official forks: approximately 0.592 seconds; not a universal worst-case bound. The terminal screen adds computation. This environment exposes five affinity CPUs but a four-CPU cgroup quota and a 4GiB memory limit; all work uses one process/worker with 30% memory headroom. Production build takes 28.92 seconds wall / 610,376KiB peak RSS; independent rebuild takes 28.51 seconds / 610,056KiB. See raw `logs/*.time`.

## Files and reproduction

Run from archive root:

```sh
python3 build.py
python3 build.py --unit
python3 tests/verify_entry.py --root . --evidence feedback --out entry_check.json
python3 research/run_terminal_pair.py submission_56149565_805057947_seat0 observed_market_idle_rival
```

The production entry is root `main.py`. `policy/a06.so` is the matching portable Linux x86-64 native; no source is selected by case/seed at runtime. `parent_r3/`, `diagnostic_r3/`, `feedback/`, and `research/` are offline only. `research/checkpoint_scripts/` preserves the exact original research scripts and original working paths; those checkpoint scripts are records, not required for production. All original and current development seeds are in `DEVELOPMENT_SEEDS.json`. Full raw official fork observations/actions, explicitly synthetic rival actions, per-unit transactions, services, production events, and cash ledgers are in `logs/official_terminal/`. No real opponent private data or source is included.
