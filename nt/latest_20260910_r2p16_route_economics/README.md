# P16 route optimization with delivery, capacity and payment constraints

This earlier local experiment is retained for development traceability. **It is not a competition replacement:** the extended panel found 12 protected-task omissions in six games. The later [complete-workflow agent](../latest_20260910_r2p16_route_workflow/main.py) repairs complete dependency chains; [latest procurement repair](../latest_20260911_r2p16_route_repair/README.md) investigates cash/resource coordination.

The baseline is commit `979fec14bdddd281f4d934494aa4a06673fba9b9`, library SHA-256 `4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b`. This package's tested mode-3 library is `build/revision5/route3.so`, SHA-256 `97d0918f7b74839050c39726ebc2f6b7daede6be9193930c0a4ddcd40bb48de7`. No economic model was retrained.

## Implemented sequence

1. Project deliveries in official worker order; markets execute afterwards. Avoid overflowing whole-bag delivery through quantitative placement or waiting. Check actual fills and preserve orders displaced by the market-order limit. Validate a complete schedule with actually expected workers/resources before cancelling a hire.
2. Keep the incumbent route and try bounded task-group reordering, intermediate warehouse visits, selective unloading and coordinated delivery. Preserve task precedence and supplies needed for later feeding/fertilization.
3. Simulate work and transactions together: completion, harvest, overflow, fills and cash. Only a few candidates receive next-day valuation and procurement checks. Do not subtract wages twice or treat inventory value as cash.
4. Register accepted commitments, reconcile real effects and attempt recovery. This stage's task-level recovery proved insufficient to preserve complete sow/water and pickup/feed chains.

Light checks run each step; heavier searches run at work start or relevant feedback, capacity or cash pressure, with trigger intervals. Each search considers up to six route alternatives, at most two with next-day valuation. Leaf simulations do not recursively start another route search. No global optimum is guaranteed.

Other fixes retain delivery actions whose goods will be harvested later, carry displaced purchases to the next preparation step, update routes after actual hiring, and verify the official first-harvest maturity boundary for all five crops.

## Extended 880-game result

Twenty seeds, 11 original live opponents, both seats and two frozen agents produce 440 games per version with the existing compiled P16 C++ simulator.

| Metric | Original P16 | Route economics |
|---|---:|---:|
| Wins / losses | 324 / 116 | 346 / 94 |
| Win rate | 73.64% | 78.64% |
| Mean cash | 110,224.9 | 107,762.0 |
| Mean wages | 4,192.7 | 4,031.1 |
| Protected omissions | 0 | 12 in six games |

The candidate rescues 30 losses and loses eight baseline wins. The 95% whole-seed bootstrap interval is [-0.91, +14.55] percentage points. Wages decrease in 232/440 cases, but cash also decreases in 120 of them. Altered empty land can change subsequent shop draws under the same seed; full-game differences include environmental and opponent feedback.

All 880 games finish without runtime errors. The 334 available prior Python games agree step-by-step with C++ observations, actions and audits. Two slow candidate trajectories reproduce all actions in serial reruns, peaking at 0.671 and 0.623 seconds. See [POOL_REPORT.md](POOL_REPORT.md) and `POOL_VALIDATION.json`.

## Earlier 132-game validation

The earlier panel used five seeds, three opponents, both seats and four versions: 120 matches. Twelve additional mode-3 old-seed games are separate regression checks.

| Version | Wins / 30 | Mean cash | Mean wages | Mean overflow |
|---|---:|---:|---:|---:|
| P16 | 24 | 109,590.9 | 3,681.8 | 9.57 |
| Execution constraints | 24 | 109,590.9 | 3,681.8 | 9.57 |
| Plus route/delivery candidates | 25 | 111,148.1 | 3,848.0 | 13.20 |
| Plus economic comparison/hire reduction | 26 | 106,428.6 | 3,895.9 | 13.47 |

All 132 games completed 720 frames; 94,908 action pairs were executed and independently replayed with the official referee. No protected omissions or invalid work actions were recorded in this small sample. Twenty mechanism checks passed, and a disabled build matched all 719 baseline actions in one complete causal trajectory.

Mode 3 cancelled only two hires, from both seats of one seed, while mean wages still rose. The delivery version fell from 10/10 to 6/10 against Market Smart. The twelve old-seed checks yielded eleven wins and must not inflate the fresh-panel rate. A maturity defect was fixed during this panel, so it was not an untouched final holdout.

`VALIDATION.json` retains an overall `FAIL`: nine parallel calls exceeded one second, peaking at 1.524 seconds. Two serial reruns reproduced every action and peaked at 0.801 and 0.776 seconds. Their timings do not erase the parallel record or certify Kaggle timing. The later omissions in `POOL_VALIDATION.json` supersede the earlier functional pass for promotion decisions.

## Build and files

Use Linux x86-64 / WSL with GCC 13. `main.py` defaults to the shipped mode-3 library. Use one `create_agent()` per game and close it after use; a new step zero resets the runtime.

```bash
python3 -B build.py --mode 3 --unit --out /tmp/route_economics_check/route3.so
```

The build refuses existing outputs. Modes are cumulative: 0 disables intervention; 1 adds execution/capacity checks; 2 adds routes and intermediate delivery; 3 adds economic/next-day comparison and cautious hire reduction. Only the final tested mode-3 runtime and its receipt are shipped; other modes can be rebuilt.

`policy/route_economics.hpp` contains the search; `triad.hpp` integrates it; `executor/policy.hpp` provides execution hooks; `public_flow_scenario.hpp` supplies forecasts. `tests/test_route_economics.cpp` contains mechanism checks.

Original panel/summarizer scripts require the source workspace's referee and opponent assets. They are audit sources, not portable standalone runners. Raw evidence remains under `experiments/r2p16_route_economics_20260910` and `experiments/r2p16_vs_route_agent_pool_20260910`; historical receipts are also in [earlier handoff evidence](../../evidence/r2p16_route_agents_20260911/economics_stage).

The policy forecasts future public flow; it does not know unopened shops, random weeds or future actions. Protected work means accepted commitments, not every unaffordable economic intention.
