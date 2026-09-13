# C07 ROUND04 economic revision

## Outcome and source boundary

**Fixed author validation: PASS. Mean terminal cash: 204,163.25.** The economic target is 200,000. All 32 of the 32 declared author games completed; runtime failures: 0. The conservative whole-panel gate mean is 204,163.25. The supplied unchanged C07 baseline on exactly the same validation seeds averaged 198,931.9375. The paired mean change is 5,231.3125, with 15 higher, 17 lower and 0 identical cash outcomes. Median paired change: -3,257.5. Worst paired change: -48,532; best: 68,936. The positive mean does not imply broad cell-by-cell dominance; more validation cells declined than improved. Revised validation cash ranges from 151,751 to 257,826.

This is economic testing against the supplied legal PASS opponent, not competitive acceptance. No round robin, fixed 12-opponent acceptance panel, or claim of an 85% competitive win rate is made. No result from the missing or unverified earlier round02 archive is counted as ROUND04 evidence.

The supplied ROUND04 ZIP has SHA256 `f97150f50853859f177fe5d382d8788a474a9c8de25255d3ad3987d6b1bb1249`. All 129 manifest entries were checked, with zero mismatches. The baseline rebuild reproduces the controller's disclosed 32-game mean of 189,323.21875 exactly. Four matched C07 supplied replays also match the rebuilt baseline in every action and complete state frame, all 719 transitions each. Source, referee and reference identities remain in the original extracted input and identity receipt.

## Frozen complete strategy

Selected development experiment: `receipt_contract_starts`. Its full 32-game development mean is **204,217.53125**, compared with the unchanged C07's 189,323.21875. Source was frozen at **2026-09-13T20:57:23.921712+00:00**, before the first author or parent validation attempt. The final native binary and all 42 Python/C++ source files are byte-identical to those tested in the selected development experiment; the effective runtime configuration is identical. Final documentation and build metadata were prepared before freezing.

The package contains the complete inherited A06/C07 economic planner, route executor, Python codec, C++ simulator and all runtime dependencies that are project files. It is not a renamed C03 submission. C07 independently implemented the rolling rules and admission guards below. C03's delivery code and economic setting block are explicitly attributed and measured separately. The selected rule has 10 live mode switches and differs in actual executed action streams from the C03-component control in 9 of the 32 development games. Its incremental mean over that reference-equivalent-cash control is only 31.59375; most of the improvement over old C07 comes from the attributed economic setting block and avoiding harmful rolling overrides. That small incremental result is not presented as broad proof of superiority to C03.

**Whole economic policy.** At an observed dawn, the inherited planner estimates affordable crop and livestock portfolios, services, feed, fertilizer, land and labor. The daily candidate search simulates execution and installs a workforce-feasible controller, preserving existing paid commitments. The executor performs market orders, movement, servicing, harvesting, transport and reinvestment using current observations. It does not merely return a forecasted portfolio.

**C07 rolling rule 7.** Keep the dawn-selected receipt-gated mode 4 contract for the current day. A conditional continuation estimate cannot override that explicit gate. For a controller in another eligible mode, inspect actual cash receipts, stock or farm-production changes at hours 4, 8, 12 and 16, at most two probes per day. Probe only when there is an idle worker, a vacant tile, cash reserve, no active admission and no final-day horizon. Compare the incumbent with allowed alternatives among modes 1 and 2. Mode 3 WAIT/NOW is unavailable.

Each alternative is a copy of the current controller simulated to the current day's end under a conditional public-state model. It must generate a different action sequence, fit the remaining assigned routes, leave no more scheduled work unfinished, and start no fewer feasible same-day projects than the incumbent. Unfunded reserved inputs are subtracted from observed cash. The model score improvement is current-day cash difference plus 0.75 times the continuation-value difference. It must exceed `25 + 0.1 * work_price * remaining_route_actions + 0.002 * min(20000, free_cash)`. Only one live switch per day is permitted. The selected reinvestment rule is read by the actual executor; existing paid routes are not replaced or cancelled by this selector.

The margin is a development-calibrated heuristic, not a statistically fitted error bound. Conditional starts, tail values and forecast gains are logged as forecasts, not realized profit. Current-day forecasts assume a rival PASS continuation and do not know future market randomness or new weeds. Runtime inputs contain no game seed, future randomness, opponent private state, or replay-derived actions. Seed values appear only in the referee, declarations and evidence metadata; the harness asserts that the policy configuration's seed is None.

**Attributed C03 components.** The exact supplied delivery changes in `policy/executor/policy.hpp` support saleable partial warehouse deposits, explicit bounded detours, carrying-capacity projections including scheduled watering, idle overflow dispatch, and retention of required feed/fertilizer. Physical capacity and warehouse room remain executor checks. The imported economic settings are `competition=0.5`, `work_price=1`, `harvest_threshold=3`; `c03_delivery_mode=3` enables delivery. No C03 inventory-sale extension or extra economic-proposal component was imported. Other settings and build flags are provided in full, not inferred from this summary.

## Development evidence and rejected alternatives

Every row below uses the same disclosed 16 seeds in both seats, against legal PASS. These are development results, not untouched validation results. No attempted variant is omitted.

| Development experiment | Mean terminal cash | Completed |
|---|---:|---:|
| baseline_full | 189,323.21875 | 32/32 |
| no_roll | 191,437.25 | 32/32 |
| no_wait | 191,902.96875 | 32/32 |
| guarded_wait | 191,437.25 | 32/32 |
| guarded_no_wait | 191,437.25 | 32/32 |
| guarded_economics | 203,638.78125 | 32/32 |
| guarded_delivery | 191,443.0625 | 32/32 |
| guarded_combined | 204,185.9375 | 32/32 |
| calibrated_base | 191,899.46875 | 32/32 |
| calibrated_combined | 202,701.59375 | 32/32 |
| no_wait_combined | 203,078.59375 | 32/32 |
| receipt_contract | 204,066.84375 | 32/32 |
| receipt_contract_starts | 204,217.53125 | 32/32 |

Removing all rolling changes improved old C07 to 191,437.25 but did not qualify. Removing WAIT alone reached 191,902.96875. The initial large work/cash margin accepted zero live switches and behaved like disabled replanning. The C03 economic setting block, with delivery disabled, raised that control to 203,638.78125. Delivery alone added only 5.8125 to its corresponding original-economic control. With the economic block, delivery added 547.15625; the component interaction matters. Requested market orders or goods left in inventory were not converted into hypothetical recoverable cash.

Calibrated rolling with both components reached 202,701.59375, below its no-switch component control. In a full-panel ablation, retaining receipt-gated mode 4 recovered 1,365.25 mean cash, improving 3 cells and worsening none. Requiring no reduction in same-day starts added 150.6875, improving one cell and worsening none. These are actual complete-game controlled comparisons on the disclosed panel, not proof of future generalization. Switch times and first action divergences are separately recorded in `SELECTED_EXECUTED_ACTION_ANALYSIS.json` to distinguish mode counters from executed behavior.

Selection was committed from four named development finalists by highest complete-panel mean, with a declared tie rule. The unchanged/reference-equivalent component control was not an eligible renamed submission. No alternative source was chosen after validation began.

## Validation design and full-panel results

The 16 author validation seeds were declared at **2026-09-13T20:35:49.924519+00:00**, before any ROUND04 game outcome. They are disjoint from all 964 seeds in the supplied inventory. That inventory explicitly warns that lost historical seed usage is unknown, so absolute non-use in unavailable prior histories cannot be proved.

Seeds: `890031804, 779113208, 1490453081, 1439415516, 1550616920, 1873633782, 273156459, 76797801, 1496046506, 1807826296, 1159744124, 609702205, 846398706, 113710341, 1384339204, 1282661023`. Both seats were played for each seed. The unchanged baseline used exactly the same panel for comparison; it was not another candidate-selection opportunity. No seeds were added or replaced. Every declared cell has a result file. Failed games would count as zero in the conservative gate mean and be identified as failures, not fabricated terminal outcomes.

Actual author validation mean: **204,163.25**. Baseline comparison mean: **198,931.9375**. Qualification status: **PASS**. The final selected source is delivered unchanged regardless of this status. Per-seat results, all cash values, action hashes, timings and source/replay identities are in the CSV and original per-game JSON records.

## Resources, builds and integrity

Measured CPU quota: 4 cores (`400000 100000`); affinity: 5 logical CPUs; cgroup memory limit: 4,294,967,296 bytes. At most two compilation, game or replay-audit workers were active concurrently. The compiler was GCC 14.2.0. Baseline rebuild took 26.795851145 seconds; the final build took 27.578 seconds. An isolated copy with no native binary rebuilt in 27.063 seconds and produced a byte-identical binary. The supplied baseline binary differs from the locally rebuilt baseline, so identity of compiler output was not assumed; replay/action reproduction was verified instead.

Mean logged game time for selected development: 4.442 seconds. Mean author-validation game time: 4.510 seconds. Peak reported game-process RSS for author validation: 98,344 KiB. Full action timings, including maxima and 99th percentiles, are recorded separately. Runtime uses Python's standard library and local project files. The native is x86-64 Linux and dynamically links ordinary system C/C++ libraries; `final_native_ldd.txt` records actual dependencies. Rebuild on a different target host as necessary. Kaggle-host execution and timing were not tested here.

The initial complete checkpoint was saved at 2026-09-13T20:35:49.988691+00:00, within five minutes. Immutable source/runtime snapshots were saved after accepted implementation revisions and before validation. The final checkpoint contains 51 source/runtime files plus its runtime manifest and is 677,228 bytes, below 5 MB.

All **486 actual game attempts** are preserved: **486 completed**, **0 failed**. The initial two-game smoke and four repeated entry/reset checks are not independent economic panels. Through the actual `main.agent` entrypoint, both seats were tested twice in one process on already-used seed 766857369 without an intervening explicit reset. 4/4 matched the selected development action streams and terminal cash; automatic step-zero reset and final explicit reset were checked.

Independent frozen-official-interpreter replay re-execution verified **486 complete replays**, **349,434 transitions**; prefix-only verifications: 0; failures: 0. The separate one-game audit-speed repetition is not counted as another unique game. Every completed game ended at transition 719 with both seats DONE and reward equal to terminal farm cash. All experiment source manifests were rechecked unchanged at delivery. An early v3 development metadata label was cosmetically stale; its exact source/config/native receipts remained authoritative and the original label is preserved with a notice. The final metadata is correctly labeled and frozen. Early panel declarations did not stamp a harness hash; this missing historical field is left null rather than reconstructed. Policy source IDs, exact settings, actual actions and replay hashes remain recorded and independently checked. A report-assembly attempt failed on that absent metadata key; its source and traceback are preserved, then the report generator was fixed without touching any policy or game record.

## Run and inspect

Extract the small runtime ZIP into one directory. Its entrypoint is `main.py:agent(observation, configuration)`. For an offline rebuild, run `python3 -B build.py --cxx g++`. `main.create_agent()` supplies an isolated callable with `.close()`; `main.reset()` releases per-seat native contexts. No feedback, evidence archive, reference agent or online service is required to run the policy.

The metadata ZIP includes supplied feedback, complete source snapshots, all settings/declarations, every per-game result, failure ledger, build receipts, English report and replay audits. Extract the independently readable replay ZIPs into the same directory for full evidence. Each evidence ZIP is below 50,000,000 bytes and has its own content manifest. `DELIVERY_CONTENT_INDEX.json` maps every file to its archive and SHA256. The external delivery receipt and SHA256SUMS record exact final archive sizes and hashes after read-back verification.

Frozen source ID: `4b1c2bc7f2da52429ab1d734a048108b50dd84e3a9c05d12d0a832b398ecb365`. Native SHA256: `5e5a5fc95563b48ba702e678aeebb11f12b182a611de3810df3fb6fab75aa02c`. Runtime checkpoint SHA256: `645a750a05192fc2a898e682383449cb17bb7c5b643ee0f944edda748d8510ae`. The original research cutoff is 2026-09-13T21:14:44.830Z; source selection ended before validation, not at a later favorable result.
