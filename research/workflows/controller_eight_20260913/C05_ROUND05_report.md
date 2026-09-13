# C05 ROUND05 — final economic revision report

Generated 2026-09-13T22:05:11.126510+00:00. Research/delivery cutoff: 2026-09-13T22:15:12.629Z. All newly reported game cash is actual terminal cash from the frozen supplied official interpreter, not a planning score or partial-game projection.

## Result: economic target MET

Selected source: `04_capital_dawn_consistent`. The exact final runtime was frozen at **2026-09-13T22:01:31.936042+00:00**, before all **32** fixed validation games. The source was chosen at 2026-09-13T21:57:31.227922+00:00 using only the complete disclosed development panels. No validation-based fallback, seed extension or subsequent economic change occurred.

The frozen strategy averaged **212,330.84375** terminal cash across **16 predeclared seeds, both seats, 32 complete games** against legal PASS. The 95% percentile bootstrap interval, resampling sixteen seed-pair means, is **197,995.62–224,678.47**. Seats sharing a seed are not treated as independent observations. This small sample does not guarantee a future shared panel will pass.

| Measure | Actual result |
|---|---:|
| Mean | 212,330.84375 |
| Seat 0 / seat 1 means | 209,029.25 / 215,632.44 |
| Median | 219,331.00 |
| Minimum / maximum | 135,375 / 258,683 |
| Individual games at least 200,000 | 23 / 32 |
| Complete 719-transition games | 32 / 32 |
| Both statuses DONE and rewards equal cash | 32 / 32 |
| PASS terminal cash | 3,000 in every game |
| Game errors | 0 |
| Mean wall time per validation game | 5.188011 seconds |
| Maximum measured validation policy call | 0.379450 seconds |

## Input identity and historical source boundary

The feedback ZIP SHA256 `9ad1fdb1490e6e0523582598fa2697e55eec1eb8eed5e62b49aff15de709be8c` and every one of its **164** manifest entries were verified before source use. Own and C08 source identities match all three disclosed controller panels. The variation in old C05 means is therefore a panel change, not evidence of an old source regression.

Historical control/reference numbers below are the supplied independently audited rows, not newly executed control panels in this chat. The local unchanged-parent smoke reproduced **every action and full state** for seed 1338192299 in both seats, including terminal cash **144,762 / 143,062**. C08's complete panel was not rerun. Matched starting seeds do not guarantee identical later shop/weed trajectories when strategies alter the board-dependent official RNG consumption.

## Complete development comparisons

Every new economic experiment used all **48 distinct disclosed seeds and both seats: 96 full games**, with no panel-specific stopping or cell omission. Each panel has 32 games. Means below are terminal cash.

| Strategy | Panel 1 | Panel 2 | Panel 3 | Pooled 96 | Role |
|---|---:|---:|---:|---:|---|
| Own supplied control | 201,762.38 | 201,458.00 | 195,535.50 | 199,585.29 | Supplied audited rows |
| C08 supplied reference | 211,787.56 | 200,061.09 | 215,897.47 | 209,248.71 | Supplied audited rows |
| 01_no_opening_prior | 199,235.50 | 196,920.75 | 201,677.81 | 199,278.02 | Isolated prior removal; rejected alone |
| 02_observed_capital | 210,621.12 | 205,598.19 | 199,668.91 | 205,296.07 | Cadence-defective prototype; excluded |
| 04_capital_dawn_consistent | 210,542.62 | 205,598.19 | 199,452.62 | 205,197.81 | Selected corrected adaptation |
| 05_capital_prior_dawn_consistent | 195,600.06 | 205,484.12 | 197,020.50 | 199,368.23 | Corrected adaptation retaining prior; rejected |

Removing the mirror prior alone did not improve the pooled mean. The adaptive rule added 5,919.79 without the prior but changed cash by -217.06 with it; the exploratory interaction contrast was +6,136.85. FACTORIAL_RESULTS.json preserves all paired intervals, so these point estimates should not be read as established causal effects. The corrected no-prior capital adaptation improved every disclosed panel mean relative to supplied own C05, with pooled change **+5,612.52**. Its approximate 95% paired seed-cluster interval is **-3,146.53–14,371.57**, crossing zero. Development selection and reuse of disclosed panels make this exploratory evidence, not a confirmatory estimate. Its third-panel mean remained below 200,000, and C08's pooled supplied mean was higher. No claim of superiority to C08 is made.

The final selection compared only complete corrected candidates under SELECTION_RULE.json. Prototype 02's slightly higher observed mean did not make it eligible after its cadence mismatch was found. Variant 03 was prepared but canceled before compilation or games, then superseded by corrected 05. Its unbuilt source and explicitly stale inherited native are preserved; it is not a runnable finalist.

## Full strategy and independent change

The strategy remains own C05's full economic planner and executor: market-aware crop/livestock valuation; conditional crop and animal dynamic programs; affordable land, workers and investment allocation; two-day public-information candidate search; paid-commitment preservation; feeding, care and collection; route/work feasibility; inventory handling and sales; observed-state intraday repairs. The animal ceiling remains 32, scenario remains 2, competition weight remains 2, and the same executable investment candidate families remain active.

The independent change is a dawn liquid-capital transform. Liquid capital is actual cash plus current sequential sale value of shed products, excluding fertilizer and reserving one wheat per existing animal. Bags, unharvested produce and predicted profit are not credited. Scarcity is `clamp(1 - liquid/6000, 0, 1)`. Base discount moves from .005 toward .08 and base capital power from .4 toward .7 as scarcity rises. The fixed 6,000 scale and endpoint choice were recorded before this experiment's outcomes; there was no threshold sweep. This changes investment ranking without adding a seed-specific portfolio rule or removing service/finance constraints.

These are base proposal/common-continuation settings. The inherited fast, slow, animal-biased and reserve-oriented candidate alternatives can override them; the installed winner retains its own day settings. Effective_* debug fields measure the adapted base, not necessarily the installed candidate weights. The transform is evaluated at actual and predicted dawn, not hourly. Future rollout continuation remains the inherited common-policy approximation rather than recursive future candidate search.

C08 receives attribution for the isolated prior-removal idea and .7/.08 endpoints. Its 26-animal cap, scenario 0 planning, short-crop maturity repairs and final idle-cargo rule were **not** imported. The source is not an unchanged or renamed C08. FINAL_VS_OWN.patch and PROVENANCE.json give exact changes.

## Economic planning versus realized execution

The initial critical own/C08 replays show different portfolios and cash growth. These are associations, not proof that idle cash or an unfilled target could have earned profit. The terminal comparisons test the whole revised planner/executor together. The frozen validation replay audit additionally checked **960 dawn capital calculations**, including **414 scarcity-active dawns**, against current observation data and official sequential prices.

The audit found **4,138 new/different-kind dawn intentions**, of which **4,031** had a confirmed same-day start. The remainder are retained in individual audit files; no profit is imputed to them. Actual new starts included 1,048 wheat, 1,504 strawberry, 832 melon, 377 carrot and 203 tomato crops, plus 363 cows, 272 sheep and 25 geese. These counts distinguish proposed work from realized work, but do not establish global economic optimality.

Validation issued 12,798 FEED and 12,103 CARE actions. Before/after observations confirmed 12,673 feed and 11,987 care effects. The 125 feed and 116 care actions crossing midnight were excluded from this effect check. There were 0 unconfirmed effects among the remaining unambiguous checks. Exclusions are retained rather than silently counted successful. Intent and service diagnostics are not extra scored games.

## Correctness and runtime tests

The final build passed **5 suites** covering the service oracle, failed-build preservation, runtime guards, full-trace entry/privacy/reset parity, and the new capital helper. The service oracle checked **15,936** value entries and **15,936** chosen actions against exhaustive official-refresh transitions under conditional prices. The adaptive helper passed 17 synthetic contracts. Two 719-frame known-development fixtures verify action parity, unknown-field exclusion, step-zero reset, close/reset and seat isolation. They are not new games and never supply runtime actions.

A controlled compiler exit 17 confirmed failed compilation preserves the previous native; this is an expected test failure, not a failed scored game. A clean extracted runtime rebuilt offline and matched the selected native SHA256 `393e207e320e78516f436c5017d91ecd04fe1978321160aaf469bccc3ec5e9fe`. The clean native build took **24.653 seconds**. Its first repeated unit-test phase was interrupted by the command runner, so the complete clean unit suite was restarted and passed in **20.349 seconds**. The interrupted logs and exact receipts are preserved in clean_check_interrupted/ and CLEAN_REBUILD.json.

Production reads only the official own observation and its whitelisted public/own fields. Actual environment seed is stripped from policy configuration. Private opponent data, future randomness and replay-derived actions are not policy inputs. Each game creates and closes an independent native context; the final 32 use the exact module `main.agent` entry. Synthetic privacy-test canaries are not real environment seeds.

## Resources and timing

| Measurement | Actual value |
|---|---|
| CPU quota | `400000 100000`: four cores |
| Affinity | `[0, 1, 2, 3, 4]`: five logical CPUs |
| Cgroup memory limit | 4,294,967,296 bytes |
| Initial measured memory current | 525,250,560 bytes |
| Compiler | g++ 14.2.0 |
| Maximum concurrent CPU workers | 2 |
| Initial own build | 25.699074 seconds; peak child RSS 578,312 KiB |
| Initial singleton full games | 5.810957 and 4.887079 seconds |
| Final selected build | 25.645813 seconds |
| Final build peak child RSS | 578,360 KiB |
| Final unit suites | 20.365724 seconds |
| Final validation peak worker RSS | 97,768 KiB |

Per-game wall/CPU time and every policy-call duration are preserved. Worker RSS is a process-lifetime high-water mark and may reflect an earlier game in that reused worker. Full-game timings include full replay serialization. Runs used at most two evaluation workers; build/tests and one-worker development audit were overlapped only within that two-worker limit.

## All attempts, failures, seeds and delivery

The new round records **418 game attempts**, **418 complete games**, and **0 game failures**. This includes two repeated parent smoke cells, four 96-game development variants and the fixed 32 validation games; do not pool these policies into a performance mean. Every planned cell has its started/result record and complete replay for successful games. The final evidence audit checked **418** records; **418** passed.

Failures and rejected work remain explicit: the initial unsupported streaming-tool launch never executed setup; prototype 02's forecast cadence mismatch was discovered and corrected, not hidden; unbuilt 03 was canceled before any games; the compiler-exit 17 unit is expected; a tool timeout interrupted the repeated clean-preflight unit phase after a successful native build, and its partial logs were preserved before the complete clean unit suite was restarted. No game evidence was lost in this round. This does not recreate unknown lost older histories.

The 16 validation seeds were declared at **2026-09-13T21:36:27.763622+00:00**, before candidate outcomes, and excluded all **1,062** recoverable IDs. The fixed seed list is `[1260184043, 558926474, 1183989484, 1550876656, 274468269, 1785475952, 1885300481, 214051485, 839400023, 1178333095, 1616543271, 877597850, 415880052, 272080820, 216928037, 2025508729]`. Exactly both seats were evaluated. Unknown lost historical IDs remain an explicit possible overlap limitation. SEED_INVENTORY.json includes the complete excluded list, per-experiment seeds/source identities and every actual attempt.

The initial complete checkpoint was attached at 21:37:04Z, within five minutes of dispatch. Each built test variant and the accepted final source has a small checkpoint; unbuilt variant 03 is preserved as source-only evidence. Final runtime is delivered separately below 5,000,000 bytes. Evidence parts are ordinary independently readable ZIPs below 50,000,000 bytes, each with source/runtime, official referee, full included-game replays and a manifest. They are not binary-split archives. The metadata ZIP includes all variants, source snapshots, supplied evidence, research scripts, declarations, failures, tests and result inventory. No inherited large archive is embedded.

Exact final artifact sizes and SHA256 hashes are in C05_ROUND05_DELIVERY_MANIFEST.json and C05_ROUND05_SHA256SUMS.txt. Every ZIP entry is read back and hash-checked after creation. Final runtime entry: `main.py:agent(observation, configuration)`; rebuild with `python3 -B build.py --cxx g++ --unit`.

**Competitive round robin and the 1,536-game competitive acceptance panel were not run. PASS cash qualification does not establish competitive acceptance, which has no cash threshold.**
