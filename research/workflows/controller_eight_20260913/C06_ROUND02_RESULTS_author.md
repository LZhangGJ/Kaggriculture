# C06 economic revision round 2: delivery report

Report generated: 2026-09-13T19:32:44.763758+00:00.
Original round: 2026-09-13 19:09:34.418 UTC to **19:54:34.418 UTC**.
All policy research ended when source was frozen at **2026-09-13T19:26:40.258929+00:00**.
Subsequent work was validation, replay reconstruction, build/reset checks and packaging.

## 1. Result and scope

The frozen revision averaged **$204,759.1875** on the **entire predeclared 16-seed, both-seat, 32-game author validation panel**, versus **$188,922.6250** for the unchanged baseline on those same cells. The paired gain was **$15,836.5625**. The revision exceeded the 200,000 author-panel target by $4,759.1875.

This is **author-side economic validation against legal PASS**, not controller qualification, a round robin, competitive acceptance, or a guarantee on fresh seeds. No competitive result is claimed. No extra validation seeds were added and no policy/settings changes were made after validation started. The controller must still draw and run its fresh shared panel.

| Panel | Seeds | Games per strategy | Baseline mean | Revised mean | Paired improvement |
|---|---:|---:|---:|---:|---:|
| Disclosed development, used for selection | 16 | 32 | $195,548.0000 | $203,098.8750 | $7,550.8750 |
| Untouched author validation | 16 | 32 | $188,922.6250 | $204,759.1875 | $15,836.5625 |

Do not pool development with validation, or average results from different experimental policies into an economic claim. All 32 candidate validation games completed exactly 719 transitions, both players were DONE, and each player's reward equaled its terminal farm money. Candidate seat means were $201,284.8125 and $208,233.5625; minimum $126,021, median $213,434.5, maximum $251,645. The candidate improved 23 cells and declined in 9; the two-seat average improved on 11 seeds and declined on 5. These are only 16 independent seed pairs, not 32 independent draws.

## 2. Complete strategy and exact changes

The inherited A06 r6 revision1 remains the complete economic planner and executor. It selects investments using conditional cash-flow and market forecasts, maintains crop/animal commitments, chooses maintenance schedules, estimates actual workforce costs, packs feasible routes with explicit material/dependency checks, and executes observed-state purchases, services, harvests, deliveries and sales. It replans from actual observations rather than following a replay or seed-specific opening tape.

The selected `work0_fertilizer` revision changes only two policy elements:

1. **Labor valuation:** `policy/config.json`, `work_price: 4 -> 0`. This removes the additional action shadow charge used in investment and maintenance valuation. It does **not** remove the model's separate wage estimate, actual Fibonacci hiring costs, current-cash checks, maximum workforce, material constraints or route deadlines. The same setting reaches the economic planner and executor; it is not merely a report-side score patch.
2. **Finite fertilizer allocation:** `policy/executor/policy.hpp`, new `c06_fertilizer_gain` and `c06_allocate_fertilizer`, called at the start of `reserve`. When currently owned shed fertilizer cannot serve the existing job set, the allocator ranks that existing set by estimated marginal crop-yield value using the visible crop, planned harvest age, current price and remaining production windows. Ties are stable. It assigns only actual current shed fertilizer, does not count uncollected or future purchased fertilizer, preserves original job ordering, and removes only the fertilizer action/material need from denied jobs. Water and other existing obligations remain. When supply is sufficient, the job list is unchanged. The marginal-value estimate is a heuristic, not a proved exact terminal-cash increment.

All other settings and original compiler flags are unchanged. Full settings are in `policy/config.json`; exact source additions/differences are in `audit/final_source.diff`. Two non-policy test files were added: `tests/run_units.py` and `tests/fertilizer_allocation.cpp`. The inherited optional `build.py --unit` path now has an actual runnable test implementation. `main.py`, the Python observation codec, all other policy files and the frozen official referee are preserved unchanged.

The extra storage safety margin, alternative staffing mode, resource exchange, reduced animal limit, higher modeled labor productivity and alternative labor penalties were **not** promoted. Their complete source, settings and records remain under `variants/` and `runs/`.

## 3. Resource diagnosis and feasibility evidence

The supplied feedback SHA-256 matched **b550c2ecae5db57522b0a0c2dad587f8a852205f6c46073d0f8486097446ed68**, and all **59** declared input-manifest entries passed. The locally rebuilt baseline reproduced all 32 controller terminal values exactly: mean $195,548. Both supplied low-cash replay/parent pairs also matched on every action and every state frame.

Both low-cash cases ended with empty shed and worker inventories and no positive-yield tiles. Full referee reconstruction found **11 overflowed units** in the $131,126 case (four fertilizer, four eggs, three wool) and **zero** in the $155,189 case. Thus neither terminal leftovers nor daily cash timing establishes a recoverable-cash explanation.

Across the complete 32-game baseline development panel, 490 units were physically lost at capacity-limited deposits. Their values at event-time quotes averaged $2,224.59375 per game, but that is **not feasible sale proceeds, an attainable counterfactual, or terminal cash**. The already feasible price-pressure delivery ablation was exactly neutral on every cell. A larger 12-unit storage safety margin with the selected economic revision reduced its development mean by $735.00. Storage constraints matter, but simply reserving more room was not an improvement.

The selected revision lost 759 units on the same panel, compared with 490 for baseline. Its measured improvement therefore must not be described as recovered overflow. Direct referee instrumentation of every actual market/hire/land transaction reconciled initial cash plus realized flows to terminal cash exactly. The selected revision generated more realized sales while also paying more wages:

| Realized cash component, development means | Baseline | Selected | Change |
|---|---:|---:|---:|
| Actual sales | $235,067.03125 | $244,252.59375 | $+9,185.56250 |
| Product purchases | $-14,614.75000 | $-14,586.96875 | $+27.78125 |
| Animal purchases | $-7,781.25000 | $-7,384.37500 | $+396.87500 |
| Seed purchases | $-7,957.50000 | $-8,348.75000 | $-391.25000 |
| Actual wages | $-5,415.53125 | $-6,833.62500 | $-1,418.09375 |
| Land purchases | $-6,750.00000 | $-7,000.00000 | $-250.00000 |
| **Net terminal-cash change** | | | **$+7,550.87500** |

These are executed transactions, not forecast prices multiplied by leftover stock. They verify that the changed planning was actually executed. They do not identify a uniquely optimal labor price or prove that the same gain will recur on another panel.

## 4. Every completed development variant

Each row below uses the same disclosed 16 seeds in both seats; all 32 games completed with the mandatory terminal checks. Selection used development only. Negative and neutral experiments are included.

| Variant | Complete games | Mean terminal cash | Difference from baseline |
|---|---:|---:|---:|
| `work0_fertilizer` | 32 | $203,098.87500 | $+7,550.87500 |
| `work0_fertilizer_capacity88` | 32 | $202,363.87500 | $+6,815.87500 |
| `work_value0` | 32 | $202,088.78125 | $+6,540.78125 |
| `work_value2` | 32 | $199,762.75000 | $+4,214.75000 |
| `work2_fertilizer` | 32 | $199,123.46875 | $+3,575.46875 |
| `fertilizer_priority` | 32 | $196,695.68750 | $+1,147.68750 |
| `feasible_hire_reduction` | 32 | $196,458.25000 | $+910.25000 |
| `resource_exchange` | 32 | $195,718.12500 | $+170.12500 |
| `delivery_pressure` | 32 | $195,548.00000 | $+0.00000 |
| `work_value1` | 32 | $194,188.40625 | $-1,359.59375 |
| `work0_fertilizer_animals14` | 32 | $192,045.59375 | $-3,502.40625 |
| `work0_fertilizer_labor12` | 32 | $187,637.18750 | $-7,910.81250 |

Hypotheses were recorded before their runs in `audit/*.hypothesis.json`. Each suite includes its declaration, actual source hashes, individual result/timing files, complete replays, diagnostics, attempt log and summary. `audit/experiment_inventory.json`, `audit/all_attempts.csv` and `audit/suite_summary.csv` index every attempt and show the incomplete initial suite separately.

## 5. Seed and freeze discipline

The validation panel was declared at **2026-09-13T19:10:12.206398+00:00**, before any validation result. The source was frozen at **2026-09-13T19:26:40.258929+00:00**. Both candidate and paired-baseline validation attempts began after the freeze. Every expected seed/seat cell appears exactly once in its validation suite. No validation seed appears in the supplied known-used ledger. The previously used smoke seed 291307001 is absent from the new panels.

Development seeds: 1781448506, 1681080428, 967718147, 1856155813, 148094548, 180752382, 370413548, 383365175, 337943030, 702093297, 320211102, 140008426, 219026998, 1501860085, 2027896586, 1658182076.

Validation seeds: 1317683761, 1464394708, 1364875040, 25400103, 1718394431, 559601743, 2028094735, 222305262, 900507858, 1221741181, 1858684874, 316412735, 433890620, 209286234, 1348965638, 1315989216. Both seats 0 and 1 were run for each.

Every completed ordinary game uses an independently created native context and closes it afterward. The four root-entry contract checks additionally reuse `main.py:agent` in one process across both seats and repeated step-zero games, verifying automatic resets. The agent receives only its own observation and seed-redacted configuration; the host asserts `configuration.seed is None` at every transition. Full referee state and seed identifiers appear in audit/replay metadata only, never in policy inputs. No external services, delegation or publication were used.

Frozen aggregate source SHA-256:
`0959bb861d28b41b35d824f19e5c95341a0c0fd35bf46c9ebdb66a2ffb6b556d`

Native library SHA-256:
`f2f4f444a5b780c5ce18cf5f2abe360c331bb3cf3fd5ff198973fb5e1dbb7037`

The aggregate is the SHA-256 of the canonical JSON source-hash map; the exact canonicalization and all 50 source/runtime hashes are in `audit/source_freeze.json`.

## 6. Actual resources, runtime and reconstruction checks

Measured CPU affinity: cores 0, 1, 2, 3 and 4; cgroup `cpu.max = 400000 100000`, a **four-CPU quota**. Memory limit: **4,294,967,296 bytes (4 GiB)**. Tested platform: Linux x86-64, Python 3.13.5, g++ (Debian 14.2.0-19) 14.2.0. Runtime uses Python's standard library and the locally built native library; it does not require network access or Kaggle installation.

Unchanged-baseline compile: **26.952400 seconds**. Selected compile: **25.715370 seconds**. Clean-extraction compile: **26.664805 seconds**. Initial sequential full-game speed, including full replay logging: **3.842047 seconds/game**. Candidate validation mean: **3.941736 seconds/game**. Timings include instrumentation and vary with local concurrency. Maximum recorded policy call across completed runs: **0.614778 seconds**. Maximum worker process RSS: **98,268 KiB**. This local CPU host is not a test of Kaggle's sandbox or schema/time-limit enforcement.

The delivery includes **504 started game attempts**, of which **502 completed** and **2 were interrupted**. Completed games include 12 development variants, baseline and speed runs, the interrupted suite's completed prefix, both validation policies and explicit duplicate-seed API/build checks. They are not one economic panel.

Every one of the **502 completed local replays** was reconstructed through the frozen official interpreter with exact whole-state equality: **360,938 transitions and 361,440 state frames**, zero mismatches. Each recorded complete replay hash matched its result receipt, and all completed games had 719 transitions, both DONE and exact reward/cash agreement. See `audit/full_replay_verification/summary.json`.

The frozen runtime checkpoint was extracted into a separate directory, rebuilt offline and tested. Its native library was byte-identical, all eight fertilizer invariant groups passed, and two complete games reproduced the selected development cash exactly ($242,738 and $210,390 on seed 1781448506, seats 0 and 1). These duplicate checks do not enter the economic mean. The final archive's root and `selected/` runtime hashes are checked against the same freeze. Final ZIP CRC, per-file manifest and root-runtime inspection receipts are delivered separately because including the final ZIP's own hash inside itself is circular.

## 7. Failures and preservation

The first baseline development command was interrupted by the tool execution timeout. Fourteen games completed; seed **383365175**, seats **0 and 1**, were interrupted after **399 and 407 readable recorded transitions**, respectively. Sixteen additional declared cells were never started. The two original partial/truncated replay files remain intact, with errors and explicit failed-result records. Only the separate complete 32-game retry is used as baseline development evidence.

Two unit-test fixtures initially had incorrect expected values. One omitted the intended early melon harvest; the other counted only one of two recurring strawberry production events within a three-day window. Original fixtures and failure outputs are retained under `audit/failures/`. Only test expectations were corrected, not policy code. Candidate and clean-extraction invariant checks then passed.

All 32 candidate validation games and all 32 paired baseline validation games completed; there were **no validation failures**. No failed or partial game was assigned terminal cash or included in a successful-panel mean.

The original failed round1 ZIP and report, repaired round1 archive/reports, original published inputs and exact round2 feedback ZIP are preserved byte-for-byte in `prior_round_archives/`, with individual hashes. The old approximately $209,142 claim remains withdrawn. Lost historical research was not recreated. The immutable round2 input source and controller records are in `input/`; the locally rebuilt but unchanged source baseline is in `baseline/`.

## 8. Validation seed-by-seat results

| Seed | Seat | Frozen candidate | Unchanged baseline | Difference |
|---|---:|---:|---:|---:|
| 1317683761 | 0 | $227,463 | $168,966 | $+58,497 |
| 1317683761 | 1 | $241,282 | $172,318 | $+68,964 |
| 1464394708 | 0 | $211,287 | $222,050 | $-10,763 |
| 1464394708 | 1 | $231,999 | $163,907 | $+68,092 |
| 1364875040 | 0 | $126,021 | $161,839 | $-35,818 |
| 1364875040 | 1 | $176,074 | $193,311 | $-17,237 |
| 25400103 | 0 | $208,903 | $180,311 | $+28,592 |
| 25400103 | 1 | $230,015 | $126,950 | $+103,065 |
| 1718394431 | 0 | $231,862 | $228,567 | $+3,295 |
| 1718394431 | 1 | $231,862 | $228,567 | $+3,295 |
| 559601743 | 0 | $213,944 | $179,088 | $+34,856 |
| 559601743 | 1 | $214,963 | $179,088 | $+35,875 |
| 2028094735 | 0 | $199,378 | $194,142 | $+5,236 |
| 2028094735 | 1 | $251,645 | $178,476 | $+73,169 |
| 222305262 | 0 | $157,733 | $198,244 | $-40,511 |
| 222305262 | 1 | $157,774 | $204,093 | $-46,319 |
| 900507858 | 0 | $217,414 | $225,777 | $-8,363 |
| 900507858 | 1 | $220,817 | $225,788 | $-4,971 |
| 1221741181 | 0 | $213,438 | $193,838 | $+19,600 |
| 1221741181 | 1 | $213,438 | $193,838 | $+19,600 |
| 1858684874 | 0 | $213,431 | $198,480 | $+14,951 |
| 1858684874 | 1 | $208,833 | $197,557 | $+11,276 |
| 316412735 | 0 | $181,815 | $151,737 | $+30,078 |
| 316412735 | 1 | $174,795 | $147,791 | $+27,004 |
| 433890620 | 0 | $228,599 | $216,645 | $+11,954 |
| 433890620 | 1 | $210,685 | $238,057 | $-27,372 |
| 209286234 | 0 | $205,729 | $176,302 | $+29,427 |
| 209286234 | 1 | $185,502 | $248,855 | $-63,353 |
| 1348965638 | 0 | $163,854 | $131,479 | $+32,375 |
| 1348965638 | 1 | $162,367 | $131,511 | $+30,856 |
| 1315989216 | 0 | $219,686 | $198,482 | $+21,204 |
| 1315989216 | 1 | $219,686 | $189,470 | $+30,216 |

## 9. Build and run

From the archive root:

```bash
python3 -B build.py --cxx g++
python3 -B tests/run_units.py --cxx g++
```

The entry point is **`main.py:agent(observation, configuration)`**. The factory **`create_agent()`** provides a separately owned context with **`close()`**; the root module also provides **`reset()`**. Required policy/runtime files are all present at the archive root. `selected/` is a byte-identical frozen runtime copy for audit tooling; it is not a different strategy.

A new local replay-producing check can be run from the archive root without external dependencies:

```bash
python3 -B research/run_suite.py --strategy selected --tag new_local_check --seeds 1781448506 --workers 1
```

The tag must be new, because the runner refuses to overwrite a prior suite. Use a separate extracted copy for rerunning tests; do not overwrite the preserved evidence. The full archive manifest covers every file except the manifest itself. Its final size, SHA-256, CRC and manifest verification counts are in the external delivery receipt.
