# C02 ROUND05 — full research and delivery report

Written 2026-09-13T22:08:38.633049+00:00. All performance figures are actual terminal cash in the unmodified supplied official interpreter, not portfolio forecasts or priced leftover inventory.

## Decision and result

**No economic behavior change was accepted.** The best checkpoint under the documented, development-only selection rule is the supplied C02 behavioral baseline. All seven tested revisions are retained. Several beat the current 32-game panel, but the larger effects reversed on older disclosed panels. This is a failed attempt to find a broadly better revision, not an unchanged C08 presented as new work.

The behaviorally unchanged checkpoint averaged **214,437.875** across the **32 fixed author-validation games, 16 distinct seeds, both seats**. That sample is above the 200,000 author gate. It remains the policy whose supplied current controller panel failed at 198,696.84375. A favorable author sample does not undo that failure or constitute a new independently qualified revision.

The matched baseline run averaged 214,437.875, a paired difference of 0. These are repeated evaluations of the same behavioral policy, not independent evidence of an improvement.
Source froze at **2026-09-13T22:02:05.055488+00:00**, before either validation panel. No seed was added, no outcome omitted, and no policy or fallback selection followed validation. The declaration was recorded at 2026-09-13T21:35:56.359912+00:00, excluding all 1,062 recoverable known IDs. Unknown lost historical histories remain a limitation.

Averaging the two seats for each seed gives a descriptive standard error of 6,393.85426 cash and an approximate 95% normal interval of 201,905.92064 to 226,969.82936. Sixteen seeds are a small panel; this is not a guarantee for the next controller draw. Seats within a seed are not treated as independent samples.

## Complete disclosed development results

All panels contain the same 16 seeds in both seats for each compared variant. The two old panels are disclosed development, not untouched validation. Current C02 was rerun on their seeds; historical scores from differently frozen C02 versions were not substituted. The old panels also contain prior development exposure, so these averages are selection-biased.

| Variant | Current panel,32 | Older panel1,32 | Older panel2,32 | All96 mean |
|---|---:|---:|---:|---:|
| baseline | 198,696.84375 | 213,143.09375 | 217,031.09375 | 209,623.67708 |
| receipt_lag | 207,726.9375 | 204,070.125 | 204,820.5625 | 205,539.20833 |
| direct_core | 206,044.34375 | 203,764.375 | 197,073.34375 | 202,294.02083 |
| direct_receipt | 207,651.65625 | 196,740.875 | 200,721.9375 | 201,704.82292 |

| Other full current-panel ablation | Games | Terminal mean | Difference from baseline |
|---|---:|---:|---:|
| maturity_current | 32 | 199,076.75 | 379.90625 |
| prefix_current | 32 | 196,848.6875 | -1,848.15625 |
| combined_current | 32 | 196,620.3125 | -2,076.53125 |
| no_unfertilized_current | 32 | 191,750.03125 | -6,946.8125 |

### Tested mechanisms and attribution

**Short-cycle maturity.** C08’s intraday shared-value admission gate was transplanted alone into C02. It allows a positive shared short-crop lifecycle to override the obsolete fixed-harvest-age output veto, while retaining money, seed, route, workforce and actual first-yield checks. The full 32-game mean rose by only 379.90625 to 199,076.75 and remained below the gate. The complete source and results survive; this repair was not accepted.

**Crop proposal coverage.** The independent wheat-first and carrot-first variants append bounded first-project constraints to the existing complete economic proposals; ordinary alternatives and the no-fertilizer family remain. Seed/input costs and execution feasibility are still checked, and the remainder of each portfolio uses the normal planner. Neither this coverage expansion nor its combination with the C08 maturity gate improved the current panel. More forecast candidates were not automatically better executed farms.

**No-fertilizer removal control.** Removing C02’s added no-fertilizer family, as a single isolation of a C08 coverage difference, reduced current-panel mean cash by 6,946.8125. The two duplicate neutral slots also disappear; the original nine proposal settings remain. Its absence in C08 does not justify removing it from this different planner.

**Receipt-date model.** Enabling the already implemented C02 calendar bit4 shifts conditional crop-output receipts one day later, capped at day29, without shifting physical work or granting new funds. This changes projected prices, capital timing, maintenance choices and crop portfolios; it is not an order to delay real sales. The model is not necessarily conservative when prices rise. It gained 9,030.09375 on the current panel but lost 9,072.96875 on older panel1 and 12,210.53125 on older panel2.

**Outer-selector control and interaction.** The C08 scenario0 setting was separately tested using the otherwise unchanged C02 daily/intraday planner, not C08’s capital, discount, animal ceiling, prior or terminal-delivery bundle. The receipt-date/direct-planner interaction was then declared on all three existing panels before its outcomes and completed regardless of the first result. The single-switch and interaction results did not beat the baseline’s equal-weight96-game mean.

The initial cross-panel selection rule was recorded before inspecting old candidate outcomes, but some old candidate games had already completed. A clarification preserves that distinction instead of silently rewriting the original stronger wording. One documented development-only amendment added the interaction. Neither the validation seed set nor the no-fallback rule changed.

## Actual execution and cash receipts

Every completed game was independently replayed with the frozen official interpreter. Read-only wrappers record actual committed trades, hired workers, purchased land, effective service/harvest actions and produced units. For each game the exact identity is: initial 3,000 + actual sale receipts + signed input costs + signed wages + signed land purchases = terminal cash. This is not a valuation of goods left in bags or storage.

| Current-panel observed mean per game | Baseline | Receipt-date shift | Direct planner |
|---|---:|---:|---:|
| sales_cash | 238,024.28125 | 247,344.46875 | 243,766.15625 |
| input_cash | -31,009.28125 | -31,364.21875 | -28,556.84375 |
| hire_cash | -4,318.15625 | -4,253.3125 | -5,164.96875 |
| land_cash | -7,000 | -7,000 | -7,000 |
| harvested_WHEAT | 183.5625 | 154.8125 | 193.34375 |
| harvested_CARROT | 35.9375 | 40 | 35.34375 |
| harvested_MELON | 160.6875 | 156.9375 | 155.25 |
| harvested_STRAWBERRY | 376.84375 | 378.78125 | 375.71875 |
| harvested_TOMATO | 70.125 | 77.8125 | 88.46875 |
| harvested_MILK | 231.375 | 248.96875 | 254.28125 |
| effective_HARVEST | 336.34375 | 336.40625 | 333.875 |
| effective_WATER | 926.1875 | 918.15625 | 943.65625 |
| effective_FERTILIZE | 135.03125 | 130.25 | 143.15625 |

These matched intervention results establish the realized effect of the entire tested switch in each seed, not that an isolated leftover pile or a daily cash difference was a feasible missed sale. The large panel reversals reject a broad causal claim that delaying forecast receipts fixes the farm. The input loss replays show differing actual crop/livestock production and expenses, not merely inventory liquidation opportunities. No hypothetical extra harvest or terminal sale is counted.

Identical seeds do not guarantee identical later exogenous shop draws: board-dependent weed draws share the official random stream. The policy never observes that stream. Conditional two-day forecast residuals also include later replanning and are not a fixed-action forecast-calibration test.

## Full retained strategy

The complete A06 r6 revision1-derived C02 farm planner remains. It evaluates crop maturity, maintenance and renewal calendars alongside livestock service, dated inputs, work and cash, expected market demand/supply and public rival production. It plans hiring, land, crops and livestock; preserves funded commitments; supplies the chosen daily portfolio to a live route executor; handles feed/fertilizer recovery, storage, whole-cargo delivery and actual selling; and reinvests same-day receipts while accounting for unfinished work. The default configuration retains the two-day conditional selector, the no-fertilizer alternative, work-price1 and harvest-threshold3.

The legal short-cycle daily planting gate remains, including real terminal-day maturity and water-before-harvest behavior. The experimental C08 intraday repair and wheat/carrot-first proposal additions are not in the delivered baseline. No accepted component changes realized production in this round because no behavioral revision was accepted. Version documentation and extra regression tests are packaging changes only.

All settings are in `final/policy/config.json` and the runtime ZIP. `ROUND05_CHANGE.json` explicitly records an empty economic change set when baseline is selected. Ordinary reference components retain their original attribution; this is not a copied/renamed C08 agent.

## Fixed validation: every seed and both seats

| Seed | Frozen seat0 cash | Frozen seat1 cash | Baseline seat0 | Baseline seat1 |
|---|---:|---:|---:|---:|
| 1743420607 | 199,856 | 202,633 | 199,856 | 202,633 |
| 1577371085 | 189,130 | 216,711 | 189,130 | 216,711 |
| 1177333824 | 194,175 | 199,645 | 194,175 | 199,645 |
| 586348586 | 214,685 | 214,196 | 214,685 | 214,196 |
| 1766118547 | 247,280 | 241,747 | 247,280 | 241,747 |
| 516280478 | 164,349 | 178,237 | 164,349 | 178,237 |
| 78172669 | 245,592 | 243,293 | 245,592 | 243,293 |
| 1773198617 | 218,387 | 244,303 | 218,387 | 244,303 |
| 1240071523 | 257,459 | 257,459 | 257,459 | 257,459 |
| 1225978264 | 246,615 | 231,638 | 246,615 | 231,638 |
| 1640516046 | 174,160 | 172,337 | 174,160 | 172,337 |
| 1422917387 | 214,603 | 214,603 | 214,603 | 214,603 |
| 1759334503 | 216,927 | 216,927 | 216,927 | 216,927 |
| 986943804 | 169,528 | 195,517 | 169,528 | 195,517 |
| 698047165 | 203,842 | 238,026 | 203,842 | 238,026 |
| 840280454 | 219,076 | 219,076 | 219,076 | 219,076 |

The two initial smoke games and four lifecycle games repeat already disclosed seeds. They are QA, not fresh performance evidence. Repeated baseline validation is a paired reproducibility control, not an expanded32-game qualification panel.

## Integrity, tests and runtime receipts

The feedback ZIP SHA256 is `a8890957d86999bea9752e9d38b2580c57573ed92350372ea294b315e607e2fc`. All 151 manifest entries verified before use and remained unchanged. Controller source identities matched all 56 supplied C02 files and all 75 C08 files; the four own/reference referee files were byte-identical.

There are **582 new game attempts**, **582 completed**, and **0 gameplay failures**. Independent replay checks cover **418,458 official transitions**. Every completed game has 719 transitions, both statuses DONE and both rewards equal terminal cash. PASS ends with 3,000. Eight additional supplied critical replays were audited separately and are not new game attempts.

The initial source was rebuilt with g++14.2.0; the supplied native had been compiled by g++11.4.0. Native bytes differ across those compilers, so byte identity to the controller native is not claimed. All 32 current baseline terminal outcomes reproduced exactly. The corrected critical parity receipt records whether the four supplied own full action/state trajectories also match. One supplementary comparison utility initially failed on four missing-seed header lookups; its failed receipt and source are preserved. The utility was repaired to use the verified EVIDENCE.json seed/seat metadata. This was not a gameplay or official-replay failure.

The source freeze digest is `e4dd2d34f625f2537cd888351ce599543666dd91b03b9720abd49eae38fa7be4`. The clean extracted rebuild was byte-identical to the selected local native. Actual `main.agent` completed four repeated full games with step-zero action reproduction in both seats; root `close()` released all contexts. The freeze, decoded plans and every replay hash were rechecked after validation.

The original crop/calendar/proposal suite passed 621 assertions and the new ROUND05 receipt-calendar/resource suite passed 1,726, for 2,347 total passing assertions; both are executable offline. Exact assertion totals and command outputs are in `receipts/final_tests.log`, with compilation/test outputs in `final/build/`. Synthetic tests are not counted as games. The tests cover planting maturity, water/harvest order, no seed or cash, route-hour feasibility, proposal preservation, conservation of crop output under receipt shifts and terminal-day clipping.

## Resources, budget and build

Measured cgroup CPU quota: four cores (`400000 100000`); affinity: five logical CPUs 0–4. Memory limit: 4,294,967,296 bytes. At most two concurrent CPU workers were used. Compiler: g++14.2.0 on Linux x86-64; no GPU or network service was used.

The first successful baseline build took 27.67 seconds with 578,216KiB peak compiler RSS. Two isolated fully traced baseline games took 5.91182 and5.98789 seconds. Direct-planner games took about 1.6 seconds with the same full-recording harness. Per-game wall/CPU times, maximum action latency and process-lifetime RSS are retained in every RESULT; panel timings include serialization and process overhead. Final and clean build `/usr/bin/time` receipts are included.

The initial verified checkpoint was delivered at 21:37:14 UTC, within five minutes of dispatch. Selection/source freeze occurred at2026-09-13T22:02:05.055488+00:00. The accepted runtime is copied byte-for-byte from checkpoint01; no final-data-driven policy edits occurred. Research/build/test work was budgeted against the 22:14:13.237 UTC hard cutoff, with the final ten minutes reserved for validation/checks/packaging.

```bash
python3 -B build.py --cxx g++
python3 -B tests/run_round05.py --cxx g++
```

`main.py:agent(observation, configuration)` is the deployment entry. `create_agent()` provides independent local contexts; `close()` frees them. The Linux x86-64 native uses standard system libraries; rebuild offline on the target host. The local official host is not Kaggle’s sandbox/timeout/schema validator, and that broader deployment validation is not claimed.

## Archive map and limitations

The source/runtime ZIP is self-contained and below 5,000,000 bytes. Metadata includes the unchanged supplied packet contents, all source variants, checkpoints, settings, seed declarations, failure/process notes, run plans, aggregate CSV/JSON results, audits and build receipts. The independently readable numbered evidence ZIPs each stay below 50,000,000 bytes and contain every new game folder exactly once, including full replays and audit receipts. Their own manifests and the top-level SHA256SUMS verify exact bytes. No inherited large archive was embedded.

No external publication, delegated research, opponent-private-state input, hidden seed input, replay action table, adaptive validation extension or postvalidation fallback was used. Previous missing history was not reconstructed or invented. Competitive acceptance is later: no round robin or 1,536-game competitive panel was run, and there is no competitive cash threshold.
