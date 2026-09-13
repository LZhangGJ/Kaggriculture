# C06 ROUND04 delivery report

Report generated: 2026-09-13T20:58:13.035853+00:00. Original hard cutoff: 2026-09-13T21:14:42.773Z.

## 1. Decision and evidence boundary

The fixed author validation panel **PASSED** the $200,000 mean-terminal-cash gate: **$212,871.9375**, all **32 declared games**, 16 seeds, both seats against legal PASS. The unchanged baseline on those same seeds averaged **$209,244.1875**; paired change **+3,627.75000**. The selected source is delivered unchanged regardless of this result. 22 cells improved and 10 declined against baseline.

This is author validation, NOT new controller qualification or competitive acceptance. A PASS opponent does not establish performance against competing agents. Later competitive acceptance has no cash threshold. The next controller shared seeds have not been evaluated here.

The original input's controller panel is development data now: unchanged C06 mean $199,392.00, C03 reference mean $204,185.9375. We rebuilt and reproduced all 32 C06 terminal cash values exactly. Reference results are supplied evidence; the eight provided critical replays were reconstructed, but we did not run a new complete C03 panel.

## 2. Complete strategy and bounded revision

The delivered strategy retains the supplied C06/A06 economic planner, investment candidate search and WAIT/NOW reinvestment controller, finite-fertilizer crop model, recurring crop and livestock service, material and cash commitments, hiring/land decisions, route packing and market sale logic. Its executor receives only public information and its own private observation. Seeds and complete referee state remain in the host and audit records, not policy inputs. Every game asserts seed-redacted configuration.

The three selected changes are **harvest_threshold=3**, **c06_delivery_mode=3**, and **c06_fertilizer_priority=0**. Work price remains **0**, competition remains **2**, and the finite-fertilizer planning model remains enabled. A zero additional action penalty does not remove real wages or workforce feasibility.

Harvest batching defers small recurring collections but preserves cap-pressure, expiry, animal retirement and terminal-day exceptions. C03's attributed delivery component estimates carried/planned goods, dispatches idle workers when storage pressure is predicted, inserts only feasible surplus-product return detours, preserves required feed/fertilizer and checks actual shared shed capacity. It retains unfinished deliveries instead of discarding undelivered inventory. The separate C06 round2 fertilizer-priority ranking was disabled because the complete paired factorial test did not support retaining it. Fertilizer itself, ordinary reservations, feeding and watering were not disabled.

The C03 delivery additions and adapted resource test fixtures are explicitly credited in the runtime and source diff. C03's economic planner, proposal changes, market sale filter and other settings were not copied. This is an integrated C06 revision, not a renamed reference. Runtime README.md explains the complete strategy and API; policy/config.json contains every actual setting.

## 3. Measured investigation and all development panels

The two largest supplied C03 comparison losses are both seats of the SAME seed1459327281, not two independent seeds. C03 produced $76,337 more actual sales and spent $1,419 more, explaining its $74,918 cash advantage. Actual harvested production and investment differed substantially; overnight inventory loss was 62 units for C06 versus34 for C03. These are descriptive matched differences, not proof that all excess leftover stock could feasibly be sold. Daily traces and exact successful-fill accounting are in audit/matched_traces/.

Every row below is the entire disclosed 16-seed, both-seat panel. All rejected candidates and controls remain in the archives. Duplicate control panels are explicitly retained rather than counted as distinct evidence.

| Suite | Mean terminal cash | Change vs baseline | Improved/declined cells |
|---|---:|---:|---:|
| baseline_development | $199,392 | +0.00000 | 0/0 |
| extension_neutral | $199,392 | +0.00000 | 0/0 |
| factor_base | $202,212.5 | +2,820.50000 | 24/8 |
| factor_delivery | $202,554.59375 | +3,162.59375 | 25/7 |
| factor_delivery_no_priority | $202,640 | +3,248.00000 | 25/7 |
| factor_no_priority | $202,574.28125 | +3,182.28125 | 26/6 |
| harvest3 | $202,212.5 | +2,820.50000 | 24/8 |
| resource_work1 | $197,660.25 | -1,731.75000 | 17/15 |
| work2 | $198,879.53125 | -512.46875 | 15/17 |
| work2_harvest3 | $201,988.46875 | +2,596.46875 | 16/16 |
| work4 | $191,335.5 | -8,056.50000 | 12/20 |

Harvest batching alone supplied most of the measured gain: +$2,820.50, 24/32 cells and 13/16 seed averages improved. The final selected combination reached $202,640.00, +$3,248.00, with25 improved and7 declined cells and13 improved seed averages. Its worst leave-one-seed-out mean delta was +$2,059.53333, a descriptive check, not an uncertainty guarantee.

The original critical seed1459327281 was NOT fixed: the selected candidate finished at $164,898 in each seat versus baseline $181,158, a decline of $16,260 per seat. Its full failure trajectories remain available even though the panel mean improved.

The fertilizer/delivery factorial used work price0 and harvest threshold3 throughout. Disabling the additional fertilizer priority changed mean cash by +$361.78125 without delivery and +$85.40625 with delivery. Delivery added +$342.09375 with priority active and only +$65.71875 with priority disabled. That last effect is small; robust unseen-seed benefit from delivery alone is not established. Positive work-price variants did not beat the selected work-price-zero candidate and were rejected. No post-validation alternative was selected.

The newly compiled extension with its delivery disabled and original priority restored reproduced all baseline actions and full states in32 games:23,008 transitions, zero differences. This checks that dormant feature wiring did not otherwise change behavior. Two-by-two source variants, hypotheses, neutral comparisons and complete replays are preserved.

## 4. Realized production, work and cash

An additional read-only reconstruction of the64 already recorded selected/baseline development games instrumented actual successful market fills, material purchases, hiring, land, harvest actions and dawn losses. It ran no policy and did not tune the source. All64 account exactly as initial cash3000 plus actual sales minus actual costs.

Selected-minus-baseline mean realized sales: **+1,351.71875**; mean actual costs: **-1,896.28125**; difference equals terminal cash change **+3,248.00000** exactly. Mean issued harvest actions changed by **-135.12500** and mean dawn overflow units by **+8.71875**. Full item-by-item harvested/sold quantities, work counters and daily investment traces are in audit/selected_accounting/. Units of different commodities are not economically interchangeable; no hypothetical value was assigned to leftovers. This is accounting of the multi-component intervention, not an isolated causal estimate for each action.

## 5. Seed and source-freeze discipline

Validation declaration: **2026-09-13T20:35:24.914695+00:00**, excluding all **964** supplied known IDs. Source freeze: **2026-09-13T20:53:00.539366+00:00**. First candidate validation start: **2026-09-13T20:53:01.127483+00:00**. Both32-game validation suites began after freeze. All declared cells are present exactly once per suite. No validation seed was added and no source/settings change or fallback was made after validation.

Validation IDs: 1092952489, 2093897826, 682827250, 145258965, 961293408, 1388652754, 2113758383, 1439486629, 1709014544, 493632357, 1108898388, 919647042, 106223666, 1971695240, 1137815032, 1502046217.

Final aggregate source SHA256: `078f87c0f769e9619382f852e2203e439c244eba8cd3c25bee02c3babc199f44`. Native SHA256: `6e034fa53900a2f2c74c9e9cda29d1a79906509cba116c0320bd4ba38417f23b`. The exact map and canonicalization are in audit/source_freeze.json and the runtime SOURCE_FREEZE.json. Documentation and offline tests were added before freezing; all policy source, executable library, configuration and entry files match the selected development variant exactly. Build outputs and circular archive metadata are explicitly excluded from the aggregate map and independently hashed by archive manifests.

There were **32 distinct actual round04 policy-game seeds**. The advisory known/executed exclusion union contains **980 IDs**, not a claim that all historical used seeds are known. The input ledger explicitly warns that lost author histories remain unknown. We cannot identify or count those missing historical seeds, so globally untouched status beyond recoverable records is unprovable.

## 6. Full author validation table

| Seed | Seat | Fixed candidate cash | Paired unchanged baseline | Difference |
|---|---:|---:|---:|---:|
| 1092952489 | 0 | $204,950 | $201,886 | +3,064 |
| 1092952489 | 1 | $192,205 | $187,118 | +5,087 |
| 2093897826 | 0 | $168,985 | $175,053 | -6,068 |
| 2093897826 | 1 | $169,292 | $175,165 | -5,873 |
| 682827250 | 0 | $246,845 | $240,050 | +6,795 |
| 682827250 | 1 | $238,777 | $238,775 | +2 |
| 145258965 | 0 | $251,177 | $248,217 | +2,960 |
| 145258965 | 1 | $251,177 | $248,217 | +2,960 |
| 961293408 | 0 | $193,804 | $199,785 | -5,981 |
| 961293408 | 1 | $193,878 | $196,706 | -2,828 |
| 1388652754 | 0 | $247,249 | $218,575 | +28,674 |
| 1388652754 | 1 | $244,584 | $246,844 | -2,260 |
| 2113758383 | 0 | $168,230 | $179,502 | -11,272 |
| 2113758383 | 1 | $175,412 | $170,674 | +4,738 |
| 1439486629 | 0 | $258,126 | $262,171 | -4,045 |
| 1439486629 | 1 | $239,693 | $235,225 | +4,468 |
| 1709014544 | 0 | $172,352 | $157,553 | +14,799 |
| 1709014544 | 1 | $176,524 | $158,946 | +17,578 |
| 493632357 | 0 | $160,896 | $155,377 | +5,519 |
| 493632357 | 1 | $161,476 | $158,873 | +2,603 |
| 1108898388 | 0 | $233,313 | $230,983 | +2,330 |
| 1108898388 | 1 | $233,313 | $230,983 | +2,330 |
| 919647042 | 0 | $158,113 | $167,299 | -9,186 |
| 919647042 | 1 | $208,903 | $198,603 | +10,300 |
| 106223666 | 0 | $224,093 | $219,802 | +4,291 |
| 106223666 | 1 | $206,326 | $210,441 | -4,115 |
| 1971695240 | 0 | $228,864 | $207,908 | +20,956 |
| 1971695240 | 1 | $242,292 | $251,159 | -8,867 |
| 1137815032 | 0 | $237,608 | $223,640 | +13,968 |
| 1137815032 | 1 | $236,568 | $225,551 | +11,017 |
| 1502046217 | 0 | $243,724 | $235,183 | +8,541 |
| 1502046217 | 1 | $243,153 | $239,550 | +3,603 |

All32 candidate and all32 paired baseline games completed719 transitions, both seats DONE and exact reward/cash agreement. The predeclared failure policy would retain every failed cell, label its zero scoring contribution as nonterminal, and fail the gate; no such failure occurred here. Failed or partial cash was never represented as a terminal result.

## 7. Resources, reproducibility and full replay checks

CPU affinity [0, 1, 2, 3, 4]; cgroup cpu.max `400000 100000` (four-CPU quota); memory cap 4,294,967,296 bytes. We used at most two game or replay workers, with sequential suites. Compiler: g++14.2.0; local Python 3.13.5; Linux x86-64. Baseline compilation26.165012 seconds; measured initial complete game3.817254 seconds. Candidate validation averaged **3.744422 seconds/game**, including complete replay logging. Maximum policy call in all local receipts was **0.380626 seconds**; maximum worker RSS **111,024 KiB**. These local timings are not a guarantee of Kaggle sandbox limits.

Initial complete checkpoint was saved20:35:24 UTC, less than one minute after dispatch. Accepted harvest-batching and resource-combination checkpoints were also retained. The final runtime checkpoint was CRC/hash checked before validation; its native library matched the selected development library byte for byte.

Configuration-only development variants intentionally reuse the binary built from identical C++ source. A copied BUILD.json describes its original compilation configuration; each suite's actual source map and config.json identify the runtime settings. The selected source was rebuilt with its final settings before freezing.

The final and clean-extraction builds passed the original eight fertilizer invariant groups and23 resource assertions. The latter cover material preservation, capacity, deadline/terminal detour rejection, partial deposits, watering forecast, batching exceptions and setting wiring. Four same-process main.agent games exercised both seats and repeated step-zero automatic reset; repeated per-seat actions and states matched exactly. Two additional clean-extraction games matched selected-development cash. These six duplicate-seed checks and the one speed game are excluded from economic panels.

| Build/check stage | Wall seconds | Return code |
|---|---:|---:|
| selected_build | 27.602587 | 0 |
| selected_fertilizer_units | 4.152895 | 0 |
| selected_resource_units | 7.950463 | 0 |
| author_validation | 60.897188 | 0 |
| paired_baseline_validation | 66.198435 | 0 |
| entry_contract | 15.899429 | 0 |
| clean_build | 26.586640 | 0 |
| clean_fertilizer_units | 3.935476 | 0 |
| clean_resource_units | 7.818822 | 0 |
| clean_entry | 8.307816 | 0 |
| full_replay_verification | 101.254909 | 0 |

There are **423 started local policy-game attempts**, **423 completed**, and **0 failed**. All successful attempts have individual start/result/timing receipts and full gzipped JSONL replays. These consist of11 disclosed32-game suites, one speed game, two32-game validation suites and six duplicate API/build checks. They are NOT one independent economic sample.

All **423** local completed replays were reconstructed through the frozen official interpreter, matching **304,137 transitions and 304,560 state frames**, with zero mismatches. This saved-action reconstruction checks official dynamics and record integrity; independent agent-action reproduction beyond the entry/reset/clean checks is left to the controller. The eight supplied replay files are also preserved and were independently matched in the initial trace analysis; they are not counted as new local policy games.

## 8. Failures and historical evidence gap

No local policy-game, validation, compile or unit-test failures occurred in this round. One execution-tool session launch failed before any harness/game was created; the error is retained in audit/tool_failures.jsonl. A pre-freeze resource-check receipt mislabeled its post-compile timestamp as started_utc; audit/receipt_field_note.json discloses this exactly. Measured elapsed durations are unaffected. Original harness versions are preserved with an explicit note where an earlier text was recovered by reversing known edits, not falsely represented as an untouched contemporaneous file.

The missing round2 full archive remains an explicit evidence gap: recorded filename C06_round02_strategy_evidence.zip,573,143,728 bytes, recorded SHA256 e828c731600d346f3a799eac264a925bcbc81137619e787f187c2d8c28a27fd3. Its claimed502 local games, original per-game evidence and full used-seed history were not recovered or reverified. No lost records or unknown seeds were invented. This round's new evidence is separate. Small surviving historical gap/report receipts are copied only as historical claims; large inherited archives are not embedded.

## 9. Delivery and offline use

The separate C06_ROUND04_runtime.zip is **701,392 bytes**, SHA256 **994fe11f67c91cf17aca2a00922a5dd9585d0cc38b9983f0eb8d35aa6ce7ed2a**. It contains complete runnable source, native library, reference host, offline build script, tests, settings, attribution and frozen source map. The runtime is immutable after validation.

Evidence is divided into independently readable ZIPs, each strictly below50,000,000 bytes. The records archive includes all current input records/source identities, baseline, every development source snapshot, final source, tools, declarations, per-game results, failures, build logs, settings and this report. Replay shards preserve original replay bytes and use the same relative paths, with per-file manifests and a master replay index. Extract all evidence archives into one directory to run the supplied inspection tools; no binary concatenation is needed. The delivery receipt lists exact sizes, hashes, CRC/manifest results and completeness counts. It is external so it can record archive hashes without a circular self-hash.

Build from the extracted runtime root:

```sh
python3 -B build.py --cxx g++
python3 -B tests/run_units.py --cxx g++
python3 -B tests/run_resource_checks.py --cxx g++
```

Entry point: `main.py:agent(observation, configuration)`. No network, delegation, publication or outside services were used. No further tuning or research is authorized by delivery.
