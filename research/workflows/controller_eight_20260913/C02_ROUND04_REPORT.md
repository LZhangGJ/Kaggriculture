# C02 ROUND04: complete strategy, economic results and evidence

Report generated 2026-09-13T20:56:56.018480+00:00. Research start 2026-09-13T20:34:38.819Z; absolute cutoff 2026-09-13T21:14:38.819Z. Source was selected and frozen before validation, with more than ten minutes left for validation and delivery.

## Result and limits

The frozen revision averaged **204,101.46875** terminal cash over all **32 predeclared author validation games**, 16 distinct seeds and both seats against the supplied legal PASS agent. All games completed. This clears the 200,000 author-panel gate by **4,101.46875**. It is not independent controller qualification or competitive acceptance.

The matched baseline averaged **203,023.06250** on those same seeds. The paired advantage is only **1,078.40625**: 18 cells improved, 14 worsened; 10 seed-pair means improved and 6 worsened. The paired seed-block standard error is 4,673.16. A descriptive approximate 95% t interval for the paired mean is [-8,882.21, 11,039.02]. This does not establish meaningful superiority. The much larger development gain mostly did not generalize. No alternate policy was selected and no validation seeds were added afterward.

| Panel | Games | Baseline mean | Revised mean | Paired difference |
|---|---:|---:|---:|---:|
| Disclosed development | 32 | 199,105.71875 | 217,031.09375 | +17,925.37500 |
| Predeclared author validation | 32 | 203,023.06250 | 204,101.46875 | +1,078.40625 |

Source: exact game results in `experiments/*/games/*.result.json`, full inventory in `reports/ALL_ATTEMPTS.csv`, paired rows in `reports/paired_baseline_validation__final_validation.json`. Development comparisons reuse the same seeds and are selection-biased; the validation source was fixed before either validation panel.

## Exactly what changed

The complete C02 planner, executor, observation codec, referee and native algorithm source are retained. Only two runtime economic settings change relative to the supplied C02:

```json
{"work_price": 1, "harvest_threshold": 3}
```

Baseline values were work_price=4 and harvest_threshold=1. Lower work shadow price changes economic valuation and maintenance/service decisions; it does not create free workers, reduce actual wages, or remove route/time/resource checks. A harvest threshold of three batches crop and animal collection when permitted by the inherited timing logic. Expiry, capacity, fertilizer-timing and terminal overrides remain. Actual staffing, purchased inputs and completed sales, not hypothetical work savings, determine terminal cash.

The two-day scenario horizon, competition=2, C02 crop calendar correction, retained no-fertilizer alternative, existing rotation-family exclusions, receipt-gated reinvestment and full farm executor remain unchanged. The C++ library is byte-identical to the rebuilt baseline; configuration-only variants use separately hashed JSON loaded at runtime. The final main.py change is a module docstring only. README/attribution files and build receipts were added before freeze.

C03 attribution: work_price=1 and harvest_threshold=3 were present in the supplied qualified C03 reference and motivated separately measured ablations. The final strategy is not a renamed or wholesale copy of C03. C03 delivery code, reduced rival weighting and one-day horizon were not installed. The independently implemented proposal-coverage experiment is retained in `snapshots/labor_proposal`, not silently discarded.

## Full economic strategy

At dawn the planner builds candidate crop-and-livestock portfolios using observed cash, public market conditions, own inventories and workforce, and public rival assets. It values investment, paid inputs, maintenance, maturity, harvest, transport and sale timing, then compares complete candidates with the inherited two-day conditional simulation and continuation value. These are approximate economic forecasts, not proofs of executable profit. Every claimed result in this report is actual terminal official cash.

The crop calendar retains legal two-day first-yield eligibility for late wheat and carrot projects and maturity-aware ongoing tomato/strawberry maintenance. There is no forced crop rotation or seed-specific opening. Previously rejected speculative rotation-forecast proposal families stay disabled; actual replacement planting after harvest or expiry remains available. The no-fertilizer whole-policy alternative remains. Livestock feed and care, fertilizer, labor, land purchases and live reinvestment all remain part of the strategy.

The executor performs procurement, hiring, land purchase, planting, animal placement, watering, fertilization, feeding, care, collection, movement, drop-off and sales. It retains physical and cash/resource admission. Harvest batching changes actual visits and later observations, so the full-run economic path can change even when only one parameter is altered. Configuration changes should not be interpreted as holding realized investment decisions constant.

No host seed, future random script, opponent private state or recorded action enters the policy. The official LocalGame clears configuration.seed, and the policy codec packs current public farms and own private inventories only. Each game uses a new independent Agent instance. Root-entry reset behavior was separately tested. The inherited learned_value.hpp ranker remains embedded but is inactive for scenario=2; no external weights or model download are required.

## Measured diagnosis and realized feasibility

All eight supplied critical replay files were replayed through the frozen official interpreter. Instrumentation recorded actual filled trades, paid wages/land, successful tile work, harvest quantities, asset ages and discarded inventory without changing official function return values. These eight files include duplicate trajectories and are not eight new policy games.

For development seed 1459327281 seat 0, supplied C03 ended at 256,076 versus baseline C02 at 141,331. The 114,745 difference reconciles as 122,668 more sales minus 6,320 extra product purchases minus 1,630 extra seeds minus 1,000 extra animals plus 1,027 less hiring. Milk and strawberry sales account for the largest positive components, offset by other products. C03 automatically discarded 34 units versus C02 four, so overflow was not a sufficient explanation. These are whole-strategy accounting differences, not isolated causal attribution to either parameter.

The selected revision was then instrumented on the two largest C03 paired-loss cells and the second lowest original C02 cell. All actual cash movements reconcile:

| Development seed / seat | Baseline cash | Revised cash | Difference | Fewer effective harvest actions |
|---|---:|---:|---:|---:|
| 1459327281 / 0 | 141,331 | 205,272 | +63,941 | 148 |
| 1459327281 / 1 | 154,821 | 200,931 | +46,110 | 148 |
| 766857369 / 0 | 145,896 | 158,195 | +12,299 | 200 |

On 1459327281 seat 0, the revised 63,941 gain equals 72,557 extra sales minus 7,223 extra products minus 2,090 extra seeds minus 400 extra animals plus 1,097 less hiring. Strawberry harvest rose 111 to 404 units, milk 270 to 318, while wheat fell 533 to 202. The actual crop mix shifted from 94 wheat/16 strawberry plantings to 39 wheat/55 strawberry plantings. These were completed official actions with paid inputs; leftover goods were not assigned hypothetical cash. Full day-by-day and transaction-level traces are in `reports/economic_traces/`.

## Every development ablation

All six interventions used all 32 disclosed development cells, not just the critical losses. Singles preceded the combined change. Every attempted game, setting, source snapshot and full replay is retained. No extra development seeds were added.

| Variant | Change from baseline | Mean cash | Difference | Improved cells | Improved seed pairs |
|---|---|---:|---:|---:|---:|
| baseline | Exact rebuilt supplied C02 | 199,105.71875 | 0 | — | — |
| work1 | work_price=1 only | 211,220.62500 | +12,114.90625 | 21/32 | 10/16 |
| harvest3 | harvest_threshold=3 only | 210,011.65625 | +10,905.93750 | 23/32 | 12/16 |
| scenario1 | scenario=1 only | 188,915.46875 | -10,190.25000 | 15/32 | 8/16 |
| competition05 | competition=0.5 only | 202,471.25000 | +3,365.53125 | 15/32 | 7/16 |
| labor_proposal | One low-labor proposal; base work_price=4 retained | 201,190.40625 | +2,084.68750 | 19/32 | 11/16 |
| work1_harvest3 | Combine separately positive work1 and harvest3 | 217,031.09375 | +17,925.37500 | 21/32 | 12/16 |

The proposal-only variant preserves all ordinary C02 plans and adds one complete labor-price-one plan evaluated by the same scorer; its 627 assertions passed, but it produced much less economic gain than the selected settings. The one-day horizon clearly regressed. Lower competition weight was fragile, with a large worst loss. Their complete negative or weaker results remain in the evidence.

The combination was selected before validation because it had the highest full development mean, improved 12/16 seed-pair means, and beat both isolated positive changes. Descriptively excluding the strongest-gain seed still leaves +13,927.67 mean delta, but this is not a new test or a selected validation subset. Its worst development cell regression was 40,578, so it was not uniformly better.

## Validation protocol and every seed

Sixteen author seeds were declared at 2026-09-13T20:35:46.034489+00:00, before results, excluding all 964 distinct seeds in the supplied registry. The source freeze was 2026-09-13T20:50:28.139790+00:00, before candidate validation began at 2026-09-13T20:50:29.509340+00:00 and before matched-baseline validation at 2026-09-13T20:52:06.346985+00:00. The baseline comparison itself was also predeclared. All 32 candidate cells and all 32 paired-baseline cells completed without failure. No source, settings, seed set, or selected fallback changed afterward.

The registry covers recoverable histories only. Lost prior author histories remain unknown, so global never-before-used status cannot be guaranteed beyond that registry. The declaration and final exclusion audit explicitly preserve this limit.

| Seed | Baseline seat 0 | Revised seat 0 | Baseline seat 1 | Revised seat 1 |
|---|---:|---:|---:|---:|
| 769841749 | 211,690 | 230,384 | 211,690 | 230,384 |
| 1254220146 | 208,156 | 202,099 | 192,533 | 181,293 |
| 313607100 | 261,935 | 239,945 | 261,935 | 235,956 |
| 368771969 | 170,511 | 156,357 | 170,511 | 156,357 |
| 935765779 | 178,834 | 192,727 | 178,203 | 192,727 |
| 1188702871 | 228,903 | 201,577 | 208,365 | 201,309 |
| 1520056219 | 259,775 | 213,840 | 249,365 | 217,164 |
| 1520517220 | 126,259 | 144,792 | 125,587 | 144,792 |
| 1563973764 | 226,616 | 240,470 | 234,731 | 233,668 |
| 65784919 | 208,984 | 240,493 | 208,984 | 239,453 |
| 919264509 | 171,487 | 180,675 | 171,487 | 180,675 |
| 58220489 | 206,511 | 208,994 | 213,378 | 223,842 |
| 562983090 | 204,135 | 186,401 | 209,833 | 199,948 |
| 1404402345 | 227,264 | 228,107 | 198,372 | 202,938 |
| 692836907 | 218,689 | 233,955 | 194,902 | 204,673 |
| 642883180 | 174,436 | 165,443 | 182,677 | 219,809 |

Validation min/max revised cash: 144,792 / 240,493. Worst paired cell delta: -45,935; best: 37,132. All are included in the mean.

## Resources, actual speed and checkpoints

Actual cgroup CPU quota was 400000/100000, or four cores, with affinity 0–4 and a 4,294,967,296-byte memory limit. At most two game/replay workers were used; compiler work was sequenced separately. Measured cgroup peak memory through final checks was 1,616,068,608 bytes. Compiler: g++ (Debian 14.2.0-19) 14.2.0.

Initial baseline build: 26.403 seconds. First full smoke game including logging: 6.593 seconds. Final build: 25.997 seconds. Clean extraction, deletion of native binary, offline rebuild and native load/close: 26.633 seconds, byte-identical to the delivered library.

The 32-game revised validation batch took 96.211 seconds using two workers; mean per-game recorded wall time was 5.999 seconds and maximum action time 0.402 seconds. Full per-action timing arrays and batch/process memory measurements are preserved. These include local observation copies, official transitions and evidence serialization, and are not promises about another deployment host.

The initial complete 654,236-byte source/runtime checkpoint was saved at 20:35:46.111918 UTC, about 67.293 seconds after dispatch. Updated checkpoints were saved for accepted work1, harvest3, the selected combination and the final frozen source. All small checkpoint ZIPs and receipts are preserved in the metadata archive.

## Verification, failures and coverage

There were **289 newly executed policy-game attempts, 289 completed and 0 failed**. This is one repeated smoke cell, 224 development games across seven policies including baseline, 32 frozen candidate validation games and 32 matched baseline validation games. They represent 32 distinct seeds/64 distinct seed-seat cells, not 289 independent seeds.

Every new replay was independently run through the frozen official interpreter: **289 games and 207,791 transitions**, no mismatches. Each game had 719 transitions, both statuses DONE, and each reward exactly matched that player's terminal cash. Every trace/replay/result hash and source/config link was checked; no declared cell or started attempt is missing.

Final-source tests passed 621 crop/calendar/proposal assertions. The rejected proposal-code variant separately passed 627 assertions; these are overlapping suites, not 1,248 distinct checks. Four complete observation/action reproductions through main.agent covered both seats and repeated step-zero resets, 719 calls each, and matched the recorded actions exactly. They are functional checks, not four extra games. All contexts closed.

Three non-game tool/progress shell issues were recorded: unavailable streaming execution and two listings of not-yet-created optional output paths. None started or interrupted a game. One generic analysis-label correction changed validation metadata wording only, not numeric results or policy source; the original helper/report are preserved. See `reports/FAILURES.json`, `receipts/tool_setup_failure.txt` and `receipts/analysis_metadata_correction.txt`.

The input ZIP hash matched 9048d9731e143fb996be86ce252ab80db62c343f1feaebdf34615e910f299b01 at both intake and final checking. All 129 supplied manifest entries stayed unchanged; C02 and C03 referee copies are identical. The rebuilt baseline reproduced all 32 controller terminal cash values exactly. Supplied C03 panel mean 204,185.9375 is controller evidence, not a fresh local 32-game C03 run. Only the critical C03 replays supplied here were locally replay-audited.

Every supplied file is preserved. Original controller replay paths outside the packet remain provenance strings, not locally available files. The packet supplied eight critical replay files, not all external historical replays; freshly generated local baseline replays do not retroactively recover missing original files. Unknown lost historical seed usage remains an explicit evidence boundary.

## Rebuild, execute and inspect the delivery

The standalone `C02_ROUND04_runtime.zip` contains complete Python/C++ source, Linux x86-64 native runtime, build flags, tests, English strategy/attribution, build receipt, source-freeze receipt and a SHA256 member manifest. It is independent of all evidence archives. Extract and run:

```bash
python3 -B build.py --cxx g++
python3 -B tests/run_units.py --cxx g++
```

Entry point: `main.py:agent(observation, configuration)`. Local API: `create_agent()` and explicit `close()`. Existing precompiled native runtime needs a compatible Linux/libstdc++ ABI; rebuilding uses only locally installed Python3 and g++ without downloads.

Runtime ZIP: 662,134 bytes. SHA256: `ed516b13a21d8321f647df8a6ed4f6398756e925fa26355a8fb03ea0d3047e70`. Frozen source ID: `6d5d002ddda9c883f307a94333f0c5ca0099423ebf53f5737f0c601082270e20`. Native binary SHA256: `f804c7775db200c5a0e6fda36f8a56a14a8c4f16adcd4c7ef62c91bd80a7aaa2`.

The separate metadata ZIP contains the unchanged feedback, all source variants and checkpoints, reports, settings, seeds, source index, failure records, tools and build/replay receipts. Independently readable evidence parts each stay below 50,000,000 bytes and contain complete game records and full replays, plus required source and referee. Parts are ordinary ZIPs, not raw binary fragments. Each PART_INDEX identifies its exact games; the entire validation result must be read across the full inventory, never cherry-picked from one part. `C02_ROUND04_DELIVERY_RECEIPT.json` and `C02_ROUND04_SHA256SUMS.txt` give every exact byte size and ZIP hash.

No competitive evaluation was run. Later competitive acceptance has no cash threshold and must use its specified strict-win panel. This delivery is a narrow author economic-panel pass with a small, uncertain matched advantage, ready for independent controller evaluation rather than a claim of competitive strength.
