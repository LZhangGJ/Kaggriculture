# C04 ROUND04 economic revision report

Final frozen strategy: **C04 baseline with three measured C03 configuration values**. Untouched author validation: **210,409.25000**, versus unchanged baseline **198,968.65625** on the same16 seeds/both seats; paired mean **+11,440.59375**. All32 candidate cells are included. This clears the200000 threshold on this author panel only, not a new controller qualification or a competitive win-rate result.

## Selected change and attribution

The final policy changes only `policy/config.json`: `work_price:4 -> 1`, `harvest_threshold:1 -> 3`, and `competition:2 -> 0.5`. All selected C++ and Python production code and compiler flags remain identical to the inherited C04 baseline. The native library is consequently byte-identical to the local baseline build. The policy is NOT unchanged: main.py loads the changed runtime configuration, and the root entrypoint reproduces the selected strategy actions.

These exact values are attributed to the supplied qualified C03 reference, and were isolated in controlled full-panel ablations. C03-specific delivery, surplus-sale and economic-choice source was not copied. This is a measured economic calibration/component reuse, not a claim to have independently invented the settings or a renamed full C03 submission. An independently coded forecast correction was tested and rejected, and remains in the evidence.

The economic rationale is to reduce the internal penalty on profitable work, batch recurring crop harvests to reduce repeated service, and reduce the forecast competition penalty in the economic objective. work_price is NOT the actual wage: payroll, hiring limits, supplies, worker counts and route feasibility constraints are unchanged. The planner still evaluates rolling crop/livestock portfolios and concrete current-day candidate scenarios; the executor performs travel, care, harvest and delivery with actual workers. The measured outcome is realized terminal cash, not forecast profit. These settings interact: their gains must not be added as independent effects.

## Full development results

Each primary row below has32 completed games on the current disclosed16-seed/both-seat panel. The baseline was rerun locally and matched every supplied cash result. No test row is hidden.

| Variant | Primary mean cash | Paired change | Better / worse / tied |
|---|---:|---:|---:|
| Baseline | 191,437.25000 | 0 | 0 / 0 / 32 |
| work1 | 189,882.43750 | -1,554.81250 | 14 / 18 / 0 |
| harvest3 | 194,895.68750 | +3,458.43750 | 22 / 10 / 0 |
| work1_harvest3 | 197,890.28125 | +6,453.03125 | 23 / 9 / 0 |
| competition05 | 191,806.00000 | +368.75000 | 16 / 16 / 0 |
| economic_C03_settings | 203,628.21875 | +12,190.96875 | 24 / 8 / 0 |
| batch_cashflow | 203,153.59375 | +11,716.34375 | 18 / 14 / 0 |

The C03 full reference supplied by the controller averaged204185.9375 on this primary panel. It was not rerun or substituted for a C04 candidate. Differences against that whole reference do not isolate the contribution of its delivery code.

Before secondary outcomes, the rule was fixed to compare the best primary calibration, the one independent cashflow correction, and unchanged baseline by pooled64-cell development mean. The secondary32 were old disclosed R02 feedback cells, not fresh validation. Their baseline cash is controller evidence;43 production-file identities against the current baseline were checked. Both revised variants were actually run on all32 secondary cells.

| Selection candidate | Secondary mean | Pooled64 mean |
|---|---:|---:|
| baseline | 195,548.00000 | 193,492.62500 |
| economic_C03_settings | 209,733.40625 | 206,680.81250 |
| batch_cashflow | 205,500.31250 | 204,326.95312 |

The combined calibration was selected before validation. No fallback was selected afterward. See `records/secondary_development/DECLARATION.json`, `FINAL_SELECTION.json` and every panel jobs/result file.

## Measured failure evidence and scope limits

The largest supplied matched C03 gap was seed1514191294: C04 cash166344 in each seat, C03 cash229474/228946. Replay-only traces verified every original transition. In seat1 C03 actually sold485 strawberries versus299, with334 versus456 HARVEST actions. It planted more strawberry seed and spent less on hiring in that complete trajectory. The first action already differed in crop/livestock purchases and hiring. These are verified whole-policy associations, not isolated causes or recoverable-profit claims. The ablations, rather than leftovers or daily cash alone, justified the selected setting reuse. The selected three-setting C04 revision on that same cell actually finished at227529:485 strawberries sold and327 HARVEST actions, versus baseline166344,299 sold and456 HARVEST actions. This is a realized combined-change outcome, not attribution to any one setting or assumed recoverability of unsold inventory. The full official-operation trace is records/traces/selected_setting_loss1_cell.json.

The independently coded batch_cashflow helper corrected a specific unit-level inconsistency: a recurring crop with one held unit below harvest threshold3 had an immediate forecast receipt but no executor harvest job. The correction waits for threshold, expiry, overflow or terminal/rotation collection, accumulates held yield to the tile cap, and does not change routes or worker counts. Its original-behavior probe passed16 assertions and corrected probe passed131 assertions. Nevertheless, pooled64 terminal cash fell by2353.859375 relative to its calibration parent. It was therefore excluded. This does not prove that the old optimistic forecast is correct; it shows that a local consistency repair was not an economic improvement in this full system.

Labor-cost reduction alone also regressed on the full primary panel. The selected strategy is not uniformly stronger: primary development has8 losing cells, and its worst primary paired loss is71923. Those full replays remain preserved. The final validation has the losses listed below. No claim is made that every planned job executes, all leftovers are recoverable, or the inherited forecast perfectly models route/storage/sale timing.

## Frozen author validation

Seeds declared: 2026-09-13T20:35:23.329009+00:00. Excluded all964 recoverable known seeds. Selection time: 2026-09-13T20:48:13.346616+00:00. Source freeze: 2026-09-13T20:48:40.522872+00:00. Candidate and matched-baseline jobs began only after freeze. The same declared16 seeds and both seats were used, with no seed addition. The optional matched32 baseline control was also declared before validation.

Candidate mean 210,409.25000; control mean 198,968.65625; paired gain +11,440.59375. Better/worse/tied cells: 20/12/0. Positive mean change across both seats on 10/16 seeds. The two seats share seed conditions and should not be treated as32 independent environments. This panel does not establish broad competitive generalization.

| Worst validation paired losses | Seat | Candidate cash | Baseline cash | Change |
|---|---:|---:|---:|---:|
| Seed 1779781600 | 1 | 192,679 | 245,104 | -52,425 |
| Seed 1779781600 | 0 | 195,043 | 245,890 | -50,847 |
| Seed 2045429545 | 1 | 209,031 | 250,811 | -41,780 |

All32 rows, successes and failures, are in `records/validation_results.csv`. No validation result changed the source. Full production/source manifests and `FINAL_SOURCE_CHECK.json` establish the freeze boundary.

## Runtime, replay and build checks

All 358 new recorded game attempts completed: 257,402 transitions. This includes repeated diagnostic/development games,64 validation/control games, and4 actual root-entrypoint games; there are48 distinct played seeds, not358 independent seeds. Each full game had719 transitions, both seats DONE and reward equal to each seat's terminal cash. The frozen interpreter replay audit passed 358/358 files, checking all recorded transitions and replay hashes. No game failure is omitted.

Four complete games exercised actual main.py:agent with the same imported module serving both seats twice, without explicit inter-game reset. All actions and states matched the selected development strategy. Explicit reset then released all contexts. The selected source rebuilt in an isolated copy to a byte-identical native library. Source manifests bind every attempt/replay to the variant used; originals and the independent rejected source survive.

The feedback archive SHA25683d60789ce6fe912f4b80f14385b933c9370b69e680606a7fe15e521e0dfb0a6 and all146 members listed in its manifest were checked initially and again after tests. The frozen official interpreter hash is bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e; referee/cpu_runtime.py is9d601268672ceb713918ad04db1804f05fe43dbd70f5e03a93ce6109f7a54726. Exact remaining identities are in REFEREE_IDENTITIES.json.

## Resources, errors and delivery

Actual quota:400000/100000 =4 CPU cores, affinity5 CPUs; memory limit4294967296 bytes (4GiB). Maximum experiment/replay worker count was2. GCC14.2.0, Linux x86-64. Initial build27.015600802 seconds; initial full games4.411329 and4.473306 seconds. Detailed per-game wall time, policy time, maximum action time and peak RSS are retained. The local host does not enforce Kaggle sandbox/schema/timeouts, so no platform compatibility certificate is implied.

Two orchestration failures are retained: an unsupported streaming-session launch did not start an experiment; a selection-script TypeError treated inherited cash[seat] as scalar and stopped before selection/build/freeze/validation. The failed script and traceback were saved before repair. These are not omitted game attempts. No policy repair followed validation. See tool_errors.jsonl and records/failed_tooling/. The supplied optional historical build --unit target references missing old test files; the documented normal offline build succeeded. Historical READMEs/test binaries are provenance, not new test claims.

Initial complete checkpoint saved 2026-09-13T20:35:33.404372+00:00 (1612504 bytes), within5 minutes of dispatch. Final frozen source/runtime checkpoint saved 2026-09-13T20:48:41.269937+00:00 (1657471 bytes), SHA256 d7c2be187aa9af265d31f675ae0de261d31bef1458ce389250711b5bbad4c198. The separate source/runtime ZIP contains every required runtime/build file and ROUND04_README.md. Rebuild offline with `python3 -B build.py --cxx g++`.

All current round evidence is delivered in independently readable ZIPs below50000000 bytes, including full source snapshots, every attempted/completed/failed record, full replays, scripts, settings, seed inventory and receipts. Extract the evidence ZIPs into one directory. Each part contains its own member manifest; the separate DELIVERY.json and SHA256SUMS.txt identify final byte sizes and hashes. No large prior archive is embedded. Prior supplied archives are referenced by exact hash; selected surviving prior records are unchanged.

Round02 validation195431.625 remains a failed historical result. Its full source/evidence archive bytes remain unavailable; no missing record, replay or used-seed history was reconstructed. Untouched validation is relative to the964 recoverable known seeds, not an unsupported claim about unknown lost histories. Round03 and old receipts remain labeled as historical evidence. No external publication or competitive evaluation was performed.
