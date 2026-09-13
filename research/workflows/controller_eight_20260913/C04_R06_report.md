# C04 ROUND06 final economic report

## Decision and source delivered

The frozen finance3 revision averaged **204538.71875 terminal cash** across the 32 declared validation games. This clears the 200000 author-panel cash gate, but it **underperformed the matched baseline by 8892.31250** (baseline 213431.03125). The development improvement did not generalize to this panel. There is no controller qualification or competitive-acceptance claim. No baseline fallback or source change was made after validation.

The delivered complete source/runtime is `C04_R06_final_source_runtime.zip`, 1672490 bytes, SHA256 `7fef72d6d82da8f8b535c09b7a201107a2cb2b84106c4c2debaf93ff4835b450`. It contains 79 files, including the native library, every required build source, Python entrypoint, configuration, portable offline API/horizon test source and provenance. The source snapshot froze at **2026-09-13T22:52:20.828521+00:00**, before all new validation outcomes. Its native SHA256 is `840c2723ba54d45b9c7d522867fd0dc2eb5d503fb40cf2becf4d12828cde89b2`. The current runtime archive was sealed before validation and is unchanged afterward.

## Inputs, identities and provenance

The supplied feedback archive is exactly 14545598 bytes with SHA256 `5850e5a2479f88eb2ec0d3f70071727a66ea31696abca6dc7e286c1bdda926e4`. All251 manifest entries and the archive CRCs were checked before use, and all251 extracted member hashes were checked again after evaluation. The manifest and exact member receipt are preserved. The unchanged official interpreter, specification and CPU referee are included as supplied; their hashes are in `receipts/SOURCE_IDENTITIES.json`.

The starting point was explicitly selected as supplied `strategy/`, the Round04 C04 baseline, before new outcomes. Supplied `failed_round05/` remains intact. That failed Round05 source changed capital_power from0.4 to0.7 and discount from0.005 to0.08. The controller's stated audit of all518 Round05 games is prior supplied evidence, not a claim that all518 were rerun here. This round locally replayed the eight supplied critical baseline/failed records before coding.

Only production header `policy/search.hpp` changes in the selected revision. The complete C04/A06 crop and animal planner, finite inventory and capital checks, daily hiring and route executor, service commitments, sale planner and intraday reinvestment remain present. All economic configuration values are unchanged from Round04. The original source provenance and prior Round04 C03 calibration are retained and attributed. No new C08 implementation or configuration component was copied. C08 and failed Round05 sources remain in the supplied-material portion of the evidence.

Earlier disclosed controller panels do not all share the same C04 configuration. The first two older panels' source configurations differ from the current baseline; the third matches it. Therefore both current baseline and revision were actually run on all four development panels. `receipts/PRIOR_PANEL_IDENTITIES.json` records the distinctions. No earlier source's cash values were substituted for a matched current-baseline run.

## Four paired critical cases diagnosed before coding

The two largest Round05 regressions are seed1743809225 seat1 and seed1580896166 seat0. The two lowest failed returns are seed1169107643 in both seats. These are four cases across three distinct seeds.

| Case | Baseline terminal cash | Failed Round05 cash | Difference |
|---|---:|---:|---:|
| 1743809225, seat1 | 215724 | 162627 | -53097 |
| 1580896166, seat0 | 243771 | 193656 | -50115 |
| 1169107643, seat0 | 190168 | 145194 | -44974 |
| 1169107643, seat1 | 190168 | 145194 | -44974 |

All eight full replays were independently stepped through the frozen interpreter:5752 verified transitions. Instrumentation recorded every actual money change and applied nonmovement field operation without changing outcomes. Initial3000 plus realized sales minus actual purchases, hiring and land cost reconciles to terminal cash in every case. This is receipt accounting, not valuation of leftovers or predicted income.

The first action divergence is day0 procurement. The baseline buys three cows, two sheep, six melon seeds, seven wheat seeds and three hires. The failed source buys two cows, one sheep,18 melon seeds, four wheat seeds and four hires. Its product purchase is three wheat rather than five. The failed source put more money into delayed melon receipts and less into the early productive portfolio.

The executor did perform the purchased planting: successful per-item seed purchases equal observed executed PLANT counts in every critical replay. Daily seed backlogs are empty, no new seeds were purchased on the final tick of a day, and no failed market fills were recorded. Applied nonmovement actions changed physical state. This does not prove that every intended operation was attempted, but it does not support blaming stranded paid planting for these losses.

Before day3 the baseline realizes844 in sales versus387 for the failed source. Baseline pre-day3 costs are565 for products,2200 for animals,550 for seeds and6 for hires; the failed source spends331,1300,1480 and9 respectively. The observed daily cash deficit becomes permanently negative after days20,14,18 and18. Earlier higher cash is not an economic gain when it accompanies a smaller productive portfolio.

Case1 loses56675 of realized sales while saving3578 net in other cash costs, giving the53097 final loss. Strawberry sales fall41071 and milk9348; strawberry units fall447 to330. Case2 loses57706 of sales and saves7591 elsewhere, for a50115 loss; milk sales alone fall71867, partly offset by other products, and milk units fall459 to255. Cases3/4 each lose46130 of sales and save1156 elsewhere; milk sales fall36493 and strawberries28403, partly offset by eggs, tomatoes and wheat. These observed differences identify investment ranking and subsequent productive capacity as the main mechanism, not a higher actual wage bill.

`analysis/PREIMPLEMENTATION.md` records the falsifiable mechanism before policy edits. Detailed daily snapshots, successful purchases/sales, actual field-operation records and checks are retained in `analysis/case*/` and `analysis/critical_execution_checks.json`.

## One coherent revision

The existing one-day public rollout transfers many not-yet-producing investments to a long conditional tail score. The new rule allocates more simulation to cash-constrained decisions, without changing capital penalties or granting additional money.

For each existing prepared proposal, it estimates immediate queue spending, known current shed sales, and the next two days' conditional feed and hiring bills plus the existing reserve. When at least one proposal leaves insufficient cash against that estimate, **all proposals use the same three-day public rollout**. Otherwise the normal one-day horizon remains. An already longer configured horizon is not shortened. The actual executor, proposals and common tail scorer are unchanged.

The service estimate selects computation only. It is not a certificate that future work, stock or cash will be feasible. Real workers, routes, inputs and affordability are still enforced by the existing executor. Future shops and market/service estimates remain a conditional public scenario, not actual future randomness or opponent private inventory. No seed or replay-derived action lookup enters the policy. No forecast profit is counted as terminal cash.

Development daily logs show1629 three-day searches and2211 one-day searches across3840 observed dawns. The three-day horizon was not enabled universally for the whole game. The synthetic tests also confirm that ample current cash preserves one day and an explicitly configured five-day horizon remains five days.

## Development, selection and fixed validation

The pilot contained the four critical cases and12 other disclosed cells. It gave critical-case paired changes of -12564,+3047,+11005 and-352. A tool timeout interrupted two pilot attempts; their partial files were retained and the affected cells retried under separate attempt names. All16 completed pilot trajectories later reproduced the corresponding full-development action/state hashes. The interrupted attempts were not silently replaced.

All four full development panels then ran the same baseline and finance3 candidate:

| Panel | Games per strategy | Baseline mean | Revision mean | Paired change |
|---|---:|---:|---:|---:|
| dev1 | 32 | 209733.40625 | 207832.59375 | -1900.81250 |
| dev2 | 32 | 203628.21875 | 206181.37500 | +2553.15625 |
| dev3 | 32 | 199943.75000 | 200661.53125 | +717.78125 |
| dev4 | 32 | 198113.34375 | 204446.34375 | +6333.00000 |
| Pooled development | 128 | 202854.6796875 | 204780.4609375 | +1925.78125 |
| Untouched author validation | 32 | 213431.03125 | 204538.71875 | -8892.31250 |

Dev1–3 are the three supplied controller panels. Dev4 is the now-disclosed Round05 author panel and is development only. The four development panels contain64 distinct seeds. The pilot is a subset and is not added again to the pooled estimate. The revision improved69 development cells and worsened59. It was selected at 2026-09-13T22:49:49.227389+00:00 on the predeclared aim of positive pooled gain and positive means on at least three of four panels. The1925.78125 development margin was modest and noisy. No additional policy ablation or parameter search was run.

The largest development loss was70523 at1781448506 seat0, and the next was65093 at1652205610 seat0. Additional receipt audits showed sales losses of75486 and68270, respectively, partly offset by cost savings. Actual lower labor costs did not explain these losses. A largest gain of67126 at2027896586 seat1 came with75513 more sales and8387 more costs. Longer conditional planning can still select poor productive portfolios; the revision is not uniformly stronger.

The16 validation seeds were declared at **2026-09-13T22:29:49.917350+00:00**, excluding all1128 recoverable supplied IDs, before any new game outcomes. Source was selected and frozen before32 candidate games plus32 matched baseline controls on exactly those seeds. No seed was added; none was removed. The final source remained unchanged regardless of the result.

Validation improved14 cells and worsened18; the seed-averaged change was positive on7 seeds and negative on9. The largest loss was97556 at231131086 seat0: revision161937 versus baseline259493. The largest gain was57267 at1788167253 seat1. All32 rows are in `validation_results.csv`, including every loss. Clearing the cash gate on this author panel does not establish a reliable gain: the unchanged baseline exceeded it by8892.3125. The evidence does not support claiming generalization success.

## Runtime, execution and verification

The current environment measured a four-core cgroup CPU quota, affinity to five logical CPUs,4GiB cgroup RAM and no swap. GCC was14.2.0. The initial native build took approximately27 seconds and peaked at570384KiB RSS. Initial fully recorded games took4.4691 and4.5923 seconds. At most two game/audit workers ran concurrently; no compiler overlapped a two-worker batch.

Completed baseline games averaged3.8474s; completed candidate development games averaged5.2125s. The largest observed game action time was0.548903s. These local measurements include gzip record writing but do not certify Kaggle sandbox or timeout enforcement. Process RSS values are high-water marks and can include earlier jobs in the same worker.

There are344 new game attempts:342 completed and two interrupted. All342 completed games have719 transitions, both players DONE and exact terminal cash/reward agreement. Every full replay passed the frozen interpreter audit, totaling245898 transitions. The two truncated attempts preserve311 and275 verified prefix transitions (586 total), but do not have terminal results or passing full-game audits. They remain failures.

Four full games reused a single imported `main.py` module across seats and games without an explicit reset between games. Every action/state hash and terminal cash matched the corresponding development run, verifying the entrypoint's step-zero reset behavior. Eleven single-observation API/horizon tests passed, and they passed again during a clean archive-extraction rebuild. The clean native rebuild is byte-identical to the evaluated library. These tests are provided as runnable Python source; old inherited test executables are not claimed as Round06 tests.

## Failures, limitations and delivery

One uncompiled draft referred to a nonexistent hiring helper. Inspection corrected it to the existing Fibonacci pricing array before compilation; the original draft remains preserved. There were no failed native builds or failed unit-test cases. Two packaging/orchestration guards failed: one checked for an API-reuse completion receipt before its final game finished; the other used the wrong seed-declaration field name. Both stopped before validation launch, were corrected, and are preserved with failed script/receipts. The pilot timeout, failed attempts, retries and audit EOF errors are all retained. `FAILURES.json` is the consolidated index.

The historical Round02 changed source and detailed evidence remain lost. Its failed195431.625 validation is recorded, not recreated. Unknown historical seed usage cannot be reconstructed; the validation declaration excludes all recoverable known IDs but cannot certify exclusion of undocumented lost IDs. Earlier inherited metadata are labeled historical and are not new test receipts.

The first linked complete checkpoint was saved at2026-09-13T22:29:50.129691+00:00, within five minutes. Research/source selection ended before the reserved final15-minute window; validation ran from2026-09-13T22:52:21.534480+00:00 to2026-09-13T22:54:47.832162+00:00. The final policy was frozen before those outcomes. The hard delivery cutoff is2026-09-13T23:28:42.340Z. Final ZIP sealing times and exact byte sizes/hashes are in the external delivery receipt and checksum file.

Evidence ZIPs are independently readable and each below50000000 bytes. Extract all into one folder; do not concatenate. No old large archive is nested. Supplied source/reference/referee and original records are preserved as individual files, along with all new source variants, four small Round06 checkpoints, full/partial replays, settings, declarations, tests, logs and this report. The original feedback ZIP is referenced by its exact hash. A portable replay verifier is included. Archive members are read back, CRC checked and SHA256 compared against staged bytes. The frozen runtime is also delivered separately below5MB.
