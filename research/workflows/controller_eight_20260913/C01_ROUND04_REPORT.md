# C01 ROUND04: economic revision and frozen validation

## Outcome

**Author-validation economic gate: PASS.** The selected complete strategy, `labor1_batch3_terminal5`, averaged **208,405.84375 terminal cash** across the exact 16 declared seeds, both seats, 32 complete games against legal PASS. Seat means were 208,414.5625 and 208,397.1250. All declared games are included. There were no added seeds, policy changes after freeze, or post-validation fallback selections.

The selected development mean was **208,759.65625**, versus the reproduced original **199,464.71875**, an increase of **9,294.93750**. Both development seat means exceed 200,000. This is an author economic result, not the controller's later fresh qualification and not competitive acceptance.

## Identities, resource measurements and timing

The input archive SHA256 is `ec2182ac24ab27df7e60464af8e5c607cffdebd045dd4662f8a8769e10863564`. All 136 supplied manifest entries verified. The three supplied referee copies were byte-identical. The official interpreter SHA256 is `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`; local runtime wrapper SHA256 is `9d601268672ceb713918ad04db1804f05fe43dbd70f5e03a93ce6109f7a54726`. The supplied baseline and reference, their prior receipts, all disclosed result rows and eight full critical replays remain unchanged in `feedback/`.

The environment had a four-core CPU quota (`400000 100000`), affinity to five logical CPUs, and a 4,294,967,296-byte memory limit. No more than two game workers were used. The compiler was g++ 14.2.0. The initial local native build took 27.0575 seconds, the horizon-enabled build 26.1160 seconds, and the final build 27.9029 seconds. The first two fully logged and replay-verified games took approximately 6.35 and 6.22 seconds. Validation averaged 6.2990 seconds per game including independent verification, of which 0.4682 seconds were verification. Mean policy time was 4.5647 seconds per game; the largest individual policy call was 0.377135 seconds. Maximum reported validation worker RSS was 98,972 KiB.

The first complete source/native checkpoint was saved at **20:35:28 UTC**, about 52 seconds after dispatch, at 670,037 bytes. Successive accepted candidates also received complete checkpoints. The fixed validation declaration was saved at **2026-09-13T20:35:28.020356+00:00**. Final selection and the end of research were **2026-09-13T20:53:13.779378+00:00**; source freeze was **2026-09-13T20:53:48.651650+00:00**. Validation began afterward. The round's absolute cutoff is 21:14:36.381 UTC. Exact final archive-verification timestamps, sizes and hashes are in the separate delivery receipt.

## What was actually changed

The submitted runtime remains the full C01/A06 implementation: daily investment planning, intraday reinvestment, crops, livestock, feed, maintenance, staffing, land, routes, deliveries and sales. It is not C03 renamed. The only changed existing policy settings are **work_price 4 → 1** and **harvest_threshold 1 → 3**. One new observed-clock setting, **c01_horizon_rule=1**, controls the explicit rollout length. The fixed base `scenario=2`, C01 payback strength, crop-maturity admission, competition setting and all other configuration values remain unchanged.

**Terminal-complete rollout.** Before the final five remaining game days, the explicit candidate rollout stays at two days. Once at most five days remain, it evaluates to the terminal boundary: five, four, three, two or one days as applicable. This replaces an intermediate residual valuation with the conditional continuation to the finish. It still uses public-state forecasts; it does not know future randomness. The filename `terminal_exact5` is an experiment identifier, not a claim of an exact stochastic forecast.

**Labor opportunity cost.** The artificial planning charge for labor is lower, while actual official wages, workforce limits and feasibility checks remain unchanged. A lower shadow charge can alter portfolios, selected staffing and actual worker counts; no extra workers or cash are granted. The planner's remaining-game investment cash-recovery test is retained.

**Harvest batching.** The executor's ordinary harvest threshold is three rather than one. Existing forced-harvest, expiry, retirement, terminal and storage-related paths remain in place. This is a scheduling calibration, not an assumption that all observed goods can feasibly be sold. Its interaction with the other selected changes was tested on the full development panel.

**Attribution.** The supplied C03 reference motivated the isolated scalar settings work_price=1 and harvest_threshold=3. No C03 source component or native implementation was copied. C01's own payback accounting, maturity admission, search, configuration and executor remain. The final horizon helper and its observation-clock regression tests were implemented in this round. Five policy files differ from baseline: `agent.py`, new `c01_horizon.hpp`, `config.json`, `search.hpp`, and `triad.hpp`. Full differences are in `evidence/SELECTED_SOURCE.diff`.

The frozen README contains one overly broad phrase about unchanged worker counts. `evidence/DOCUMENTATION_ERRATA.md` clarifies that the **rules and limits** are unchanged, whereas actual chosen staffing can change. The frozen runtime was not edited after validation began.

## Development tests and selection

All nine planned development panels completed all 32 disclosed cells, with the same seeds, both seats, and the same frozen legal PASS opponent. The baseline exactly reproduced all 32 controller cash values. The negative control checks that the new helper/settings plumbing does not itself change actions. No validation outcome was consulted in selecting the highest complete development mean.

| Variant | Games | Mean cash | Seat 0 | Seat 1 | Delta vs original | Better / worse / tied |
|---|---:|---:|---:|---:|---:|---:|
| Original C01, two-day rollout | 32 | 199,464.71875 | 201,893.3125 | 197,036.1250 | +0.0000 | 0 / 0 / 32 |
| New code, rule 0 negative control | 32 | 199,464.71875 | 201,893.3125 | 197,036.1250 | +0.0000 | 0 / 0 / 32 |
| Terminal-complete final ≤5 days | 32 | 200,097.71875 | 202,422.3750 | 197,773.0625 | +633.0000 | 24 / 2 / 6 |
| Four-day rollout from day 18 | 32 | 200,654.03125 | 203,387.8125 | 197,920.2500 | +1,189.3125 | 20 / 12 / 0 |
| One-day rollout from day 18 | 32 | 199,616.09375 | 201,957.1250 | 197,275.0625 | +151.3750 | 13 / 19 / 0 |
| Labor shadow price 4 → 1 | 32 | 206,081.21875 | 207,274.6875 | 204,887.7500 | +6,616.5000 | 18 / 14 / 0 |
| Harvest batch threshold 1 → 3 | 32 | 202,229.71875 | 206,617.3125 | 197,842.1250 | +2,765.0000 | 19 / 13 / 0 |
| Lower labor shadow + terminal completion | 32 | 206,368.65625 | 207,632.0625 | 205,105.2500 | +6,903.9375 | 18 / 14 / 0 |
| Selected: lower labor shadow + batching + terminal completion | 32 | 208,759.65625 | 209,852.6875 | 207,666.6250 | +9,294.9375 | 19 / 13 / 0 |

The final candidate improved 19 cells and worsened 13. At the paired-seed level it improved 11 of 16 seeds and worsened five. Its worst cell regression was 44,277 cash; its best gain was 72,518. Mean development cash after dropping any one seed pair ranged from 204,978.70 to 212,113.17, so the 200,000 development crossing does not depend on retaining one particular seed pair. This is descriptive sensitivity analysis, not a statistical guarantee. The paired-seed standard error of the mean improvement was 7,472.57; the small development panel and multiple tested variants limit generalization claims.

The measured sequence of conditional mean gains was +6,616.5 from the labor setting, then +287.4375 from terminal completion with that labor setting, then +2,391 from batching on that combination. These are controlled full-panel ablation differences, not proof that a particular observed leftover or single action caused the overall improvement.

## Supplied failure evidence and causal limits

All eight supplied critical full replays were re-executed under the unchanged official interpreter, with full-frame equality and terminal reward/cash agreement. Matched baseline/reference trajectories already diverged in early investment actions. Therefore later cash gaps and terminal inventory differences do not isolate a late-game cause.

An offline diagnostic observed official inventory-drop losses without changing mutations or policy inputs; full-frame replay equality was checked. Both low-cash C01 trajectories and high-cash reference trajectories contained lost inventory. For example, the 250,092-cash C03 reference discarded 46 strawberries, while a 199,423-cash C01 trajectory discarded 35 wool. These are quantities, not valuations or feasible recoverable profits. The evidence does not justify claiming that liquidation of those goods would have been legal, timely or profitable. The full quantity/event records and action/production summaries are preserved.

## Untouched author validation: full fixed panel

The declaration excluded every recoverable seed in the supplied known-seed file before any outcome was inspected. That file explicitly warns that some old author histories are lost. Accordingly, these seeds were untouched within this round and excluded from all **known** earlier use; universal non-use across unrecoverable past histories cannot be proved. No historical unknown seeds were invented.

| Declared seed | Seat 0 cash | Seat 1 cash | Status |
|---:|---:|---:|---|
| 1690074963 | 236,572 | 236,572 | Both complete and replay-verified |
| 346166193 | 216,078 | 229,419 | Both complete and replay-verified |
| 1721513566 | 197,763 | 215,342 | Both complete and replay-verified |
| 1314085243 | 210,599 | 194,495 | Both complete and replay-verified |
| 621306983 | 188,974 | 187,909 | Both complete and replay-verified |
| 2034220553 | 231,394 | 231,394 | Both complete and replay-verified |
| 1731369443 | 136,440 | 136,372 | Both complete and replay-verified |
| 142652784 | 261,081 | 261,081 | Both complete and replay-verified |
| 803811759 | 218,576 | 191,620 | Both complete and replay-verified |
| 1299507916 | 235,041 | 228,252 | Both complete and replay-verified |
| 1414714124 | 213,403 | 212,683 | Both complete and replay-verified |
| 1685113311 | 212,925 | 219,143 | Both complete and replay-verified |
| 457541210 | 189,260 | 190,322 | Both complete and replay-verified |
| 931834493 | 177,741 | 177,526 | Both complete and replay-verified |
| 1120711045 | 196,008 | 210,781 | Both complete and replay-verified |
| 487862802 | 212,778 | 211,443 | Both complete and replay-verified |

**Full-panel mean: 208,405.84375.** Minimum individual cash: 136,372; maximum: 261,081. All 32 results, including low-cash outcomes, are retained. The gate is a mean requirement, not a per-game minimum. The source/native/configuration remained byte-identical to the freeze throughout the entire panel.

## Transition, source and runtime verification

There were **322 new complete game attempts**: 288 full development games, two initial speed-measurement games on disclosed cells, and 32 frozen validation games. These contain **231,518 recorded official transitions**. Every complete replay has an initial frame plus 719 transitions; both players finished DONE and both terminal rewards equaled cash. Every replay was independently re-executed against the frozen official interpreter and every saved frame compared. The passive opponent was checked as the exact legal PASS action. No failed or incomplete game was excluded, and no unmatched start receipt remained.

The eight supplied historical replays add 5,752 checked transitions but are not counted as newly run economic games. The API tests intentionally used two partial contexts with three total official transitions; those are separately labeled and fully traced, not terminal performance claims.

The final build passed **620** native payback/cash/maturity checks, **5,752** horizon-boundary assertions and **three** public-entrypoint/reset/isolation checks. Its native binary hash exactly matches the selected development binary: `8fa3feea85737bf649e90f4bd2196c3c48f291db9b8e39a19173a9188721f8f0`. All frozen source files, all per-experiment source manifests, every replay hash and every start/result pair were reconciled. This establishes source-to-record linkage and independent simulator replay correctness; it is not a claim that a second policy process regenerated every action of every game.

| Action-prefix comparison | Cells | Required equal prefix | Result |
|---|---:|---:|---|
| baseline vs horizon_mode0 | 32 | 719 transitions | PASS |
| baseline vs terminal_exact5 | 32 | 600 transitions | PASS |
| baseline vs late_h4 | 32 | 432 transitions | PASS |
| baseline vs late_h1 | 32 | 432 transitions | PASS |
| labor_shadow1 vs labor1_terminal5 | 32 | 600 transitions | PASS |

## Failures and evidence limitations

No game, native build, regression test or final replay verification failed. The original baseline and one-day late rollout remained below the mean gate; all their complete results are retained. Other tested candidates were not selected, even where they passed 200,000.

An early diagnostic parser raised `KeyError: state` because the supplied initial frame uses `initial`; it was corrected without changing replay bytes. The error and correction are recorded, but the original diagnostic stderr was not separately retained before rerunning. A rejected interactive-container request executed no command or game; subsequent execution used supported calls. These operational issues and the documentation clarification are in the failure inventory. There is no missing new game replay or source snapshot.

## Delivery and reproduction

`C01_ROUND04_RUNTIME.zip` is **726,988 bytes**, contains the complete frozen source, native library, entrypoint, referee, build instructions, tests and source hashes, and is verified independently of the evidence parts. Its SHA256 is:

```text
f96c115ed06899f6dbfccb8a1667e6a626780696290be6352973d62741a28191
```

Evidence is split into ordinary independently readable ZIPs, each strictly below 50,000,000 bytes. Extract all parts into one directory to reconstruct the complete `c01_r04/` tree. Every part has an entry-level SHA256 manifest; `C01_ROUND04_EVIDENCE_INDEX.json` maps every preserved file to its part. All original inputs, source snapshots, checkpoint ZIPs, settings, declared seeds, full results, failures, build receipts and full replays are included without embedding the large inherited R1/R2 archives.

`REPRODUCE.md` describes the exact runner and verifier. Use a new experiment name when rerunning to avoid overwriting historical evidence. The standalone results CSV, summary and delivery receipt provide inspection without opening the full evidence archives.

No round robin, external publication, fresh controller qualification or later 1,536-game competitive acceptance is claimed. Later competitive acceptance has no cash threshold.
