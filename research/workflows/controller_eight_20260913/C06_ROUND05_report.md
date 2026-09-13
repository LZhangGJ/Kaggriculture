# C06 ROUND05 — final source and evidence report

Assembled from original receipts at **2026-09-13T22:00:13.363068+00:00**. Research/delivery cutoff: **2026-09-13T22:15:12.921Z**.

## Result and limits

Frozen candidate author validation: **$207,898.15625**, all 32 declared games, 16 distinct seeds, both seats versus the legal supplied PASS agent. Paired unchanged-current-baseline validation: **$195,514.40625**; paired improvement **$12,383.75000**. This is economic author validation, not a fresh shared-controller result and not competitive acceptance.

The revision improved 21 cells and declined in 11; candidate cash ranged from $152,189.00000 to $257,205.00000. 15 individual cells remained below $200,000; the requested gate concerns the panel mean, not every individual game. The two seats share each seed and are not 32 independent seed draws. Standard error of the paired mean using 16 seed-level paired averages is $6,679.28369; this small panel does not establish universal improvement.

## Full strategy, changes and attribution

### C06 ROUND05: complete runnable strategy

Entry point: `main.py:agent(observation, configuration)`. `create_agent()` creates an independent native context; both explicit `close()` and `reset()` are supported. The module entry maintains separate seat contexts and automatically resets on each new step-zero observation.

#### Offline build and checks

```bash
python3 -B build.py --cxx g++
python3 -B tests/run_units.py --cxx g++
python3 -B tests/run_resource_checks.py --cxx g++
python3 -B tests/run_round05_contracts.py --cxx g++
```

The included native library targets Linux x86-64. Rebuild from the included C++20 source for the destination environment. Python uses the standard library only; normal system C/C++ libraries and an offline C++ compiler are required. No network, package download, trained-weight file or external service is used. Frozen local-referee files are included for audit, not needed by the live agent.

#### Full strategy

The retained A06/C06 planner constructs crop and livestock cash-flow portfolios from the current observed farm, owned inventory, market and public rival farm. It accounts for feed, seeds, fertilizer, crop maturation, animal service, land and estimated labor costs. The conditional market forecast uses observed shops plus inherited expected future demand; it never reads future shops or a hidden seed. Public opponent estimates are forecasts, not opponent private information.

The selected daily control is direct adaptive planning (`scenario=0`), rather than the outer one-day scenario selector. At each dawn it retains funded commitments, values incumbents and feasible new investments, then checks the funded plan with the route executor. Intraday repairs use actual observations and existing supply reservations. Early investment preferences use capital power 0.7, discount 0.08, and an animal ceiling of 26. Action shadow price is 0; actual wages remain payable and modeled. The hand ceiling remains 14.

C06's harvest batching threshold of 3 remains, with capacity, expiry and terminal harvest overrides. Its extra fertilizer-priority heuristic remains disabled; the ordinary finite-fertilizer and service models remain active and constrain actual stock. C03-attributed feasible surplus delivery remains at mode 3. This round adds no hauling mechanism. Workers are hired through existing funded route estimates, and every actual action is checked against observed location, inventory, job dependencies and time.

#### Provenance

The complete ancestor is the supplied frozen C06 ROUND04 strategy, derived from A06 r6 revision1. C06's allocation switches, harvest batching and retained executor are preserved. Its pre-existing delivery component is attributed to C03. The supplied C08 reference receives credit for the direct adaptive control choice and investment settings. C08 opening-prior/maturity changes were tested in a separate variant, not incorporated in this selected policy. The closed-handle check follows the C08 safety pattern. This is not an unchanged renamed C08: C06 harvest batching, work-price choice where different, partial-delivery implementation and overall source ancestry remain explicit.

#### Evidence boundary

Selected development variant: `adaptive_investment`. Selection used three disclosed panels before the separately declared validation panel. Development outcomes are selection-biased and do not establish competitive performance. Full attempt records, source snapshots, declarations and replay files are delivered in separate evidence ZIPs. Final validation outcomes are in the external report, not embedded in policy inputs. No seed, full referee state, reference replay action or private rival inventory enters the policy.

The old missing ROUND02 archive and claimed 502 games remain an explicit gap and are not included as verified evidence. Unknown seeds from lost historical research remain unknown. Later competitive acceptance is separate and has no cash threshold.


### Exact selected setting changes versus supplied own source

| Setting | Supplied C06 | Final C06 |
|---|---:|---:|
| scenario | 1 | 0 |
| capital_power | 0.4 | 0.7 |
| discount | 0.005 | 0.08 |
| max_animals | 20 | 26 |

Retained: work_price=0, harvest_threshold=3, max_hands=14, c06_delivery_mode=3 and c06_fertilizer_priority=0. Exact complete settings are in selected/policy/config.json. Production C++ ancestry remains supplied C06; final native library is independently rebuilt here. The additional closed-handle guard and main.close() alias affect lifecycle safety, not tested live economic trajectories. All 32 final-wrapper development trajectories exactly matched the selected economic variant before freeze.

## Actual resources and budget

CPU affinity [0, 1, 2, 3, 4]; cgroup quota 400000/100000 = 4 CPUs; memory ceiling 4,294,967,296 bytes. Python 3.13.5; g++ (Debian 14.2.0-19) 14.2.0. At most 2 simultaneous game/build/replay workers. Initial unchanged baseline build 29.86 seconds; initial full game 3.610597465 seconds including replay/accounting logging. Candidate validation mean 1.683096 seconds/game; paired baseline 3.768230 seconds/game. Clean extraction build 27.796135 seconds. Maximum game-worker process RSS 111,332 KiB; maximum individual measured policy call 0.221777 seconds. Compiler peak RSS is separately recorded in the /usr/bin/time build receipts. No Kaggle sandbox timing claim is made.

A complete baseline checkpoint was published at 21:37:17 UTC, within five minutes; accepted direct and combined-investment checkpoints followed at 21:40:41 and21:42:23. Final source froze at 21:52:31 UTC, before validation. No further policy research occurred after freeze. More than the final ten-minute reserve was available for fixed checks and delivery.

## Development: coherent component comparisons

All following current-panel rows use the same 16 disclosed controller seeds, both seats. Baseline and reference rebuilds exactly reproduced all 32 supplied terminal cash rows. These outcomes were used for selection and are not fresh validation.

| Variant | Games | Mean terminal cash |
|---|---:|---:|
| baseline |32| $198,253.56250 |
| direct_adaptive |32| $207,154.25000 |
| investment_only |32| $198,833.81250 |
| adaptive_investment |32| $219,060.53125 |
| worker_relief |32| $207,063.87500 |
| adaptive_worker_relief |32| $219,060.53125 |
| adaptive_facts |32| $218,872.34375 |
| adaptive_cost4 |32| $217,010.87500 |
| reference |32| $215,897.46875 |

direct_adaptive changes scenario only. investment_only changes capital power/discount/animal ceiling only. Their combination is adaptive_investment. worker_relief and adaptive_worker_relief raise the hand ceiling 14→15 in their respective bases. adaptive_facts adds the attributed C08 daily/intraday maturity corrections and removes the opening mirror prior as a bundle. adaptive_cost4 restores work_price=4 in the combined investment variant. All hypotheses and exact source/config maps precede their runs. Rejected variants remain included.

### Predeclared selection across three disclosed panels

Before reading the additional disclosed-panel outcomes, selection was fixed to the highest equally weighted 96-game mean among the three coherent leading revisions. All three and the unchanged current C06 baseline were rerun on both earlier panels. The original historical panel source identities remain in input/DISCLOSED_PANEL_HISTORY.json; those old results are not mislabeled as runs of the current source.

| Source | Current panel | Earlier panel1 | Earlier panel2 | All96 games |
|---|---:|---:|---:|---:|
| baseline |$198,253.56250|$204,453.84375|$202,640.00000|$201,782.46875|
| adaptive_investment |$219,060.53125|$213,061.84375|$204,024.59375|$212,048.98958|
| adaptive_facts |$218,872.34375|$213,739.84375|$202,120.25000|$211,577.47917|
| adaptive_cost4 |$217,010.87500|$211,830.00000|$205,371.03125|$211,403.96875|

Selected adaptive_investment at 2026-09-13T21:50:06.459365+00:00, before validation, with a 96-cell paired gain of $10,266.52083. The maturity/prior bundle and positive action-price variant were not selected. Their aggregate comparisons do not isolate the merits of each bundled correctness change. The final strategy still retains the inherited gates/prior, a documented limitation rather than a claim they are ideal.

## Economic and executor diagnosis

| Current matched panel accounting | Baseline | Selected | Difference |
|---|---:|---:|---:|
| sales |$236,530.68750|$257,033.28125|$20,502.59375|
| total_costs |$41,277.12500|$40,972.75000|$-304.37500|
| wages |$4,069.09375|$4,265.03125|$195.93750|
| animals |$7,246.87500|$8,878.12500|$1,631.25000|
| seeds |$8,293.75000|$7,925.31250|$-368.43750|
| land |$7,000.00000|$6,875.00000|$-125.00000|
| inputs |$14,667.40625|$13,029.28125|$-1,638.12500|
| terminal_cash |$198,253.56250|$219,060.53125|$20,806.96875|

Every receipt reconciles starting cash 3000 plus actual successful sales minus actual payments to terminal cash exactly. These are realized operations from call-through logging around the unchanged referee, not hypothetical values of leftover cargo. Replay reconstruction uses a fresh interpreter without those hooks and matches every full state.

Direct adaptive planning alone helped substantially; changing investment preferences under the old selector alone helped little. The combined intervention improved much more, supporting an interaction between the control policy and investment valuation. The 15-hand combined variant reproduced the 14-hand cash outcomes exactly on all 32 current cells, while the direct-only 15-hand test declined slightly. Those tests did not demonstrate the hard staffing ceiling as the main bottleneck under the tested settings; they do not rule out other labor inefficiencies. No mandatory overflow haul or extra delivery heuristic was added. Current investment sales rose while total costs fell slightly; wages rose, so the gain is not explained by suppressing real labor payments.

### Critical supplied low-cash seed1047284227

| Policy | Seat0 | Seat1 |
|---|---:|---:|
|baseline|$127,257.00000|$129,303.00000|
|direct_adaptive|$148,548.00000|$146,735.00000|
|investment_only|$133,690.00000|$132,321.00000|
|adaptive_investment|$187,397.00000|$189,412.00000|
|adaptive_facts|$217,933.00000|$217,933.00000|
|adaptive_cost4|$191,588.00000|$190,186.00000|
|reference|$216,654.00000|$216,654.00000|

The selected revision improves this failure case but still trails C08 and the rejected maturity/prior bundle there. Selection used all 96 disclosed cells, not these two cases. Same initial seeds can lead to policy-dependent later official random draws; paired outcomes do not imply identical later shops or weeds. The critical raw trajectories and actual transactions are retained.

## Validation discipline and complete panel

Declaration: 2026-09-13T21:36:01.881503+00:00. Source freeze: 2026-09-13T21:52:31.299708+00:00. Candidate start: 2026-09-13T21:54:41.035326+00:00. Candidate finish: 2026-09-13T21:55:08.327581+00:00. Both policies began after source freeze. All 16 validation seeds were excluded from all 1,062 recoverable supplied IDs. No validation IDs were added; no fallback was chosen afterward; all 32 candidate and 32 paired-baseline cells completed.

Frozen source-map aggregate SHA256: `3eb57e6d6da5e8144a4868203641dd91ac5f9a247bd7a001a97f617085cc9584`. Canonicalization is SHA256 of the sorted compact JSON source-path→SHA256 map, exactly retained in audit/source_freeze.json. Native SHA256: `6e034fa53900a2f2c74c9e9cda29d1a79906509cba116c0320bd4ba38417f23b`. The final runtime contains 57 entries including CHECKPOINT.json; the policy/source map has 56 entries. All remain unchanged after freeze.

| Seed | Seat | Candidate cash | Baseline cash | Difference |
|---|---:|---:|---:|---:|
|352766599|0|$206,078.00000|$214,803.00000|$-8,725.00000|
|352766599|1|$200,438.00000|$190,275.00000|$10,163.00000|
|2142573695|0|$196,762.00000|$154,438.00000|$42,324.00000|
|2142573695|1|$200,480.00000|$148,525.00000|$51,955.00000|
|1512047632|0|$199,149.00000|$197,260.00000|$1,889.00000|
|1512047632|1|$193,011.00000|$196,511.00000|$-3,500.00000|
|2015901684|0|$256,071.00000|$217,673.00000|$38,398.00000|
|2015901684|1|$256,071.00000|$217,673.00000|$38,398.00000|
|496287773|0|$190,683.00000|$190,905.00000|$-222.00000|
|496287773|1|$191,208.00000|$190,905.00000|$303.00000|
|784874686|0|$227,221.00000|$172,097.00000|$55,124.00000|
|784874686|1|$227,221.00000|$172,097.00000|$55,124.00000|
|783510989|0|$207,757.00000|$218,000.00000|$-10,243.00000|
|783510989|1|$199,289.00000|$215,691.00000|$-16,402.00000|
|593854150|0|$207,986.00000|$226,589.00000|$-18,603.00000|
|593854150|1|$208,967.00000|$226,589.00000|$-17,622.00000|
|1392307343|0|$158,672.00000|$148,994.00000|$9,678.00000|
|1392307343|1|$179,406.00000|$185,777.00000|$-6,371.00000|
|105437625|0|$192,271.00000|$170,062.00000|$22,209.00000|
|105437625|1|$152,189.00000|$150,060.00000|$2,129.00000|
|1567599970|0|$178,618.00000|$171,398.00000|$7,220.00000|
|1567599970|1|$176,187.00000|$161,270.00000|$14,917.00000|
|1356128972|0|$257,205.00000|$236,739.00000|$20,466.00000|
|1356128972|1|$257,205.00000|$236,739.00000|$20,466.00000|
|1567220014|0|$220,751.00000|$224,587.00000|$-3,836.00000|
|1567220014|1|$244,458.00000|$171,219.00000|$73,239.00000|
|1650176190|0|$246,945.00000|$204,294.00000|$42,651.00000|
|1650176190|1|$252,226.00000|$205,303.00000|$46,923.00000|
|663227382|0|$203,847.00000|$245,842.00000|$-41,995.00000|
|663227382|1|$199,554.00000|$245,842.00000|$-46,288.00000|
|1553132332|0|$183,752.00000|$174,152.00000|$9,600.00000|
|1553132332|1|$181,063.00000|$174,152.00000|$6,911.00000|

The report table is a view of original individual result receipts, not their replacement. audit/validation_paired_results.csv and validation_checks.json point to the full evidence. Full result/action timings and replays are in runs/validation_candidate and runs/validation_baseline.

## All attempts, replay reconstruction and safety checks

**647 newly started and completed local games in 23 suites**, zero game failures or unstarted declared cells. Every game completed 719 transitions, both DONE, exact reward/cash agreement, and actual-cash accounting. The total is not one economic panel: it includes reference comparisons, variants, earlier-panel reruns, a speed duplicate,32 final-wrapper parity duplicates,4 same-process API duplicates and2 clean-extraction duplicates as well as64 paired validation games.

All 647 local replays passed fresh uninstrumented official-interpreter reconstruction: **465,193 transitions and 465,840 state frames**. All 8 supplied controller replay paths also passed (4 unique hashes; overlapping cases intentionally retained). Across 655 files, 470,945 transitions and 471,600 state frames matched with zero mismatches. Complete per-file receipts and hashes are in audit/full_replay_verification and audit/replay_index.json.

The clean extraction rebuild produced a byte-identical native library and both clean games reproduced the full selected trajectory. Four same-module games retained the same seat contexts without explicit inter-game reset and exactly reproduced selected trajectories, validating automatic step-zero resets. Explicit close is idempotent; a closed Agent raises before native access. All 8 inherited fertilizer invariant groups, 23 resource assertions and 9 new ROUND05 assertions passed. No seed or full referee state entered policy input: harness checks configuration.seed is None at every turn; reference replays were comparison evidence, never action tables.

### Preserved failure

The first newly written ROUND05 unit fixture failed to compile because its struct F collided with the inherited fertilizer symbol F. The failed fixture, compiler output and full pre-fix source snapshot are retained under audit/failures/test_fixture_name_collision and snapshots/selected_before_test_fix. The fixture was renamed to Round05Fixture; production policy code was unchanged. The retry passed before source freeze. This test failure is separate from game failures, of which there were none. The pre-freeze parity plan used the pre-fix test file; the snapshot and freeze receipt explicitly account for that sole test-only difference.

### Historical evidence gap and seeds

The missing C06 ROUND02 full ZIP (recorded 573,143,728 bytes, SHA256 e828c731600d346f3a799eac264a925bcbc81137619e787f187c2d8c28a27fd3) and its claimed 502 games remain unverified and unavailable. No lost games, declarations, traces or unknown seed IDs were invented. This report does not count that claim. Existing prior attachments are preserved unchanged outside this round package; audit/prior_artifacts_inventory.json records their observed hashes. No inherited large archives are embedded.

Exactly 64 distinct IDs were actually used in ROUND05: 48 disclosed development seeds and 16 newly declared validation seeds. The complete per-attempt ledger preserves duplicates by suite/seat. The supplied recoverable IDs plus this round form a 1,078-ID advisory exclusion list. Unknown IDs from unrecoverable historical work remain unknown in identity and count; this list cannot certify complete all-history freshness.

## Packaging and reproduction

Use the separately delivered C06_ROUND05_runtime.zip for the agent. All evidence archives are independent ZIPs below 50,000,000 bytes, with paths relative to a common evidence root; extract into one directory, do not concatenate. Per-archive manifests verify every member except themselves. The top-level external delivery receipt records exact archive bytes and SHA256, final CRC/hash inspection and full cross-archive result/replay coverage. No claim depends on an undelivered monolithic archive.

Audit reproduction tools are under tools/. The frozen referee is under input/referee; all source variants and originals remain included. Example review commands (not necessary to run the live policy): `python3 -B tools/verify_replays.py --name independent_recheck --include-input`. To rerun games use a new suite name and a declared seed-file path with tools/run_suite.py; the recorded results are immutable evidence of this round. Do not treat an independent rerun as part of the fixed original validation panel.

**No fresh controller qualification, round-robin score or competitive acceptance result is claimed.** Later competitive acceptance is separate and has no cash threshold.
