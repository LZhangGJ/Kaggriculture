# C04 round 03 report

## Outcome

**No new economic revision was accepted.** The sole action-changing candidate, append_candidate_v1, failed the predeclared development decision rule. The delivered source is the unchanged supplied baseline, selected and frozen before any validation outcome.

The frozen baseline averaged **203,828.21875 terminal cash** on the entire predeclared author-validation panel: 16 new seeds, both seats, 32 games against legal PASS. This exceeds 200,000 on this panel only. It is not a gain versus the earlier disclosed panel, does not overturn the earlier controller qualification result, and is not competitive acceptance. Different seed panels cannot establish an improvement.

| Panel | Source | Games | Mean terminal cash |
|---|---|---:|---:|
| Disclosed development | Supplied baseline, independently reproduced by no-action-change diagnostic | 32 | 195,548.00000 |
| Disclosed development | Rejected append-only rescue | 32 | 194,877.84375 |
| Untouched author validation | Selected unchanged baseline | 32 | 203,828.21875 |

The development difference was **-670.15625 per game**: one pair gained 168, one lost 21,613, and 30 pairs were identical in every recorded action and state. No second action-changing candidate, parameter search, or validation-based fallback was used.

## Economic idea and measured failure

The existing bounded fleet optimizer can compact routes after an earlier greedy pass has dropped a job. The diagnostic observed five additional feasible append opportunities in two of 32 development games. It reproduced all 32 supplied terminal cash values; four complete baseline/critical replay comparisons also matched all 719 transitions.

The tested repair appended only an already-reserved existing-asset job. It required a wholly unowned tile, no new pickup type, depot access for material-dependent work, and completion within the actual route budget, including required return and DROP. It did not reorder or remove existing jobs, admit investments, change economic settings, alter the workforce-sizing function, or enable broad regret insertion. Work was capped at 256 route checks per final compilation.

One measured success was concrete: on development seed 967718147, seat 0, an idle worker was assigned a 14-action route within a 22-action budget on day 29. At transition 704 the official interpreter records that worker harvesting two tomatoes at (9, 7); the paired final cash gain was 168.

The counterexample was larger: on seed 337943030, seat 0, the first order divergence occurred at policy input step 144, day 6, hour 0, before any logged installed-core recovery. The candidate purchased seven melon seeds and hired four hands; the baseline purchased one melon seed, one wheat seed, and hired three hands. Final cash fell from 226,976 to 205,363. The routing routine participates in existing simulation-based planning as well, so forecast sensitivity is a plausible mechanism. It is **not a proved causal diagnosis**; no isolation experiment followed. Feasible local work was insufficient evidence for accepting the change.

## Preregistration and source integrity

Validation seeds were declared at **2026-09-13T20:02:39.797718+00:00**, before any new game. The exclusion union contains 852 distinct seeds: all 836 supplied known seeds, the disclosed development seeds, and the 16 surviving round-02 validation seeds. The disclosed development seeds were already in the supplied known set. Lost historical records limit any stronger assertion of globally never-used seeds.

Validation seeds, unchanged throughout the round:

`1447343866, 174869052, 584769068, 2088682755, 301419550, 296894131, 1923254772, 1899818043, 1227889858, 1663483468, 2118332570, 1136017406, 1910797787, 191579541, 408037147, 1484282601`

Selection and source freeze: **2026-09-13T20:09:18.077022+00:00**. First validation game: **2026-09-13T20:09:19.505128+00:00**. All 32 predeclared validation combinations completed. No additional seeds were drawn or selected after outcomes, and no fallback was chosen from validation.

All 43 production source/settings files in the selected runtime match the supplied feedback baseline byte-for-byte. The native library was rebuilt locally before evaluation. An isolated clean rebuild later produced the identical native SHA-256:

`174ba2ad749cb3213a72a03302ef50b08a37a579848ba922bde132a19b8b6c23`

The exact source freeze, settings, compiler flags, source hashes and per-game native hashes are in the records. Added round-03 descriptions and packaging metadata do not alter production source.

## Actual resources and timing

The initial complete source/runtime checkpoint was sealed at 2026-09-13T20:02:39.986662+00:00, about 33 seconds after the first execution timestamp and within the first five minutes. It was 1,606,271 bytes; SHA-256 1313b7a2320008efd164bfabadebd0421893105b16a8daf035249ca238d91530. A separate candidate checkpoint and selected-source checkpoint were also preserved.

The first execution timestamp was 2026-09-13T20:02:07.229759+00:00; this is the first measured tool timestamp, not a claimed exact message-receipt timestamp. A conservative delivery cutoff of 2026-09-13T20:31:30+00:00 was used. Economic candidate research ended at source selection, 20:09:18 UTC, leaving more than the required eight minutes for validation and delivery.

CPU quota: 400,000 microseconds per 100,000-microsecond period, equivalent to four cores; affinity exposed five logical CPUs. RAM ceiling: 4,294,967,296 bytes (4 GiB). Maximum parallel game/replay workers: two. Compiler: GCC 14.2.0. Initial baseline build: 27.50750626 seconds. Isolated clean rebuild: 26.916 seconds. Initial complete games with full replay writing took 3.753–3.906 seconds. Maximum recorded policy action over all games: 0.181035 seconds. Recorded cgroup memory peak before packaging: 1,352,507,392 bytes.

## Verification and evidence completeness

All **98 recorded game attempts completed**, across 32 distinct game seeds. These are repeated comparisons plus the single new validation panel, not 98 independent validation samples. Every game has 719 transitions, both seats DONE, and terminal reward equal to the corresponding farm's cash: 70,462 transitions and 196 terminal reward/cash checks in total.

All 98 complete replay files were re-executed through the unchanged official interpreter, with every recorded state checked. All passed. The rejected patch passed 1,000 generated structural route cases and 62,573 assertions, covering deadlines, return/DROP, material pickup types, unchanged worker identity, preserved route prefixes, excluded shared-tile dependencies and the route-check bound. These checks establish tested scheduling properties, not profitability. Six exported-entrypoint/reset invocations on existing initial observations also passed; they are interface tests, not extra full games.

There were no recorded game exceptions, build failures, unit-test failures or replay-audit failures. The failed economic experiment and all 32 of its full replays remain preserved. Source variants include the initial checkpoint, unchanged rebuilt baseline, no-action-change diagnostic, rejected candidate, selected baseline and isolated rebuild copy. No game attempt, negative result or source variant was discarded.

The first evidence-packaging attempt failed with a Python ZipFile constructor TypeError caused by a duplicated allowZip64 argument, before an evidence archive was opened. The failed packaging script and traceback are preserved. Only the packaging argument was corrected; no policy source, game, build or test was rerun.

## Prior source/evidence loss and limitations

The original round-02 evidence ZIP remains unavailable: recorded 328,192,668 bytes, SHA-256 3dbc5d8a5e43ac9cf7d386e538e1c0dfa30ad7e1e3d92d9735c9e3e4fe48b499. It was not recreated. The prior failed 195,431.625 validation remains a surviving report/CSV claim, not a new round-03 result. Prior surviving metadata is retained unchanged and explicitly labeled.

LocalGame uses the frozen official interpreter but is not Kaggle's sandbox, timeout or action-schema validator. Platform compatibility is therefore not certified by these tests. The inherited build.py --unit option references old test source absent from the input; its older binaries/receipts are preserved as provenance, not presented as newly rerun tests.

## Run and inspect

The selected small source/runtime ZIP is delivered separately, under 5 MB. Unzip it and use `main.py:agent(observation, configuration)`. Offline rebuild: `python3 -B build.py --cxx g++`. The native library is Linux x86-64; another platform requires a compatible rebuild. Python 3 and a C++20 compiler suffice; no network is needed.

Every evidence ZIP is an independent normal ZIP under 50,000,000 bytes. Extract all evidence archives into one directory to obtain the complete source, tools, settings, seed inventory, receipts, every attempt/result and all full replays. No large prior ZIP is nested inside. Input archive identities are referenced by exact hashes; the small supplied feedback contents are preserved as individual files. Per-archive manifests and game/seed inventories support separate inspection.

The rejected candidate must not be confused with the selected unchanged baseline. It is retained for review, not silently substituted after validation.
