# C02 economic revision round 2

## Result and qualification boundary

**The frozen revision averaged 201,405.03125 terminal cash over the complete, predeclared 16-seed, both-seat author validation panel: 32 games against the supplied legal PASS agent.** This exceeds the requested 200,000 sample-mean target by 1,405.03125. The rebuilt supplied C02 baseline averaged 200,617.93750 on the identical 32 games. The paired advantage is only **+787.09375**.

The candidate was better in 16 cells and worse in 16. This is a narrow author-panel pass, **not proof of reliable superiority, not independent controller qualification, and not competitive acceptance**. The controller's fresh shared panel is still required. No round robin or 1,536-game competitive acceptance panel was run. No cash threshold is being asserted for that later competitive stage.

Candidate seat means were 202,949.00000 and 199,861.06250. Its minimum game cash was 153,302; maximum was 255,670. As a descriptive uncertainty check, a Student-t interval over the 16 both-seat seed averages is approximately [188,636, 214,174] for mean cash, and [-7,755, +9,329] for the paired advantage. These intervals include the target and zero, respectively. They assume independent seed averages and do not establish representative competitive performance.

Evidence: `research/audit/VALIDATION_RESULTS.json`, `research/experiments/validation_candidate16/`, and `research/experiments/validation_baseline16/`.

## What changed

The revision removes the two extra crop-rotation **forecast proposal families** added in round 1, while retaining the extra no-fertilizer proposal. It does not prohibit planting a replacement crop after a harvest and does not replace the whole-farm strategy with a fixed crop script.

The final setting changes only `c02_crop_candidates: 1 -> 3`. The two-day conditional lookahead (`scenario=2`), first-yield planting calendar (`c02_crop_calendar=1`), complete inherited proposal families, livestock, worker hiring, land acquisition, procurement, maintenance, storage, transport, sales, and intraday replanning remain enabled as before. Base `rotation=0` and `repeat=0` remain unchanged. The retained no-fertilizer alternative competes with ordinary fertilized plans; it does not globally turn fertilizer off.

`policy/proposals.hpp` now exposes bounded ablation modes. Mode 1 preserves the round-1 options; mode 2 omits repeated rotations; selected mode 3 omits both the one-successor and repeated-rotation extras. Disabled slots are replaced with the ordinary base plan and deduplicated. Stable proposal IDs are retained, so removing a family does not renumber the remaining alternatives. Modes 4 and 5 are implemented and unit-checked but were not evaluated as separate game candidates this round.

Only four text files differ from the supplied source:

1. `policy/proposals.hpp`: family gating described above.
2. `policy/config.json`: selects mode 3.
3. `tests/test_crop_contract.cpp`: 76 additional proposal-mode assertions, for 621 total.
4. `main.py`: provenance docstring only; its behavior is unchanged.

No executor, simulator, observation codec, market model, or referee source was changed. Complete before/after hashes and the patch are in `research/audit/SOURCE_CHANGES.json` and `.diff`. The compiled native naturally changes and is included with its matching source and build receipt.

## Why this test, and what the replays do and do not establish

The task prioritized matched crop-rotation tests because the earlier C02 development advantage did not generalize. The disclosed controller panel is now development data. I did not select a new candidate from the old high round-1 development mean.

The rebuilt supplied strategy exactly reproduced all 32 disclosed controller cash values, including the two worst cells. Both critical full replays also matched the supplied records. The two critical cases are the two seats of the same seed, 2027896586, not two independent seeds.

The low-cash cases showed crop-mix divergence well before the final day. A diagnostic for watering already-full mature finite crops found zero such actions in both cases. Harvest-batching opportunities were counted but not treated as proved savings, and no harvest-batching or terminal-liquidation patch was implemented. Unsold goods and daily cash gaps were not counted as achievable profit.

Removing the two added forecasting families changed the actual whole-farm allocation and executor actions. On the critical seed, the candidate reached 230,312 and 222,733 cash, compared with the supplied C02's 133,592 and 146,694. The supplied matched parent replays ended at 175,815 and 188,501.

An independent instrumentation pass replayed actual successful transactions and reconciled each of six critical traces exactly: starting 3,000 cash plus realized sales minus actual purchases, hires, and land costs equals terminal cash. In seat 0, candidate milk receipts were 118,154 versus 35,665; strawberry receipts were 57,514 versus 28,520. Costs and other product receipts also changed. These are realized receipts, not projected values or hypothetical liquidation of leftovers. The evidence supports a whole-policy allocation effect; it does **not** isolate milk, strawberries, a particular harvest, or one forecast error as the sole causal mechanism.

Detailed placements, harvested quantities, actual transactions, daily states, and exact replay comparisons are in `research/audit/economic_traces/`, `economic_trace.log`, `baseline_controller_reproduction.json`, `full_crop_water_diagnostic.json`, and `harvest_batch_diagnostic.json`.

## Matched development ablations

All five rows below use the same 16 disclosed seeds in both seats. Four ablations were declared before their results. No partial games or predicted cash enter any mean.

| Panel | Completed games | Mean terminal cash | Difference from supplied C02 |
|---|---:|---:|---:|
| dev_baseline | 32 | 197,116.43750 | +0.00000 |
| dev_no_repeat | 32 | 210,360.21875 | +13,243.78125 |
| dev_no_rotations | 32 | 210,806.59375 | +13,690.15625 |
| dev_no_extras | 32 | 203,345.12500 | +6,228.68750 |
| dev_horizon1 | 32 | 195,761.46875 | -1,354.96875 |

The selected candidate is `dev_no_rotations`. The runner-up, removing repeated rotations alone, was only 446.375 lower on this development panel. That difference is not claimed statistically significant. Removing all three extras performed worse than retaining the no-fertilizer option. Reducing the search horizon to one day also performed worse. The selected candidate improved 22 of the 32 development cells and worsened 10; excluding the two disclosed critical cells, its mean improvement was still 8,844.2.

Development seeds, all both seats:

```text
140008426, 148094548, 180752382, 219026998, 320211102, 337943030, 370413548, 383365175, 702093297, 967718147, 1501860085, 1658182076, 1681080428, 1781448506, 1856155813, 2027896586
```

Evidence: `research/audit/DEVELOPMENT_DESIGN.json`, `DEVELOPMENT_SELECTION.json`, and all `research/experiments/dev_*/` directories. Each panel has the exact effective configuration and source identity in `declaration.json` and each result record.

## Untouched validation declaration and freeze

The validation panel was declared at **2026-09-13T19:10:07.148822+00:00**, before any validation outcomes. Source was frozen at **2026-09-13T19:18:06.786798+00:00**, before either validation panel began. The candidate completed at 19:19:01.358730Z; the paired baseline completed at 19:20:06.729100Z. There was no tuning, source change, seed addition, or validation exclusion afterward.

The 16 declared seeds are disjoint from every known seed in the supplied known-use material and from the current development seeds. The supplied known-use inventory explicitly warns that some other authors' lost histories are unavailable; universal prior-use coverage cannot be proved. Smoke seed 291307001 was not used this round and is not presented as fresh.

| Validation seed | Candidate seat 0 | Candidate seat 1 | Supplied C02 seat 0 | Supplied C02 seat 1 |
|---|---:|---:|---:|---:|
| 221883146 | 252,336 | 235,620 | 247,877 | 235,041 |
| 193722781 | 192,865 | 193,998 | 179,175 | 182,140 |
| 478491823 | 210,671 | 180,042 | 202,980 | 141,753 |
| 1502030196 | 180,948 | 180,948 | 181,349 | 181,349 |
| 1594998908 | 222,305 | 255,670 | 226,223 | 255,618 |
| 263289160 | 200,516 | 200,516 | 180,008 | 181,077 |
| 879405060 | 205,957 | 166,423 | 208,350 | 232,151 |
| 285509795 | 153,629 | 153,302 | 162,840 | 162,840 |
| 510483081 | 219,403 | 219,805 | 224,226 | 204,171 |
| 1333414495 | 216,273 | 231,974 | 216,659 | 231,997 |
| 589423679 | 184,626 | 190,994 | 196,910 | 215,006 |
| 339799306 | 219,179 | 219,179 | 190,291 | 190,259 |
| 1064328730 | 217,344 | 170,117 | 208,300 | 212,641 |
| 2019841597 | 231,390 | 203,984 | 226,638 | 198,113 |
| 789747602 | 179,113 | 179,113 | 184,730 | 175,346 |
| 208109925 | 160,629 | 216,092 | 166,343 | 217,373 |

The primary mean uses exactly the 32 candidate cells above, not all 64 candidate-plus-control games. Four additional repeated entry-point checks used the already-disclosed critical development seed and are not extra independent validation observations.

Evidence: `research/audit/VALIDATION_DECLARATION.json`, `FINAL_SOURCE_FREEZE.json`, `POST_VALIDATION_SOURCE_CHECK.json`, and `SOURCE_AND_SEED_BOUNDARY_CHECK.json`.

## Complete game and failure accounting

There were **231 new policy-game attempts; all 231 completed**. This consists of one baseline smoke game, 32 baseline reproduction games, two round-1-mode identity games, four 32-game development ablations, 32 candidate validation games, 32 paired baseline validation games, and four repeated root-entry games.

All completed games have 719 transitions, both players DONE, and each player's terminal reward equal to that player's farm money. Every attempt has its seed, seat, source/configuration identity, wall/CPU/action timings, terminal checks, full replay, and result file. Ordinary panel games also have full daily native diagnostic traces. Repeated games are identified as checks rather than additional independent seeds.

An independent official-interpreter audit re-executed all 231 stored traces and compared every recorded state: **166,089 transitions passed**. The audit itself took 13.070 seconds using four local workers. Verification replays and the six economic instrumentation passes are not counted as additional strategy trials.

There were zero gameplay failures, but two non-game setup/audit errors are preserved: an unsupported interactive-container session request, and a comparator that initially compared a scalar with the controller's two-seat cash array. The former started no game; the latter was corrected to use `own_cash` and then all 32 comparisons passed. No errored or unfinished game was silently removed. Earlier-round failures and records remain unchanged inside the original round-1 archive.

Index files: `research/audit/experiment_inventory.json` (all ten panels), `ALL_GAMES.csv`, `ALL_GAMES.json`, `ALL_REPLAY_AUDIT.json`, and `FAILURES.json`.

## Actual resources, speed, and build receipts

The current environment reports a CPU quota of `400000/100000`, or **four CPU equivalents**, with affinity to five logical CPUs. The memory limit is **4 GiB**. It has Python 3.13.5 and g++ 14.2.0. These were measured before selecting the test budget.

The initial baseline compiler receipt was 26.758 seconds; whole command wall time was 27.34 seconds, with peak compiler RSS 577,964 KiB. The first full baseline game, including complete replay and debug output, took 7.726 seconds. Four-worker batches were therefore practical within the deadline.

The final in-workspace compiler receipt was 25.442 seconds. A separate clean copy removed the native library, rebuilt it offline in 26.143 seconds whole-command wall time, and produced a **byte-identical** library. The clean copy also passed all 621 assertions in 4.956 seconds. It loaded and created/closed an agent from an unrelated working directory without the research workspace.

The candidate's 32 validation games took 53.645 seconds as a four-worker batch. Individual full-game wall time averaged 6.626 seconds, range 5.801–7.267 seconds; the largest observed candidate validation action was 0.456 seconds. The matched baseline batch took 64.762 seconds and averaged 7.976 seconds per concurrent game. These are actual host measurements, including serialization, not a guarantee of Kaggle sandbox timing. Peak recorded candidate validation worker RSS was 97,796 KiB.

Receipts: `research/audit/resources.json`, `TIMING_REPORT.json`, build stdout/stderr files, `CLEAN_PACKAGE_BUILD.json`, root `policy/a06.BUILD.json`, and `build/crop_units.json`.

## Runtime, entry point, and offline reproduction

The ZIP root is the standalone production agent. It includes every Python/C++ source and runtime file, `main.py`, `build.py`, `COMPILER_FLAGS.json`, `policy/config.json`, the compiled `policy/a06.so`, and tests. It requires Python's standard library and normal Linux C/C++ runtime libraries only; no Python package installation, network access, external service, replay file, or research directory is needed to run the agent. The supplied binary targets Linux x86-64; rebuild for the evaluation host with a suitable C++20 g++ compiler.

From the extracted ZIP root:

```bash
python3 -B build.py --cxx g++
python3 -B tests/run_units.py --cxx g++
```

Load `main.py:agent(observation, configuration)`. For explicit independent contexts, use `main.create_agent()` and call `close()` when done. `main.close()` releases module-level contexts. The root entry was tested for four complete games without manual resets between them, including both seats twice; every action matched the appropriate stored reference and new step zero reset state correctly.

For an optional local reproduction using the preserved referee and research harness:

```bash
python3 -B research/strategy/build.py --cxx g++
python3 -B research/tools/run_panel.py \
  --strategy research/strategy \
  --panel local_reproduction \
  --seeds '[2027896586]' --seats 0,1 --workers 2
```

Use a new panel name for any further run: the harness refuses to overwrite attempts. This command deliberately uses a known development seed; it does not extend the frozen author validation panel. `research/tools/audit_replays.py` independently checks saved full replays. `research/tools/verify_package.py` verifies a ZIP, extracts it, rehashes all files, checks the frozen source and exact validation cells, and tests native loading.

The numeric observation codec uses only allowlisted current official observation fields: public farm/market/town data and the current player's private inventory. It does not consume configuration or a game RNG seed. Full two-seat states and seeds are retained only by offline evaluation and replay verification. No replay action, future randomness, or opponent private state is passed to the policy.

## Provenance and archive map

The feedback archive SHA-256 matched `3a81219875fcd8991353336c664b50cc8cc7fef7b9777136908ed7fa6b940c6b`; all 59 internal manifest entries passed before work and again after testing. `research/supplied/` is its unchanged extraction. `research/baseline/` is the locally rebuilt baseline; `research/snapshots/modes/` is the exact ablation source; `research/strategy/` is the frozen final source. The standalone root duplicates the latter runtime intentionally. `research/referee/` is the unchanged official engine.

`provenance/` contains exact originals of the round-2 feedback ZIP, round-1 strategy ZIP, round-1 report, and original author-input ZIP. The 41,808,117-byte round-1 ZIP retains SHA-256 `13062d1229dca0ab6aea83e0c6e1ab2b04725dd1b8eb6d26175a30254584a0b0`. Its old results, failures, and source are preserved, not relabeled as new evidence. The controller's wider 256-game verification is supplied history; the new local work reproduced our 32 disclosed cells and audited the traces described above, not the other authors' undisclosed traces.

`PACKAGE_MANIFEST.json` records the exact byte count and SHA-256 of every other file in this delivery. `SOURCE_HASHES.json` records the frozen runtime mapping. Final ZIP verification includes CRC, member-set equality, every file hash, extraction and rehashing, both root and research runtime hashes, all 231 game-record/replay links, exact validation cells/means, and native create/close. The outer ZIP hash is delivered alongside the archive, avoiding a self-referential hash inside it.

Frozen source-set identity: `ab200b53b17d304e732e69b5cef4ae123ef9bf27b9f0ffcb8054369847b19fbe`.

Native SHA-256: `f804c7775db200c5a0e6fda36f8a56a14a8c4f16adcd4c7ef62c91bd80a7aaa2`.

The round's absolute cutoff is 2026-09-13T19:54:33.786Z. Policy research and changes stopped at the source freeze, leaving over 36 minutes for fixed validation, auditing, clean rebuild, and packaging. No further research or validation-seed expansion followed the sample-mean result.

## Outer ZIP delivery receipt

ZIP verification completed at 2026-09-13T19:26:05.087112+00:00. All 1,080 archive members passed CRC checks; all 1,079 manifest entries were checked, extracted, and rehashed. The extracted runtime loaded successfully.

File: `C02_round2_strategy.zip`

Exact size: 262,780,104 bytes.

SHA-256: `5729f5bd617b2a4439e9fd123d5f8d208bd077950a594ec91c16c75d2fa27a75`

The machine-readable outer receipt is `C02_round2_DELIVERY_RECEIPT.json`. This appended outer-archive receipt is outside the ZIP to avoid a self-referential ZIP hash.
