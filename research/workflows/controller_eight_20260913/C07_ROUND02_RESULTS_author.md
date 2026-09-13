# C07 economic revision round 2

## Outcome: fixed author validation missed the 200,000 target

The frozen revised strategy completed all **32 predeclared author-validation games** (16 seeds, both seats) against the supplied legal PASS opponent. Mean actual terminal cash was **197,938.93750**, which is **2,061.06250 below 200,000**. This is a failed author-validation threshold, not an economic qualification claim. The complete revised source is delivered despite that miss.

On the disclosed development panel, the selected revision increased mean terminal cash from 195,548.00000 to **206,398.59375**, a paired change of **+10,850.59375 (5.55%)**. It improved 20 cells and worsened 12. The unchanged parent subsequently scored **194,222.62500** on the exact same author-validation seeds. The frozen candidate's paired validation change was **+3,716.31250**. Neither the control nor the validation results were used to revise the selected source.

Every reported terminal result is from the frozen official Python interpreter after exactly 719 transitions, with both seats DONE and both rewards equal to their own terminal farm money. The supplied LocalGame host is not Kaggle's sandbox, timeout implementation, or schema validator; no platform submission is claimed. No partial-game cash, forecast value, leftover valuation, or replay-derived action is counted as terminal performance.

## Evidence and baseline identity

Input: `C07_ROUND02_FEEDBACK.zip`, SHA256 `495a9817c6b8d70a8e8b84b4a714e2467e5ec69fda6cbcb0380ab52cf8ef9883`. All 56 supplied MANIFEST entries passed verification before development and again after testing. `research/feedback/` is preserved unchanged. `research/baseline/` is the locally rebuilt parent. The local parent reproduced all 32 controller-reported terminal cash cells exactly.

`TASK.md`, `EVIDENCE.json`, all 32 results, both lowest-cash full replays and their matched parent replays were inspected. Both low-case replay pairs match in every recorded frame. At the final frame, those two cases have no positive crop/animal yield left, no shed goods, and empty worker inventories. Their late carried goods were deposited and sold before termination. This does not establish that every earlier route or sale was optimal, but it rules out an unsupported claim that visible terminal leftovers caused these two low scores. Daily cash differences alone were not treated as a causal diagnosis.

The supplied ZIP contains only the two selected full candidate replays and their matched parent replays, not all 32 original controller replays. This round generated new full local parent replays for all 32 disclosed cells. Previous C07 round-1 files and the original author-input archive are preserved as original ZIPs under `provenance/`; their historical limitations remain in their original reports.

## Implemented strategy

The root runtime remains the complete inherited A06 r6 revision1 economic planner, service/calendar logic, market executor, daily candidate selector, Python codec and native C++ simulator support. It is not a replacement component or a proposal. The economic settings in `policy/config.json` are byte-identical to the parent.

The selected C07 change is a **single observed-state economic refresh at hour 8 of each nonterminal day**. It is eligible only after initial preparation has finished (`core.phase == 3`) and while no paid intraday admission is pending. This is a bounded periodic checkpoint, not a measured prediction-error trigger.

At that checkpoint, the policy copies its current controller and reruns the inherited economic planner from the actual own farm, current cash, own held and stored goods, seeds, workers, public market and visible opponent farm. It retains the inherited funded-commitment bookkeeping. The dawn staffing estimator returns a total workforce, not an incremental hire count. To prevent an intraday refresh from purchasing that workforce a second time, the copied planner temporarily uses zero additional dawn hires, then restores the normal hand limit. Existing observed workers remain available. Ordinary later executor admissions still use their existing feasibility and financing checks.

The policy compares KEEP against the revised complete controller. Each branch uses the same observation, runs its actual remaining-day executor through the inherited own-state scenario, then adds the inherited conditional tail valuation to simulated day-end cash. A strictly positive finite difference is required; ties retain KEEP. Forecast values are never reported as observed terminal cash. The accepted controller, not merely its score, is installed and executed against the next real observations.

Recursive replanning is disabled within these comparison probes. There are at most 29 eligible checkpoints on a real game trajectory and two conditional continuations per eligible decision. The inherited daily candidate search also calls the same mechanism on its bounded candidate trajectories; the stated real-trajectory bound is not a claim that there are only 58 simulator calls across all daily candidate searches. New C07 refreshes are disabled on terminal day 29; the inherited terminal execution remains in place.

The ordinary compiler reserves actual inputs, uses current worker positions and remaining-hour route budgets, and rechecks real fills. This is a feasibility guard, not a proof that all planning forecasts are correct. Unknown new weeds and unknown opponent actions are absent from the own-day scenario, and the tail is still approximate. The policy never receives the actual game seed, future random state, opponent private inventory, or replay actions. Seeds exist only in host-side test declarations and records. The inherited scenario's fixed constructor seed is a dummy for a reconstructed conditional world, not the real episode seed.

The selected development games recorded **923 eligible refresh checks and 186 accepted controller changes**. Both originally supplied low-cash cells improved in development: seed 1681080428 seat 0 rose from 131,126 to 146,243; seed 1658182076 seat 0 rose from 155,189 to 221,201. This was not uniform: seed 702093297 lost 46,034 in each seat, for example. All cells, including regressions, remain in the means and records.

## Experiments and failures

Every complete development comparison used the same disclosed 16-seed, both-seat panel. The original paired-route prototype used only the two disclosed low-case seeds in both seats before its runtime repair. It is not a fresh panel and its three completed games do not form a valid four-game mean.

| Experiment | Completed / attempted | Mean terminal cash | Paired difference from parent |
|---|---:|---:|---:|
| Rebuilt unchanged parent, disclosed development | 32/32 | 195,548.00000 | +0.00000 |
| Five periodic route repairs per day | 32/32 | 193,648.37500 | -1,899.62500 |
| Frequent route repair, hours 2 through 20 | 32/32 | 195,548.00000 | +0.00000 |
| Inherited own-phase boundary feature enabled | 32/32 | 195,548.00000 | +0.00000 |
| Original paired-route prototype | 3/4 | not a full-panel mean | not used |
| Runtime-fixed paired-route prototype | 32/32 | 195,059.43750 | -488.56250 |
| SELECTED: economic refresh at hour 8 | 32/32 | 206,398.59375 | +10,850.59375 |
| Alternative: remaining-job recompile at hour 8 | 32/32 | 199,222.34375 | +3,674.34375 |
| FROZEN CANDIDATE: fixed author validation | 32/32 | 197,938.93750 | +3,716.31250 |
| Unchanged parent on the same author panel | 32/32 | 194,222.62500 | +0.00000 |
| Repeated source-only entrypoint/reset checks | 4/4 | repeated checks only | not used |

The five-checkpoint route-only repair was rejected because the full matched mean fell. Frequent repair and the inherited own-phase boundary feature produced no terminal cash improvement on any disclosed cell. Runtime-fixed paired route selection also fell below the parent. A complete remaining-job recompile improved development cash but remained below both 200,000 and the selected economic refresh. The final selection followed the recorded rule: highest complete disclosed-panel mean among failure-free buildable candidates, with replay verification, followed by one fixed validation panel.

Two build/game failures are preserved rather than omitted:

1. The first blocking parent build was interrupted by the execution tool's timeout and had no completed build receipt. Its interruption record and captured output files remain in `research/audit/initial_build_interruption.json` and associated logs. A separately logged rebuild succeeded.
2. The original paired-route prototype, seed 1681080428 seat 0, exceeded the host five-second action limit at step 288: **5.399563679 seconds**. Its 288-transition replay prefix, failure state, traceback, action timings and seed declaration are preserved. No terminal cash is asserted for that attempt. Independent replay confirmed all 288 recorded transitions and that the game was not done. A changed-plan guard removed redundant identical probes in the follow-up implementation. All three completed original prototype trajectories are byte-identical after decompression to their corresponding optimized trajectories. The final selected economic refresh uses a different, once-daily controller comparison.

A later container transport timeout interrupted the audit command's visible response, not the games or the completed data audit. The surviving files and running verifier were inspected; no experiments were restarted and no source changed. That recovery is recorded in `research/audit/audit_driver_transport_timeout.json`.

See `research/audit/all_attempts.csv` and `.json`, all experiment `panel_declaration.json`, per-game `.attempt.json`, `.result.json`, `.replay.jsonl.gz`, and debug files. The failed prototype remains part of the attempt inventory even though it cannot enter a terminal mean.

## Fixed validation protocol and results

Validation seeds were declared at **2026-09-13T19:10:16.833196+00:00**, before development results. Candidate executable source, settings and native binary were frozen at **2026-09-13T19:29:08.320566+00:00**. Validation began at **2026-09-13T19:29:09.097848+00:00** and ended at **2026-09-13T19:31:36.447311+00:00**. Source hashes remained unchanged throughout. No extra candidate-validation seeds were added, no failed sample was silently replaced, and no policy change was made after validation. The unchanged parent was run on those same seeds as a paired control, not as an extension of the candidate panel.

The seed selection excluded every recoverable known seed and every disclosed development seed. Lost pre-recovery round-1 histories cannot be excluded beyond the supplied known-used list; no claim is made to have recovered those missing histories.

| Predeclared seed | Seat 0 terminal cash | Seat 1 terminal cash | Seed mean |
|---:|---:|---:|---:|
| 1765915810 | 224,717 | 183,121 | 203,919.0 |
| 1061951453 | 137,220 | 144,761 | 140,990.5 |
| 686978169 | 235,532 | 227,229 | 231,380.5 |
| 1516688936 | 198,010 | 239,341 | 218,675.5 |
| 1316587864 | 227,961 | 255,410 | 241,685.5 |
| 1449278379 | 170,419 | 183,465 | 176,942.0 |
| 1878994638 | 175,793 | 169,214 | 172,503.5 |
| 364832534 | 200,595 | 199,647 | 200,121.0 |
| 566784632 | 151,248 | 151,248 | 151,248.0 |
| 775159170 | 224,304 | 212,918 | 218,611.0 |
| 13368487 | 181,967 | 170,958 | 176,462.5 |
| 1442673330 | 195,953 | 195,953 | 195,953.0 |
| 1407770778 | 182,877 | 182,877 | 182,877.0 |
| 711138023 | 226,520 | 226,207 | 226,363.5 |
| 1114964601 | 240,022 | 217,491 | 228,756.5 |
| 1025924508 | 198,950 | 202,118 | 200,534.0 |

Validation mean: **197,938.93750**. Range: **137,220 to 255,410**. Seat means: **198,255.50000** and **197,622.37500**. All 32 candidate games are included. These 16 seed pairs are not 32 independent random environments. Repeated smoke games and fixed-action replay re-executions are not extra validation samples. The controller's forthcoming fresh shared panel and competitive tests have not been run here.

## Build, resources and runtime receipts

Measured environment: CPU quota `400000 100000` (four CPU cores of quota), affinity `[0, 1, 2, 3, 4]`, memory limit **4 GiB**, `g++ (Debian 14.2.0-19) 14.2.0`, and `Python 3.13.5`. The affinity exposes five logical CPUs but does not raise the four-core quota.

The successful parent rebuild took **38.580 seconds**. Full-game timings include the official interpreter and full replay logging. The disclosed parent panel averaged **8.465 seconds/game**; the selected development panel averaged **18.831 seconds/game** under overlapping test workloads. Those are not isolated native-only benchmarks. The final two-worker validation averaged **8.941 seconds/game**, with a maximum measured action call of **1.148924 seconds**.

| Successful build receipt | Exit code | Wall seconds | Peak child RSS, KiB |
|---|---:|---:|---:|
| baseline_retry | 0 | 38.580 | 570,380 |
| econ_mpc | 0 | 38.856 | 571,452 |
| final_candidate | 0 | 36.590 | 578,728 |
| isolated_rebuild | 0 | 35.944 | 571,456 |
| recompile_mpc | 0 | 57.951 | 579,016 |
| roll1 | 0 | 51.395 | 570,356 |
| roll4 | 0 | 54.930 | 570,332 |
| roll_mpc | 0 | 61.440 | 579,024 |
| roll_mpc2 | 0 | 35.092 | 578,796 |

A separate source-only copy had all native binaries and generated native receipts removed, then rebuilt offline. Its binary is byte-identical to the selected development binary and delivered root binary:

```text
a8ab6e3c42b57c17f49598494cc337d9d59ed5caab69e4c0e9fc70144bec0a2e
```

That isolated rebuild was tested through the actual `main.agent` entrypoint in one Python process, both seats twice, without manually resetting between games. All four complete games reproduced every action and official frame from the selected development reference. Automatic step-0 state reset and explicit final `reset()` both passed. These repeated stress checks used already-disclosed seed 1658182076, selected because it had the slowest observed development action in each seat. Their mean full-game time, including replay comparison, was **8.964 seconds**, and their maximum action call was **0.477164 seconds**.

Across this round, **296 policy-game attempts** are recorded: **295 complete games** and **1 incomplete failed game**. The completed games contain **212,105 terminal-checked transitions**, plus **288 preserved failed-prefix transitions**. Independent frozen-engine replay re-execution passed every frame of all **295 completed games**; the failed prefix was checked separately. These totals include development, control and repeated checks and are not a larger qualification sample.

## Offline build and execution

From the extracted ZIP root:

```sh
python3 -B verify_package.py
python3 -B build.py --cxx g++
```

Run package verification before rebuilding: a rebuild legitimately updates generated build receipts and therefore changes their delivered-file hashes. The tested target is Linux x86-64 with C++20 and GCC 14.2. The Python runtime uses the standard library and ctypes. No external model weights, remote repository, network request or private project directory is required. The ZIP includes the compiled native runtime and every C++/Python source and vendored header needed for rebuilding. `build.py --unit` is not an advertised command: it is inherited but its historical unit-test directory was not supplied.

The interface is `main.py:agent(observation, configuration)`. For separate local game contexts, use `create_agent()` and call the returned object's `close()`. `reset()` releases the module's per-seat handles. Starting a new game at step 0 resets reused seat state automatically.

An optional local reproduction on a disclosed seed, without touching original records:

```sh
cd research
python3 -B run_panel.py --root candidate --tag reproduction_1658182076 --seeds 1658182076 --workers 1
python3 -B verify_replays.py --tag reproduction_1658182076 --workers 1
```

The runner refuses to overwrite an existing panel tag. This is a repeated development check, not fresh validation. Host scripts keep seeds separate from policy observations and configuration. Original audit receipts preserve their original absolute execution paths; the complete runtime at the ZIP root is portable independently of those historical paths.

## Contents and integrity

The root contains `main.py`, `build.py`, `policy/`, the native runtime, compiler flags, current build settings, source provenance, this report, build instructions, source hashes and a package verifier. `research/` contains the original frozen feedback, unchanged rebuilt baseline, every experiment source snapshot, all declarations/results/failures/replays, build receipts, checkpoint ZIPs, audit scripts and source-only rebuild. `provenance/` contains original round-1 deliverables and supplied archives without rewriting their bytes.

`SOURCE_HASHES.json` identifies the frozen production files. `research/audit/C07_SOURCE_CHANGES.patch` and `source_changes.json` describe changes against the rebuilt parent. The substantive C++ additions are in `policy/triad.hpp` and `policy/executor/policy.hpp`; the latter also retains inactive experimental gating for audit provenance. The final macros are `C07_ROLL_MODE=4` and `C07_ECON_HOUR=8`. The native filename remains the inherited `policy/a06.so`.

`MANIFEST.json` hashes every packaged file except itself. The separately delivered ZIP hash covers the manifest as well. Packaging verifies ZIP CRCs, every manifest entry, the root runtime's frozen hashes and the actual downloadable file. No source revision, test result or lost historical record is invented. The author-validation target remains missed, and no competitive acceptance claim is made.
