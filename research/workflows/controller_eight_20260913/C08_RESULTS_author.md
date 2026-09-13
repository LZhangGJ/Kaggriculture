# C08 round 1 — measured report

Generated 2026-09-13T18:02:53.233416+00:00. All game cash figures below are actual terminal cash from the supplied frozen official interpreter, not forecasts or partial-game values.

## Result

**The local economic target was met on the predeclared held-out panel:** C08 averaged **201,529.23** terminal cash over **128 distinct seeds, both seats, 256 complete games**, against the legal PASS agent. The selected runtime was frozen at 2026-09-13T17:42:36.853957+00:00, before any held-out outcomes were available. No policy tuning followed those outcomes.

| Panel | Distinct seeds | Games | Mean terminal cash | Per-game range |
|---|---:|---:|---:|---:|
| Selected strategy — development | 104 | 208 | 203,935.58 | 123,882.00–265,097.00 |
| Frozen C08 — held out | 128 | 256 | 201,529.23 | 130,421.00–260,816.00 |
| Supplied parent — identical held-out panel | 128 | 256 | 200,480.57 | 111,868.00–268,225.00 |
| Unpacked rebuild — module API check | 8 | 16 | 202,206.81 | 131,040.00–261,408.00 |

The held-out mean changed by **1,048.65 cash (+0.52%)** relative to the supplied parent on the same seeds and seats. C08's seed-averaged cash was higher on 68 seeds, equal on 0, and lower on 60. This is a cash comparison between two PASS evaluations, **not** a head-to-head competitive win rate.

The development mean was used to choose among tested candidates and is selection-biased. The unpacked-module panel is a separate runtime check; it was not used for tuning. The four isolated speed tests reuse development seeds and are not counted as fresh evidence.

### Sampling uncertainty

Averaging both seats before calculating uncertainty gives a held-out standard error of 2,762.60 cash and an approximate 95% normal interval of **196,114.53–206,943.92**. The paired mean difference interval is approximately -4,353.17–6,450.48. It includes zero: this panel does not establish a reliable cash advantage over the parent. These are descriptive seed-level estimates, not a guarantee of the controller's fresh-panel result. Seats sharing a seed are not treated as independent samples.

Identical episode seeds do not guarantee identical later realized random shops under different strategies: board-dependent weed draws share the official RNG stream. The paired comparison still uses identical starting seeds, but no claim of identical exogenous trajectories is made.

## What changed

The supplied A06 r6 revision 1 strategy remains the documented ancestor. The complete executor and planner were retained, not replaced by a component-only proposal.

* Removed the unsupported day-zero mirror-opponent production prior while retaining public-rival forecasts and competition weight 2.
* Repaired daily and intraday maturity gates so legally productive short wheat/carrot cycles can reach the executor. Unfunded, seed-deficient, immature or unrouteable starts remain rejected.
* Changed early investment ranking: capital power 0.7, discount 0.08, animal ceiling 26, and direct adaptive planning instead of the outer rollout selector. The planner still replans at dawn and repairs work against live observations.
* Added feasible idle-cargo delivery on the final scored day and hardened context close/reset handling. The terminal-only change contributed about 14.88 cash per game on matched development tests; it is not presented as the main economic gain.

See `STRATEGY.md` for the full design and `evidence/FINAL_VS_PARENT.patch` for the exact seven-file source/configuration change. The source freeze, compiler flags, native hash and full configuration are preserved. The inherited `learned_value.hpp` ranker remains in the source but is not called by the released `scenario = 0` configuration.

## Terminal and execution checks

Every final held-out game completed all **719 transitions**, both statuses were **DONE**, and both rewards equaled their terminal cash. PASS finished with 3,000 cash in every scored game. None of these figures come from a planning simulator.

Across the entire recorded research session, **2,862 full-game attempts** are logged, of which **2,862 completed** and **0 failed**. This includes rejected variants and repeated controls; their pooled cash mean is deliberately not used as a performance claim. The final evidence audit passed all recorded plan/result/terminal/trace checks. It rehashed 388,260 recorded actions across 540 fully traced games. Earlier screens retain per-game action hashes and dawn records rather than complete action traces.

A separate audit probe checked **184,064 one-step projections** across the 256 final held-out games, and **327,864** frames across all audited research games. It matched known own physical state and market transitions against the official interpreter. Newly random midnight weeds and shops are excluded from that comparison. The probe does not send results or hidden information back to the policy.

The held-out action receipts recorded 0 aggregate seed-rejection conflicts and 0 units discarded by explicit DROP. Productive no-op receipts were `{}`. Residual automatic-deposit overflow totaled **4,508 units**, and terminal bags retained **413 units** across the panel. These losses are disclosed: the terminal repair does not make the whole inventory system optimal.

## Regression and numerical tests

The final source and the clean unpacked rebuild each passed **68 regression checks**: 33 C++ executor contracts, 16 official-kernel/pricing tests and 19 Python API/lifecycle tests. The pricing tests include 21,789 native/official unit-price comparisons and 484 sequential-trade cases, with no mismatches. Unit tests use synthetic states and are not counted as full scored games.

The checked primitive price/trade functions are exact on the tested cases. The selected economic objective still uses the inherited three-point aggregate-flow approximation where applicable. An exact-integral forecast variant was tested, averaged less cash on the same 104 development seeds, and was rejected. Forecasts of unknown shops and rival production remain conditional estimates rather than privileged knowledge.

## Resources and timing

Measurements were made in this execution environment, not copied from the controller's smoke test.

| Measurement | Observed value |
|---|---|
| CPU quota | 4 cores; `cpu.max = 400000 100000` |
| CPU affinity | 5 logical CPUs: 0–4 |
| Memory limit | 4,294,967,296 bytes (4 GiB) |
| Python / compiler | 3.13.5 / g++ 14.2.0 |
| Platform | Linux x86-64; glibc 2.41 |
| Initial successful parent build | 37.764 s; peak compiler child RSS 570,384 KiB |
| Final selected build | 37.784 s; peak compiler child RSS 570,980 KiB |
| Clean unpacked rebuild | 33.255 s; peak compiler child RSS 570,980 KiB |
| Four isolated C08 full games | 1.553–1.806 s each |
| Isolated maximum single-action latency | 0.009550 s |
| Maximum isolated game-process RSS | 101,676 KiB |

Four-process audited panel timings include process contention, per-action receipt checks, serialization and projection comparisons. They must not be interpreted as isolated policy speed. Each game's actual wall time, CPU time, action time and process high-water RSS are in its result row. RSS is the process lifetime high-water mark and may include earlier games in a reused worker.

The included `.so` was built for the measured Linux host and uses standard system C/C++ libraries. Rebuild offline on the destination host; no package download or network is required. The clean unpacked build produced the exact frozen native SHA-256:

`1f1c3f1c9b29da41c42ed4424d1fc9fe86e10a07695da004bd58c292bca00436`

## Reproducibility and failure record

All 71 entries in the input MANIFEST matched their hashes. The original archive SHA-256 is:

`75356640e384979f504fa7393796ab3c4e8dd22b051d5284599d7e4266b2e76d`

Two early foreground build attempts were interrupted by the command runner before a build receipt was written. Their actual durations are unknown and marked as such in `evidence/failures.jsonl`. The subsequent measured parent build succeeded. They are build interruptions, not omitted scored games. Game and audit failures would also be retained rather than silently retried or deleted.

The runtime was cleanly rebuilt from an unpacked preflight ZIP, then the exact `main.agent(observation, configuration)` module entry point completed 16 additional games on eight separately reserved seeds. Its source and rebuilt native matched the frozen candidate. The final archive adds reports and complete research evidence without changing that tested runtime. Package-level manifests verify the delivered bytes; build receipts naturally change when a user rebuilds.

### Seed ledger

| Role | Seeds | Seats |
|---|---|---|
| Screen development | 88091301–88091308 | Both |
| Extended development | 88091401–88091432 | Both |
| Tertiary development | 88091433–88091496 | Both |
| Frozen holdout | 88091501–88091564 and 88091701–88091764 | Both |
| Unpacked-module validation | 88091601–88091608 | Both |
| Previously used controller smoke | 291307001 | Recorded separately; excluded from fresh claims |

`evidence/ALL_GAMES.csv` is the unfiltered attempt index. Every panel's `PLAN.json` records the exact source hashes, seed list, seats and options before execution. The corresponding per-game result, event log and dawn record are included. Episode seeds go only to the referee, never the policy. The final configuration contains no seed-specific exceptions or replay action lookup.

## Independent design findings and rejected work

The exercise cross-checked inherited assumptions rather than assuming more detailed forecasts always improve cash. Source inspection and official parity checks confirmed that the single-product shop's doubled consumption, floor-specific stock accounting and real labor-price schedule were genuine game mechanics.

Changing labor cost assumptions downward and extending the opening rollout did not improve the selected screens. Raising animal capacity without matching capital preference was not reliably beneficial. Removing every observed overflow with mandatory hauling worsened the eight-seed mean to 200,463.69 from the capacity-26 control's 216,327.94 despite eliminating overflow; a larger safety buffer worsened it further. Several earlier warehouse forecast switches produced no action change. These failed and no-effect experiments remain available under `experiments/` and `evidence/`, with their actual source/configuration and results.

The final configuration was selected using 104 development seeds, not only the favorable eight-seed screen. Selection and every comparison are recorded in `evidence/SELECTION_DECISION.json`. The held-out panel was declared before the runtime freeze and no final-data-driven policy revision was made.

## What this does not establish

This is the requested local economic qualification evidence and a complete runnable submission, not a proof of optimality. It does not run or predict the controller's separate fresh shared panel. No author round robin and no 64-seed × 12-opponent × 2-seat competitive acceptance panel was executed. No claim of 1,306 strict wins, >85% acceptance, or improvement over the supplied historical competitive record is made. Those require actual competitive games; PASS cash alone cannot establish them.
