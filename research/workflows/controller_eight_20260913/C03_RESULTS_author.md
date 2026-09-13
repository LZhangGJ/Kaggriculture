# C03 round 1 — completed results

**Local economic target met.** The frozen strategy averaged **207,077.02343750 terminal cash** over **128 distinct fresh seeds × both seats = 256 complete games** against the supplied legal PASS opponent. The policy was unchanged across the two panels.

## Results by panel

| Panel | Distinct seeds | Complete games | C03 mean terminal cash | Supplied parent mean |
|---|---:|---:|---:|---:|
| Development (used for selection) | 8 | 16 | 221,414.1250 | 189,721.3125 |
| First held-out panel | 64 | 128 | 208,535.828125 | 206,068.921875 |
| Additional robustness panel | 64 | 128 | 205,618.218750 | Not run |
| **Combined fresh C03 panels** | **128** | **256** | **207,077.02343750** | Not a paired 128-seed comparison |

Both seats were tested on every listed seed. The first panel was declared and the policy frozen before any held-out outcome. The second panel was declared after the first panel as an additional check of the same frozen policy. Neither panel selected a successor. Controller smoke seed **291307001** was excluded. The local smoke seed **291303101** and all development/reproduction repeats are labeled separately and are not counted as fresh evidence.

Fresh seat-zero mean was **207,898.125**; seat-one mean was **206,255.921875**. Individual terminal cash ranged from **124,519** to **275,461**. The target is a mean, not a claim that every game exceeds 200,000.

A descriptive seed-pair bootstrap gives a 95% interval of approximately **201,934–212,261** for the combined panel mean. It resamples whole two-seat seed pairs (10,000 draws) rather than pretending that the two seats are independent. This is descriptive uncertainty, not a sequential hypothesis test, guarantee, or substitute for the controller's independent panel.

## Parent comparison: modest out-of-sample gain

On the same first 64 seeds and both seats, the parent itself averaged **206,068.921875**, also above the local cash target. C03's observed mean improvement was **2,466.90625 cash (1.197%)**, much smaller than the development-panel improvement.

C03 earned more cash than the parent in **61 of 128** matched cases and less in **67**; there were no ties. Each policy separately faced PASS. These counts are **not head-to-head competitive wins**, and the mean difference is not presented as a proven competitive advantage.

## Full terminal and economic checks

All **256 fresh C03 games** completed exactly **719 transitions**, with **DONE for both players** and exact **reward/cash agreement**. External cash-flow accounting also reconciled every game:

`3000 + actual sale receipts − actual purchase costs − actual hiring fees − actual land costs = terminal cash`

No failed market commit and no non-PASS unit no-op were detected in these 256 games. The harness checked the action grammar, market-order count, current worker count and aggregate seed demand. The interpreter source was unchanged; auditing wrappers were outside the policy and never supplied private referee state to it.

The complete research log contains **1,054 full-game attempts across 94 experiments**, all complete, with **zero full-game errors**. This larger count includes repeated development seeds and comparisons and must not be confused with 256 independent fresh games. The final audit matched every STARTED record to a result, checked all **1,054 traces**, verified **757,826 consecutive transitions**, recomputed action hashes and checked final trace cash against terminal results. No trace was missing.

Earlier runs were not all instrumented for every economic/no-op metric. CSV and summary fields are blank/null when a measurement was absent; that means **not measured**, not zero. All fresh C03 runs had both economic and execution auditing enabled.

## Inventory cost was reduced selectively, not eliminated

There were no manual-deposit overflow losses in the fresh C03 panels. Automatic end-of-day overflow still destroyed **6,338 units**, or **24.7578 units per game**. Quantities are heterogeneous products, not a fabricated cash-loss estimate.

On the matched first panel, automatic lost units averaged **22.8281 for C03** versus **15.4844 for the parent**. Thus the full C03 configuration does **not** establish that overall warehouse loss is lower: larger production/harvest batches can outweigh the new selective detours. The selected configuration won the development cash comparison, not an overflow-minimization contest.

The new detours preserve feed/fertilizer obligations, have explicit route/deadline checks and use capacity-clipped partial deposits. The finite-water forecast includes late melon watering. Safe harvest batching reduces collection/travel but can increase cargo waves. `STRATEGY.md` explains these tradeoffs and the actual executor changes.

## Tests and rebuild

The final test suite passed **1,041 C++ assertions**, **2,321 comparisons with the frozen official integer-price rule**, and **12 entry/reset/privacy checks**. Tests cover price-floor stock behavior, holding liquidity/capacity, final liquidation, partial deposits, shared capacity, retained feed jobs, melon timing, batching exceptions, observation whitelisting and context reset.

A clean directory was populated from a source ZIP **without a native library**. `python3 -B build.py --cxx g++ --unit` succeeded and reproduced native SHA-256:

`394c8ec7fb358de66d9cd0a52721b774ff2050f23ee98a5c60a5080440354e44`

The rebuilt root entry completed both seats of development seed 1600081280 and exactly reproduced the previously recorded terminal cash and all 719-action hashes. The portable experiment-replay helper then reproduced those results again. These repeats are not fresh qualification games.

## Measured resources and timing

| Measurement | Result |
|---|---|
| CPU affinity | [0, 1, 2, 3, 4] (five logical CPUs visible) |
| Actual cgroup CPU quota | 4 CPUs (`400000 100000`) |
| Memory limit | 4 GiB |
| Compiler | g++ (Debian 14.2.0-19) 14.2.0 |
| Python | 3.13.5 |
| Parent rebuild | 44.932 seconds |
| C03 clean source rebuild | 45.695 seconds |
| Fresh C03 mean full-game elapsed time | 5.858 seconds |
| Slowest measured fresh policy call | 0.470 seconds |
| Maximum evaluator-process peak RSS, fresh C03 | 109,964 KiB |

Per-game time includes the Python referee and audit/trace work. Measurements came from this container, not the controller's quoted smoke timings. More detailed measurements and build receipts are in `TIME_AND_RESOURCES.json` and `reports/`.

## Negative results and failure records

Longer stock holding, fully integral forecast pricing, extra planning alternatives and multiple other settings were implemented and tested. Several promising small screens failed to improve the complete development panel. Multi-day holding and the additional candidate-cost grid are included but **disabled** in the delivered configuration.

Tooling/test failures were not hidden: an initial 20-second build call timed out; a streaming-execution request was unsupported; the first price-oracle test used the wrong function name; and an added test fixture initially used the wrong C++ container type. The supplied unit runner was missing and a prototype melon forecast omission was corrected. Failed outputs, the erroneous added test source, repairs and successful reruns are preserved in `research/logs/KNOWN_FAILURES.json` and the referenced files. No such tooling/test failure is labeled as a successful full game.

## Acceptance boundary

This delivery clears the requested **local PASS cash target only**. The supplied local host runs the frozen official interpreter but is not Kaggle's sandbox, full schema validator or timeout wrapper. The controller's fresh economic qualification, eight-author round robin and **1,536-game competitive acceptance test** were not run here. No >85% competitive win-rate claim is made; cash has no competitive acceptance threshold.

All sources, binary, offline build/test tools, source hashes, exact seed panels, per-game results, traces, failure records and historical checkpoints are in the ZIP. There was no delegation, external publication, seed-specific rule, use of actual future randomness, or use of opponent private state in the submitted policy.
