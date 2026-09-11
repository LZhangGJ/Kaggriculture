# P16 route and procurement repair candidate

This candidate derives from `latest_20260910_r2p16_route_recovery`, retaining its economic configuration. It changes execution coordination and optional-search budgeting, with no retraining or access to seeds, private opponent stock or future replay actions.

**Do not promote it to the competition baseline.** Execution checks improved, but the fresh panel against original JointAFS R2 shows a clear win-rate regression. This publication preserves the tested implementation and its negative findings.

## Changes

1. Keep recovery pending when required work is only partly scheduled; retry after actual cash, materials or workers increase. Continue a feasible route when state has not improved.
2. Remove the cash-below-prediction-only alarm. Check worker actions, sales and purchases in real order; true missing fills and resource shortages still trigger recovery.
3. Protect existing assets first, then try up to four ranked investment subsets. Validate accepted schedules against a state reconstructed from the observation. Keep a feasible solution when optional search stops.
4. Let unaffected workers continue during replenishment. Fund purchases and hires only from stock actually deliverable and saleable in this step. Require all recovery purchases, including animals, to fill.
5. Compare feasible recovery, deferral and incumbent plans using current-day results plus a common next-day value estimate. This bounded local search can misvalue investment timing.
6. Check whole-bag and quantitative deliveries against warehouse capacity in worker order. Retain goods that do not fit, retry actions and required future capacity.
7. Give optional search approximately 0.65 seconds of process CPU time per decision. Necessary action generation still runs; this is not an absolute one-second wall-clock guarantee.

## Use and rebuild

`main.py` loads the frozen Linux x86-64 GCC 13 library `build/revision4/route3.so`. Native Windows Python cannot load it. Use a separate `create_agent()` for each game and call `close()` afterwards. The runtime resets at a new step zero.

```python
import importlib.util
from pathlib import Path

entry = Path("main.py").resolve()
spec = importlib.util.spec_from_file_location("route_repair_entry", entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
player = module.create_agent()
# action = player(observation)
player.close()
```

Rebuild to a new destination; existing outputs are intentionally protected:

```bash
python3 -B build.py --unit --out /tmp/route_repair_check/route3.so
```

The recorded build passes 105 checks: 20 economics, 58 workflow and 27 recovery cases. Historical absolute paths in receipts are provenance; the build script resolves current paths relative to this package. `PACKAGE_MANIFEST.json` records shipped bytes.

| Source | Role |
|---|---|
| `policy/route_recovery.hpp` | Recovery triggering, retries, scheduling and funding |
| `policy/route_economics.hpp` | Route valuation, capacity, work commitments and next-day comparison |
| `policy/decision_budget.hpp` | Optional-search CPU budget |
| `policy/triad.hpp`, `policy/search.hpp` | Economic-controller integration |
| `tests/test_route_recovery.cpp` | Recovery contracts and inherited economics/workflow checks |

## Frozen evaluation

The three local programs played 3,150 live games using the existing compiled P16 C++ simulator. Original opponents were not rebuilt or replaced with replay tapes. All seeds were played from both seats.

| Panel | Complete workflow | Old recovery | This repair |
|---|---:|---:|---:|
| 13 historical opponents, 25 fresh seeds each | 421/650 (64.77%) | 417/650 (64.15%) | 440/650 (67.69%) |
| Original JointAFS R1, 100 seeds | 101/200 (50.50%) | 102/200 (51.00%) | 99/200 (49.50%) |
| Original JointAFS R2, same 100 seeds | 122/200 (61.00%) | 125/200 (62.50%) | 97/200 (48.50%) |

There were no draws. Historical and R1 paired difference intervals include zero. Against R2, repair is down 12.5 percentage points versus workflow (95% seed-cluster interval [-22.0, -3.0]) and 14.0 versus recovery ([-23.5, -4.5]). These are fixed-pool results, not leaderboard estimates.

All 4,536,000 bilateral observations and task metrics passed the existing transition checks; a separate script recounted terminal outcomes. The 100 formal seeds do not overlap 155 checked prior seeds or seven development seeds. Fourteen development regression games are excluded from formal rates.

The repair's 1,050 formal games have zero invalid actions and zero unexplained omissions of registered commitments, but **3,227 explicit task deferrals**. Registration differs by version, so count differences are not causal comparisons of identical task sets. The maximum recorded repair decision is 0.69209 seconds; none exceeds one second in this formal panel.

## Remaining problems

Against R2, the repair defers 1,891 tasks. Relative to workflow, mean wages increase by 44.38, own cash by 13.94 and opponent cash by 1,313.71; margin falls by 1,299.77. Cleaner execution accounting has not improved competitive choices.

Six selected large win-to-loss comparisons across four distinct seeds first diverge on day 8, step 169 after identical observations, with no optional-search cutoff at that first difference. Five defer 44 tasks; one changes same-step seed purchases and hires. These are selected diagnostics, not an estimate of all loss causes; full-game cash differences cannot be attributed to one order.

Ten serial timing reruns are excluded from win rates. One earlier repair outcome changes with runtime load. The original R2 archive also takes 2.660 seconds in a serial sample. The host records timing without Kaggle timeout forfeits.

Next work should compare staged fundraising that preserves investment, same-step replenishment and deferral over a common multi-day horizon, while retaining retry conditions for recoverable investments. No tuning or new tournament was performed during this publication.

## Identity and evidence

The user-supplied `submission (6).tar.gz` is the original R2 opponent, whose own rates against workflow/recovery/repair are 39.0%, 37.5% and 51.5%. Its association with submission `56149565` uses context and its entry label, not an independently authenticated official hash.

The separate `latest_20260911_afs_workflow_repair_r1_r2/r2` default library is a different, workflow-repaired build. It was **not** tested in this 100-seed panel and is not this local procurement-repair candidate.

- [English report and full limitations](../../evidence/r2p16_route_repair_20260911/REPORT.md)
- [Results](../../evidence/r2p16_route_repair_20260911/RESULTS.json)
- [Identity verification](../../evidence/r2p16_route_repair_20260911/IDENTITY.json)
- [Portable verifier](../../evidence/r2p16_route_repair_20260911/verify.py)

Full replays remain at `experiments/r2p16_route_repair_20260911` on the source workspace. The handoff publishes compressed per-game records, protocols, QA receipts, diagnostics and the original R2 runtime, excluding the multi-gigabyte replay collection and intermediate builds.
