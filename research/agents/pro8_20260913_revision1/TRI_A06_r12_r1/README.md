# A06 r12 calendar revision 1

A correction connecting the finite harvest calendar to planting admission. Legal late wheat and carrot plans that mature by day 29 can pass admission when their complete shared-calendar value is positive. Cash, seed, labor, routing, reserve and execution constraints remain active. The change does not force the earliest harvest date or guarantee that estimated future value will be realized.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and zero draws. Overall strict wins: **1099/1536 (71.55%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 984/1408 | 69.89% |
| Original AFS R2 | 115/128 | 89.84% |
| Representative seeds | 844/1152 | 73.26% |
| Stress seeds | 255/384 | 66.41% |

Mean final cash: 103,808.81; cash is diagnostic. The panel is mixed development data (48 representative and 16 stress seeds), not sealed Holdout. It differs from the original-agent baseline panel, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision1/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text and parent-only matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `d642c5650c916365136bfc5ce5c47f82240c27dc34d9baf691a580376b56ec0b`; original ZIP SHA256 is `eae4ca287d258e20fb4536a506c261bcb03f23a219c1b146e26128d5ceb24eba`. Production bytes were rechecked against the frozen evaluation, all 369 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
