# A06 r12 calendar revision 3, fix 1

The calendar strategy funds selected ongoing maintenance when real stock is insufficient and checks that the additional services can actually execute. Within 48 transitions of the terminal state, it also compares complete continuation plans with and without the extra purchase under two public supply assumptions. Additional procurement must improve terminal own cash and the configured competitive cash objective; unsold inventory receives no terminal credit. Earlier maintenance, investment and calendar capabilities remain available.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1217/1536 (79.23%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1094/1408 | 77.70% |
| Original AFS R2 | 123/128 | 96.09% |
| Representative seeds | 908/1104 | 82.25% |
| Stress seeds | 309/432 | 71.53% |

Mean final cash: 104,778.15; cash is diagnostic. The panel is mixed development data (46 representative and 18 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision3/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `ed0d275ca9caf13fdaaf2c833e164c5b8c1ccf313dbf96296fb9f7896254ab90`; original ZIP SHA256 is `0c15b55c7cf7c29e57ad632432586a18f089df8a6f642d07d354d19c6a24f7c4`. Production bytes were rechecked against the frozen evaluation, all 339 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
