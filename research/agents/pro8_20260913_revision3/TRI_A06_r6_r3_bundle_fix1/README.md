# A06 r6 revision 3, combined funding and land comparison, fix 1

The strategy compares complete investment plans with and without new land, prioritizes sale-funded planned labor after current animal feed, and can recover missing maintenance labor after actual receipts. The fix supplies fertilizer required by already selected services throughout the game, subject to the existing cash and order constraints. The separate focused-test launcher requires an explicit policy include directory; see BUILD_PORTABILITY.md for the original failure and successful central command.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1286/1536 (83.72%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1171/1408 | 83.17% |
| Original AFS R2 | 115/128 | 89.84% |
| Representative seeds | 917/1104 | 83.06% |
| Stress seeds | 369/432 | 85.42% |

Mean final cash: 104,705.78; cash is diagnostic. The panel is mixed development data (46 representative and 18 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision3/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.author.txt) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `6f1ea0c50035b5c33a313cf7e61e2f98f7da826fda52863d67d7c086301250de`; original ZIP SHA256 is `ed0723c90fba40808b597b042800e90dbf6e45a6b5d2b7918412267b1d479dd5`. Production bytes were rechecked against the frozen evaluation, all 243 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
