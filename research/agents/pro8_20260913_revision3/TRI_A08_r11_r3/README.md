# A08 r11 revision 3

The executor can transfer a complete group of resource-free tasks between workers to let a loaded worker visit the depot earlier. It checks travel, deadlines, carrying capacity and existing resource commitments before installing the two revised routes. It retains the animal-service labor correction and the native action/value build check. The subsequent experimental economic gate was retained for research and was not selected for this panel.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1087/1536 (70.77%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1000/1408 | 71.02% |
| Original AFS R2 | 87/128 | 67.97% |
| Representative seeds | 785/1104 | 71.11% |
| Stress seeds | 302/432 | 69.91% |

Mean final cash: 104,564.11; cash is diagnostic. The panel is mixed development data (46 representative and 18 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision3/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `49c98f2c1930cf741bdd9e2eb7da9af398117bb64f903f9375c7da9350e8fa64`; original ZIP SHA256 is `00d0afa3cc2f9c34acdaec50c313ca3acd36e22a82da1d8010c573c88c223e27`. Production bytes were rechecked against the frozen evaluation, all 1071 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
