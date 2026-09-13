# A06 r12 calendar revision 2

The executor protects an ongoing tomato or strawberry crop from a second unwatered-day death when future production and a feasible watering job remain. For qualifying sale-funded hires, it orders sales, current priority wheat-feed purchases, already planned hires, and the remaining admitted purchases so that planned labor can actually be funded. Ungated execution retains the inherited order; the change does not invent cash or add unplanned hires.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1087/1536 (70.77%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 969/1408 | 68.82% |
| Original AFS R2 | 118/128 | 92.19% |
| Representative seeds | 737/1080 | 68.24% |
| Stress seeds | 350/456 | 76.75% |

Mean final cash: 98,706.82; cash is diagnostic. The panel is mixed development data (45 representative and 19 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision2/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `638626de9398f0aed45e98394742ee22df875dd805f2647dee6d5666eb1c7755`; original ZIP SHA256 is `0bffa7ee3bdcf27c8683b4ae528c92adc587109ac2fb34f4e950d110a0ce3d87`. Production bytes were rechecked against the frozen evaluation, all 289 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
