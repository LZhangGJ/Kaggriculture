# A06 r6 revision 2

The proposal search retains a matched option to defer new land purchases when its original proposal set would otherwise always expand land. The alternative uses the current owned-land limit while preserving the other proposal settings and downstream execution pairing. It adds a choice to the inherited economic search without imposing a general ban on expansion.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1017/1536 (66.21%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 916/1408 | 65.06% |
| Original AFS R2 | 101/128 | 78.91% |
| Representative seeds | 678/1080 | 62.78% |
| Stress seeds | 339/456 | 74.34% |

Mean final cash: 97,487.42; cash is diagnostic. The panel is mixed development data (45 representative and 19 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision2/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `b10d86c102f68e5863559e6c049369571cbc123aa4db71db908a10c1ab14eab1`; original ZIP SHA256 is `76bbf478998777803188074ee5cf205f9c2d519767a52c65f33e389589a1beda`. Production bytes were rechecked against the frozen evaluation, all 452 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
