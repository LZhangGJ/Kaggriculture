# A06 r6 revision 1

A focused market-state correction for the inherited investment and sale-timing planner. Positive supply that crosses the price-floor boundary uses the existing per-unit execution path, so predicted stock stops consistently with the official market. Production configuration, candidate generation and the remaining executor are inherited unchanged.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and zero draws. Overall strict wins: **1196/1536 (77.86%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1078/1408 | 76.56% |
| Original AFS R2 | 118/128 | 92.19% |
| Representative seeds | 899/1152 | 78.04% |
| Stress seeds | 297/384 | 77.34% |

Mean final cash: 105,068.61; cash is diagnostic. The panel is mixed development data (48 representative and 16 stress seeds), not sealed Holdout. It differs from the original-agent baseline panel, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision1/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text and parent-only matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `174ba2ad749cb3213a72a03302ef50b08a37a579848ba922bde132a19b8b6c23`; original ZIP SHA256 is `38a1fc1863ecb03535f8ea79a4fa86409913d492b8fd8181811e11871e8e7677`. Production bytes were rechecked against the frozen evaluation, all 308 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
