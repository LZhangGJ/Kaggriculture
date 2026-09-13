# A08 r11 revision 2, fix 1

The animal-service dynamic program accounts for collection labor on the days when animal yield and manure are actually collected. It selects the value and feed/care actions from the same explicitly evaluated legal candidate. A build-time check validates the actual compiled native before promotion; the delivered GCC 14 library and local GCC 13 rebuild passed the central action-consistency checks.

This exact revision completed 1,536 official local games: 64 fresh shared seeds, 11 fixed public entries plus original AFS R2, and both seats. All games completed 719 transitions with zero errors and 0 draws. Overall strict wins: **1097/1536 (71.42%)**. This version did not meet the requested overall target of greater than 85% (at least 1,306 wins).

| Scope | Strict wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1015/1408 | 72.09% |
| Original AFS R2 | 82/128 | 64.06% |
| Representative seeds | 740/1080 | 68.52% |
| Stress seeds | 357/456 | 78.29% |

Mean final cash: 102,636.31; cash is diagnostic. The panel is mixed development data (45 representative and 19 stress seeds), not sealed Holdout. It differs from previous evaluation panels, so the rate difference is not a paired estimate of the code change. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision2/README.md).

The runnable callable is `agent/main.py:agent(observation, configuration)`. Keep the entire `agent/` directory together. It contains every file in the unmodified author delivery, including production Python/C++ sources, fixed configuration, matching native, original build receipts, tests and raw logs. `original_delivery.zip` preserves the same complete delivery. Historical text, component checks and diagnostic matches inside the delivery are development records, not this public-pool result.

Run on compatible Linux x86-64 or WSL. For an offline rebuild, copy `agent/` to a disposable directory and run `python3 -B build.py --cxx g++`; see [the original build instructions](agent/BUILD.md) for flags and output options. The delivered compiler was GCC 14.2.0; a different compiler need not reproduce identical binary bytes. The local evaluation does not certify Kaggle execution-time limits.

The tested native SHA256 is `91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed`; original ZIP SHA256 is `4094b9808a6cd07f922e199fea4d2c47349f7aa14a7154178cd90959b4d402e3`. Production bytes were rechecked against the frozen evaluation, all 1575 extracted files matched the original ZIP, and the completed full64 result passed independent row/aggregate reconciliation. `CENTRAL_CHECK.json` records focused pre-panel source and behavior checks; it is not an acceptance certificate.
