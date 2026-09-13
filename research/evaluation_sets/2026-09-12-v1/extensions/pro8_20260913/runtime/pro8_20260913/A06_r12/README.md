# A06 r12 — harvest-calendar version

A whole-portfolio finite harvest-calendar dynamic program for wheat, carrots and melons. It scores legal harvest-date labels using the same conditional portfolio value, accounting for existing investments, resource needs, labor and competing supply. Selected dates feed the inherited scheduling and execution layers. Optimality is limited to the finite date-label set; future labor, sales and investment opportunities remain conditional estimates.

## Latest verified local comparison

The frozen official Python 1.32.7 evaluator ran 20 previously unused environment seeds against 11 pinned public entries and original AFS R2, in both seats: 480 games per agent. All games completed 719 transitions with zero errors and zero draws. These are measured local results, not online leaderboard or Kaggle resource-limit certification.

| Scope | Wins / games | Win rate |
|---|---:|---:|
| All 12 opponents | 349/480 | 72.71% |
| Public 11 entries | 315/440 | 71.59% |
| Original AFS R2 | 34/40 | 85.00% |

Mean terminal cash: 102,194.93. Cash is a diagnostic; strict wins are determined by terminal cash being greater than the opponent's. This version did not exceed 85% overall on this panel. The earlier 22-opponent evaluation has a different scope and seed set.

## Run and rebuild

The callable entry is `main.py:agent(observation, configuration)`. Keep `main.py` and `policy/` together. The supplied native library targets Linux x86-64; use Linux or WSL. Reset both policies for every game and never expose the hidden environment seed to either policy.

Production Python/C++ sources and the original builder are included. To rebuild in a disposable copy with a C++20 compiler:

```sh
python3 -B build.py --cxx g++ --out /tmp/A06_r12_rebuilt.so
```

The authors' recorded compiler was GCC 14.2.0; other compiler versions are not promised to reproduce identical binary bytes. Builders write receipts/logs, so rebuild in a copy. `original_delivery.zip` preserves the full unmodified author delivery, including historical tests, fixtures, failure records and original documentation. Historical acceptance text inside that archive is dated evidence, not this release's current result.

The tested native SHA256 is `1400cdd7ef2677dc16b9c2b46a34a7ad41419d038c6e9bf5ebf60137707add3a`. See `PROVENANCE.json`, `BUILD.json` and the release-level `SHA256SUMS.txt` for exact identities.
