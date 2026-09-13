# A06 r6

A competitive economic agent that combines investment and execution planning with a finite sale-timing dynamic program. Its selected sale-risk mechanism compares waiting with and without estimated public opponent supply. It uses one fixed production configuration across games and preserves the original economic candidate generation and joint action selection.

## Latest verified local comparison

The frozen official Python 1.32.7 evaluator ran 20 previously unused environment seeds against 11 pinned public entries and original AFS R2, in both seats: 480 games per agent. All games completed 719 transitions with zero errors and zero draws. These are measured local results, not online leaderboard or Kaggle resource-limit certification.

| Scope | Wins / games | Win rate |
|---|---:|---:|
| All 12 opponents | 374/480 | 77.92% |
| Public 11 entries | 339/440 | 77.05% |
| Original AFS R2 | 35/40 | 87.50% |

Mean terminal cash: 102,915.96. Cash is a diagnostic; strict wins are determined by terminal cash being greater than the opponent's. This version did not exceed 85% overall on this panel. The earlier 22-opponent evaluation has a different scope and seed set.

## Run and rebuild

The callable entry is `main.py:agent(observation, configuration)`. Keep `main.py` and `policy/` together. The supplied native library targets Linux x86-64; use Linux or WSL. Reset both policies for every game and never expose the hidden environment seed to either policy.

Production Python/C++ sources and the original builder are included. To rebuild in a disposable copy with a C++20 compiler:

```sh
python3 -B build.py --cxx g++ --out /tmp/A06_r6_rebuilt.so
```

The authors' recorded compiler was GCC 14.2.0; other compiler versions are not promised to reproduce identical binary bytes. Builders write receipts/logs, so rebuild in a copy. `original_delivery.zip` preserves the full unmodified author delivery, including historical tests, fixtures, failure records and original documentation. Historical acceptance text inside that archive is dated evidence, not this release's current result.

The tested native SHA256 is `4119ae39b07c5fd2ef7f79d2f30ee3ee55f0dcd8980707d133a75b92e1140342`. See `PROVENANCE.json`, `BUILD.json` and the release-level `SHA256SUMS.txt` for exact identities.
