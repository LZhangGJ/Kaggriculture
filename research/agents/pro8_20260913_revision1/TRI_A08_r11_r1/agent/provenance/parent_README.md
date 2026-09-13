# A08 r11

A competitive economic agent with a finite sale-window dynamic program. It allocates only verified surplus already in its own warehouse across a short known demand window, compares observed opponent-supply scenarios, and replans the current sell quantity from each new observation. Existing maintenance reserves and execution commitments are preserved. The scenario comparison is finite and approximate; it does not assume access to hidden opponent inventory or future events.

## Latest verified local comparison

The frozen official Python 1.32.7 evaluator ran 20 previously unused environment seeds against 11 pinned public entries and original AFS R2, in both seats: 480 games per agent. All games completed 719 transitions with zero errors and zero draws. These are measured local results, not online leaderboard or Kaggle resource-limit certification.

| Scope | Wins / games | Win rate |
|---|---:|---:|
| All 12 opponents | 393/480 | 81.88% |
| Public 11 entries | 368/440 | 83.64% |
| Original AFS R2 | 25/40 | 62.50% |

Mean terminal cash: 106,542.79. Cash is a diagnostic; strict wins are determined by terminal cash being greater than the opponent's. This version did not exceed 85% overall on this panel. The earlier 22-opponent evaluation has a different scope and seed set.

## Run and rebuild

The callable entry is `main.py:agent(observation, configuration)`. Keep `main.py` and `policy/` together. The supplied native library targets Linux x86-64; use Linux or WSL. Reset both policies for every game and never expose the hidden environment seed to either policy.

Production Python/C++ sources and the original builder are included. To rebuild in a disposable copy with a C++20 compiler:

```sh
python3 -B build.py --cxx g++ --out /tmp/A08_r11_rebuilt.so
```

The authors' recorded compiler was GCC 14.2.0; other compiler versions are not promised to reproduce identical binary bytes. Builders write receipts/logs, so rebuild in a copy. `original_delivery.zip` preserves the full unmodified author delivery, including historical tests, fixtures, failure records and original documentation. Historical acceptance text inside that archive is dated evidence, not this release's current result.

The tested native SHA256 is `3b95f2c633242e31329adffae2438440a76b5eb6b11ece4897fe69f5710bb9cb`. See `PROVENANCE.json`, `BUILD.json` and the release-level `SHA256SUMS.txt` for exact identities.
