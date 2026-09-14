# A06 r6 revision 4

Adds delivery commitments to the inherited investment, labor-funding and rolling executor. Selected work is checked together with the physical delivery needed to realize its output, while resource and route constraints remain active. The earlier investment and maintenance funding repairs are retained.

Overall strict wins: **1280/1536 (83.33%)**. This revision **did not meet** the overall target of greater than 85%, which requires at least 1,306 wins. All 1,536 games completed 719 transitions, with zero errors and 0 draws.

| Scope | Wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1160/1408 | 82.39% |
| Original AFS R2 | 120/128 | 93.75% |
| Representative | 751/888 | 84.57% |
| Stress | 529/648 | 81.64% |

The shared panel has 37 representative and 27 stress seeds, sampled once from 128 remaining public source seeds after candidate identities were frozen. Both seats and all fixed 12 opponent entries are included. This is mixed development data, not sealed Holdout; comparisons to older panels are not paired estimates of a code change. Mean final cash is 107,692.17, a diagnostic with no acceptance gate. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision4/README.md).

The controller applied the author source delta to the exact parent and built the tested native with g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0. The original author native was not received. The archive here is that verified central reconstruction.

Keep the complete `agent/` directory together and call `main.py:agent(observation, configuration)` on Linux x86-64 or WSL. To rebuild offline, copy it to a disposable directory and run `python3 -B build.py` with Python 3 and a C++20 compiler. A different compiler need not produce identical binary bytes. This local panel does not certify Kaggle execution-time limits.

The archive [reconstructed_from_source.zip](reconstructed_from_source.zip) preserves all 254 payload files. The extracted R6 root `BUILD` text is named `BUILD.author.txt` to coexist with `build/` on Windows; archive bytes and production inputs are unchanged. Retained parent README, BUILD and historical manifests inside reconstructed packages describe earlier revisions; use this README, `RECONSTRUCTION.json`, source delta, current `policy/a06.BUILD.json` and central receipts for current provenance. Historical or author-side tests are not this full-pool acceptance result.

Tested native SHA256: `09d000103cbdfda138f38f99d0a9f9bdb8c061b4dae7dde22ae54f15a5c84065`. Published archive SHA256: `3c35737c7fe7abf34f9a7285c4c15f17d76e7ddf25eaed059d2622d4b6ef3d18`. Every extracted file was checked against the frozen archive, and every runtime file against the frozen panel. Central receipts record pre-panel checks, not acceptance.
