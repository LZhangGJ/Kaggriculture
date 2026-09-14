# A08 r11 revision 4

A rolling route strategy with complete crop stops, worker task handoff, paid continuation, animal-service labor accounting and price-floor sale planning. This is the frozen r4 production build; the experimental cash-ledger and realization modes are disabled in its recorded compiler flags.

Overall strict wins: **1190/1536 (77.47%)**. This revision **did not meet** the overall target of greater than 85%, which requires at least 1,306 wins. All 1,536 games completed 719 transitions, with zero errors and 0 draws.

| Scope | Wins / games | Win rate |
|---|---:|---:|
| Public 11 entries | 1105/1408 | 78.48% |
| Original AFS R2 | 85/128 | 66.41% |
| Representative | 662/888 | 74.55% |
| Stress | 528/648 | 81.48% |

The shared panel has 37 representative and 27 stress seeds, sampled once from 128 remaining public source seeds after candidate identities were frozen. Both seats and all fixed 12 opponent entries are included. This is mixed development data, not sealed Holdout; comparisons to older panels are not paired estimates of a code change. Mean final cash is 105,827.87, a diagnostic with no acceptance gate. [Full report](../../../evaluation_runs/pro8_tri64_20260913_revision4/README.md).

The tested native and complete lightweight production archive were delivered by the author, built with g++ (Debian 14.2.0-19) 14.2.0.

Keep the complete `agent/` directory together and call `main.py:agent(observation, configuration)` on Linux x86-64 or WSL. To rebuild offline, copy it to a disposable directory and run `python3 -B build.py` with Python 3 and a C++20 compiler. A different compiler need not produce identical binary bytes. This local panel does not certify Kaggle execution-time limits.

The archive [author_delivery.zip](author_delivery.zip) preserves all 141 payload files. The extracted R6 root `BUILD` text is named `BUILD.author.txt` to coexist with `build/` on Windows; archive bytes and production inputs are unchanged. Retained parent README, BUILD and historical manifests inside reconstructed packages describe earlier revisions; use this README, `RECONSTRUCTION.json`, source delta, current `policy/a06.BUILD.json` and central receipts for current provenance. Historical or author-side tests are not this full-pool acceptance result.

Tested native SHA256: `acf904cb79039a80612d69437f26b358cec471e74400f1ac1275369466e9c27f`. Published archive SHA256: `12eff50cc55a4f5dfa31ad0789503b478ae6f11898b57b0a1f62c578da860cee`. Every extracted file was checked against the frozen archive, and every runtime file against the frozen panel. Central receipts record pre-panel checks, not acceptance.
