# Representative results

All 49,152 representative games are complete: 8,192 per candidate, using 256 seeds, 16 opponents and both seats. Zero invalid, missing or duplicate cells. Stress and holdout are still pending.

| Candidate | Wins | Draws | Losses | Strict win rate | 95% seed-cluster interval |
|---|---:|---:|---:|---:|---:|
| v37 feed reserve | 7,442 | 0 | 750 | 90.84% | 90.20%–91.49% |
| Day3 v2 | 6,869 | 0 | 1,323 | 83.85% | 81.10%–86.39% |
| Day9 | 6,860 | 0 | 1,332 | 83.74% | 81.07%–86.28% |
| Day3 v1 | 6,821 | 0 | 1,371 | 83.26% | 80.86%–85.53% |
| Terminal suffix | 6,333 | 144 | 1,715 | 77.31% | 74.63%–79.86% |
| AFS R2 | 6,238 | 288 | 1,666 | 76.15% | 73.36%–78.64% |

Draws contribute zero wins. Intervals use the frozen 4,000-draw bootstrap and keep all games for a seed together. These are local CPU results under the official 1.32.7 rules, not Kaggle leaderboard scores.

[Win rates by opponent](BY_OPPONENT.md) · [Opponent-by-seat tables](BY_OPPONENT_AND_SEAT.md) · [Exact counts, intervals and paired differences](RESULTS.json).

The v37 feed-reserve policy uses a public route-based parent; its score does not establish the team’s Three-Layer architecture goal. All six candidates remain frozen for the rest of this comparison. Optional terminal-suffix debug JSON errors are recorded separately; action and terminal checks passed.
