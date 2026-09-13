# Stress results

All 24,576 stress games are complete: 4,096 per candidate, using 128 seeds, 16 opponents and both seats. Zero invalid, missing or duplicate cells. Holdout results are pending.

| Candidate | Wins | Draws | Losses | Strict win rate | 95% seed-cluster interval |
|---|---:|---:|---:|---:|---:|
| v37 feed reserve | 3,697 | 0 | 399 | 90.26% | 89.40%–91.11% |
| Day9 | 3,561 | 0 | 535 | 86.94% | 83.67%–89.94% |
| Day3 v1 | 3,561 | 0 | 535 | 86.94% | 83.86%–89.79% |
| Day3 v2 | 3,557 | 0 | 539 | 86.84% | 83.50%–89.92% |
| Terminal suffix | 3,338 | 86 | 672 | 81.49% | 78.12%–84.57% |
| AFS R2 | 3,250 | 162 | 684 | 79.35% | 75.90%–82.45% |

Draws count as zero wins. The unchanged analysis uses 4,000 paired bootstrap draws, resampling whole seeds. These are local CPU results under the official 1.32.7 rules.

v37 leads the aggregate, but its paired lead over Day9 is 3.32 percentage points with a 95% interval of −0.20 to +7.15 points. Against native AFS R2, v37 wins 58/256 games (22.66%); Day9 wins 243/256 (94.92%). Day9 and Day3 v1 each win 3,561 games, while Day3 v2 wins four fewer. The paired comparisons do not establish a gain over Day9 for either Day3 variant.

[Win rates by opponent](BY_OPPONENT.md) · [Opponent-by-seat tables](BY_OPPONENT_AND_SEAT.md) · [Exact counts and paired differences](RESULTS.json).

Stress covers 128 selected extremes and contrasting conditions from a 4,096-seed pool. Its win rate does not estimate the default seed distribution, and selected extremes need not be harder for every policy. Reference shop labels use PASS/PASS; actual shop and price paths depend on both policies. Sixteen opponent names include shared strategy families. No holdout outcome enters this report; all six policies remain frozen.

[Reference-condition results](REFERENCE_CONDITIONS.json) and [realized market summaries](REALIZED_MARKETS.json) retain the frozen analysis definitions. Realized market groups are descriptive because policy actions affect them.

The v37 policy uses the public route-based parent and does not establish the team’s Three-Layer architecture goal. Optional terminal-suffix debug JSON errors are recorded separately; action and terminal checks passed. No candidate is promoted by this report.
