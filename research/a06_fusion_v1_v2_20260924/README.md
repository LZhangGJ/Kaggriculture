# A06 Fusion V1 / V2

[中文说明](README_ZH.md) · [English](README_EN.md)

| Version | Entry point | Status |
|---|---|---|
| V1 | [v1/main.py](v1/main.py) | Retained baseline: `cf_liq_h12_nointraday` |
| V2 | [v2/main.py](v2/main.py) | Comparison candidate, not promoted: `v2_animal06` |

Both versions include unchanged evaluated Python/C++ sources, Linux x86-64 runtime, configuration, and build script. All 67 files in each version match the original frozen candidate. Neither version uses ML/RL.

Latest matched live-policy results: 100 shared environment seeds, both seats, 200 games per cell. All 800 games completed; zero errors, draws, opponent model fallbacks, or illegal model choices.

| Opponent | V1 wins | V2 wins |
|---|---|---|
| Teammate V306 | 119/200 (59.5%) | 123/200 (61.5%) |
| Teammate V463 | 110/200 (55.0%) | 123/200 (61.5%) |

The earlier internal/public dual-90% target was **not met**. Do not interpret V2's small teammate-panel advantage as evidence of broader superiority. Serial checks found calls above one second in both versions; the local benchmark did not impose timeout forfeits.

- [Latest full report](evidence/teammate/REPORT_ZH.md), [HTML](evidence/teammate/REPORT_ZH.html), [CSV](evidence/teammate/RESULTS.csv)
- [Earlier dual-panel decision](evidence/FINAL_REPORT_EN.md)
- [Exact version source hashes](VERSIONS.json)
- [Portable verification and reproduction](README_EN.md#reproduce)

Historical files inside `v1/`, `v2/`, and `evidence/` preserve upstream provenance. Their old `agent/` or `agent_v1/` path names refer to the previous local release; this handoff uses **v1/** and **v2/**. Follow this README for current status.
