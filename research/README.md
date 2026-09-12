# Kaggriculture research

## Shared evaluation tools — 2026-09-12

Use [evaluation tools v2](evaluation_tools/v2/README.md) for new comparisons with strict input checks, a four-seed parity preflight, separate economic stress opponents and paired analysis. These tools use the pinned 1.32.7 runtime in the [shared evaluation bundle](evaluation_sets/2026-09-12-v1/README.md). The original bundle and its frozen six-candidate comparison remain unchanged. The older engine notes below describe the August research state.

## Live competition survey (2026-08-11)

- Competition window: 2026-07-29 through 2026-09-30; entry/team-merger
  deadline 2026-09-23. Final episodes continue for about two weeks, followed by
  one Bradley-Terry tournament.
- Only the newest two submissions remain active; the leaderboard shows the best
  of them. Ranking uses win/loss/tie, not final coin margin.
- Runtime resources are tight (6.5 GiB RAM, 1.6 vCPU, 100 MiB submission), so a
  large training network may be used offline but the submitted policy must be
  distilled or otherwise sized for the evaluator.
- Official balance update: town-center demand is now one unit/product/day with no
  later multiplier, and shops are sampled with replacement. Staff explicitly
  requires `kaggle-environments>=1.32.6`.
- The official daily episode dataset publishes up to 20 GB/day, selected by the
  average contemporaneous rating of both agents. Its index is the correct entry
  point for replay BC/IL; do not scrape the episode viewer at scale.
- Current public meta emphasizes action efficiency, compact land use, market dump
  timing, strict-future evaluation, and held-out seeds. Public high-score notebooks
  are already clustered around shared openings, so a replay-cloned public policy is
  a warm start rather than a durable edge.

High-signal live sources:

- https://www.kaggle.com/competitions/kaggriculture/discussion/733431
- https://www.kaggle.com/competitions/kaggriculture/discussion/731215
- https://www.kaggle.com/competitions/kaggriculture/discussion/731587
- https://www.kaggle.com/competitions/kaggriculture/discussion/734033
- https://www.kaggle.com/competitions/kaggriculture/discussion/734308
- https://www.kaggle.com/code/boatlee/v16-rc2-high-score-near-mirror-market-relay
- https://www.kaggle.com/code/raykkretzschmar/kaggriculture-rank-your-agent
- https://www.kaggle.com/code/cjlcjlcjl/kaggriculture-what-the-top-farms-do-a-live-meta
- https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons
- https://www.kaggle.com/code/kaitofukami/15-16-strict-future-v25-meta-reset

## Sources captured locally

- Official competition `README.md` and `AGENTS.md`: `../official/competition/`
- Official engine source at Kaggle/kaggle-environments commit
  `bded87b0d7879078c726a93a4884d044f79c4eed`: `../vendor/kaggle-environments/`
- Eight selected public notebooks: `notebooks/`
- Daily top-episode index manifest: `episodes-index/manifest.csv`

The selected notebooks cover the official quick start, a fixed reference ladder,
held-out local tuning, benchmark failure modes, replay mining, top-episode downloads,
economic policy, and public-meta analysis.

## Findings that affect the training system

1. Pin `kaggle-environments==1.32.6`. The balance update reduced town demand and
   changed shop sampling to sampling with replacement. Earlier versions simulate a
   materially different economy.
2. Verify engine behaviour, not only the package version string. Public notebooks
   document stale-import cases where metadata and imported code disagree.
3. Play both seats. Market orders are processed in player order, creating measurable
   seat asymmetry.
4. Use fixed tuning seeds and a disjoint held-out seed set. Several public local wins
   reverse on held-out seeds.
5. Treat a seed's two seat-swapped games as one statistical block.
6. Replay actions are shifted: the action used for observation step `t` is stored at
   replay `steps[t + 1]`.
7. Kaggle limits episode views to 3,600 per 24 hours. Prefer the daily top-episodes
   dataset for imitation learning, behaviour cloning, and meta statistics.
8. The final leaderboard uses a single Bradley–Terry tournament after an additional
   two-week episode window. Local evaluation should therefore use diverse opponents,
   both seats, and a Bradley–Terry-style league rather than raw bank alone.
9. The current replay index covers 12 days, 9,104 top episodes, and 232.5 GiB of raw
   replay data. Download it selectively; the manifest is a routing index, not a reason
   to mirror the entire corpus before the training objective is fixed.

## Performance conclusion

Profiling a 720-turn `starter` vs `starter` game shows that the official framework's
per-step schema validation, deep copies, and `Struct` conversion dominate runtime.
The actual game interpreter is branch-heavy Python over small dictionaries and a
10×10 board. Moving that transition directly to a GPU would add transfer and kernel
launch overhead.

The fast CPU runner calls the official 1.32.6 interpreter directly and omits
evaluation-only framework work. For the requested maximum-throughput RL workload,
the engine is additionally rewritten as fixed-shape structure-of-arrays tensors in
`gpu_engine.py`; this changes the data model rather than attempting to CUDA-compile
the dictionary interpreter. The CPU runner remains the differential-test oracle.

## Measured throughput

Hardware: AMD Ryzen 7 3800X (8 cores / 16 threads), 64 GiB RAM, NVIDIA RTX 5070 Ti
(16 GiB). Agents: built-in `starter` vs `starter`, 720-turn episodes.

| Runner | Episodes | Seconds | Episodes/s | Relative to official |
| --- | ---: | ---: | ---: | ---: |
| Official framework | 1 | 1.954 | 0.51 | 1.0× |
| Fast runner, serial | 100 | 8.131 | 12.30 | 24.0× |
| Fast runner, 8 processes | 1,024 | 17.887 | 57.25 | 111.9× |

The 8-process result is about 41,200 joint environment turns per second and 206,000
complete games per hour. Complex policies reduce those figures in proportion to their
own inference cost.
