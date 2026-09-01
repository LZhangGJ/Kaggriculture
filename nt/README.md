# NT Kaggriculture research

This directory contains the NT branch of local Kaggriculture simulation and
Orbit Wars transfer research. It is intentionally isolated from the repository's
existing engines so the implementations can be compared without changing their
APIs or validation assumptions.

## Contents

- `latest_20260901_candidate8_width12_mcts/` — current Candidate8 offline
  search suite. It adds the eight-stage sequence search, Beam width 2/4/8/12
  sweep, C++ MCTS, R8 executor sources, exact replay checks, compact evidence,
  and a one-command runner. Hasegawa-specific analysis is intentionally
  excluded. Start with
  `latest_20260901_candidate8_width12_mcts/README_ZH.md`.
- `latest_20260829_candidate8_rolling_oracle/` — original 2026-08-29
  Candidate8 rolling-Oracle handoff. Use the 2026-09-01 package above for the
  current width=12 and MCTS implementation.
- `latest_20260827_macro_route_submit_agents/` — two submit-ready single-file
  route-tree agents trained from the deduplicated 2026-08-23/25 Top40 Replay
  pool. It contains the pipeline-selected G034 opening, the alternative G096
  forced opening, comparison metrics, and official 1.32.7 receipts. Start with
  `latest_20260827_macro_route_submit_agents/README_ZH.md`.
- `latest_20260827_complete_68_agent_pool/` — incremental completion layer for
  the full 68-Agent validation inventory. It adds the nine missing Rank40
  validation routers, three current-Top20 reconstruction banks/maps/receipts,
  and the authoritative complete-pool config and inventory. Start with
  `latest_20260827_complete_68_agent_pool/README_ZH.md`.
- `replay_collection/` — daily Top 60 live-leaderboard snapshot and Replay
  downloader. It freezes each team's currently scoring submission ID, then
  downloads every Public completed Replay currently exposed for that exact
  submission. Start with `replay_collection/agent.md`.
- `latest_20260825_front40_public_v48/` — Rank1–40 current-submission
  reconstruction snapshot plus the 34-notebook 2026-08-25 Recently Run scan,
  final route banks/maps/receipts, and the strictly parity-accepted Kaito V48
  JAX family. V48 beats frozen FC24B in 851/1,024 independent dual-seat games.
  Start with `latest_20260825_front40_public_v48/README_ZH.md`.
- `latest_20260823_fc24b/` — current FC24B handoff: the complete FC0-to-FC24B
  development-document chain, frozen source/configuration, six 2026-08-22
  public-agent JAX migrations, 8,704-game 34-Agent acceptance evidence,
  official-1.32.7 CPU/JAX parity traces, and Kaggle submission `55708153`.
  Start with `latest_20260823_fc24b/README_ZH.md`.
- `latest_20260821_arena_fusion_fc2a/` — reproducible 28-Agent strict-JAX
  Arena snapshot, the complete 37,800-game pairwise result, current
  rule-fusion development evidence, and the broken/fixed FC2A CPU submissions.
  Start with `latest_20260821_arena_fusion_fc2a/README.md`.
- `latest_20260819/` — latest official-1.32.7 development snapshot: the M3.9
  three-layer planner, current JAX rules core, five high-potential public
  agents reproduced as exact GPU opponents, acceptance receipts, and the
  planner-convergence design review. Start with
  `latest_20260819/README_ZH.md`.
- `gpu_sim/` — parity-first JAX GPU rewrite of the official
  `kaggle-environments==1.32.6` Kaggriculture interpreter, including source,
  tests, tools, frozen reference material, benchmark receipts, and the final
  acceptance report.
- `agents/eba26v2/` — exact EBA26v2 submission artifact, official 1.32.7
  holdout evidence, upload receipts, and the timestamped Public score.
- `handoff/gpt_route_bundle_1327_v2/` — verified 1.32.7 JAX route-search and
  official-referee bundle prepared for review on a four-core CPU machine.
- `orbit_wars/` — source-grounded summaries of all 11 publicly linked gold
  writeups, one Kaggriculture migration design per writeup, a shared modelling
  specification, and an RTX 3090 implementation comparison.
- `docs/` — detailed Chinese Kaggriculture rules and modelling reference used
  by both the simulator and the migration designs. The current end-to-end
  status is in `docs/KAGGRICULTURE_PROGRESS_SUMMARY_20260818_ZH.md`.

## Important boundaries

- `gpu_sim/reference/` is the immutable rule/parity oracle for this simulator.
- Do not train against a changed rules core until parity receipts have been
  regenerated.
- The local `.venv-wsl`, pytest caches, and Python bytecode are intentionally
  excluded from Git. Recreate the WSL environment using
  `gpu_sim/requirements-wsl.in` or the frozen lock file.
- Orbit Wars articles are stored as detailed Chinese research digests with
  official source links, not verbatim copies of third-party writeups.

Start with
[`latest_20260819/README_ZH.md`](latest_20260819/README_ZH.md),
[`docs/KAGGRICULTURE_COMPETITION_AND_GAME_RULES_ZH.md`](docs/KAGGRICULTURE_COMPETITION_AND_GAME_RULES_ZH.md),
[`docs/KAGGRICULTURE_PROGRESS_SUMMARY_20260818_ZH.md`](docs/KAGGRICULTURE_PROGRESS_SUMMARY_20260818_ZH.md),
[`gpu_sim/README.md`](gpu_sim/README.md), and
[`orbit_wars/README.md`](orbit_wars/README.md).
