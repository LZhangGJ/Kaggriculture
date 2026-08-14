# Local model archive

The `checkpoints/` directory is tracked with Git LFS. `manifest.json` records the
SHA-256 digest, byte size, parameter count, and checkpoint metadata for every
locally produced model included in this branch.

The current validated learned-policy champion is
`teacher_bc_v2_v17_pool_ft1.pt`. PPO checkpoints are retained as experimental
history because neither the one- nor two-iteration candidate exceeded that BC
checkpoint in matched deterministic evaluation.

The current overall script-policy champion is stored in `league_champion/`. It
uses the frozen V17 production route plus a sparse, observation-gated market
preemption residual. Its 220-game official-CPU frozen-pool report is included
alongside the agent.
