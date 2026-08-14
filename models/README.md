# Local model archive

The `checkpoints/` directory is tracked with Git LFS. `manifest.json` records the
SHA-256 digest, byte size, parameter count, and checkpoint metadata for every
locally produced model included in this branch.

The current validated pure-neural champion is
`teacher_bc_nn_lookup_ar_dagger4_epoch2.pt`. It has no route prior or action
lookup table, canonicalizes public farm features into own/opponent order, and
decodes the ten ordered market slots autoregressively. It scored 106/220
(48.18%) in matched deterministic evaluation against the 11-agent frozen pool,
compared with 85/220 for the previous independent-slot model on the same games.
The paired evaluation reports are in `neural_champion/`.

The first conservative market-only PPO iteration retained the same deterministic
actions and score, so it was not promoted over the BC/DAgger checkpoint.

The current overall script-policy champion is stored in `league_champion/`. It
uses the frozen V17 production route plus a sparse, observation-gated market
preemption residual. Its 220-game official-CPU frozen-pool report is included
alongside the agent. It is retained only as the one-time Kaggle baseline and BC
teacher, not as a final learned-policy candidate.
