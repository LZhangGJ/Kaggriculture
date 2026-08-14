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

Autoregressive PPO now samples each market slot from the actual sampled prefix
and replays that prefix when recomputing log probabilities. Conservative PPO
candidates through a 5e-5 learning rate retained the same deterministic action
boundaries, while 2e-4 collapsed; no PPO checkpoint was promoted over DAgger4.

The current overall script-policy champion is stored in `league_champion/`. It
uses the frozen V17 production route plus a sparse, observation-gated market
preemption residual. Its 220-game official-CPU frozen-pool report is included
alongside the agent. It is retained only as the one-time Kaggle baseline and BC
teacher, not as a final learned-policy candidate.
