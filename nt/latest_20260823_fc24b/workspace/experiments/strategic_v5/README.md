# Kaggriculture Strategic V5

This directory contains the contracts, rule evidence, tests, and receipts for
the V5 full-farm strategic controller.

The accepted JAX simulator in `../../gpu_sim` already implements the official
crop, land, fertilizer, animal, market, shed, worker, and terminal mechanics.
V5 therefore extends the strategic task/candidate/ledger/executor layer without
rewriting that simulator.

Execution order:

1. E0 baseline freeze.
2. E1 official rule matrix and CPU/JAX differential coverage.
3. E2 Land-core and Fertilizer-core.
4. E3 Animal-core.
5. E4 unified Full-core controller and ledger.
6. E5 Full-econ simple and Full RULE_ONLY acceptance.
7. F physical-batch Full-core PPO performance and event-bank registry.
8. G zero-update, tiny update, and fair initialization screening.
9. H formal Current Self-play.
10. I CPU export and frozen public-Agent panel.

Additional accepted diagnostic work package:

- H1C exact Boatlee V16-RC2 GPU frozen opponent, official stepwise parity,
  batch-2048 benchmark, and one tiny fixed-opponent PPO experiment.  The GPU
  opponent is accepted; the PPO checkpoint is diagnostic-only because its
  fixed-evaluation effect was NEUTRAL.  See
  `reports/H1C_BOATLEE_V16_GPU_OPPONENT_ACCEPTANCE_ZH.md`.

No formal long-horizon PPO run is authorized before E5, F, and G pass.

Every completed work package must also have both a machine-readable receipt in
`receipts/` and a human-readable Chinese acceptance report in `reports/`.
Missing either artifact means that the work package is not closed.
