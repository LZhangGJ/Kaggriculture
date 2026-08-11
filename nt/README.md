# NT Kaggriculture research

This directory contains the NT branch of local Kaggriculture simulation and
Orbit Wars transfer research. It is intentionally isolated from the repository's
existing engines so the implementations can be compared without changing their
APIs or validation assumptions.

## Contents

- `gpu_sim/` — parity-first JAX GPU rewrite of the official
  `kaggle-environments==1.32.6` Kaggriculture interpreter, including source,
  tests, tools, frozen reference material, benchmark receipts, and the final
  acceptance report.
- `orbit_wars/` — source-grounded summaries of all 11 publicly linked gold
  writeups, one Kaggriculture migration design per writeup, a shared modelling
  specification, and an RTX 3090 implementation comparison.
- `docs/` — detailed Chinese Kaggriculture rules and modelling reference used
  by both the simulator and the migration designs.

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
[`docs/KAGGRICULTURE_COMPETITION_AND_GAME_RULES_ZH.md`](docs/KAGGRICULTURE_COMPETITION_AND_GAME_RULES_ZH.md),
[`gpu_sim/README.md`](gpu_sim/README.md), and
[`orbit_wars/README.md`](orbit_wars/README.md).
