# Route Playbook V1

This experiment implements the end-to-end route discovery and stress-test
contract in:

`docs/route_exploration/ROUTE_PLAYBOOK_DISCOVERY_AND_STRESS_TEST_DESIGN_ZH.md`

Current status: `M0_CONTRACT_AND_ROUTE_SCHEMA`.

The implementation is isolated from the accepted official reference simulator
and previous strategic experiments. It reuses their public APIs but does not
overwrite their sources or receipts.

## Runtime

GPU runs use the accepted WSL environment:

```text
Ubuntu-24.04
gpu_sim/.venv-wsl/bin/python
JAX CUDA on NVIDIA GeForce RTX 3090
```

Python paths required by development commands:

```text
gpu_sim/src
experiments/strategic_v5/src
experiments/route_playbook_v1/src
```

No E0 search or later-stage match result is valid until its stage receipt has
status `PASS`.
