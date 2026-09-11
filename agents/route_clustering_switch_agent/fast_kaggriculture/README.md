# Fast Kaggriculture environment

Typed C++20/pybind11 implementation of the `kaggle-environments 1.32.7`
Kaggriculture interpreter. Licensed under Apache-2.0.

The official environment remains the authority for evaluation. This simulator
must only replace it in training after differential tests pass for the exact
configuration in use.

## Build and verify

```bash
cd fast_kaggriculture
/root/miniforge3/envs/torch-npu/bin/python setup.py build_ext --inplace
PYTHONPATH=python /root/miniforge3/envs/torch-npu/bin/python -m pytest -q tests
```

## APIs

- `FastEnv.step(actions)`: string-compatible debug API returning both observations.
- `FastEnv.step_raw(actions)`: C++ state update without constructing observations.
- `FastEnv.step_packed_raw(...)`: integer action hot path for a single environment.
- `FastBatchEnv.step_packed(...)`: OpenMP-parallel `[B,2,N,3]` integer action path.
- `observation(...)`: materialize a Python observation only when the policy needs it.

Integer action tuples are `(Op, Item, quantity)`. Unit and market action counts
are passed separately, so unused padded rows do not become actions.

## Fidelity coverage

`tests/test_differential.py` compares both players after every official step and
covers:

- initialization and terminal rewards;
- unit actions and atomic multi-unit PLANT validation;
- per-unit two-player lockstep market pricing and commits;
- town demand and dynamic prices;
- HIRE and BUY_LAND;
- daily crop/animal refresh, inventory drop, weeds and shop RNG;
- full 719-action episodes and seeded mixed-action fuzz traces.

Current limitation: sparse `marketParams` configuration overrides are not yet
implemented. Default competition parameters are covered. The batch API exposes
full observations on demand; a compact feature-only view should be added when
integrating the tree-Q selector to avoid Python object creation at every turn.
