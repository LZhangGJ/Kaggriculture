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

## Component repair shadow audit

The experimental component-scoped repair planner has a read-only G001
certificate/owner-coverage runner. It never applies its proposed manifest and
does not report reward metrics. Its conservative mapping boundary, fixed seed
panel, validation fields, and current negative coverage result are documented
in [COMPONENT_SHADOW_EVAL.md](COMPONENT_SHADOW_EVAL.md).

## Experimental selective market arms

`experimental_market_arm=0` is a hard bypass: it neither constructs nor calls
the public-belief adapter, selective input builder, or selective runtime. Its
submitted action remains the legacy action byte-for-byte.

Arms 1 and 2 use one observation/continuation state per seat. At step 0 the
adapter resets from the official public state plus that seat's own state. Every
returned full `PlayerAction` is staged, and the next call first derives a
conservative own-SELL fill interval by shed conservation. Only an
observation-confirmed interval for the pending option's product is supplied to
the runtime. Builder/runtime failure returns and stages exact legacy; no native
path executes `FullTakeover`.

The downstream 24-tick input uses the caller-selected raw G001 core tape as a
causal baseline (current exact final action plus 23 frames and an optional 24th
sentinel). It is rebuilt every tick. It is not a claim that future K320/R5/MD,
capacity, route or repair overlays are already known. A route switch inside the
window, enabled/active dynamic repair, a gap, or an invalid/unknown future
market queue fails closed. Only a certified typed market round-trip can replace
the current market vector; unit/MOVE actions are outside the replacement ABI.

The Python audit retains the old phased fields and adds observation failures,
builder rejection counts, runtime fallback classification, selected/changed
steps, and separate `runtime_total_us` (runtime only) versus
`native_overlay_total_us` (whole native seam) timing. `observation_us` and
`builder_us` isolate the two native-boundary costs. For bounded smoke/audit,
`NativeTeammateExecutor.play(..., stop_after_steps=N)` for positive `N` returns the current
money and accumulated audit after exactly `N` simulator steps; the default
`-1` remains a full episode and preserves all existing callers.
