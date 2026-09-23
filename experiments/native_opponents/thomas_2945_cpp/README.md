# Native Thomas 2945 opponent

Pure-C++ offline rollout port of `opponents/thomas_2945/main.py`. The runtime
does not import Python or parse JSON. Python is used only to generate the frozen
binary asset, expose a calibration binding, and run the authoritative Python
agent as the parity oracle.

## Runtime shape

This is not a mechanical translation of one apparent `agent()` function. The
Python file repeatedly wraps that symbol and ultimately executes this chain:

1. frozen replay router/tape executor and Chassis repair;
2. Thomas-family market and worker overlays through R85;
3. stateful V9 CARROT, V9 HERD, V9 FERT and public-sale predictor;
4. OVERFLOW, CARROT2, ORDERPRI2, CAPHARV, SHEDROOM, HERD2/COWSWAP.

The implementation reuses the MetaV4 binary tape loader, official native
simulator and common Thomas-family primitives in one translation unit. Thomas
adds only its ordered wrapper profile, profile constants and genuinely
different state machines. It does not copy a second simulator/runtime core.
`thomas_2945.assets.bin` contains 41 routes x 719 frames and all 64 predictor
shop-pair panels; its provenance is pinned by `assets.manifest.json`.

## Build and parity

```bash
/root/miniforge3/envs/torch-npu/bin/python generate_assets.py
bash build.sh
PYTHONPATH=../../.. /root/miniforge3/envs/torch-npu/bin/python check_parity.py \
  --seed 2609500000 --seed-count 4 --steps 719
```

The release/O3 build passes all 160 full-game runs in `parity_report.json`:

- calibration block: 4 seeds x both seats with a passive rival and active raw
  routes 0, 2 and 100, 32/32;
- disjoint holdout block 1: 8 seeds x both seats with a passive rival and
  active raw routes 0, 2 and 100, 64/64;
- disjoint holdout block 2: 8 seeds x both seats with a passive rival and
  active raw routes 9, 105 and 118, 64/64.

The holdouts were not ceremonial: they exposed and led to fixes for wrapper
order, duplicated ordering, HERD2 state/credit, sale-debt attribution, V9
CARROT state and FARMERS_MARKET strawberry public-flow handling. The component
reports retain every seed/seat result and all point to O3 build SHA-256
`40a1d5baec1b523fd934be8d1e42c642dd46496837ec17c6c9f0888426c063f4`.

The active raw routes are interaction fixtures, not claims that a replay tape
is a complete policy opponent. They exercise official board, market and public
belief feedback while the Thomas action is compared step by step with Python.
Parity normalizes only engine no-ops (zero-quantity/PASS market entries and an
implicit quantity of one); executable order position and quantity, every unit
action, and the final selected route must match.

## Native integration

```cpp
thomas_2945::Opponent opponent(asset_path);  // one per environment/thread
auto action = opponent.action(simulator, player);
opponent.reset();                            // before reusing for another game
```

`fixture_route_action()` is calibration-only. RL should call `action()` with
the actual shared `fastkag::Simulator`; there is no Python call or GIL in that
path. The pybind `benchmark_vs_route()` runs the whole game loop in C++, releases
the GIL and uses OpenMP with one mutable opponent instance per worker.

On the 192-core Kunpeng-920 host, O3 measurements for 3,840 complete 719-step
games give 1.19M--1.26M steps/s on the two warm repeats (1,651--1,751 games/s);
the first cold repeat, including 192 independent asset loads, gives 608k
steps/s. A 64-game single-thread run gives 8.77k steps/s. All repeated parallel
runs produced the same checksum and reward sums; raw measurements are in
`benchmark_report.json`.
