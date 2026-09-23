# Native 2965 Master Hybrid opponent

Offline-only C++ port of
`work/new_public_opponents/the-2965-master-hybrid-engine/output/main.py`.
It reuses the repository simulator and native Thomas-family replay executor;
production agent/configuration files are untouched.

```bash
/root/miniforge3/envs/torch-npu/bin/python generate_assets.py
bash build.sh
PYTHONPATH=../../.. /root/miniforge3/envs/torch-npu/bin/python check_parity.py \
  --seed 2609500000 --seed-count 4 --steps 719 --rival-route 0 \
  --output parity_active_report.json
```

`Opponent` in `metav4_2965.hpp` is the rollout API.  One instance owns one
episode and keeps independent mutable state for both seats.  The binary asset
contains all 41 replay tapes, all 64 public-shop predictor streams, routing
tables, and the one public rival-state override; rollout does not import or
parse Python.

The final `-O3 -DNDEBUG -march=native` build passes the fixed gates on seeds
`2609500000..2609500003`, both seats, and all 719 actions:

- `parity_active_report.json`: 8/8 exact while the rival executes route 0.
- `parity_report.json`: 8/8 exact against a passive rival.

It also passes `parity_active_holdout.json`: 32/32 exact on the disjoint seeds
`2609500100..2609500115`, both seats, with the rival executing route 0.

`batch_vs_route` is the trace-free pybind smoke/throughput entry point.  It
loads the immutable assets once, resets the stateful opponent between games,
and alternates seats.  The C++ rollout integration should call
`Opponent::action` directly and keep one `Opponent` per environment worker.
On this 192-core Kunpeng-920 host, one worker completed 128 full 719-step games
in 14.317 s (8.94 games/s, 155.6 us/step); see `throughput_report.json`.

Parity is semantic: ignored empty/zero market markers are normalized, while
unit commands and executable market order/quantity/list position remain exact.
