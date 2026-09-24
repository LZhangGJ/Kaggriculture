# Native replay pool

This is an offline RL-opponent pool, not a Kaggle deployment agent and not a
claim that a replay route equals its original author. It reuses the existing
`fastkag::NativeTeammateExecutor`: route JSON is decoded once, then every game,
action overlay and environment step stays in C++/OpenMP.

The pool deliberately runs with `neutral_special_economy=true`, switch columns
set to `-1`, and no Thomas prefix. A route is selected once at episode reset,
either by a fixed cycle or a deterministic seed hash; neither an opponent id nor
a replay-match fingerprint reaches the policy.

Retained from the mature native shell:

- per-game/per-seat `NativeAgentState`, action/hand alignment and the legacy
  weed transaction;
- late/day-end capacity guards, unusable late-seed suppression, feed-value
  guard, terminal crop salvage and liquidation;
- the official typed C++ simulator and market transitions.

Excluded:

- replay routing/switching and every Thomas shop/rival route selector;
- opponent-plan/front-run logic and Thomas prefix market overlays;
- K320/R5/MD/Moon/FC opponent-family and frozen-reference economic overlays;
- all experimental repair masks by default.

This is therefore a route-agnostic *execution shell*, not byte-for-byte raw
replay playback. It also does not include the many Thomas-specific wrappers
defined after the base `Chassis` (`V/R/CA/OR2/CH/HD2/CS`, courier/herd/race and
market forecast layers).

```bash
cd /root/kaggriculture-replay-switch-clean-v1
PYTHONPATH=.:fast_kaggriculture/python OMP_NUM_THREADS=192 \
  /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_opponents/replay_pool/smoke.py \
  --mode hash --games 1024 \
  --output work/native-replay-pool/smoke-all.json
```

Use `--max-source-failures 0` for the clean-source subset. The full 245-family
pool is useful for diversity, but `macro_failure_*` must be used as a quality
filter before assigning sampling weights; engine completion alone does not make
an invalid tape action valid. `--repair-mask` exposes the already-built native
repair bits for measurement; it defaults to `0` until an audit shows a bit
actually reduces failures on this pool.

## Measured smoke (96 CPU threads)

All runs used complete 719-step games, paired seeds, both seats and no route
switches. The reported failures are rejected macro unit/market orders, not
simulator crashes.

| pool | games | routes exercised | games/s | failure-free audited player-games |
|---|---:|---:|---:|---:|
| all 245 representatives | 512 | 214 | 9,309 | 34 / 512 |
| 68 source-clean routes | 512 | 68 | 13,136 | 208 / 512 |

On a separate 512-game clean-pool comparison, repair masks `1`, `2`, `4`, `7`,
`16`, `32`, and `64` did not improve failure-free coverage over mask `0` (the
animal retry slightly reduced unit failures but slowed simulation). The initial
RL pool therefore deliberately uses mask `0` and should filter/weight routes
from fresh multi-seed audits rather than silently changing their actions.

The current asset contains 245 representatives clustered from 609 replay
routes. New replay JSON should first go through the existing offline manifest,
macro extraction, clustering and carrier-export scripts; it is not parsed or
matched online by this pool.

## Incremental FastEnv API

`NativeReplayOpponent(executor, route, neutral_special_economy=True,
repair_mask=0)` owns one native `NativeAgentState`. Call `reset(route)` at
episode reset, then exactly once per turn:

```python
opponent.reset(route)
opponent_action = opponent.action(env, player, env.step_count)
joint = [None, None]
joint[player], joint[1 - player] = opponent_action, student_action
env.step(joint)
```

`action` rejects a stale/wrong step, so the C++ route state cannot silently
drift from the shared `FastEnv`. The full-pool index is the position in
`metadata["opponent_routes"]` (`0..244`). After filtering to the 68
source-clean entries, indices are compacted to `0..67`; retain the returned
`entries` list and use `entries[index]["route_id"]`/`["family"]` for logging.
The smoke runs the same external-student episode twice and requires identical
native action hashes and terminal rewards.
