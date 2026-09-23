# Paused native student plan seam

This experiment turns the synchronous v3 planner callback into a batched
request/response protocol without changing the planner or production agent.
Each borrowed R1 handle runs in one 2 MiB pthread stack. `collect_ready()`
returns active session indices and event arrays, and `apply()` wakes those
sessions after one batched model step.

```bash
bash experiments/native_student_rollout/build.sh
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_rollout/smoke.py

# Freeze Thomas/Meta x both-seat openings, replay them natively, then hand the
# live step-288 handles to the paused planner.
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_rollout/build_prefix_cache.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_rollout/smoke_prefix.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_rollout/smoke_suffix_oracle.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_rollout/scale_suffix_oracle.py
```

`PlanBatch.contexts` is `[B,2233]`. `collect_ready()` returns
`session_indices`, `stages`, `cells`, `suggested`, `legal_masks`, `sequences`,
`resources [N,347]`, and a per-session `terminal` vector. Handles are borrowed:
their Python owners must outlive the batch, and no other thread may touch them
until `join()` or `close()`.

`PrefixBatch` owns its R1 handles. It replays each cache from a seed reset,
recomputes and checks every native opponent action, warms `td_observe_external`,
and verifies the step-288 packed/context hashes. Keep it alive while a
`PlanBatch(..., prefix.handles, prefix.packed, activate_external=False)` uses
those handles.
