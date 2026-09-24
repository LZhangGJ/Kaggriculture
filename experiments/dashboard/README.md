# Experiment dashboard

Pure-stdlib, read-only monitor for the current training pipeline. It exposes only:

- `/` — dashboard
- `/healthz` — health check
- `/api/status` — aggregate metrics from fixed repository paths

It never serves trajectory/private records, absolute paths, commands, or arbitrary files.

BC/DAgger metrics fail closed: the file must declare the action-event ABI-v3
contract and its manifest/schema hashes must match an `accepted` local shard.
Artifacts named by `work/student-v1/DO_NOT_USE*`, candidate-ranking metrics,
and old slot ABI metrics are ignored. PPO records are automatically discovered
from the fixed `work/student-v1/*.metrics.json` path when their algorithm
starts with `on_policy_clipped_ppo`; formal status additionally
requires an attested day-bundle leave-one-out PPO algorithm, a native C++
rollout scope, matching checkpoint/rollout hashes (including native `.npz`),
and attested native-opponent artifacts. Older PPO
smokes remain visible only as audit-only history.

```bash
/root/miniforge3/envs/torch-npu/bin/python experiments/dashboard/serve.py --self-check
/root/miniforge3/envs/torch-npu/bin/python experiments/dashboard/serve.py \
  --host 0.0.0.0 --port 39765 \
  --pid-file work/dashboard/dashboard.pid
```

The managed instance writes its PID and log to `work/dashboard/dashboard.pid`
and `work/dashboard/dashboard.log`. The host listens on all interfaces, but the
port 39765 is inside the host's existing UFW allow-range (39000--40000); no
system firewall or shared reverse-proxy configuration is changed by this tool.
