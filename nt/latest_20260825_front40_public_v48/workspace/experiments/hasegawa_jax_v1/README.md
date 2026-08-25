# Hasegawa JAX V1

This is a state-driven JAX reproduction of the current rank-1 Hasegawa policy,
compiled from 102 winning official 1.32.7 Public Replays.

It reproduces the confirmed opening transaction exactly, then executes a
high-level daily business calendar through the shared fulfillment kernel.  At
each day boundary an on-device router selects the closest winning calendar
using only currently visible shop demand, market price, cash, own production,
workers and unlocked land.  It never reads future shops and never stores or
plays raw Replay actions or coordinates.

The Arena entry point supports passive plus the local Exact5 JAX opponents in
both seats:

```powershell
wsl.exe -d Ubuntu-24.04 -- /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim/.venv-wsl/bin/python `
  /mnt/e/ai_coding/kaggle/kaggriculture/experiments/hasegawa_jax_v1/tools/run_hasegawa_jax_arena.py `
  --hasegawa-bank /mnt/e/ai_coding/kaggle/kaggriculture/experiments/hasegawa_jax_v1/artifacts/hasegawa_plan_bank_v1.npz `
  --exact-bank /mnt/e/ai_coding/kaggle/kaggriculture/experiments/expert_business_agent_v2/artifacts/high_potential_route_bank_v1.npz `
  --runtime /mnt/e/ai_coding/kaggle/kaggriculture/experiments/expert_business_agent_v2/artifacts/high_potential_runtime_tables_v1.npz `
  --opponents all5 --batch 256 `
  --output /mnt/e/ai_coding/kaggle/kaggriculture/experiments/hasegawa_jax_v1/receipts/exact5_arena_v1.json
```

V1 is a behavioral reproduction scaffold, not a claim of exact source-code
identity.  Full promotion requires official Python replay/alignment checks and
the paired Exact5 Arena receipt.
