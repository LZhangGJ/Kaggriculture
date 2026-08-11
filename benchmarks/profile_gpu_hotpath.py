"""Short torch.profiler trace for the CUDA transition hot path."""

from __future__ import annotations

import argparse

import torch

from kaggriculture_lab.gpu_engine import CudaKaggricultureEnv, GpuEngineConfig, U_DIG, U_EAST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--profile", choices=("move", "mixed"), default="move")
    args = parser.parse_args()

    config = GpuEngineConfig(use_triton=True)
    env = CudaKaggricultureEnv(args.envs, device="cuda", config=config)
    actions = env.empty_actions()
    actions.unit_ops.fill_(U_EAST)
    env.state.hands_count.fill_(config.max_hands)
    env.state.unit_active.fill_(True)
    if args.profile == "mixed":
        actions.unit_ops[:, :, 0] = U_DIG

    for _ in range(4):
        env.step(actions)
    env.reset()
    env.state.hands_count.fill_(config.max_hands)
    env.state.unit_active.fill_(True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=False,
    ) as prof:
        for _ in range(args.steps):
            env.step(actions)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
