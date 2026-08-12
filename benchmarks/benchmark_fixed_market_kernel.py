"""Microbenchmark one fixed-market Triton launch without environment overhead."""

from __future__ import annotations

import argparse

import torch

from kaggriculture_lab.gpu_engine import CudaKaggricultureEnv, GpuEngineConfig, M_BUY_SEED
from kaggriculture_lab.triton_ops import run_fixed_market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=700)
    args = parser.parse_args()
    config = GpuEngineConfig(max_market_orders=2, starting_money=1_000_000_000)
    env = CudaKaggricultureEnv(args.envs, device="cuda", config=config)
    actions = env.empty_actions()
    actions.market_ops[:, :, 0] = M_BUY_SEED
    actions.market_quantities[:, :, 0] = 1

    for _ in range(10):
        run_fixed_market(env.state, actions, env, 0, M_BUY_SEED)
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(args.steps):
        run_fixed_market(env.state, actions, env, 0, M_BUY_SEED)
    finished.record()
    finished.synchronize()
    elapsed = started.elapsed_time(finished) / 1000
    print(f"envs={args.envs} steps={args.steps} seconds={elapsed:.6f} launches/s={args.steps / elapsed:,.1f}")


if __name__ == "__main__":
    main()
