"""Benchmark CPU game transitions with CUDA-batched policy inference."""

from __future__ import annotations

import argparse
import time

import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.gpu_policy import KaggriculturePolicy, flatten_environment_observations, policy_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = KaggriculturePolicy(args.hidden_size).to(device).eval()
    environments = VectorFastEnv(args.envs)
    observations = environments.reset(range(args.envs))

    # Warm CUDA kernels and allocator before timing.
    flat = flatten_environment_observations(observations)
    policy_batch(model, flat, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    policy_seconds = 0.0
    env_seconds = 0.0
    started = time.perf_counter()
    for _ in range(args.steps):
        flat = flatten_environment_observations(observations)
        tick = time.perf_counter()
        batch = policy_batch(model, flat, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        policy_seconds += time.perf_counter() - tick
        paired_actions = [[batch.actions[2 * index], batch.actions[2 * index + 1]] for index in range(args.envs)]
        tick = time.perf_counter()
        results = environments.step(paired_actions)
        env_seconds += time.perf_counter() - tick
        observations = [result.observations for result in results]

    elapsed = time.perf_counter() - started
    joint_turns = args.envs * args.steps
    print(f"device={device} gpus={torch.cuda.device_count()} batch_players={args.envs * 2}")
    print(f"joint_turns={joint_turns} seconds={elapsed:.3f} joint_turns/s={joint_turns / elapsed:.1f}")
    print(f"policy_seconds={policy_seconds:.3f} env_seconds={env_seconds:.3f}")


if __name__ == "__main__":
    main()

