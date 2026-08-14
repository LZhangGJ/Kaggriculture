"""Linearly interpolate compatible PyTorch policy checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be in [0, 1]")

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    base_state = base["model"]
    candidate_state = candidate["model"]
    if base_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint state dictionaries have different keys")

    mixed_state = {}
    for name, base_tensor in base_state.items():
        candidate_tensor = candidate_state[name]
        if base_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"shape mismatch for {name}")
        if base_tensor.is_floating_point():
            mixed_state[name] = torch.lerp(base_tensor, candidate_tensor, args.alpha)
        else:
            mixed_state[name] = base_tensor.clone()

    output = dict(base)
    output["model"] = mixed_state
    output["interpolation"] = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "alpha": args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved={args.output} alpha={args.alpha}")


if __name__ == "__main__":
    main()
