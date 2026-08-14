"""Scale selected checkpoint tensors while leaving the rest byte-equivalent."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = dict(checkpoint["model"])
    selected = []
    for name, tensor in state.items():
        if any(name.startswith(prefix) for prefix in args.prefix):
            if not tensor.is_floating_point():
                raise TypeError(f"selected tensor is not floating point: {name}")
            state[name] = tensor * args.scale
            selected.append(name)
    if not selected:
        parser.error("no checkpoint tensors matched the requested prefixes")

    output = dict(checkpoint)
    output["model"] = state
    output["parameter_scaling"] = {
        "source": str(args.checkpoint),
        "prefixes": args.prefix,
        "scale": args.scale,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved={args.output} scale={args.scale} tensors={len(selected)}")


if __name__ == "__main__":
    main()
