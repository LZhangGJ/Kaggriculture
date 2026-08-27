#!/usr/bin/env python3
"""Append aligned target columns to a native selector matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--addition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base, allow_pickle=False) as base, np.load(
        args.addition, allow_pickle=False
    ) as addition:
        for key in ("openings", "opponents", "checkpoints", "seeds"):
            if not np.array_equal(base[key], addition[key]):
                raise ValueError(f"unaligned selector matrices: {key}")
        if not np.array_equal(base["states"], addition["states"]):
            raise ValueError("target matrices do not share exact checkpoint states")
        base_targets = base["targets"].astype(str)
        addition_targets = addition["targets"].astype(str)
        new_indices = np.asarray([
            index for index, family in enumerate(addition_targets)
            if family not in set(base_targets)
        ], dtype=np.int64)
        payload = {
            "outcome": np.concatenate((
                base["outcome"], addition["outcome"][:, :, new_indices]
            ), axis=2),
            "margin": np.concatenate((
                base["margin"], addition["margin"][:, :, new_indices]
            ), axis=2),
            "states": base["states"],
            "openings": base["openings"],
            "targets": np.concatenate((base["targets"], addition["targets"][new_indices])),
            "opponents": base["opponents"],
            "checkpoints": base["checkpoints"],
            "seeds": base["seeds"],
            "engine": np.asarray("C++ native selector matrix with evolved targets"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"{args.output.resolve()} targets={len(payload['targets'])}")


if __name__ == "__main__":
    main()
