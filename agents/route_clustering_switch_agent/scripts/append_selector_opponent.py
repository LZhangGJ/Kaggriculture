#!/usr/bin/env python3
"""Append aligned opponent rows to a native selector matrix."""

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
        for key in ("openings", "targets", "checkpoints", "seeds"):
            if not np.array_equal(base[key], addition[key]):
                raise ValueError(f"unaligned selector matrices: {key}")
        payload = {
            "outcome": np.concatenate((base["outcome"], addition["outcome"]), axis=3),
            "margin": np.concatenate((base["margin"], addition["margin"]), axis=3),
            "states": np.concatenate((base["states"], addition["states"]), axis=2),
            "openings": base["openings"],
            "targets": base["targets"],
            "opponents": np.concatenate((base["opponents"], addition["opponents"])),
            "checkpoints": base["checkpoints"],
            "seeds": base["seeds"],
            "engine": np.asarray("C++ native pool plus dynamic tree agent"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"{args.output.resolve()} opponents={len(payload['opponents'])}")


if __name__ == "__main__":
    main()
