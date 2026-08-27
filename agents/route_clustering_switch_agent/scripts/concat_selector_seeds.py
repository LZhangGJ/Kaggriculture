#!/usr/bin/env python3
"""Concatenate aligned selector matrices along their seed axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [np.load(path, allow_pickle=False) for path in args.input]
    try:
        first = loaded[0]
        for other in loaded[1:]:
            for key in ("openings", "targets", "opponents", "checkpoints"):
                if not np.array_equal(first[key], other[key]):
                    raise ValueError(f"unaligned selector matrices: {key}")
        seeds = np.concatenate([saved["seeds"] for saved in loaded])
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("seed matrices overlap")
        payload = {
            "outcome": np.concatenate([saved["outcome"] for saved in loaded], axis=4),
            "margin": np.concatenate([saved["margin"] for saved in loaded], axis=4),
            "states": np.concatenate([saved["states"] for saved in loaded], axis=3),
            "openings": first["openings"], "targets": first["targets"],
            "opponents": first["opponents"], "checkpoints": first["checkpoints"],
            "seeds": seeds,
            "engine": np.asarray("concatenated C++ selector seed cohorts"),
        }
    finally:
        for saved in loaded:
            saved.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"{args.output.resolve()} seeds={len(seeds)}")


if __name__ == "__main__":
    main()
