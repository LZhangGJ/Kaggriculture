#!/usr/bin/env python3
"""Collect new exact-prefix targets against an existing opponent matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--extra-actions", type=Path, required=True)
    parser.add_argument("--families", type=_csv, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extra = load_action_tapes(args.extra_actions)
    with np.load(args.template, allow_pickle=False) as template:
        opening = str(template["openings"][0])
        opponents = template["opponents"].astype(str)
        seeds = template["seeds"].astype(np.int64)
        checkpoints = template["checkpoints"].astype(np.int16)
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata,
        additional_routes={family: extra[family] for family in args.families},
        included_families=(opening, *opponents),
    )
    result = bundle.executor.switch_search(
        [bundle.index(opening)],
        [bundle.index(family) for family in args.families],
        checkpoints.tolist(), seeds.tolist(),
        [bundle.index(family) for family in opponents],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        outcome=np.asarray(result["outcome"], dtype=np.uint8),
        margin=np.asarray(result["margin"], dtype=np.float32),
        states=np.asarray(result["states"], dtype=np.float32),
        openings=np.asarray([opening]),
        targets=np.asarray(args.families),
        opponents=opponents,
        checkpoints=checkpoints,
        seeds=seeds,
        engine=np.asarray("C++ NativeTeammateExecutor.switch_search extra targets"),
    )
    print(f"{args.output.resolve()} targets={len(args.families)} opponents={len(opponents)}")


if __name__ == "__main__":
    main()
