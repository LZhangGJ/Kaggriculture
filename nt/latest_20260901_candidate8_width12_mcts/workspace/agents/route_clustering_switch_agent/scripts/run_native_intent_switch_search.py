#!/usr/bin/env python3
"""Thin serializer for the compact all-C++ counterfactual switch search."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
import fast_kaggriculture._fast_kaggriculture as native_extension


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--openings", type=_csv, required=True)
    parser.add_argument("--targets", type=_csv)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    openings = args.openings
    targets = args.targets or bundle.families
    opening_indices = [bundle.index(value) for value in openings]
    target_indices = [bundle.index(value) for value in targets]
    setup_seconds = time.perf_counter() - setup_started

    started = time.perf_counter()
    result = bundle.executor.switch_search(
        opening_indices, target_indices, args.checkpoints, args.seeds
    )
    simulation_seconds = time.perf_counter() - started
    outcome = np.asarray(result["outcome"], dtype=np.uint8)
    margin = np.asarray(result["margin"], dtype=np.float32)
    states = np.asarray(result["states"], dtype=np.float32)
    games = int(outcome.size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        outcome=outcome,
        margin=margin,
        states=states,
        openings=np.asarray(openings),
        targets=np.asarray(targets),
        opponents=np.asarray(bundle.families),
        checkpoints=np.asarray(args.checkpoints, dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        engine=np.asarray("C++ NativeTeammateExecutor.switch_search"),
    )
    native_path = Path(native_extension.__file__).resolve()
    native_source = Path(__file__).resolve().parents[1] / "fast_kaggriculture/src/native_teammate.cpp"
    payload = {
        "schema_version": 2,
        "engine": "C++ NativeTeammateExecutor.switch_search",
        "layout": "opening, checkpoint, target, opponent, seed, seat",
        "openings": list(openings),
        "target_count": len(targets),
        "opponent_count": len(bundle.families),
        "checkpoints": list(args.checkpoints),
        "seed_count": len(args.seeds),
        "pairing": "identical seeds and both seats",
        "games": games,
        "state_vectors": int(np.prod(states.shape[:-1])),
        "feature_count": int(states.shape[-1]),
        "setup_seconds": setup_seconds,
        "simulation_seconds": simulation_seconds,
        "games_per_second": games / simulation_seconds,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "targets": list(targets),
        "seeds": list(map(int, args.seeds)),
        "inputs": {
            "source": {"path": str(args.source), "sha256": _sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": _sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
            "native_extension": {"path": str(native_path), "sha256": _sha256(native_path)},
            "native_source": {"path": str(native_source), "sha256": _sha256(native_source)},
        },
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
