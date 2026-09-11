#!/usr/bin/env python3
"""Screen every frozen native route against G001 with fixed seeds and both seats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _wilson(wins: float, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    denominator = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    setup_seconds = time.perf_counter() - setup_started
    g001 = bundle.index("G001")
    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int64)

    tasks: list[list[int]] = []
    owners: list[tuple[int, int]] = []
    for candidate in range(len(bundle.families)):
        if candidate == g001:
            continue
        for seed in seeds:
            tasks.append([candidate, g001, int(seed), -1, -1, -1, -1])
            owners.append((candidate, 0))
            tasks.append([g001, candidate, int(seed), -1, -1, -1, -1])
            owners.append((candidate, 1))

    simulation_started = time.perf_counter()
    rewards_raw, audit_raw = bundle.executor.play_audit_batch(np.asarray(tasks, dtype=np.int64))
    simulation_seconds = time.perf_counter() - simulation_started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    audit = np.asarray(audit_raw, dtype=np.int32)

    rows: list[dict[str, object]] = []
    for candidate, family in enumerate(bundle.families):
        if candidate == g001:
            continue
        indices = [i for i, owner in enumerate(owners) if owner[0] == candidate]
        candidate_reward = np.asarray([
            rewards[i, 0] if owners[i][1] == 0 else rewards[i, 1] for i in indices
        ])
        opponent_reward = np.asarray([
            rewards[i, 1] if owners[i][1] == 0 else rewards[i, 0] for i in indices
        ])
        delta = candidate_reward - opponent_reward
        wins = int(np.count_nonzero(delta > 0))
        ties = int(np.count_nonzero(delta == 0))
        losses = int(np.count_nonzero(delta < 0))
        score_wins = wins + 0.5 * ties
        lower, upper = _wilson(score_wins, len(indices))
        candidate_audit = np.asarray([
            audit[i, 0] if owners[i][1] == 0 else audit[i, 1] for i in indices
        ])
        rows.append({
            "family": family,
            "games": len(indices),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "score_rate": score_wins / len(indices),
            "wilson95_low": lower,
            "wilson95_high": upper,
            "mean_candidate_reward": float(candidate_reward.mean()),
            "mean_g001_reward": float(opponent_reward.mean()),
            "mean_margin": float(delta.mean()),
            "median_margin": float(np.median(delta)),
            "seat0_score": float(np.mean(delta[0::2] > 0) + 0.5 * np.mean(delta[0::2] == 0)),
            "seat1_score": float(np.mean(delta[1::2] > 0) + 0.5 * np.mean(delta[1::2] == 0)),
            "mean_unit_failures": float(candidate_audit[:, 0].mean()),
            "mean_market_failures": float(candidate_audit[:, 1].mean()),
            "first_macro_failure_min": int(candidate_audit[:, 2].min()),
        })

    rows.sort(key=lambda row: (-float(row["score_rate"]), -float(row["mean_margin"]), str(row["family"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_path = Path(native_extension.__file__).resolve()
    payload = {
        "schema_version": 1,
        "status": "complete",
        "engine": "C++ NativeTeammateExecutor.play_audit_batch",
        "target": "G001",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "pairing": "identical seeds, both seats",
        "route_count": len(bundle.families),
        "candidate_count": len(rows),
        "games": len(tasks),
        "setup_seconds": setup_seconds,
        "simulation_seconds": simulation_seconds,
        "games_per_second": len(tasks) / simulation_seconds,
        "inputs": {
            "source": {"path": str(args.source), "sha256": _sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": _sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
            "native_extension": {"path": str(native_path), "sha256": _sha256(native_path)},
        },
        "ranking": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "games": payload["games"],
        "simulation_seconds": simulation_seconds,
        "games_per_second": payload["games_per_second"],
        "top20": rows[:20],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
