#!/usr/bin/env python3
"""Replay frozen R8 joint traces in official kaggle-environments 1.32.7."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from kaggle_environments import make

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][index].get("values", {}))
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def canonical(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--opponents", nargs="+", default=["PASSIVE", "G001"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    config = (24, 8, 8, 4, 0.01)
    rows: list[dict[str, object]] = []
    for opponent_name in args.opponents:
        opponent = -1 if opponent_name.upper() == "PASSIVE" else bundle.index(opponent_name)
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for seat in (0, 1):
                native = bundle.adaptive_executor.play_r8(
                    genome, opponent, seed, seat, *config, True
                )
                trace = canonical(native["trace"])
                official = make(
                    "kaggriculture",
                    configuration={"episodeSteps": 720, "seed": seed},
                    debug=False,
                )
                state = official.reset(2)
                for joint_action in trace:
                    state = official.step(joint_action)
                official_rewards = [float(value.reward) for value in state]
                native_rewards = [float(value) for value in native["rewards"]]
                statuses = [str(value.status) for value in state]
                complete = len(trace) == 719 and statuses == ["DONE", "DONE"]
                reward_exact = official_rewards == native_rewards
                rows.append({
                    "opponent": opponent_name,
                    "seed": seed,
                    "candidate_seat": seat,
                    "trace_steps": len(trace),
                    "statuses": statuses,
                    "complete": complete,
                    "native_rewards": native_rewards,
                    "official_rewards": official_rewards,
                    "reward_exact": reward_exact,
                    "hard_failures": int(native["end_overflow"]) +
                        int(native["r8_unresolved_hard_day_tasks"]) +
                        int(native["r8_duplicate_reservation_violations"]),
                })
                print(
                    f"official {len(rows)}: {opponent_name} seed={seed} "
                    f"seat={seat} exact={reward_exact}", flush=True
                )

    passed = all(
        row["complete"] and row["reward_exact"] and
        row["hard_failures"] == 0 and
        all(math.isfinite(value) for value in row["official_rewards"])
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.r8-official-replay.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "official_package_version": version,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "dual_seat": True,
        "opponents": args.opponents,
        "games": len(rows),
        "summary": {
            "complete_games": sum(int(row["complete"]) for row in rows),
            "reward_exact_games": sum(int(row["reward_exact"]) for row in rows),
            "hard_failure_count": sum(int(row["hard_failures"]) for row in rows),
        },
        "config": {
            "day_horizon_steps": config[0],
            "joint_horizon_steps": config[1],
            "joint_candidate_limit": config[2],
            "feature_level": config[3],
            "lookahead_scale": config[4],
        },
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": sha256(args.backbone)
            },
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
