#!/usr/bin/env python3
"""Audit native G001 as a complete-plan arm and combine it with Candidate8."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def outcome(own: float, opponent: float) -> float:
    return 1.0 if own > opponent else (0.5 if own == opponent else 0.0)


def summary(rows: list[dict], prefix: str) -> dict:
    own = np.asarray([row[f"{prefix}_cash"] for row in rows], dtype=np.float64)
    rival = np.asarray(
        [row[f"{prefix}_opponent_cash"] for row in rows], dtype=np.float64
    )
    margin = own - rival
    return {
        "states": len(rows),
        "wins": int(np.count_nonzero(margin > 0)),
        "ties": int(np.count_nonzero(margin == 0)),
        "losses": int(np.count_nonzero(margin < 0)),
        "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        "mean_cash": float(own.mean()),
        "mean_opponent_cash": float(rival.mean()),
        "mean_margin": float(margin.mean()),
        "minimum_margin": float(margin.min()),
        "maximum_margin": float(margin.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--candidate8-receipt", type=Path)
    parser.add_argument("--sequence-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    g001 = bundle.index("G001")
    tasks = np.asarray(
        [
            [g001, g001, seed, -1, -1, -1, -1]
            for seed in range(args.seed_start, args.seed_start + args.seed_count)
        ],
        dtype=np.int64,
    )
    started = time.perf_counter()
    rewards_raw, audit_raw = bundle.executor.play_audit_batch(tasks)
    simulation_seconds = time.perf_counter() - started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    audit = np.asarray(audit_raw, dtype=np.int32)

    rows: list[dict] = []
    for offset, seed in enumerate(range(args.seed_start, args.seed_start + args.seed_count)):
        for seat in (0, 1):
            rows.append(
                {
                    "actual_seed": seed,
                    "seat": seat,
                    "g001_cash": float(rewards[offset, seat]),
                    "g001_opponent_cash": float(rewards[offset, 1 - seat]),
                    "g001_margin": float(rewards[offset, seat] - rewards[offset, 1 - seat]),
                    "g001_outcome": outcome(rewards[offset, seat], rewards[offset, 1 - seat]),
                    "g001_unit_failures": int(audit[offset, seat, 0]),
                    "g001_market_failures": int(audit[offset, seat, 1]),
                }
            )

    combined = None
    if args.candidate8_receipt is not None:
        candidate_payload = json.loads(
            args.candidate8_receipt.read_text(encoding="utf-8")
        )
        candidate_by_state = {
            (int(row["actual_seed"]), int(row["seat"])): row
            for row in candidate_payload["states"]
        }
        for row in rows:
            candidate = candidate_by_state[(row["actual_seed"], row["seat"])]
            candidate_own = float(candidate["rolling_cash"])
            candidate_opponent = float(candidate["opponent_cash"])
            g001_key = (
                row["g001_outcome"], row["g001_margin"], row["g001_cash"]
            )
            candidate_key = (
                outcome(candidate_own, candidate_opponent),
                candidate_own - candidate_opponent,
                candidate_own,
            )
            if candidate_key > g001_key:
                selected = "CANDIDATE8_ROLLING"
                own, rival = candidate_own, candidate_opponent
            else:
                selected = "G001_COMPLETE_PLAN"
                own, rival = row["g001_cash"], row["g001_opponent_cash"]
            row.update(
                {
                    "candidate8_cash": candidate_own,
                    "candidate8_opponent_cash": candidate_opponent,
                    "selected_arm": selected,
                    "combined_cash": own,
                    "combined_opponent_cash": rival,
                    "combined_margin": own - rival,
                }
            )
        combined = summary(rows, "combined")
        combined["selected_arm_counts"] = {
            arm: sum(row["selected_arm"] == arm for row in rows)
            for arm in ("G001_COMPLETE_PLAN", "CANDIDATE8_ROLLING")
        }

    three_arm = None
    if args.sequence_receipt is not None:
        if args.candidate8_receipt is None:
            raise ValueError("--sequence-receipt requires --candidate8-receipt")
        sequence_payload = json.loads(
            args.sequence_receipt.read_text(encoding="utf-8")
        )
        depth_keys = sorted(
            sequence_payload["states"],
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        sequence_key = depth_keys[-1]
        sequence_by_state = {
            (int(row["actual_seed"]), int(row["seat"])): row
            for row in sequence_payload["states"][sequence_key]
        }
        for row in rows:
            sequence = sequence_by_state[(row["actual_seed"], row["seat"])]
            arms = [
                (
                    "G001_COMPLETE_PLAN",
                    float(row["g001_cash"]),
                    float(row["g001_opponent_cash"]),
                ),
                (
                    "CANDIDATE8_ROLLING",
                    float(row["candidate8_cash"]),
                    float(row["candidate8_opponent_cash"]),
                ),
                (
                    f"CANDIDATE8_{sequence_key.upper()}",
                    float(sequence["cash"]),
                    float(sequence["opponent_cash"]),
                ),
            ]
            selected, own, rival = max(
                arms,
                key=lambda arm: (
                    outcome(arm[1], arm[2]), arm[1] - arm[2], arm[1]
                ),
            )
            row.update(
                {
                    "three_arm_selected": selected,
                    "three_arm_cash": own,
                    "three_arm_opponent_cash": rival,
                    "three_arm_margin": own - rival,
                }
            )
        three_arm = summary(rows, "three_arm")
        three_arm["sequence_arm"] = sequence_key
        three_arm["selected_arm_counts"] = {
            arm: sum(row["three_arm_selected"] == arm for row in rows)
            for arm in sorted({row["three_arm_selected"] for row in rows})
        }

    payload = {
        "schema": "kaggriculture.g001_complete_plan_arm.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "C++ NativeTeammateExecutor.play_audit_batch",
        "parameters": {
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": [0, 1],
            "pairing": "native G001 versus native G001",
            "combined_objective": "win_then_margin_then_own_cash",
        },
        "summary": {
            "g001_complete_plan": summary(rows, "g001"),
            "best_of_complete_plan_and_candidate8": combined,
            "best_of_complete_plan_candidate8_and_sequence": three_arm,
            "simulation_seconds": simulation_seconds,
        },
        "states": rows,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "candidate8_receipt": None if args.candidate8_receipt is None else {
                "path": str(args.candidate8_receipt),
                "sha256": sha256(args.candidate8_receipt),
            },
            "sequence_receipt": None if args.sequence_receipt is None else {
                "path": str(args.sequence_receipt),
                "sha256": sha256(args.sequence_receipt),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
