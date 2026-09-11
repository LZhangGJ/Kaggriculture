#!/usr/bin/env python3
"""Select robust semantic representatives from every episode of one submission.

The selector never uses reward to choose the representative.  It first labels
the two observed public macro decisions (animal mix and late crop suffix), then
chooses the episode nearest the robust median of phase-level economic actions
inside each branch.  Highest-reward episodes are reported only as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PHASES = range(6)
MACRO_ACTIONS = (
    "HIRE:", "BUY_LAND:",
    "BUY_ANIMAL:GOOSE", "BUY_ANIMAL:COW", "BUY_ANIMAL:SHEEP",
    "BUY_SEED:WHEAT", "BUY_SEED:CARROT", "BUY_SEED:TOMATO",
    "BUY_SEED:STRAWBERRY", "BUY_SEED:MELON",
    "SELL:WHEAT", "SELL:CARROT", "SELL:TOMATO", "SELL:STRAWBERRY",
    "SELL:MELON", "SELL:EGG", "SELL:MILK", "SELL:WOOL",
    "SELL:FERTILIZER",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, default=55714246)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def number(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    return float(raw) if raw not in ("", None) else 0.0


def total(row: dict[str, str], action: str, phases=PHASES) -> float:
    return sum(number(row, f"P{phase}:{action}") for phase in phases)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    args = parse_args()
    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["submission_id"]) == args.submission_id
        ]
    if not rows:
        raise SystemExit(f"no rows for submission {args.submission_id}")

    clusters = sorted({int(row["cluster"]) for row in rows})
    sheep_mean = {
        cluster: float(np.mean([
            total(row, "BUY_ANIMAL:SHEEP")
            for row in rows if int(row["cluster"]) == cluster
        ]))
        for cluster in clusters
    }
    sheep_cluster = max(sheep_mean, key=sheep_mean.get)

    feature_names = [
        f"P{phase}:{action}" for phase in PHASES for action in MACRO_ACTIONS
    ]
    x = np.asarray([
        [number(row, name) for name in feature_names] for row in rows
    ], dtype=np.float64)
    labels: list[str] = []
    for row in rows:
        animal = (
            "sheep_heavy" if int(row["cluster"]) == sheep_cluster
            else "cow_mixed"
        )
        late_carrot = sum(
            number(row, f"P{phase}:BUY_SEED:CARROT") for phase in (4, 5)
        )
        suffix = "carrot" if late_carrot >= 10 else "wheat"
        labels.append(f"{animal}__{suffix}")

    branches: dict[str, object] = {}
    for label in sorted(set(labels)):
        indices = np.asarray(
            [index for index, value in enumerate(labels) if value == label],
            dtype=np.int64,
        )
        local = x[indices]
        median = np.median(local, axis=0)
        mad = np.median(np.abs(local - median), axis=0)
        scale = np.where(mad > 0, mad, np.maximum(1.0, np.abs(median) * 0.10))
        distance = np.mean(np.abs(local - median) / scale, axis=1)
        medoid_index = int(indices[int(np.argmin(distance))])
        rewards = np.asarray([float(rows[index]["reward"]) for index in indices])
        margins = np.asarray([float(rows[index]["margin"]) for index in indices])
        highest_index = int(indices[int(np.argmax(rewards))])

        def episode_record(index: int) -> dict[str, object]:
            episode_id = int(rows[index]["episode_id"])
            replay = args.episodes_dir / f"{episode_id}.json"
            if not replay.exists():
                raise FileNotFoundError(replay)
            return {
                "episode_id": episode_id,
                "player": int(rows[index]["seat"]),
                "opponent": rows[index]["opponent"],
                "reward": float(rows[index]["reward"]),
                "margin": float(rows[index]["margin"]),
                "replay": str(replay.resolve()),
                "replay_sha256": sha256(replay),
            }

        branches[label] = {
            "episodes": int(len(indices)),
            "reward": {
                "mean": float(rewards.mean()),
                "p10": float(np.quantile(rewards, 0.10)),
                "median": float(np.median(rewards)),
                "max": float(rewards.max()),
            },
            "margin": {
                "mean": float(margins.mean()),
                "median": float(np.median(margins)),
            },
            "macro_median": {
                "buy_cow": float(np.median([
                    total(rows[index], "BUY_ANIMAL:COW") for index in indices
                ])),
                "buy_sheep": float(np.median([
                    total(rows[index], "BUY_ANIMAL:SHEEP") for index in indices
                ])),
                "buy_wheat_seed": float(np.median([
                    total(rows[index], "BUY_SEED:WHEAT") for index in indices
                ])),
                "buy_carrot_seed": float(np.median([
                    total(rows[index], "BUY_SEED:CARROT") for index in indices
                ])),
            },
            "robust_medoid": {
                **episode_record(medoid_index),
                "standardized_l1_distance": float(distance.min()),
                "selection_uses_reward": False,
            },
            "highest_reward_diagnostic_only": episode_record(highest_index),
        }

    payload = {
        "schema": "kaggriculture.rank1-submission-branch-representatives.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_id": args.submission_id,
        "episodes": len(rows),
        "episode_ids_unique": len({int(row["episode_id"]) for row in rows}),
        "branch_counts": dict(sorted(Counter(labels).items())),
        "animal_cluster_semantics": {
            "sheep_heavy_cluster": sheep_cluster,
            "mean_sheep_buys_by_cluster": sheep_mean,
        },
        "selection_rule": (
            "robust medoid of phase-level hires, land, animal, seed and sell "
            "actions; reward is excluded from selection"
        ),
        "feature_names": feature_names,
        "branches": branches,
        "inputs": {
            "features": str(args.features.resolve()),
            "features_sha256": sha256(args.features),
            "episodes_dir": str(args.episodes_dir.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "submission_id": args.submission_id,
        "episodes": len(rows),
        "branch_counts": payload["branch_counts"],
        "representatives": {
            label: data["robust_medoid"]
            for label, data in branches.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
