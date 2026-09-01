#!/usr/bin/env python3
"""Causal C++ ablation for generic public-state market preemption profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


PROFILES = [
    {"name": "OFF", "fraction": 0.0, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 30},
    {"name": "P25_R4_Q0_M30", "fraction": 0.25, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 30},
    {"name": "P50_R4_Q0_M30", "fraction": 0.50, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 30},
    {"name": "P100_R4_Q0_M30", "fraction": 1.00, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 30},
    {"name": "P50_R8_Q0_M30", "fraction": 0.50, "min_ready": 8, "min_price_ratio": 0.0, "max_quantity": 30},
    {"name": "P50_R4_Q80_M30", "fraction": 0.50, "min_ready": 4, "min_price_ratio": 0.80, "max_quantity": 30},
    {"name": "P50_R4_Q100_M30", "fraction": 0.50, "min_ready": 4, "min_price_ratio": 1.00, "max_quantity": 30},
    {"name": "P50_R4_Q0_M12", "fraction": 0.50, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 12},
    {"name": "P50_R4_Q0_M60", "fraction": 0.50, "min_ready": 4, "min_price_ratio": 0.0, "max_quantity": 60},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def competitive(margins: np.ndarray) -> dict[str, float | int]:
    wins = int(np.count_nonzero(margins > 0))
    ties = int(np.count_nonzero(margins == 0))
    losses = int(np.count_nonzero(margins < 0))
    games = int(margins.size)
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "score_rate": (wins + 0.5 * ties) / games,
        "win_rate": wins / games,
        "mean_margin": float(margins.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument("--group", choices=("hard16", "all"), default="hard16")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    if args.group == "hard16":
        opponent_names = list(merged["route_bands"]["oracle_below_50pct"])
    else:
        opponent_names = [row["opponent"] for row in merged["opponents"]]

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    genome = np.asarray(load_genome(args.genomes, args.genome_index), dtype=np.float64)
    genomes = genome.reshape(1, -1)

    state_keys: list[tuple[str, int, int]] = []
    tasks: list[list[int]] = []
    profile_index: list[int] = []
    for opponent_name in opponent_names:
        opponent = bundle.index(opponent_name)
        for offset in range(args.seed_count):
            seed = args.seed_start + offset
            for seat in (0, 1):
                state_keys.append((opponent_name, seed, seat))
                for index, profile in enumerate(PROFILES):
                    profile_index.append(index)
                    tasks.append([
                        0,
                        opponent,
                        seed,
                        seat,
                        int(round(1000 * profile["fraction"])),
                        int(profile["min_ready"]),
                        int(round(1000 * profile["min_price_ratio"])),
                        int(profile["max_quantity"]),
                    ])

    task_matrix = np.asarray(tasks, dtype=np.int64)
    started = time.perf_counter()
    rewards, diagnostics = bundle.adaptive_executor.play_market_preempt_batch(
        genomes, task_matrix
    )
    simulation_seconds = time.perf_counter() - started
    rewards = np.asarray(rewards, dtype=np.float64)
    diagnostics = np.asarray(diagnostics, dtype=np.int64)
    profile_index_array = np.asarray(profile_index, dtype=np.int64)

    candidate_seats = task_matrix[:, 3]
    own = rewards[np.arange(rewards.shape[0]), candidate_seats]
    opponent = rewards[np.arange(rewards.shape[0]), 1 - candidate_seats]
    margins = own - opponent

    profile_rows = []
    per_opponent: dict[str, dict[str, dict]] = defaultdict(dict)
    for index, profile in enumerate(PROFILES):
        mask = profile_index_array == index
        summary = competitive(margins[mask])
        summary.update({
            "mean_cash": float(own[mask].mean()),
            "avoidable_crop_losses": int(diagnostics[mask, 0].sum()),
            "avoidable_animal_losses": int(diagnostics[mask, 1].sum()),
            "end_overflow": int(diagnostics[mask, 2].sum()),
        })
        profile_rows.append({"profile": profile, "summary": summary})
        for opponent_name in opponent_names:
            opponent_mask = mask & np.asarray(
                [key[0] == opponent_name for key in state_keys for _ in PROFILES],
                dtype=bool,
            )
            opponent_summary = competitive(margins[opponent_mask])
            opponent_summary["mean_cash"] = float(own[opponent_mask].mean())
            per_opponent[opponent_name][profile["name"]] = opponent_summary

    # Every state has exactly one row per profile in the same profile order.
    state_count = len(state_keys)
    profile_count = len(PROFILES)
    own_matrix = own.reshape(state_count, profile_count)
    opponent_matrix = opponent.reshape(state_count, profile_count)
    margin_matrix = own_matrix - opponent_matrix
    win_matrix = np.where(
        margin_matrix > 0, 1.0, np.where(margin_matrix == 0, 0.5, 0.0)
    )
    selected = []
    for state in range(state_count):
        best = max(
            range(profile_count),
            key=lambda profile: (
                win_matrix[state, profile],
                margin_matrix[state, profile],
                own_matrix[state, profile],
                -profile,
            ),
        )
        selected.append(best)
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_own = own_matrix[np.arange(state_count), selected_array]
    selected_opponent = opponent_matrix[np.arange(state_count), selected_array]
    selected_margin = selected_own - selected_opponent
    oracle_summary = competitive(selected_margin)
    oracle_summary["mean_cash"] = float(selected_own.mean())
    oracle_summary["selected_profile_counts"] = {
        PROFILES[index]["name"]: int(np.count_nonzero(selected_array == index))
        for index in range(profile_count)
    }

    best_global_index = max(
        range(profile_count),
        key=lambda index: (
            profile_rows[index]["summary"]["score_rate"],
            profile_rows[index]["summary"]["mean_margin"],
            profile_rows[index]["summary"]["mean_cash"],
            -index,
        ),
    )

    output = {
        "schema": "kaggriculture.runtime-market-preempt-profile-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "group": args.group,
            "opponents": opponent_names,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": [0, 1],
            "states": state_count,
            "profiles": profile_count,
            "games": int(task_matrix.shape[0]),
        },
        "profiles": profile_rows,
        "best_global_profile": profile_rows[best_global_index],
        "clairvoyant_profile_oracle": oracle_summary,
        "per_opponent": per_opponent,
        "diagnostics": {
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "games_per_second": task_matrix.shape[0] / simulation_seconds,
        },
        "inputs": {
            "source": str(args.source),
            "actions": str(args.actions),
            "actions_sha256": sha256(args.actions),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256(args.metadata),
            "genomes": str(args.genomes),
            "genomes_sha256": sha256(args.genomes),
            "merged_receipt": str(args.merged_receipt),
            "merged_receipt_sha256": sha256(args.merged_receipt),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "scope": output["scope"],
        "profiles": [
            {"name": row["profile"]["name"], **row["summary"]}
            for row in profile_rows
        ],
        "best_global_profile": output["best_global_profile"],
        "clairvoyant_profile_oracle": oracle_summary,
        "diagnostics": output["diagnostics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
