#!/usr/bin/env python3
"""Search one identity-blind adaptive genome against a frozen route pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from search_native_adaptive_passive import sample_genomes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_anchors(path: Path | None) -> np.ndarray:
    if path is None:
        return np.empty((0, len(adaptive_genome_names())), dtype=np.float64)
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = list(adaptive_genome_names())
    defaults = dict(zip(names, adaptive_default_genome(), strict=True))
    defaults.update({
        str(name): float(value)
        for name, value in payload.get("base_values", {}).items()
    })
    return np.asarray([
        [float(row["values"].get(name, defaults[name])) for name in names]
        for row in payload["genomes"]
    ], dtype=np.float64)


def evaluate(
    executor,
    genomes: np.ndarray,
    opponent_indices: list[int],
    opponent_labels: list[str],
    seed_start: int,
    seed_count: int,
) -> tuple[list[dict[str, object]], int, float]:
    tasks = np.asarray([
        [genome, opponent, seed, seat]
        for genome in range(len(genomes))
        for opponent in opponent_indices
        for seed in range(seed_start, seed_start + seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)
    started = time.perf_counter()
    rewards_raw, diagnostics_raw = executor.play_batch(genomes, tasks)
    elapsed = time.perf_counter() - started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    diagnostics = np.asarray(diagnostics_raw, dtype=np.int32)
    rows: list[dict[str, object]] = []
    for genome in range(len(genomes)):
        indices = np.flatnonzero(tasks[:, 0] == genome)
        own = np.asarray([rewards[index, tasks[index, 3]] for index in indices])
        rival = np.asarray([rewards[index, 1 - tasks[index, 3]] for index in indices])
        margin = own - rival
        diag = diagnostics[indices]
        hard = diag[:, 2] + diag[:, 3]
        overflow = diag[:, 4]
        by_opponent: dict[str, dict[str, float | int]] = {}
        score_rates: list[float] = []
        for opponent, label in zip(opponent_indices, opponent_labels, strict=True):
            local = indices[tasks[indices, 1] == opponent]
            local_own = np.asarray([rewards[index, tasks[index, 3]] for index in local])
            local_rival = np.asarray([rewards[index, 1 - tasks[index, 3]] for index in local])
            local_margin = local_own - local_rival
            score_rate = float(np.mean(
                (local_margin > 0) + 0.5 * (local_margin == 0)
            ))
            score_rates.append(score_rate)
            by_opponent[label] = {
                "games": int(len(local)),
                "mean_reward": float(local_own.mean()),
                "p10_reward": float(np.quantile(local_own, 0.10)),
                "mean_margin": float(local_margin.mean()),
                "score_rate": score_rate,
            }
        mean_hard = float(hard.mean())
        mean_overflow = float(overflow.mean())
        min_score = min(score_rates)
        objective = float(
            own.mean()
            + 0.50 * margin.mean()
            + 20_000.0 * min_score
            - 12_000.0 * mean_hard
            - 800.0 * mean_overflow
        )
        rows.append({
            "genome_index": genome,
            "objective": objective,
            "mean_reward": float(own.mean()),
            "p10_reward": float(np.quantile(own, 0.10)),
            "mean_margin": float(margin.mean()),
            "mean_score_rate": float(np.mean(score_rates)),
            "min_score_rate": float(min_score),
            "hard_losses": int(hard.sum()),
            "end_overflow": int(overflow.sum()),
            "mean_replans": float(diag[:, 0].mean()),
            "by_opponent": by_opponent,
        })
    rows.sort(key=lambda row: (
        -float(row["objective"]),
        -float(row["min_score_rate"]),
        -float(row["p10_reward"]),
    ))
    return rows, len(tasks), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--opponents", default="G001,G136,G006,G036")
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--search-seed", type=int, default=20_590_001)
    parser.add_argument("--screen-seed-start", type=int, default=2_059_001)
    parser.add_argument("--screen-seed-count", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=48)
    parser.add_argument("--validation-seed-start", type=int, default=2_069_001)
    parser.add_argument("--validation-seed-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    labels = [value.strip() for value in args.opponents.split(",") if value.strip()]
    indices = [-1 if label.upper() == "PASSIVE" else bundle.index(label) for label in labels]
    names = list(adaptive_genome_names())
    genomes = sample_genomes(args.candidate_count, args.search_seed)
    anchors = load_anchors(args.anchors)
    if len(anchors):
        genomes[: min(len(genomes), len(anchors))] = anchors[: len(genomes)]

    screen, screen_games, screen_seconds = evaluate(
        bundle.adaptive_executor, genomes, indices, labels,
        args.screen_seed_start, args.screen_seed_count,
    )
    finalist_indices = [int(row["genome_index"]) for row in screen[: args.finalists]]
    finalist_genomes = genomes[finalist_indices]
    validation, validation_games, validation_seconds = evaluate(
        bundle.adaptive_executor, finalist_genomes, indices, labels,
        args.validation_seed_start, args.validation_seed_count,
    )
    for row in validation:
        local_index = int(row["genome_index"])
        original_index = finalist_indices[local_index]
        row["screen_genome_index"] = original_index
        row["values"] = {
            name: float(value)
            for name, value in zip(names, genomes[original_index], strict=True)
        }

    payload = {
        "schema": "kaggriculture.native-adaptive-pool-search.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "policy_semantics": (
            "one general public-state planner; opponent identities are used only "
            "to construct the offline validation pool"
        ),
        "opponents": labels,
        "candidate_count": args.candidate_count,
        "genome_names": names,
        "search_seed": args.search_seed,
        "screen": {
            "seed_start": args.screen_seed_start,
            "seed_count": args.screen_seed_count,
            "games": screen_games,
            "seconds": screen_seconds,
            "games_per_second": screen_games / screen_seconds,
            "top": screen[:100],
        },
        "validation": {
            "seed_start": args.validation_seed_start,
            "seed_count": args.validation_seed_count,
            "games": validation_games,
            "seconds": validation_seconds,
            "games_per_second": validation_games / validation_seconds,
            "ranking": validation,
        },
        "setup_seconds": setup_seconds,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": sha256(args.backbone)
            },
            "anchors": None if args.anchors is None else {
                "path": str(args.anchors), "sha256": sha256(args.anchors)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "screen_games": screen_games,
        "screen_games_per_second": screen_games / screen_seconds,
        "validation_games": validation_games,
        "validation_games_per_second": validation_games / validation_seconds,
        "top_validation": validation[:10],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
