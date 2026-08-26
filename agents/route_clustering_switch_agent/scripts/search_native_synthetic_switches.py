#!/usr/bin/env python3
"""Evaluate early switches from a mature opening into synthetic tape mutations."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_counter_schedules import _policy_response_grid
from search_native_tape_mutations import _apply_genes, _ints


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _confidence(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    lower = mean - float(student_t.ppf(.95, len(values) - 1)) * standard_error
    return lower, standard_error


def _row(
    bundle: NativeTeammateBundle,
    rewards: np.ndarray,
    candidate_seats: np.ndarray,
    seeds: Sequence[int],
    policy_targets: np.ndarray,
) -> dict[str, Any]:
    indices = np.arange(len(rewards))
    own = rewards[indices, candidate_seats]
    other = rewards[indices, 1 - candidate_seats]
    margins = own - other
    scores = (margins > 0).astype(np.float64) + .5 * (margins == 0)
    paired_scores = scores.reshape(len(seeds), 2).mean(axis=1)
    paired_margins = margins.reshape(len(seeds), 2).mean(axis=1)
    score_lower, score_se = _confidence(paired_scores)
    margin_lower, margin_se = _confidence(paired_margins)
    target_names = [bundle.families[int(value)] for value in policy_targets]
    return {
        "mean_score": float(np.mean(scores)),
        "one_sided_95pct_score_lower": score_lower,
        "paired_score_standard_error": score_se,
        "mean_margin": float(np.mean(margins)),
        "one_sided_95pct_margin_lower": margin_lower,
        "paired_margin_standard_error": margin_se,
        "minimum_margin": float(np.min(margins)),
        "minimum_paired_seed_margin": float(np.min(paired_margins)),
        "mean_reward": float(np.mean(own)),
        "minimum_reward": float(np.min(own)),
        "paired_seed_scores": paired_scores.tolist(),
        "paired_seed_margins": paired_margins.tolist(),
        "policy_target_counts": dict(sorted(Counter(target_names).items())),
    }


def _fitness(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(value["one_sided_95pct_score_lower"]),
        float(value["mean_score"]),
        float(value["one_sided_95pct_margin_lower"]),
        float(value["mean_margin"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--genes-from", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--parent", default="G006")
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--checkpoints", type=_csv_ints, default=(1, 24, 48, 72))
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parent_entry = next(
        value for value in metadata["opponent_routes"]
        if str(value["family"]) == args.parent
    )
    parent_tape = load_action_tapes(args.actions)[str(parent_entry["route_id"])]
    gene_payload = json.loads(args.genes_from.read_text(encoding="utf-8"))
    gene_rows = list(gene_payload["ranking"])[:args.top]
    if not any(not (row.get("genes") or []) for row in gene_rows):
        gene_rows.append({"rank": None, "genes": []})
    names = [f"S{index:04d}" for index in range(len(gene_rows))]
    routes = {
        name: _apply_genes(parent_tape, list(row.get("genes") or []))
        for name, row in zip(names, gene_rows)
    }
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, additional_routes=routes
    )
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = sorted(
        (
            value["selected"] for value in policy["nodes"]
            if value["selected"].get("enabled", True)
            and str(value["selected"]["opening"]) == args.opening
        ),
        key=lambda value: int(value["checkpoint"]),
    )
    samples = [(int(seed), seat) for seed in args.seeds for seat in (0, 1)]
    candidate_seats = np.asarray([seat for _seed, seat in samples], dtype=np.int64)
    opening_route = bundle.index(args.opening)
    policy_route = bundle.index(args.opening)
    ranking: list[dict[str, Any]] = []
    game_seconds = 0.0

    for checkpoint in args.checkpoints:
        policy_step, policy_target = _policy_response_grid(
            bundle, nodes, args.opening, args.opening, tuple(names), checkpoint,
            tuple(args.seeds),
        )
        tasks = np.empty((len(names) * len(samples), 7), dtype=np.int64)
        for candidate_number, name in enumerate(names):
            candidate_route = bundle.index(name)
            offset = candidate_number * len(samples)
            for sample_number, (seed, seat) in enumerate(samples):
                index = offset + sample_number
                if seat == 0:
                    tasks[index] = (
                        opening_route, policy_route, seed,
                        checkpoint, candidate_route,
                        policy_step[candidate_number, sample_number],
                        policy_target[candidate_number, sample_number],
                    )
                else:
                    tasks[index] = (
                        policy_route, opening_route, seed,
                        policy_step[candidate_number, sample_number],
                        policy_target[candidate_number, sample_number],
                        checkpoint, candidate_route,
                    )
        game_started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
        game_seconds += time.perf_counter() - game_started
        for candidate_number, (name, gene_row) in enumerate(zip(names, gene_rows)):
            offset = candidate_number * len(samples)
            result = _row(
                bundle,
                rewards[offset:offset + len(samples)],
                candidate_seats,
                args.seeds,
                policy_target[candidate_number],
            )
            result.update({
                "synthetic_family": name,
                "source_rank": gene_row.get("rank"),
                "checkpoint": checkpoint,
                "genes": list(gene_row.get("genes") or []),
            })
            ranking.append(result)
        best = max(ranking[-len(names):], key=_fitness)
        print(json.dumps({
            "checkpoint": checkpoint,
            "evaluated": len(names),
            "best_score": best["mean_score"],
            "best_score_lower": best["one_sided_95pct_score_lower"],
            "best_margin": best["mean_margin"],
            "source_rank": best["source_rank"],
        }), flush=True)

    ranking.sort(key=_fitness, reverse=True)
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    games = len(names) * len(args.checkpoints) * len(samples)
    payload = {
        "schema": "native-synthetic-early-switch-search-v1",
        "parent": args.parent,
        "opening": args.opening,
        "checkpoints": list(args.checkpoints),
        "seeds": list(args.seeds),
        "candidate_count": len(ranking),
        "games": games,
        "game_seconds": game_seconds,
        "games_per_second": games / game_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": ranking,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "games": games,
        "games_per_second": payload["games_per_second"],
        "top10": ranking[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
