#!/usr/bin/env python3
"""Search one-switch route schedules against an existing switch policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fast_kaggriculture import native_tree_predict
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if separator:
        return tuple(range(int(start), int(stop)))
    return tuple(int(part) for part in value.split(",") if part.strip())


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _tree_predict(tree: dict, features: np.ndarray) -> np.ndarray:
    leaf_class = np.argmax(np.asarray(tree["value"]), axis=1).astype(np.int32)
    indices = np.asarray(native_tree_predict(
        np.asarray(tree["left"], dtype=np.int32),
        np.asarray(tree["right"], dtype=np.int32),
        np.asarray(tree["feature"], dtype=np.int32),
        np.asarray(tree["threshold"], dtype=np.float64),
        leaf_class,
        np.ascontiguousarray(features, dtype=np.float32),
    ))
    return np.asarray(tree["classes"], dtype=str)[indices]


def _policy_response_grid(
    bundle: NativeTeammateBundle,
    nodes: list[dict],
    policy_opening: str,
    candidate_opening: str,
    candidate_targets: tuple[str, ...],
    candidate_checkpoint: int,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    policy_route = bundle.index(policy_opening)
    candidate_route = bundle.index(candidate_opening)
    samples = [(int(seed), candidate_seat) for seed in seeds for candidate_seat in (0, 1)]
    switch_step = np.full(
        (len(candidate_targets), len(samples)), -1, dtype=np.int64
    )
    switch_target = np.full(
        (len(candidate_targets), len(samples)), policy_route, dtype=np.int64
    )
    for node in nodes:
        checkpoint = int(node["checkpoint"])
        tasks = np.empty((len(candidate_targets) * len(samples), 10), dtype=np.int64)
        for target_number, target in enumerate(candidate_targets):
            target_route = bundle.index(target)
            offset = target_number * len(samples)
            for sample_number, (seed, candidate_seat) in enumerate(samples):
                index = offset + sample_number
                if candidate_seat == 0:
                    tasks[index] = (
                        candidate_route, policy_route, seed, checkpoint, 1, policy_route,
                        candidate_checkpoint, target_route, -1, -1,
                    )
                else:
                    tasks[index] = (
                        policy_route, candidate_route, seed, checkpoint, 0, policy_route,
                        -1, -1, candidate_checkpoint, target_route,
                    )
        predictions = _tree_predict(
            node["tree"],
            np.asarray(bundle.executor.features_with_switch_batch(tasks)),
        ).reshape(len(candidate_targets), len(samples))
        for target_number in range(len(candidate_targets)):
            for sample_number, prediction in enumerate(predictions[target_number]):
                if (
                    switch_step[target_number, sample_number] >= 0
                    or prediction == policy_opening
                ):
                    continue
                switch_step[target_number, sample_number] = checkpoint
                switch_target[target_number, sample_number] = bundle.index(str(prediction))
    return switch_step, switch_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-opening", default="G001")
    parser.add_argument("--openings", type=_csv)
    parser.add_argument("--opening-ranking", type=Path)
    parser.add_argument("--top-openings", type=int, default=20)
    parser.add_argument("--targets", type=_csv)
    parser.add_argument("--checkpoint", type=int, default=216)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = sorted(
        (
            value["selected"] for value in policy["nodes"]
            if value["selected"].get("enabled", True)
            and str(value["selected"]["opening"]) == args.policy_opening
        ),
        key=lambda value: int(value["checkpoint"]),
    )
    if not nodes:
        raise ValueError(f"no enabled policy nodes for {args.policy_opening}")
    if args.openings:
        openings = args.openings
    elif args.opening_ranking:
        ranking = json.loads(args.opening_ranking.read_text(encoding="utf-8"))["ranking"]
        openings = tuple(str(row["family"]) for row in ranking[:args.top_openings])
    else:
        openings = tuple(bundle.families)
    targets = args.targets or tuple(bundle.families)
    for family in (*openings, *targets):
        bundle.index(family)

    rows = []
    games = 0
    game_seconds = 0.0
    samples = [(int(seed), candidate_seat) for seed in args.seeds for candidate_seat in (0, 1)]
    matrix_shape = (len(openings), len(targets), len(args.seeds), 2)
    margin_matrix = np.empty(matrix_shape, dtype=np.float64)
    score_matrix = np.empty(matrix_shape, dtype=np.float64)
    own_reward_matrix = np.empty(matrix_shape, dtype=np.float64)
    policy_step_matrix = np.empty(matrix_shape, dtype=np.int64)
    policy_target_matrix = np.empty(matrix_shape, dtype=np.int64)
    for opening_number, opening in enumerate(openings, 1):
        policy_step, policy_target = _policy_response_grid(
            bundle, nodes, args.policy_opening, opening, targets,
            args.checkpoint, args.seeds,
        )
        opening_route = bundle.index(opening)
        policy_route = bundle.index(args.policy_opening)
        tasks = np.empty((len(targets) * len(samples), 7), dtype=np.int64)
        for target_number, target in enumerate(targets):
            target_route = bundle.index(target)
            offset = target_number * len(samples)
            for sample_number, (seed, candidate_seat) in enumerate(samples):
                index = offset + sample_number
                if candidate_seat == 0:
                    tasks[index] = (
                        opening_route, policy_route, seed,
                        args.checkpoint, target_route,
                        policy_step[target_number, sample_number],
                        policy_target[target_number, sample_number],
                    )
                else:
                    tasks[index] = (
                        policy_route, opening_route, seed,
                        policy_step[target_number, sample_number],
                        policy_target[target_number, sample_number],
                        args.checkpoint, target_route,
                    )
        game_started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
        game_seconds += time.perf_counter() - game_started
        games += len(tasks)
        for target_number, target in enumerate(targets):
            offset = target_number * len(samples)
            block = rewards[offset:offset + len(samples)]
            candidate_seats = np.asarray([seat for _seed, seat in samples], dtype=np.int64)
            indices = np.arange(len(samples))
            own = block[indices, candidate_seats]
            other = block[indices, 1 - candidate_seats]
            margins = own - other
            scores = (margins > 0).astype(np.float64) + 0.5 * (margins == 0)
            matrix_index = (opening_number - 1, target_number)
            margin_matrix[matrix_index] = margins.reshape(len(args.seeds), 2)
            score_matrix[matrix_index] = scores.reshape(len(args.seeds), 2)
            own_reward_matrix[matrix_index] = own.reshape(len(args.seeds), 2)
            policy_step_matrix[matrix_index] = policy_step[target_number].reshape(
                len(args.seeds), 2
            )
            policy_target_matrix[matrix_index] = policy_target[target_number].reshape(
                len(args.seeds), 2
            )
            paired_margins = margins.reshape(len(args.seeds), 2).mean(axis=1)
            paired_scores = scores.reshape(len(args.seeds), 2).mean(axis=1)
            if len(args.seeds) > 1:
                margin_se = float(
                    np.std(paired_margins, ddof=1) / math.sqrt(len(args.seeds))
                )
                margin_lower = float(
                    np.mean(paired_margins)
                    - student_t.ppf(.95, len(args.seeds) - 1) * margin_se
                )
            else:
                margin_se = 0.0
                margin_lower = float(paired_margins[0])
            response_names = [
                bundle.families[int(value)] for value in policy_target[target_number]
            ]
            rows.append({
                "opening": opening,
                "checkpoint": args.checkpoint,
                "target": target,
                "mean_score": float(np.mean(scores)),
                "mean_margin": float(np.mean(margins)),
                "minimum_margin": float(np.min(margins)),
                "minimum_paired_seed_margin": float(np.min(paired_margins)),
                "one_sided_95pct_margin_lower": margin_lower,
                "paired_margin_standard_error": margin_se,
                "mean_reward": float(np.mean(own)),
                "minimum_reward": float(np.min(own)),
                "policy_switch_rate": float(
                    np.mean(policy_step[target_number] >= 0)
                ),
                "policy_target_counts": dict(sorted(Counter(response_names).items())),
                "paired_seed_scores": paired_scores.tolist(),
                "paired_seed_margins": paired_margins.tolist(),
            })
        best_for_opening = max(
            rows[-len(targets):], key=lambda row: (
                row["mean_score"], row["one_sided_95pct_margin_lower"], row["mean_margin"]
            )
        )
        print(
            f"{opening_number}/{len(openings)} {opening}->{best_for_opening['target']} "
            f"score={best_for_opening['mean_score']:.4f} "
            f"margin={best_for_opening['mean_margin']:.1f}",
            flush=True,
        )

    rows.sort(key=lambda row: (
        -float(row["mean_score"]),
        -float(row["one_sided_95pct_margin_lower"]),
        -float(row["mean_margin"]),
        str(row["opening"]),
        str(row["target"]),
    ))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "native-one-switch-counter-schedule-search-v1",
        "engine": "C++ NativeTeammateExecutor with exact final-checkpoint policy response",
        "policy": str(args.policy.resolve()),
        "policy_opening": args.policy_opening,
        "candidate_checkpoint": args.checkpoint,
        "openings": list(openings),
        "targets": list(targets),
        "seeds": list(args.seeds),
        "candidate_count": len(rows),
        "games": games,
        "game_seconds": game_seconds,
        "games_per_second": games / game_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": rows,
    }
    if args.matrix_output:
        matrix_output = args.matrix_output.resolve()
        matrix_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_output,
            openings=np.asarray(openings),
            targets=np.asarray(targets),
            seeds=np.asarray(args.seeds, dtype=np.int64),
            margins=margin_matrix,
            scores=score_matrix,
            own_rewards=own_reward_matrix,
            policy_switch_steps=policy_step_matrix,
            policy_switch_targets=policy_target_matrix,
        )
        payload["matrix_output"] = str(matrix_output)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "games": games,
        "games_per_second": payload["games_per_second"],
        "matrix_output": payload.get("matrix_output"),
        "top10": [
            {key: row[key] for key in (
                "rank", "opening", "checkpoint", "target", "mean_score",
                "mean_margin", "minimum_paired_seed_margin",
                "one_sided_95pct_margin_lower", "mean_reward",
                "policy_switch_rate", "policy_target_counts",
            )}
            for row in rows[:10]
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
