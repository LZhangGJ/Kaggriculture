"""Collect teacher labels on states visited by a structured BC policy."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_policy import encode_batch
from kaggriculture_lab.policy_v2 import (
    policy_from_checkpoint,
    structured_action_targets,
    structured_policy_batch,
)


def _isolated_agent(spec: str, tag: str) -> Callable[[Any, Any], dict[str, Any]]:
    path = Path(spec)
    if not path.is_file():
        return resolve_agent(spec)
    module_spec = importlib.util.spec_from_file_location(f"dagger_agent_{tag}", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load agent from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    agent = module.agent
    positional = [
        parameter
        for parameter in inspect.signature(agent).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return agent
    return lambda observation, configuration: agent(observation)


def _safe_name(spec: str) -> str:
    path = Path(spec)
    return path.parent.name if path.suffix == ".py" else spec.replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--opponent", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Kaggriculture\data\processed\dagger_bc_v2"),
    )
    parser.add_argument("--seed-start", type=int, default=42_000)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--mask-unit-actions", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("at least five seeds are required for train/validation/test splits")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device).eval()

    games = [
        (seed, seat, offset, _safe_name(opponent), opponent)
        for offset, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds))
        for opponent in args.opponent
        for seat in (0, 1)
    ]
    environments = VectorFastEnv(len(games))
    observations = environments.reset(seed for seed, _, _, _, _ in games)
    teachers = [_isolated_agent(args.teacher, f"teacher_{index}") for index in range(len(games))]
    opponents = [
        _isolated_agent(opponent, f"opponent_{index}")
        for index, (_, _, _, _, opponent) in enumerate(games)
    ]

    keys = (
        "features",
        "unit_context",
        "unit_active",
        "unit_targets",
        "unit_quantity_targets",
        "unit_quantity_active",
        "market_targets",
        "market_quantity_targets",
        "market_quantity_active",
        "market_order_active",
    )
    episode_arrays: list[dict[str, list[np.ndarray]]] = [
        {key: [] for key in keys} for _ in games
    ]
    mismatch_counts = np.zeros(len(games), dtype=np.int32)
    results = None

    while not environments.envs[0].done:
        policy_observations = [
            pair[seat] for pair, (_, seat, _, _, _) in zip(observations, games, strict=True)
        ]
        teacher_actions = [
            teacher(observation, environments.envs[index].configuration)
            for index, (teacher, observation) in enumerate(
                zip(teachers, policy_observations, strict=True)
            )
        ]
        policy_actions = structured_policy_batch(
            model,
            policy_observations,
            device,
            deterministic=True,
            mask_unit_actions=args.mask_unit_actions,
        ).actions
        features, unit_context, _ = encode_batch(policy_observations)
        targets = structured_action_targets(policy_observations, teacher_actions)
        for index in range(len(games)):
            episode_arrays[index]["features"].append(features[index].astype(np.float16))
            episode_arrays[index]["unit_context"].append(unit_context[index].astype(np.float16))
            for key, values in targets.items():
                episode_arrays[index][key].append(values[index])
            mismatch_counts[index] += policy_actions[index] != teacher_actions[index]

        action_pairs = []
        for index, (pair, (_, seat, _, _, _)) in enumerate(zip(observations, games, strict=True)):
            other = 1 - seat
            opponent_action = opponents[index](
                pair[other], environments.envs[index].configuration
            )
            action_pairs.append(
                [policy_actions[index], opponent_action]
                if seat == 0
                else [opponent_action, policy_actions[index]]
            )
        results = environments.step(action_pairs)
        observations = [result.observations for result in results]

    assert results is not None
    total: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    episodes = []
    for index, ((seed, seat, offset, opponent_name, opponent_spec), result) in enumerate(
        zip(games, results, strict=True)
    ):
        fraction = offset / args.seeds
        split = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"
        filename = f"dagger_vs_{opponent_name}_seed{seed}_seat{seat}.npz"
        output = args.output_dir / split / filename
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists; pass --overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        stacked = {key: np.stack(values) for key, values in episode_arrays[index].items()}
        policy_reward = float(result.rewards[seat])
        opponent_reward = float(result.rewards[1 - seat])
        outcome = float((policy_reward > opponent_reward) - (policy_reward < opponent_reward))
        stacked["value_targets"] = np.full(len(stacked["features"]), outcome, dtype=np.float16)
        np.savez(output, **stacked)
        assignments[filename] = split
        total["examples"] += len(stacked["features"])
        total["mismatches"] += int(mismatch_counts[index])
        total["wins"] += policy_reward > opponent_reward
        total["ties"] += policy_reward == opponent_reward
        total["losses"] += policy_reward < opponent_reward
        episodes.append(
            {
                "seed": seed,
                "seat": seat,
                "opponent": opponent_spec,
                "opponent_name": opponent_name,
                "split": split,
                "output": str(output),
                "mismatches": int(mismatch_counts[index]),
                "policy_reward": policy_reward,
                "opponent_reward": opponent_reward,
            }
        )
        print(
            f"[{index + 1}/{len(games)}] seed={seed} seat={seat} "
            f"mismatches={mismatch_counts[index]} policy={policy_reward:.0f} "
            f"opponent={opponent_reward:.0f} output={output}"
        )

    manifest = {
        "schema_version": 2,
        "kind": "dagger",
        "checkpoint": str(args.checkpoint),
        "teacher": args.teacher,
        "opponents": args.opponent,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "mask_unit_actions": args.mask_unit_actions,
        "assignments": assignments,
        "stats": dict(total),
        "episodes": episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(dict(total), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
