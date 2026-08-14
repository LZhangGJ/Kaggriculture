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
    parser.add_argument(
        "--seat",
        type=int,
        choices=(0, 1),
        action="append",
        help="collect only selected policy seats; repeat for both (default: both)",
    )
    parser.add_argument("--mask-unit-actions", action="store_true")
    parser.add_argument(
        "--mismatch-only",
        action="store_true",
        help="store only states where the deterministic policy action differs from the teacher",
    )
    parser.add_argument(
        "--mismatch-component",
        choices=("any", "unit", "market"),
        default="any",
        help="when filtering mismatches, compare the full action or only one component",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("at least five seeds are required for train/validation/test splits")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device).eval()
    selected_seats = tuple(dict.fromkeys(args.seat or (0, 1)))

    games = [
        (seed, seat, offset, _safe_name(opponent), opponent)
        for offset, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds))
        for opponent in args.opponent
        for seat in selected_seats
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
        features, unit_context, _ = encode_batch(
            policy_observations,
            include_unit_inventory=model.unit_inventory_context,
        )
        targets = structured_action_targets(policy_observations, teacher_actions)
        for index in range(len(games)):
            if args.mismatch_component == "unit":
                mismatch = (
                    policy_actions[index].get("farmer")
                    != teacher_actions[index].get("farmer")
                    or policy_actions[index].get("hands", [])
                    != teacher_actions[index].get("hands", [])
                )
            elif args.mismatch_component == "market":
                mismatch = policy_actions[index].get("market", []) != teacher_actions[
                    index
                ].get("market", [])
            else:
                mismatch = policy_actions[index] != teacher_actions[index]
            mismatch_counts[index] += mismatch
            if args.mismatch_only and not mismatch:
                continue
            episode_arrays[index]["features"].append(features[index].astype(np.float16))
            episode_arrays[index]["unit_context"].append(unit_context[index].astype(np.float16))
            for key, values in targets.items():
                episode_arrays[index][key].append(values[index])

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
        selected_examples = len(episode_arrays[index]["features"])
        if selected_examples and output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists; pass --overwrite: {output}")
        policy_reward = float(result.rewards[seat])
        opponent_reward = float(result.rewards[1 - seat])
        outcome = float((policy_reward > opponent_reward) - (policy_reward < opponent_reward))
        if selected_examples:
            output.parent.mkdir(parents=True, exist_ok=True)
            stacked = {
                key: np.stack(values) for key, values in episode_arrays[index].items()
            }
            stacked["value_targets"] = np.full(
                selected_examples, outcome, dtype=np.float16
            )
            np.savez(output, **stacked)
            assignments[filename] = split
        total["examples"] += selected_examples
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
                "output": str(output) if selected_examples else None,
                "selected_examples": selected_examples,
                "mismatches": int(mismatch_counts[index]),
                "policy_reward": policy_reward,
                "opponent_reward": opponent_reward,
            }
        )
        print(
            f"[{index + 1}/{len(games)}] seed={seed} seat={seat} "
            f"mismatches={mismatch_counts[index]} selected={selected_examples} "
            f"policy={policy_reward:.0f} "
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
        "seats": list(selected_seats),
        "mask_unit_actions": args.mask_unit_actions,
        "mismatch_only": args.mismatch_only,
        "mismatch_component": args.mismatch_component,
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
