"""Find the first on-policy action divergence between two neural checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.gpu_policy import _get
from kaggriculture_lab.policy_v2 import policy_from_checkpoint, structured_policy_batch

from build_dagger_bc_dataset import _isolated_agent


def _summary(observation: object) -> dict[str, object]:
    player = int(_get(observation, "player", 0))
    farms = list(_get(observation, "farms", []) or [])
    farm = farms[player]
    return {
        "step": int(_get(observation, "step", 0)),
        "player": player,
        "money": float(_get(farm, "money", 0.0)),
        "hands": len(_get(farm, "hands", []) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, default=75_001)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mask-unit-actions", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints = [
        torch.load(path, map_location=device, weights_only=True)
        for path in (args.left, args.right)
    ]
    models = [policy_from_checkpoint(checkpoint, device).eval() for checkpoint in checkpoints]
    environments = VectorFastEnv(2)
    observations = environments.reset((args.seed, args.seed))
    opponents = [
        _isolated_agent(args.opponent, f"trajectory_opponent_{index}")
        for index in range(2)
    ]
    first_divergence = None
    results = None

    while not environments.envs[0].done:
        policy_observations = [pair[args.seat] for pair in observations]
        actions = [
            structured_policy_batch(
                model,
                [policy_observations[index]],
                device,
                deterministic=True,
                mask_unit_actions=args.mask_unit_actions,
            ).actions[0]
            for index, model in enumerate(models)
        ]
        if first_divergence is None and actions[0] != actions[1]:
            first_divergence = {
                "observation": _summary(policy_observations[0]),
                "left_action": actions[0],
                "right_action": actions[1],
            }

        action_pairs = []
        for index, pair in enumerate(observations):
            other = 1 - args.seat
            opponent_action = opponents[index](
                pair[other], environments.envs[index].configuration
            )
            action_pairs.append(
                [actions[index], opponent_action]
                if args.seat == 0
                else [opponent_action, actions[index]]
            )
        results = environments.step(action_pairs)
        observations = [result.observations for result in results]

    assert results is not None
    output = {
        "left": str(args.left),
        "right": str(args.right),
        "opponent": args.opponent,
        "seed": args.seed,
        "seat": args.seat,
        "first_divergence": first_divergence,
        "left_rewards": list(results[0].rewards),
        "right_rewards": list(results[1].rewards),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
