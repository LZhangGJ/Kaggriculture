"""Evaluate a PolicyV2 checkpoint against any local Kaggriculture agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.policy_v2 import (
    policy_from_checkpoint,
    structured_policy_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--no-unit-mask", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device).eval()

    games = [
        (seed, seat)
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in (0, 1)
    ]
    environments = VectorFastEnv(len(games))
    observations = environments.reset(seed for seed, _ in games)
    opponent = resolve_agent(args.opponent)
    results = None
    while not environments.envs[0].done:
        policy_observations = [
            pair[seat] for pair, (_, seat) in zip(observations, games, strict=True)
        ]
        policy_actions = structured_policy_batch(
            model,
            policy_observations,
            device,
            deterministic=True,
            mask_unit_actions=not args.no_unit_mask,
        ).actions
        action_pairs = []
        for index, (pair, (_, seat)) in enumerate(zip(observations, games, strict=True)):
            other = 1 - seat
            opponent_action = opponent(pair[other], environments.envs[index].configuration)
            action_pairs.append(
                [policy_actions[index], opponent_action]
                if seat == 0
                else [opponent_action, policy_actions[index]]
            )
        results = environments.step(action_pairs)
        observations = [result.observations for result in results]

    assert results is not None
    policy_rewards = [float(result.rewards[seat]) for result, (_, seat) in zip(results, games, strict=True)]
    opponent_rewards = [
        float(result.rewards[1 - seat]) for result, (_, seat) in zip(results, games, strict=True)
    ]
    margins = [left - right for left, right in zip(policy_rewards, opponent_rewards, strict=True)]
    wins = sum(margin > 0 for margin in margins)
    losses = sum(margin < 0 for margin in margins)
    ties = len(margins) - wins - losses
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "opponent": args.opponent,
                "games": len(games),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "score_rate": (wins + 0.5 * ties) / len(games),
                "mean_policy_reward": sum(policy_rewards) / len(games),
                "mean_opponent_reward": sum(opponent_rewards) / len(games),
                "mean_margin": sum(margins) / len(games),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
