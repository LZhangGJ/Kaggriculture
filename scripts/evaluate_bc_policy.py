"""Evaluate a BC-v0 checkpoint against a fixed opponent in both seats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_policy import KaggriculturePolicy, policy_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--model-seed", type=int, default=20260814)
    parser.add_argument("--mask-actions", action="store_true", help="apply the conservative pre-action legality mask")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.model_seed)
    device = torch.device(args.device)
    hidden_size = args.hidden_size
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        hidden_size = int(checkpoint["hidden_size"])
    model = KaggriculturePolicy(hidden_size).to(device).eval()
    if checkpoint:
        model.load_state_dict(checkpoint["model"])

    games = [(seed, seat) for seed in range(args.seed_start, args.seed_start + args.seeds) for seat in (0, 1)]
    environments = VectorFastEnv(len(games))
    observations = environments.reset(seed for seed, _seat in games)
    opponent = resolve_agent(args.opponent)
    results = None
    while not environments.envs[0].done:
        policy_observations = [pair[seat] for pair, (_seed, seat) in zip(observations, games, strict=True)]
        policy_actions = policy_batch(
            model,
            policy_observations,
            device,
            deterministic=True,
            mask_actions=args.mask_actions,
        ).actions
        action_pairs = []
        for index, (pair, (_seed, seat)) in enumerate(zip(observations, games, strict=True)):
            other = 1 - seat
            opponent_action = opponent(pair[other], environments.envs[index].configuration)
            action_pairs.append(
                [policy_actions[index], opponent_action] if seat == 0 else [opponent_action, policy_actions[index]]
            )
        results = environments.step(action_pairs)
        observations = [result.observations for result in results]

    assert results is not None
    policy_rewards = []
    opponent_rewards = []
    for result, (_seed, seat) in zip(results, games, strict=True):
        policy_rewards.append(float(result.rewards[seat]))
        opponent_rewards.append(float(result.rewards[1 - seat]))
    wins = sum(left > right for left, right in zip(policy_rewards, opponent_rewards, strict=True))
    losses = sum(left < right for left, right in zip(policy_rewards, opponent_rewards, strict=True))
    ties = len(games) - wins - losses
    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "random-initialization",
        "opponent": args.opponent,
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "score_rate": (wins + 0.5 * ties) / len(games),
        "mean_policy_reward": sum(policy_rewards) / len(games),
        "mean_opponent_reward": sum(opponent_rewards) / len(games),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
