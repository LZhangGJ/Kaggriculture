"""Find the first closed-loop divergences between PolicyV2 and its teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaggriculture_lab.fast_env import FastKaggricultureEnv, resolve_agent
from kaggriculture_lab.policy_v2 import policy_from_checkpoint, structured_policy_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, default=41_000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-mismatches", type=int, default=20)
    parser.add_argument("--mask-unit-actions", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device).eval()
    teacher = resolve_agent(args.teacher)
    opponent = resolve_agent(args.opponent)
    env = FastKaggricultureEnv()
    env.reset(args.seed)
    mismatches = []

    while not env.done:
        observations = env.observations()
        policy_observation = observations[args.seat]
        other_observation = observations[1 - args.seat]
        teacher_action = teacher(policy_observation, env.configuration)
        policy_action = structured_policy_batch(
            model,
            [policy_observation],
            device,
            deterministic=True,
            mask_unit_actions=args.mask_unit_actions,
        ).actions[0]
        opponent_action = opponent(other_observation, env.configuration)
        if policy_action != teacher_action and len(mismatches) < args.max_mismatches:
            farm = policy_observation["farms"][args.seat]
            mismatches.append(
                {
                    "step": int(policy_observation["step"]),
                    "money": float(farm["money"]),
                    "hands": len(farm["hands"]),
                    "policy": policy_action,
                    "teacher": teacher_action,
                }
            )
        actions = (
            [policy_action, opponent_action]
            if args.seat == 0
            else [opponent_action, policy_action]
        )
        env.step(actions)

    state = env._require_state()
    print(
        json.dumps(
            {
                "seed": args.seed,
                "seat": args.seat,
                "mismatches_shown": len(mismatches),
                "mismatches": mismatches,
                "policy_reward": float(state[args.seat].reward),
                "opponent_reward": float(state[1 - args.seat].reward),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
