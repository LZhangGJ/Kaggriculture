"""Run a generated V17-family candidate without its broad exception fallback."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from kaggriculture_lab.fast_env import FastKaggricultureEnv, resolve_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("strict_v17_candidate", args.candidate)
    if spec is None or spec.loader is None:
        raise ImportError(args.candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opponent = resolve_agent(args.opponent)
    env = FastKaggricultureEnv()
    observations = env.reset(args.seed)
    while not env.done:
        observation = observations[args.seat]
        step = min(max(0, int(module._get(observation, "step", 0) or 0)), len(module._ACTIONS) - 1)
        try:
            action = module._weed_repair_action(
                observation, module._copy_action(module._ACTIONS[step]), step
            )
            action = module._v17_feed_guard(observation, action, step)
            action = module._v17_room_evac(observation, action, step)
            action = module._repay_shift(observation, action, step)
            action = module._rank_sell_slots(observation, action, None)
            action = module._preempt_shift(observation, action, step)
            action = module._soil_route_counter(observation, action, step)
            action = module._v17_r5_counter(observation, action, step)
            action = module._v17_md_counter(observation, action, step)
            action = module._v17_room_guard(observation, action, step)
            action = module._terminal_liquidation(observation, action, step)
            action = module._align_hands(action, observation)
        except Exception as error:
            raise RuntimeError(f"candidate failed at step={step} seat={args.seat}") from error
        other = opponent(observations[1 - args.seat], env.configuration)
        result = env.step([action, other] if args.seat == 0 else [other, action])
        observations = result.observations
    state = env._require_state()
    print(
        {
            "seed": args.seed,
            "seat": args.seat,
            "candidate_reward": float(state[args.seat].reward),
            "opponent_reward": float(state[1 - args.seat].reward),
        }
    )


if __name__ == "__main__":
    main()
