#!/usr/bin/env python3
"""Submission-contract regression: drive the official engine as Kaggle does.

The interpreter writes ``step`` into player 0's observation only
(``kaggle_environments`` core sets ``new_state[0].observation.step``); player 1
receives ``day`` and ``hour`` alone.  Every local A/B harness in this project
synthesises the missing key, so a policy that silently depends on it still
scores normally in experiments while failing on seat one.

This test therefore feeds the *raw* observation to the real entry point on both
seats and asserts that the agent survives and that the replay-to-R1 handoff is
actually reached.  The opponent passes every turn: only the policy under test
is exercised, which keeps the run short and deterministic.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "agent" / "main.py"
# episodeSteps is 720 and the interpreter fires DONE at step >= episodeSteps - 2,
# so one complete episode is frames 0..718.
EPISODE_FRAMES = 719


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_seat(seat, steps, seed, entry):
    from kaggle_environments import make

    module = load(entry, f"contract_seat_{seat}")
    # agent/main.py exposes create_agent(); policy/r1/agent.py is a bare
    # module-level agent() whose live instance sits in the module's _instances.
    policy = module.create_agent(seat) if hasattr(module, "create_agent") else module.agent

    def instance():
        if hasattr(policy, "debug"):
            return policy
        return getattr(module, "_instances", {}).get(seat)

    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    seat_one_without_step = 0
    frames = 0
    error = None
    try:
        while not env.done and frames < steps:
            actions = []
            for player in (0, 1):
                observation = json.loads(json.dumps(state[player].observation))
                observation["player"] = player
                if player == 1 and "step" not in observation:
                    seat_one_without_step += 1
                actions.append(
                    policy(observation, env.configuration) if player == seat else ["PASS"]
                )
            state = env.step(actions)
            frames += 1
        owner = instance()
        debug = owner.debug() if owner is not None and hasattr(owner, "debug") else {}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        error = repr(exc)
        debug = {}
    finally:
        owner = instance()
        close = getattr(owner, "close", None)
        if close:
            close()
    return {
        "seat": seat,
        "frames": frames,
        "error": error,
        "handoff_step": debug.get("dynamic_handoff_step"),
        "dynamic_plan_calls": debug.get("plan_calls"),
        "dynamic_search_calls": debug.get("search_calls"),
        "replay_opening": debug.get("replay_opening"),
        "seat_one_observations_without_step": seat_one_without_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=EPISODE_FRAMES,
                        help=f"frames per seat ({EPISODE_FRAMES} is a complete episode)")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--seed", type=int, default=2609400000)
    parser.add_argument("--entry", type=Path, default=ENTRY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seats = [int(value) for value in args.seats.split(",")]
    if args.steps < 1 or any(seat not in (0, 1) for seat in seats):
        parser.error("steps must be positive and seats must be 0 and/or 1")
    if not args.entry.is_file():
        parser.error(f"entry does not exist: {args.entry}")
    target = min(args.steps, EPISODE_FRAMES)

    rows = [run_seat(seat, args.steps, args.seed, args.entry) for seat in seats]
    failures = []
    for row in rows:
        if row["error"]:
            failures.append(f"seat {row['seat']} raised {row['error']}")
            continue
        if row["frames"] < target:
            failures.append(f"seat {row['seat']} stopped after {row['frames']}/{target} frames")
        if row["handoff_step"] is None:
            # Pure dynamic entry: it plans from the first day boundary onwards.
            if target >= 48 and not row["dynamic_plan_calls"]:
                failures.append(f"seat {row['seat']} never planned a turn")
            continue
        if args.steps > row["handoff_step"] and not row["dynamic_plan_calls"]:
            failures.append(f"seat {row['seat']} never reached the dynamic handoff")

    result = {
        "engine": "kaggle_environments:kaggriculture",
        "entry": str(args.entry),
        "step_patch": "none; raw observations as delivered to the agent",
        "seed": args.seed,
        "seats": rows,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
