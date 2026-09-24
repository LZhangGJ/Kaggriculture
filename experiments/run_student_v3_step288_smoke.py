#!/usr/bin/env python3
"""Run one terminal FastEnv game for the step-288-only v3 student."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(state):
    result = []
    for player in (0, 1):
        observation = json.loads(json.dumps(state[player]))
        if observation.get("step") is None:
            observation["step"] = observation["day"] * 24 + observation["hour"]
        observation["player"] = player
        result.append(observation)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2620000000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--opponent", default="ahmed_v47")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-first-handoff-128-"
        "free-run-seed2620000000.json"))
    args = parser.parse_args()

    from fast_kaggriculture import Config, FastEnv

    student_module = load(
        ROOT / "experiments/student_action_event_agent.py", "student_v3_smoke")
    opponent_path = ROOT / "opponents" / args.opponent / "main.py"
    opponent_module = load(opponent_path, f"opponent_{args.opponent}")
    candidate = student_module.create_agent(args.seat)
    opponent = opponent_module.agent
    opponent_takes_configuration = len(inspect.signature(opponent).parameters) > 1
    env = FastEnv(Config(), args.seed)
    state = list(env.reset(args.seed))
    decision_seconds = 0.0
    frames = 0
    started = time.perf_counter()
    try:
        while not env.done:
            current = observations(state)
            actions = []
            for player, observation in enumerate(current):
                if player == args.seat:
                    decision_started = time.perf_counter()
                    action = candidate(observation, {})
                    decision_seconds += time.perf_counter() - decision_started
                else:
                    action = (opponent(observation, {})
                              if opponent_takes_configuration
                              else opponent(observation))
                actions.append(action)
            state = env.step(actions)
            frames += 1
        wall_seconds = time.perf_counter() - started
        own = float(env.rewards[args.seat])
        rival = float(env.rewards[1 - args.seat])
        dynamic = getattr(candidate, "dynamic", candidate)
        student = dynamic.student_summary()
        result = {
            "status": "PASS" if (frames == 719 and student["days"] == 1 and
                                    not student["fallbacks"] and
                                    not student["illegal"]) else "FAIL",
            "scope": "first_handoff_only_step288",
            "post288_policy": "frozen_r1",
            "engine": "fast_kaggriculture:FastEnv",
            "seed": args.seed,
            "seat": args.seat,
            "opponent": args.opponent,
            "frames": frames,
            "checkpoint": str(dynamic.checkpoint_path),
            "checkpoint_sha256": digest(dynamic.checkpoint_path),
            "binary": str(dynamic.binary_path),
            "binary_sha256": digest(dynamic.binary_path),
            "cash": own,
            "opponent_cash": rival,
            "margin": own - rival,
            "won": own > rival,
            "wall_seconds": wall_seconds,
            "decision_seconds": decision_seconds,
            "student": student,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(1)
    finally:
        close = getattr(candidate, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
