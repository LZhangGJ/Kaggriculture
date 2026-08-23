#!/usr/bin/env python3
"""Reproduce the FC2A stall and verify the stateful PRT shadow fix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path


def load_agent(path: Path, suffix: str):
    name = f"fc2a_stall_fix_{suffix}_{hashlib.sha256(path.read_bytes()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.agent
    with_configuration = len(inspect.signature(function).parameters) >= 2

    def agent(obs, configuration):
        return function(obs, configuration) if with_configuration else function(obs)

    return agent


def is_all_pass(action: object) -> bool:
    if not isinstance(action, dict):
        return True
    farmer = action.get("farmer") or ["PASS"]
    hands = action.get("hands") or []
    market = action.get("market") or []
    return (
        (not farmer or farmer[0] == "PASS")
        and all(not row or row[0] == "PASS" for row in hands)
        and not market
    )


def longest_pass(actions: list[object]) -> dict:
    best_start = best_end = -1
    current_start = -1
    for index, action in enumerate(actions):
        if is_all_pass(action):
            if current_start < 0:
                current_start = index
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
        else:
            current_start = -1
    return {
        "start": best_start,
        "end": best_end,
        "length": 0 if best_start < 0 else best_end - best_start + 1,
    }


def run_game(candidate_path: Path, opponent_path: Path, seed: int, seat: int, label: str) -> dict:
    from kaggle_environments import make

    candidate = load_agent(candidate_path, f"{label}_{seed}_{seat}")
    agents = [candidate, str(opponent_path)]
    if seat == 1:
        agents.reverse()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(agents)
    actions = [frame[seat].action for frame in env.steps]
    final = env.steps[-1]
    return {
        "label": label,
        "seed": seed,
        "seat": seat,
        "frames": len(env.steps),
        "candidate_status": str(final[seat].status),
        "candidate_reward": float(final[seat].reward),
        "opponent_reward": float(final[1 - seat].reward),
        "nonpass_action_frames": sum(not is_all_pass(action) for action in actions),
        "longest_all_pass_span": longest_pass(actions),
        "frame289_action": actions[289],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buggy", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=545102)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label, path in (("buggy", args.buggy), ("fixed", args.fixed)):
        for seat in (0, 1):
            rows.append(run_game(path.resolve(), args.opponent.resolve(), args.seed, seat, label))

    buggy = [row for row in rows if row["label"] == "buggy"]
    fixed = [row for row in rows if row["label"] == "fixed"]
    reproduced = all(row["longest_all_pass_span"] == {"start": 289, "end": 719, "length": 431} for row in buggy)
    repaired = all(
        row["candidate_reward"] >= 70000
        and row["longest_all_pass_span"]["length"] < 48
        and row["nonpass_action_frames"] > 600
        for row in fixed
    )
    receipt = {
        "schema": "kaggriculture.fc2a.shadow-prt-fix.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if reproduced and repaired else "FAIL",
        "bug_reproduced": reproduced,
        "fix_verified": repaired,
        "semantic_gates": {
            "known_bug_span": {"start": 289, "end": 719, "length": 431},
            "fixed_min_cash": 70000,
            "fixed_max_all_pass_streak": 47,
            "fixed_min_nonpass_action_frames": 601,
        },
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
