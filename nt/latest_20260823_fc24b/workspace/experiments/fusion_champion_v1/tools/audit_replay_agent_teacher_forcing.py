#!/usr/bin/env python3
"""Teacher-force a local Agent on recorded observations and compare actions.

The result measures same-observation behavioral agreement only.  It does not
prove free-running equivalence because a single action difference changes later
states in a real match.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


EPISODE_STEPS = 719


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_agent(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "agent", None)
    if not callable(fn):
        raise AttributeError(f"{path} has no callable agent")
    return fn


def canonical_atom(value: object) -> list:
    if not isinstance(value, (list, tuple)) or not value:
        return ["PASS"]
    return [item.item() if hasattr(item, "item") else item for item in value]


def canonical_action(value: object, unit_count: int) -> dict:
    action = value if isinstance(value, dict) else {}
    hands = list(action.get("hands", []) or [])
    hand_count = max(0, unit_count - 1)
    canonical_hands = [
        canonical_atom(hands[index]) if index < len(hands) else ["PASS"]
        for index in range(hand_count)
    ]
    market = list(action.get("market", []) or [])
    return {
        "farmer": canonical_atom(action.get("farmer", ["PASS"])),
        "hands": canonical_hands,
        "market": [canonical_atom(order) for order in market],
    }


def deep_overlay(base: object, delta: object) -> object:
    """Rehydrate Kaggle's seat-1 delta observation over seat-0 shared state."""
    if not isinstance(base, dict) or not isinstance(delta, dict):
        return copy.deepcopy(delta)
    result = copy.deepcopy(base)
    for key, value in delta.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_overlay(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def rehydrated_observation(frame: list[dict], seat: int) -> dict:
    own = frame[seat]["observation"]
    if seat == 0:
        return copy.deepcopy(own)
    return deep_overlay(frame[0]["observation"], own)


def prefix_length(mask: list[bool]) -> int:
    for index, value in enumerate(mask):
        if not value:
            return index
    return len(mask)


def audit_replay(
    replay_path: Path,
    agent_path: Path,
    team: str,
    serial: int,
    checkpoint_steps: set[int],
) -> dict:
    document = json.loads(replay_path.read_text(encoding="utf-8"))
    if len(document.get("steps", [])) != EPISODE_STEPS + 1:
        raise ValueError(f"{replay_path}: expected 720 frames")
    teams = [str(value) for value in document.get("info", {}).get("TeamNames", [])]
    matches = [index for index, value in enumerate(teams) if value == team]
    if len(matches) != 1:
        raise ValueError(f"{replay_path}: team={team!r}, teams={teams}")
    seat = matches[0]
    agent = load_agent(agent_path, f"_teacher_forced_agent_{serial}_{replay_path.stem}")
    complete_mask: list[bool] = []
    unit_mask: list[bool] = []
    market_mask: list[bool] = []
    first_differences = []
    exceptions = []
    unit_difference_steps: Counter[str] = Counter()
    market_difference_steps: Counter[str] = Counter()
    checkpoints = []

    for step in range(EPISODE_STEPS):
        observation = rehydrated_observation(document["steps"][step], seat)
        farm = observation["farms"][seat]
        unit_count = 1 + len(farm.get("hands", []) or [])
        recorded = canonical_action(
            document["steps"][step + 1][seat].get("action") or {}, unit_count
        )
        try:
            predicted = canonical_action(agent(observation, None), unit_count)
        except Exception as error:  # capture evidence instead of losing the corpus
            predicted = canonical_action({}, unit_count)
            exceptions.append(
                {
                    "step": step,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        units_equal = (
            predicted["farmer"] == recorded["farmer"]
            and predicted["hands"] == recorded["hands"]
        )
        market_equal = predicted["market"] == recorded["market"]
        complete_equal = units_equal and market_equal
        unit_mask.append(units_equal)
        market_mask.append(market_equal)
        complete_mask.append(complete_equal)
        if step in checkpoint_steps:
            checkpoints.append(
                {
                    "step": step,
                    "recorded": recorded,
                    "predicted": predicted,
                    "unit_equal": units_equal,
                    "market_equal": market_equal,
                    "complete_equal": complete_equal,
                }
            )
        if not units_equal:
            recorded_ops = [recorded["farmer"][0], *(item[0] for item in recorded["hands"])]
            predicted_ops = [predicted["farmer"][0], *(item[0] for item in predicted["hands"])]
            unit_difference_steps.update(f"{left}->{right}" for left, right in zip(recorded_ops, predicted_ops) if left != right)
        if not market_equal:
            recorded_ops = [item[0] for item in recorded["market"]]
            predicted_ops = [item[0] for item in predicted["market"]]
            market_difference_steps[f"{'/'.join(recorded_ops) or 'NONE'} -> {'/'.join(predicted_ops) or 'NONE'}"] += 1
        if not complete_equal and len(first_differences) < 20:
            first_differences.append(
                {
                    "step": step,
                    "day": step // 24,
                    "turn": step % 24,
                    "recorded": recorded,
                    "predicted": predicted,
                    "unit_equal": units_equal,
                    "market_equal": market_equal,
                }
            )

    rewards = [int(value) for value in document.get("rewards", [0, 0])]
    shops = document["steps"][-2][seat]["observation"].get("town", {}).get("unlocked_shops", [])
    return {
        "episode_id": int(document.get("info", {}).get("EpisodeId", replay_path.stem)),
        "seat": seat,
        "opponent": teams[1 - seat],
        "reward": rewards[seat],
        "opponent_reward": rewards[1 - seat],
        "won": rewards[seat] > rewards[1 - seat],
        "shops": [str(value) for value in shops],
        "complete_steps": sum(complete_mask),
        "unit_steps": sum(unit_mask),
        "market_steps": sum(market_mask),
        "exact_prefix_steps": prefix_length(complete_mask),
        "day_complete_steps": [
            sum(complete_mask[day * 24 : min((day + 1) * 24, EPISODE_STEPS)])
            for day in range(30)
        ],
        "exception_count": len(exceptions),
        "exceptions": exceptions,
        "first_differences": first_differences,
        "unit_op_difference_histogram": unit_difference_steps.most_common(),
        "market_difference_histogram": market_difference_steps.most_common(),
        "checkpoints": checkpoints,
        "replay": str(replay_path),
        "replay_sha256": sha256(replay_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-steps",
        default="168,193,216,264,288,312",
        help="Comma-separated zero-based action steps retained verbatim.",
    )
    args = parser.parse_args()
    checkpoint_steps = {
        int(value.strip())
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    if any(step < 0 or step >= EPISODE_STEPS for step in checkpoint_steps):
        raise ValueError("checkpoint step outside 0..718")

    replay_paths = sorted(args.replay_dir.resolve().glob("*.json"))
    if not replay_paths:
        raise ValueError("no replay JSON files found")
    agent_path = args.agent.resolve()
    episodes = [
        audit_replay(path, agent_path, args.team, index, checkpoint_steps)
        for index, path in enumerate(replay_paths)
    ]
    payload = {
        "schema": "kaggriculture.fusion_champion.replay-agent-teacher-forcing.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["exception_count"] == 0 for row in episodes) else "FAIL",
        "truth_boundary": (
            "Same-observation teacher-forcing agreement only. It does not prove free-running "
            "equivalence, common authorship, copying, or equal counterfactual recovery."
        ),
        "team": args.team,
        "agent": str(agent_path),
        "agent_sha256": sha256(agent_path),
        "replay_count": len(episodes),
        "aggregate": {
            "mean_complete_steps": sum(row["complete_steps"] for row in episodes) / len(episodes),
            "mean_unit_steps": sum(row["unit_steps"] for row in episodes) / len(episodes),
            "mean_market_steps": sum(row["market_steps"] for row in episodes) / len(episodes),
            "exception_count": sum(row["exception_count"] for row in episodes),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "replay_count": payload["replay_count"],
        "aggregate": payload["aggregate"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
