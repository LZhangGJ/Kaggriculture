#!/usr/bin/env python3
"""Dump one seat's public board and private carrying state at one official frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve(value: str) -> str:
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (ROOT / path).resolve())


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): plain(item) for key, item in dict(value).items()}
    try:
        return [plain(item) for item in value]
    except TypeError:
        return str(value)


def tile_label(tile) -> str:
    if not isinstance(tile, dict):
        return "."
    kind = str(tile.get("kind") or "?")
    content = str(tile.get("animal") or tile.get("crop") or "")
    return f"{kind}:{content}" if content else kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from kaggle_environments import make

    agents = [resolve(args.candidate), resolve(args.opponent)]
    if args.seat == 1:
        agents.reverse()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": args.seed},
        debug=False,
    )
    env.run(agents)
    frame_index = min(max(0, args.frame), len(env.steps) - 1)
    actor = env.steps[frame_index][args.seat]
    obs = plain(actor.observation)
    farm = dict((obs.get("farms") or [{}, {}])[args.seat] or {})
    private = dict(obs.get("private") or {})
    payload = {
        "schema": "kaggriculture-official-board-step-v1",
        "seed": args.seed,
        "seat": args.seat,
        "frame": frame_index,
        "observation_step": int(obs.get("step", 0) or 0),
        "status": str(actor.status),
        "reward": actor.reward,
        "money": int(farm.get("money", 0) or 0),
        "farmer": farm.get("farmer"),
        "hands": farm.get("hands", []),
        "inventories": private.get("inventories", []),
        "shed": private.get("shed", {}),
        "tiles": [[tile_label(tile) for tile in row] for row in farm.get("tiles", [])],
        "raw_tiles": farm.get("tiles", []),
        "action": plain(actor.action),
        "final_rewards": [float(row.reward) for row in env.steps[-1]],
        "final_status": [str(row.status) for row in env.steps[-1]],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "seed", "seat", "frame", "observation_step", "money", "farmer",
        "hands", "inventories", "shed", "tiles", "action", "final_rewards",
        "final_status",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
