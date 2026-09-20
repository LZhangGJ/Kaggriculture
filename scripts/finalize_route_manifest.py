#!/usr/bin/env python3
"""Build a replay manifest from a completed selection without redownloading files."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _summary(task: tuple[int, str]) -> tuple[int, dict[str, Any]]:
    import orjson

    episode_id, path_text = task
    path = Path(path_text)
    payload = orjson.loads(path.read_bytes())
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError(f"invalid replay {episode_id}")
    return episode_id, {
        "bytes": path.stat().st_size,
        "team_names": [
            str(value) for value in (payload.get("info", {}) or {}).get("TeamNames", [])
        ],
        "rewards": list(payload.get("rewards", []) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--team-alias", action="append", default=[], metavar="CURRENT=REPLAY",
        help="Explicit current leaderboard name to historical replay-name mapping.",
    )
    args = parser.parse_args()
    aliases: dict[str, set[str]] = {}
    for raw in args.team_alias:
        current, separator, replay = raw.partition("=")
        if not separator or not current or not replay:
            parser.error(f"invalid --team-alias {raw!r}; expected CURRENT=REPLAY")
        aliases.setdefault(current, set()).add(replay)
    selection = json.loads((args.root / "selection.json").read_text(encoding="utf-8"))
    episode_targets = {
        int(key): list(values) for key, values in selection["episode_targets"].items()
    }
    tasks = [
        (episode_id, str(args.root / "replays" / f"episode-{episode_id}-replay.json"))
        for episode_id in sorted(episode_targets)
    ]
    summaries = {}
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        for episode_id, value in pool.imap_unordered(_summary, tasks, chunksize=1):
            summaries[episode_id] = value
            if len(summaries) % 100 == 0 or len(summaries) == len(tasks):
                print(f"parsed {len(summaries)}/{len(tasks)} replays", flush=True)
    replay_rows = []
    for episode_id, requested in sorted(episode_targets.items()):
        value = summaries[episode_id]
        names, rewards = value["team_names"], value["rewards"]
        targets = []
        for target in requested:
            for player, name in enumerate(names):
                current_name = str(target["team_name"])
                if name == current_name or name in aliases.get(current_name, set()):
                    targets.append({
                        **target, "player_index": player,
                        "replay_team_name": name,
                        "final_reward": rewards[player] if player < len(rewards) else None,
                    })
        replay_rows.append({
            "episode_id": episode_id,
            "replay": f"replays/episode-{episode_id}-replay.json",
            "requested_targets": requested, "error": None,
            "bytes": int(value["bytes"]), "team_names": names,
            "rewards": rewards,
            "targets": targets, "demonstrations_per_target": 719,
            "alignment": "steps[t-1][player_index].observation -> steps[t][player_index].action, t=1..719",
        })
    manifest = {
        "competition": selection["competition"],
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "top_teams": int(selection["top_teams"]),
            "episodes_per_team": int(selection["episodes_per_team"]),
            "public_completed_only": True,
        },
        "leaderboard_targets": selection["targets"],
        "replays": replay_rows,
    }
    path = args.root / "manifest.json"
    temporary = args.root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps({
        "manifest": str(path), "replays": len(replay_rows),
        "targets": sum(len(row["targets"]) for row in replay_rows),
        "teams": len(selection["targets"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
