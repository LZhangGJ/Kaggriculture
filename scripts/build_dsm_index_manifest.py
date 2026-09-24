#!/usr/bin/env python3
"""Adapt the deduplicated DSM replay index to the mature macro manifest format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import orjson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for number, row in enumerate(index["replays"], 1):
        episode = int(row["episode_id"])
        if episode in seen:
            raise ValueError(f"duplicate episode {episode}")
        seen.add(episode)
        path = Path(row["path"]).resolve()
        replay = orjson.loads(path.read_bytes())
        if int(replay["info"]["EpisodeId"]) != episode or len(replay["steps"]) != 720:
            raise ValueError(f"incomplete or mismatched replay: {path}")
        rewards = replay.get("rewards")
        seat = int(row["dsm_seat"])
        if not rewards or len(rewards) != 2 or seat not in (0, 1):
            raise ValueError(f"missing DSM reward: {path}")
        records.append({
            "episode_id": episode, "source": row["source"],
            "replay": os.path.relpath(path, args.output.parent),
            "error": None, "rewards": rewards,
            "targets": [{"team_name": "DSM", "player_index": seat,
                         "final_reward": rewards[seat]}],
        })
        if number % 50 == 0:
            print(f"indexed {number}/{len(index['replays'])}", flush=True)
    args.output.write_text(json.dumps({"replays": records}, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} unique DSM replays to {args.output}", flush=True)


if __name__ == "__main__":
    main()
