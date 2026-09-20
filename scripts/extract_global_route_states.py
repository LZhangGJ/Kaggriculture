#!/usr/bin/env python3
"""Extract compact causal checkpoint states for every labelled replay side."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from meta_agent.src.recurrent_meta import (
    market_town_vector,
    private_plan_vector,
    public_farm_vector,
)


DEFAULT_CHECKPOINTS = (
    0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240,
    288, 336, 384, 432, 480, 528, 576, 624, 648, 672, 696,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    return parser.parse_args()


def _extract(task: tuple[str, list[dict[str, Any]], tuple[int, ...]]) -> list[dict[str, Any]]:
    import orjson

    path, targets, checkpoints = task
    replay = orjson.loads(Path(path).read_bytes())
    steps = list(replay.get("steps", []) or [])
    names = [str(value) for value in (replay.get("info", {}) or {}).get("TeamNames", [])]
    result = []
    for target in targets:
        player = int(target["player_index"])
        states = {}
        for checkpoint in checkpoints:
            index = min(int(checkpoint), max(0, len(steps) - 1))
            observation = dict(steps[index][player].get("observation") or {})
            observation["player"] = player
            observation["step"] = index
            farms = list(observation.get("farms", []) or [])
            own = farms[player] if player < len(farms) else {}
            opponent = farms[1 - player] if 1 - player < len(farms) else {}
            states[int(checkpoint)] = {
                "self_public": public_farm_vector(own).astype(np.float16),
                "opponent_public": public_farm_vector(opponent).astype(np.float16),
                # The candidate continuation descriptor is joined later.  This
                # vector deliberately contains only current private inventory.
                "self_private": private_plan_vector(observation).astype(np.float16),
                "market_town": market_town_vector(observation).astype(np.float16),
            }
        result.append({
            "episode_id": int(target["episode_id"]),
            "player_index": player,
            "team": str(target["team"]),
            "opponent": names[1 - player] if len(names) == 2 else "",
            "states": states,
        })
    return result


def main() -> None:
    args = arguments()
    with np.load(args.macro_cache, allow_pickle=True) as cached:
        macro_rows = list(cached["rows"])
    wanted = {
        (int(row["episode_id"]), int(row["player_index"]), str(row["team"]))
        for row in macro_rows
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = []
    found = set()
    for replay in manifest.get("replays", []):
        if replay.get("error") is not None:
            continue
        targets = []
        for target in replay.get("targets", []) or []:
            key = (
                int(replay["episode_id"]), int(target["player_index"]),
                str(target["team_name"]),
            )
            if key in wanted:
                found.add(key)
                targets.append({
                    "episode_id": key[0], "player_index": key[1], "team": key[2],
                })
        if targets:
            tasks.append((
                str((args.manifest.parent / replay["replay"]).resolve()),
                targets, tuple(sorted(set(args.checkpoints))),
            ))
    missing = wanted - found
    if missing:
        raise RuntimeError(f"manifest is missing {len(missing)} labelled replay sides")
    rows = []
    workers = min(max(1, args.workers), len(tasks))
    with mp.get_context("spawn").Pool(workers) as pool:
        for values in pool.imap_unordered(_extract, tasks, chunksize=1):
            rows.extend(values)
            if len(rows) % 100 < len(values):
                print(f"extracted {len(rows)}/{len(wanted)} sides", flush=True)
    rows.sort(key=lambda row: (
        int(row["episode_id"]), int(row["player_index"]), str(row["team"])
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray([1], dtype=np.int16),
        checkpoints=np.asarray(sorted(set(args.checkpoints)), dtype=np.int16),
        rows=np.asarray(rows, dtype=object),
    )
    print(json.dumps({
        "output": str(args.output), "sides": len(rows), "replays": len(tasks),
        "checkpoints": len(set(args.checkpoints)), "bytes": args.output.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
