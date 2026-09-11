#!/usr/bin/env python3
"""Export low-level replay carriers for failure-independent intent routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    args = parser.parse_args()

    import orjson

    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    replay_paths = {}
    for root in args.replay_root:
        for path in root.glob("episode-*-replay.json"):
            replay_paths.setdefault(int(path.name.split("-")[1]), path)

    action_tapes = {}
    entries = []
    for route in specs["routes"]:
        if not route.get("selected_for_evaluation", True):
            continue
        route_id = str(route["intent_medoid_source"]["route_id"])
        episode_text, player_text = route_id.split(":")
        episode, player = int(episode_text), int(player_text)
        replay = orjson.loads(replay_paths[episode].read_bytes())
        tape = []
        for step in range(1, 720):
            action = replay["steps"][step][player].get("action") or {}
            tape.append({
                "farmer": list(action.get("farmer") or ["PASS"]),
                "hands": [list(value or ["PASS"]) for value in action.get("hands") or []],
                "market": [list(value) for value in action.get("market") or []],
            })
        action_tapes[route_id] = tape
        entries.append({
            "family": route["family"],
            "alias": route["alias"],
            "route_id": route_id,
            "team": route["intent_medoid_source"]["team"],
            "support": route["support"],
            "selected": True,
            "drop_reason": route["drop_reason"],
            "source_execution_hard_failures": route["intent_medoid_source"]["execution_hard_failures"],
        })
    packed = zlib.compress(orjson.dumps(action_tapes), level=9)
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    args.actions.write_bytes(packed)
    payload = {
        "schema_version": 1,
        "taxonomy": "unified failure-independent intent families",
        "carrier_role": "movement and maintenance skeleton only; intent specs are authoritative",
        "opponent_routes": entries,
        "selected": [entry for entry in entries if entry["selected"]],
        "actions_file": str(args.actions),
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "routes": len(entries), "selected": len(payload["selected"]),
        "compressed_bytes": len(packed),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
