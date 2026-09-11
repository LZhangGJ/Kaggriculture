#!/usr/bin/env python3
"""Select a deterministic action medoid for one Replay route family."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


EPISODE_ACTIONS = 719


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical(action: object) -> str:
    return json.dumps(
        action or {}, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--result", default="WIN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.episode_csv)
    selected_rows = frame[
        (frame["capital_signature"].astype(str) == args.route)
        & (frame["result"].astype(str) == args.result)
    ].copy()
    if selected_rows.empty:
        raise RuntimeError("route/result filters selected no Replay")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {int(row["episode_id"]): row for row in manifest["episodes"]}
    trajectories = []
    step_counts = [Counter() for _ in range(EPISODE_ACTIONS)]
    for row in selected_rows.to_dict(orient="records"):
        episode_id = int(row["episode_id"])
        entry = entries.get(episode_id)
        if entry is None:
            raise KeyError(f"episode absent from manifest: {episode_id}")
        path = Path(entry["path"])
        if sha256(path) != str(entry["sha256"]).upper():
            raise RuntimeError(f"Replay SHA256 mismatch: {path}")
        replay = json.loads(path.read_text(encoding="utf-8"))
        if str(entry.get("module_version", "")) != "1.32.7" or len(replay["steps"]) != 720:
            raise RuntimeError(f"incompatible Replay: {episode_id}")
        seat = int(entry["own_seat"])
        actions = [
            replay["steps"][index][seat].get("action") or {}
            for index in range(1, 720)
        ]
        signatures = [canonical(action) for action in actions]
        for index, signature in enumerate(signatures):
            step_counts[index][signature] += 1
        trajectories.append(
            {
                "episode_id": episode_id,
                "seat": seat,
                "reward": float(row["reward"]),
                "opponent_reward": float(row["opponent_reward"]),
                "path": path,
                "path_sha256": sha256(path),
                "signatures": signatures,
            }
        )

    population = len(trajectories)
    for trajectory in trajectories:
        trajectory["distance"] = sum(
            population - step_counts[index][signature]
            for index, signature in enumerate(trajectory["signatures"])
        )
        trajectory["modal_steps"] = sum(
            step_counts[index][signature] == max(step_counts[index].values())
            for index, signature in enumerate(trajectory["signatures"])
        )
    medoid = min(
        trajectories,
        key=lambda row: (
            int(row["distance"]),
            -int(row["modal_steps"]),
            -float(row["reward"]),
            int(row["episode_id"]),
        ),
    )
    ranked = sorted(
        trajectories,
        key=lambda row: (
            int(row["distance"]),
            -int(row["modal_steps"]),
            -float(row["reward"]),
            int(row["episode_id"]),
        ),
    )
    payload = {
        "schema": "kaggriculture.fusion-champion.replay-route-medoid.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Winning-Replay action medoid for an isolated route probe. This is "
            "not source-code recovery and not a deployable policy."
        ),
        "official_environment_version": "1.32.7",
        "episode_csv": str(args.episode_csv.resolve()),
        "episode_csv_sha256": sha256(args.episode_csv),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "route": args.route,
        "result_filter": args.result,
        "candidate_count": population,
        "selected_episode_id": int(medoid["episode_id"]),
        "selected_seat": int(medoid["seat"]),
        "selected_reward": float(medoid["reward"]),
        "selected_opponent_reward": float(medoid["opponent_reward"]),
        "selected_replay": str(medoid["path"]),
        "selected_replay_sha256": medoid["path_sha256"],
        "selected_total_hamming_distance": int(medoid["distance"]),
        "selected_modal_steps": int(medoid["modal_steps"]),
        "ranking": [
            {
                "episode_id": int(row["episode_id"]),
                "seat": int(row["seat"]),
                "reward": float(row["reward"]),
                "opponent_reward": float(row["opponent_reward"]),
                "total_hamming_distance": int(row["distance"]),
                "modal_steps": int(row["modal_steps"]),
                "replay": str(row["path"]),
                "replay_sha256": row["path_sha256"],
            }
            for row in ranked
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "candidate_count": population,
                "selected_episode_id": payload["selected_episode_id"],
                "selected_reward": payload["selected_reward"],
                "selected_total_hamming_distance": payload[
                    "selected_total_hamming_distance"
                ],
                "selected_modal_steps": payload["selected_modal_steps"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
