#!/usr/bin/env python3
"""Build a frozen C++ opponent library from deduplicated Top-60 replays.

This exporter is deliberately non-destructive.  It selects one representative
from every near-route family for the requested ranks, preserves the source seat
and official random seed, and writes the normal compressed route-tape format
consumed by ``NativeTeammateBundle``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib

import ijson


def parse_ranks(raw: str) -> set[int]:
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_seed(path: Path) -> int:
    with path.open("rb") as handle:
        values = ijson.items(handle, "info.seed", use_float=True)
        try:
            return int(next(values))
        except StopIteration as exc:
            raise RuntimeError(f"missing info.seed: {path}") from exc


def normalize_action(action: Any) -> dict[str, Any]:
    action = action or {}
    farmer = action.get("farmer") or ["PASS"]
    # Official actions encode the farmer as one command, not a command list.
    if farmer and isinstance(farmer[0], list):
        farmer = farmer[0]
    return {
        "farmer": list(farmer),
        "hands": [list(command or ["PASS"]) for command in action.get("hands") or []],
        "market": [list(command) for command in action.get("market") or []],
    }


def read_action_tape(path: Path, seat: int) -> list[dict[str, Any]]:
    tape: list[dict[str, Any]] = []
    flat_count = 0
    with path.open("rb") as handle:
        for flat_index, action in enumerate(
            ijson.items(handle, "steps.item.item.action", use_float=True)
        ):
            flat_count += 1
            frame = flat_index // 2
            action_seat = flat_index % 2
            if frame == 0 or action_seat != seat:
                continue
            tape.append(normalize_action(action))
    if flat_count != 1440:
        raise RuntimeError(f"expected 1440 agent frames, got {flat_count}: {path}")
    if len(tape) != 719:
        raise RuntimeError(f"expected 719 actions, got {len(tape)}: {path}")
    return tape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--dedup-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exclude-ranks", default="1,4,9,49", type=parse_ranks)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = sorted((args.dedup_root / "submission_reports").glob("rank*.json"))
    if len(reports) != 60:
        raise RuntimeError(f"expected 60 submission reports, got {len(reports)}")

    action_tapes: dict[str, list[dict[str, Any]]] = {}
    entries: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    selected_member_count = 0
    excluded: list[dict[str, Any]] = []

    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rank = int(report["rank"])
        if rank in args.exclude_ranks:
            excluded.append(
                {
                    "rank": rank,
                    "team_name": report["team_name"],
                    "submission_id": int(report["submission_id"]),
                    "trajectory_count": int(report["episode_count"]),
                    "family_count": int(report["near_family_count"]),
                }
            )
            continue

        episode_rows = {int(row["episode_id"]): row for row in report["episodes"]}
        for family in report["near_families"]:
            episode_id = int(family["representative_episode_id"])
            episode = episode_rows[episode_id]
            seat = int(episode["seat"])
            group_id = str(family["group_id"])
            route_id = f"R{rank:02d}_{group_id}_E{episode_id}_P{seat}"
            replay_path = args.snapshot_root / "episodes" / f"{episode_id}.json"
            if not replay_path.is_file():
                raise FileNotFoundError(replay_path)
            seed = read_seed(replay_path)
            action_tapes[route_id] = read_action_tape(replay_path, seat)

            member_count = int(family["member_count"])
            selected_member_count += member_count
            entry = {
                "family": route_id,
                "alias": f"rank{rank:02d}-{group_id}",
                "route_id": route_id,
                "team": report["team_name"],
                "support": member_count,
                "selected": True,
            }
            entries.append(entry)
            cases.append(
                {
                    "route_id": route_id,
                    "family": route_id,
                    "rank": rank,
                    "team_id": int(report["team_id"]),
                    "team_name": report["team_name"],
                    "submission_id": int(report["submission_id"]),
                    "leaderboard_score": float(report["score"]),
                    "near_family_id": group_id,
                    "near_similarity_threshold": float(
                        report["near_family_similarity_threshold"]
                    ),
                    "member_count": member_count,
                    "member_episode_ids": [
                        int(value) for value in family["member_episode_ids"]
                    ],
                    "representative_episode_id": episode_id,
                    "opponent_seat": seat,
                    "candidate_seat": 1 - seat,
                    "official_seed": seed,
                    "representative_result": episode["result"],
                    "representative_own_reward": float(episode["own_reward"]),
                    "representative_opponent_reward": float(episode["opponent_reward"]),
                    "representative_margin": float(episode["margin"]),
                    "raw_action_sha256": episode["raw_action_sha256"],
                    "semantic_step_sha256": episode["semantic_step_sha256"],
                    "daily_macro_sha256": episode["daily_macro_sha256"],
                    "source_replay_path": str(replay_path),
                    "source_replay_sha256": sha256(replay_path),
                }
            )

    if len(cases) != 153:
        raise RuntimeError(f"expected 153 representative families, got {len(cases)}")
    if selected_member_count != 5007:
        raise RuntimeError(
            f"expected 5007 represented trajectories, got {selected_member_count}"
        )
    if len(action_tapes) != len(cases) or len(entries) != len(cases):
        raise RuntimeError("route IDs are not unique")

    actions_path = args.output_root / "top60_nondynamic_family_actions.json.zlib"
    metadata_path = args.output_root / "top60_nondynamic_route_library.json"
    cases_path = args.output_root / "top60_nondynamic_candidate8_cases.json"
    packed = zlib.compress(
        json.dumps(action_tapes, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        level=9,
    )
    actions_path.write_bytes(packed)
    metadata = {
        "schema_version": 1,
        "taxonomy": "top60 2026-09-01 non-dynamic near-family representatives",
        "opponent_routes": entries,
        "selected": entries,
        "actions_file": str(actions_path),
        "actions_sha256": sha256(actions_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "schema": "candidate8-top60-nondynamic-cases-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_root": str(args.snapshot_root),
        "dedup_root": str(args.dedup_root),
        "near_family_similarity_threshold": 0.90,
        "excluded_dynamic_submissions": sorted(excluded, key=lambda row: row["rank"]),
        "included_submission_count": 56,
        "representative_family_count": len(cases),
        "represented_source_trajectory_count": selected_member_count,
        "actions_path": str(actions_path),
        "actions_sha256": sha256(actions_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "cases": cases,
        "status": "PASS",
    }
    cases_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "representative_families": len(cases),
                "represented_trajectories": selected_member_count,
                "excluded": excluded,
                "actions_compressed_bytes": len(packed),
                "output_root": str(args.output_root),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
