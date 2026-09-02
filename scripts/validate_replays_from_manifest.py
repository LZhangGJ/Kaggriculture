"""Validate replay files referenced by a Kaggriculture CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Parse every complete JSON document instead of checking its boundaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    missing: list[str] = []
    unique_paths: dict[str, Path] = {}
    for row in rows:
        replay_path = Path(row["replay_path"])
        if not replay_path.is_file() or replay_path.stat().st_size == 0:
            missing.append(row["episode_id"])
            continue
        unique_paths.setdefault(row["episode_id"], replay_path)

    invalid: list[tuple[str, str]] = []
    for episode_id, replay_path in unique_paths.items():
        try:
            if args.full_json:
                with replay_path.open(encoding="utf-8") as stream:
                    json.load(stream)
            else:
                with replay_path.open("rb") as stream:
                    first = stream.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
                    stream.seek(max(0, replay_path.stat().st_size - 4096))
                    last = stream.read().rstrip(b" \t\r\n")
                if not first or first[:1] not in (b"{", b"["):
                    raise ValueError("invalid JSON start boundary")
                if not last or last[-1:] not in (b"}", b"]"):
                    raise ValueError("invalid JSON end boundary")
        except Exception as exc:
            invalid.append((episode_id, type(exc).__name__))

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(row["rank"])].append(row)

    for rank in sorted(groups):
        group = groups[rank]
        reused = sum(row["was_present"].lower() == "true" for row in group)
        missing_for_team = sum(
            not Path(row["replay_path"]).is_file()
            or Path(row["replay_path"]).stat().st_size == 0
            for row in group
        )
        print(
            "TEAM|{rank}|{team}|{submission}|{score}|total={total}|new={new}|"
            "reused={reused}|missing={missing}".format(
                rank=rank,
                team=group[0]["team_name"],
                submission=group[0]["submission_id"],
                score=group[0]["public_score"],
                total=len(group),
                new=len(group) - reused,
                reused=reused,
                missing=missing_for_team,
            )
        )

    print(
        f"VALIDATION|rows={len(rows)}|unique={len(unique_paths)}|"
        f"missing={len(missing)}|invalid_json={len(invalid)}"
    )
    for episode_id, error_type in invalid:
        print(f"INVALID|{episode_id}|{error_type}")
    return 1 if missing or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
