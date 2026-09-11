#!/usr/bin/env python3
"""Offline independent audit of frozen leaderboard, raw lists and replay bytes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def verify_replay(item):
    """Pure local-file check; top-level function supports Windows process pools."""
    root, eid, row = item
    path = root / "episodes" / f"{eid}.json"
    try:
        replay = read_json(path)
        assert int(replay.get("info", {}).get("EpisodeId", -1)) == eid, "EpisodeId mismatch"
        assert isinstance(replay.get("steps"), list) and len(replay["steps"]) >= 2, "no usable steps"
        assert len(replay["steps"]) == int(row["step_frames"]), "frame count mismatch"
        size = path.stat().st_size
        assert size == int(row["bytes"]), "size mismatch"
        hashed = digest(path)
        assert hashed == row["sha256"].upper(), "SHA256 mismatch"
        return {
            "episode_id": eid,
            "status": "PASS",
            "bytes": size,
            "sha256": hashed,
            "step_frames": len(replay["steps"]),
        }
    except Exception as exc:
        return {"episode_id": eid, "status": "FAILED", "error": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--executor", choices=("threads", "processes"), default="processes")
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        raise ValueError("workers must be 1..16")
    root = a.root.resolve()
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    lb = read_json(root / "leaderboard_service_raw.json")
    frozen = {int(e["rank"]): e for e in lb["publicLeaderboard"] if 1 <= int(e["rank"]) <= a.top}
    with (root / "selected_teams.csv").open(encoding="utf-8-sig", newline="") as f:
        selected = list(csv.DictReader(f))
    check(len(selected) == a.top, "selected count")
    check(sorted(int(t["rank"]) for t in selected) == list(range(1, a.top + 1)), "ranks not exactly 1..top")
    check(len({t["team_id"] for t in selected}) == a.top, "duplicate teams")
    check(len({t["submission_id"] for t in selected}) == a.top, "duplicate submissions")
    check(len(list((root / "submission_manifests").glob("*.json"))) == a.top, "manifest count")
    expected = {}
    row_count = 0
    exclusions = Counter()
    for team in selected:
        rank, sid, tid = (int(team[k]) for k in ("rank", "submission_id", "team_id"))
        check(
            int(frozen[rank]["submissionId"]) == sid and int(frozen[rank]["teamId"]) == tid,
            f"frozen identity rank {rank}",
        )
        raw = read_json(root / "api_raw" / f"list_episodes_submission_{sid}.json")
        check(not any(raw.get(k) for k in ("nextPageToken", "nextPage", "hasMore")), f"pagination {sid}")
        eligible = {}
        for ep in raw["episodes"]:
            agents = ep.get("agents", [])
            if ep.get("state") != "COMPLETED":
                exclusions["not_completed"] += 1
                continue
            if ep.get("type") not in ("PUBLIC", "EPISODE_TYPE_PUBLIC"):
                exclusions["not_public"] += 1
                continue
            if len(agents) != 2 or not all(ag.get("teamId") and ag.get("submissionId") for ag in agents):
                errors.append(f"incomplete public agent metadata {ep['id']}")
                continue
            own = [i for i, ag in enumerate(agents) if int(ag["submissionId"]) == sid]
            if not own:
                exclusions["different_submission"] += 1
                continue
            check(len(own) == 1, f"ambiguous own seat {ep['id']}")
            eligible[int(ep["id"])] = (ep, own[0])
        manifest = read_json(root / "submission_manifests" / f"rank{rank:02d}_{sid}.json")
        check(int(manifest["submission_id"]) == sid and int(manifest["team_id"]) == tid, f"manifest identity {sid}")
        rows = manifest["episodes"]
        check(len(rows) == manifest["episode_count"] == len(eligible), f"manifest count {sid}")
        check({int(r["episode_id"]) for r in rows} == set(eligible), f"manifest omitted/extra episodes {sid}")
        row_count += len(rows)
        for row in rows:
            eid = int(row["episode_id"])
            ep, own = eligible[eid]
            check(row["own_seat"] == own, f"seat mismatch {sid}/{eid}")
            check(int(ep["agents"][own]["teamId"]) == tid, f"team mismatch {sid}/{eid}")
            check(
                int(row["opponent_submission_id"]) == int(ep["agents"][1 - own]["submissionId"]),
                f"opponent mismatch {sid}/{eid}",
            )
            check(row["status"] not in ("failed", "not_attempted"), f"incomplete download {eid}")
            if eid in expected:
                check(expected[eid].get("sha256") == row.get("sha256"), f"duplicate hash mismatch {eid}")
            expected[eid] = row

    verified = []
    executor = ProcessPoolExecutor if a.executor == "processes" else ThreadPoolExecutor
    with executor(max_workers=a.workers) as pool:
        items = ((root, eid, row) for eid, row in sorted(expected.items()))
        for n, result in enumerate(pool.map(verify_replay, items), 1):
            verified.append(result)
            if result["status"] != "PASS":
                errors.append(f"replay {result}")
            if n % 250 == 0 or n == len(expected):
                print(f"audit {n}/{len(expected)} errors={len(errors)}", flush=True)
    summary = read_json(root / "DOWNLOAD_SUMMARY.json")
    check(summary["selected_teams"] == a.top, "summary selected count")
    check(summary["unique_episode_count"] == summary["valid_unique_replays"] == len(expected), "summary unique count")
    check(summary["failed_unique_replays"] == 0, "summary failures")
    check(summary["per_submission_episode_rows"] == row_count, "summary row count")
    check(summary["latest_per_submission"] is None, "latest limit is forbidden")
    check(not list((root / "episodes").glob("*.part")), "partial files remain")
    lb_hash = digest(root / "leaderboard_service_raw.json")
    check(summary["leaderboard_sha256"] == lb_hash, "leaderboard hash mismatch")
    result = {
        "status": "FAILED" if errors else "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all completed public replays returned by frozen per-submission official API lists",
        "top": a.top,
        "unique_replays": len(expected),
        "per_submission_rows": row_count,
        "local_executor": a.executor,
        "local_workers": a.workers,
        "valid_replays": sum(r["status"] == "PASS" for r in verified),
        "excluded_api_rows": dict(exclusions),
        "leaderboard_sha256": lb_hash,
        "download_counts": summary["materialization_counts"],
        "bytes": sum(r.get("bytes", 0) for r in verified),
        "errors": errors,
    }
    (root / "ACCEPTANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "VERIFIED_REPLAY_HASHES.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
