#!/usr/bin/env python3
"""Continuously append stable, complete new replays to a RouteGenome pool."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from extract_route_genomes import (
    _existing_records,
    _iter_results,
    _open_text,
    _episode_id,
    _path_reference,
    _players,
    _source,
)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "files": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported watcher state: {path}")
    value.setdefault("files", {})
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _discover_replays_fast(
    sources: list[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan only the documented replay layouts, not every agent source tree."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    patterns = (
        "episode-*-replay.json",
        "*/replays/episode-*-replay.json",
        "*/*/replays/episode-*-replay.json",
    )
    for priority, (label, root) in enumerate(sources):
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"replay source does not exist: {resolved}")
        seen: set[Path] = set()
        for pattern in patterns:
            for path in resolved.glob(pattern):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                episode = _episode_id(path)
                grouped[episode].append(_path_reference(label, resolved, path, priority))
                source_counts[label] += 1
    tasks = []
    conflicts = []
    for episode, references in sorted(grouped.items()):
        ordered = sorted(
            references,
            key=lambda value: (int(value["priority"]), str(value["absolute_path"])),
        )
        sizes = sorted({int(value["bytes"]) for value in ordered})
        if len(sizes) > 1:
            conflicts.append({"episode_id": episode, "sizes": sizes, "copies": len(ordered)})
        tasks.append({
            "episode_id": episode,
            "path": ordered[0]["absolute_path"],
            "references": [
                {key: item for key, item in value.items() if key != "priority"}
                for value in ordered
            ],
        })
    discovered = sum(source_counts.values())
    return tasks, {
        "source_files": dict(sorted(source_counts.items())),
        "discovered_files": discovered,
        "unique_episodes": len(tasks),
        "duplicate_file_copies": discovered - len(tasks),
        "duplicate_size_conflicts": conflicts,
    }


def _pending_tasks(
    tasks: list[dict[str, Any]],
    completed_ids: set[str],
    players: tuple[int, ...],
    state: dict[str, Any],
    stable_seconds: float,
    now: float,
) -> list[dict[str, Any]]:
    pending = []
    files = state["files"]
    for task in tasks:
        if not any(f"{task['episode_id']}:{player}" not in completed_ids for player in players):
            continue
        path = Path(str(task["path"]))
        signature = _signature(path)
        key = str(path.resolve())
        previous = files.get(key, {})
        files[key] = {
            **previous,
            **signature,
            "episode_id": int(task["episode_id"]),
            "last_seen": now,
        }
        if now - path.stat().st_mtime < stable_seconds:
            files[key]["status"] = "settling"
            continue
        if (
            previous.get("status") == "error"
            and previous.get("bytes") == signature["bytes"]
            and previous.get("mtime_ns") == signature["mtime_ns"]
        ):
            continue
        pending.append(task)
    return pending


def scan_once(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    tasks, discovery = _discover_replays_fast(list(args.source))
    completed_ids, genome_ids, result_counts, record_count = _existing_records(args.pool)
    pending = _pending_tasks(
        tasks,
        completed_ids,
        args.players,
        state,
        args.stable_seconds,
        started,
    )
    pending.sort(
        key=lambda task: Path(str(task["path"])).stat().st_mtime_ns,
        reverse=True,
    )
    eligible_pending = len(pending)
    pending = pending[: args.max_replays_per_scan]
    new_genomes = 0
    errors = 0
    team_counts: Counter[str] = Counter()
    output_mode = "a" if args.pool.exists() else "w"
    error_mode = "a" if args.errors.exists() else "w"
    args.pool.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(args.pool, output_mode) as output_handle, args.errors.open(
        error_mode, encoding="utf-8", newline="\n"
    ) as error_handle:
        for result in _iter_results(pending, args.players, False, args.workers):
            task = next(value for value in pending if value["episode_id"] == result["episode_id"])
            key = str(Path(str(task["path"])).resolve())
            if result["error"] is not None:
                errors += 1
                state["files"][key].update({
                    "status": "error",
                    "last_attempt": time.time(),
                    "error": result["error"],
                })
                error_handle.write(json.dumps({
                    "observed_at": time.time(),
                    "episode_id": result["episode_id"],
                    **result["error"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
                continue
            for genome in result["genomes"]:
                source_id = str(genome["source"]["source_id"])
                if source_id in completed_ids:
                    continue
                output_handle.write(json.dumps(
                    genome,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")
                completed_ids.add(source_id)
                genome_ids.add(str(genome["genome_id"]))
                result_counts[str(genome["source"].get("result", "unknown"))] += 1
                team_counts[str(genome["source"].get("team_name", ""))] += 1
                record_count += 1
                new_genomes += 1
            state["files"][key].update({
                "status": "ingested",
                "last_attempt": time.time(),
                "error": None,
            })
    state["last_scan"] = time.time()
    state["last_new_genomes"] = new_genomes
    _write_json_atomic(args.state, state)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "sources": [
            {"label": label, "root": str(path.resolve())} for label, path in args.source
        ],
        **discovery,
        "eligible_pending_episodes": eligible_pending,
        "processed_episode_batch": len(pending),
        "new_genomes": new_genomes,
        "errors_this_scan": errors,
        "genome_records": record_count,
        "unique_genome_ids": len(genome_ids),
        "result_counts": dict(sorted(result_counts.items())),
        "new_team_counts": dict(team_counts.most_common()),
        "pool": str(args.pool.resolve()),
        "elapsed_seconds": time.time() - started,
        "scanned_at": state["last_scan"],
    }
    _write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_source, action="append", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--players", type=_players, default=(0, 1))
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-seconds", type=float, default=30.0)
    parser.add_argument("--max-replays-per-scan", type=int, default=64)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.workers <= 0
        or args.poll_seconds <= 0
        or args.stable_seconds < 0
        or args.max_replays_per_scan <= 0
    ):
        parser.error(
            "workers/poll/max-replays must be positive and stable-seconds non-negative"
        )
    args.pool = args.pool.resolve()
    args.state = args.state.resolve()
    args.summary = args.summary.resolve()
    args.errors = args.errors.resolve()
    state = _load_state(args.state)
    while True:
        scan_once(args, state)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
