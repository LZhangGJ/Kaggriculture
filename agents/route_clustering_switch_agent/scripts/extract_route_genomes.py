#!/usr/bin/env python3
"""Build a mutation-ready RouteGenome seed library from replay trees."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, TextIO


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.route_genome import ROUTE_GENOME_SCHEMA_VERSION, extract_route_genome


EPISODE_PATTERN = re.compile(r"episode-(\d+)-replay\.json$")


def _source(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source must be LABEL=/path/to/replay-root")
    return label.strip(), Path(raw_path.strip()).expanduser()


def _players(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("players must be a comma-separated integer list") from exc
    if not result or any(player not in (0, 1) for player in result):
        raise argparse.ArgumentTypeError("players must contain player 0 and/or player 1")
    return result


def _episode_id(path: Path) -> int:
    match = EPISODE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"not an episode replay filename: {path}")
    return int(match.group(1))


def _path_reference(label: str, root: Path, path: Path, priority: int) -> dict[str, Any]:
    relative = path.relative_to(root)
    parts = relative.parts
    value: dict[str, Any] = {
        "dataset": label,
        "relative_path": relative.as_posix(),
        "absolute_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "priority": priority,
    }
    if len(parts) >= 4 and parts[-2].lower() == "replays":
        team_directory, submission_directory = parts[-4], parts[-3]
        value["team_directory"] = team_directory
        value["submission_directory"] = submission_directory
        team_id, separator, team_name = team_directory.partition("_")
        submission_id = submission_directory.removesuffix("_agent")
        if separator and team_id.isdigit():
            value["team_id"] = int(team_id)
            value["team_slug"] = team_name
        if submission_id.isdigit():
            value["submission_id"] = int(submission_id)
    return value


def discover_replays(sources: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    for priority, (label, root) in enumerate(sources):
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"replay source does not exist: {resolved}")
        for path in resolved.rglob("episode-*-replay.json"):
            if not path.is_file():
                continue
            episode = _episode_id(path)
            reference = _path_reference(label, resolved, path, priority)
            grouped[episode].append(reference)
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
    summary = {
        "source_files": dict(sorted(source_counts.items())),
        "discovered_files": sum(source_counts.values()),
        "unique_episodes": len(tasks),
        "duplicate_file_copies": sum(source_counts.values()) - len(tasks),
        "duplicate_size_conflicts": conflicts,
    }
    return tasks, summary


def _loads_replay(path: str) -> dict[str, Any]:
    data = Path(path).read_bytes()
    try:
        import orjson
    except ImportError:
        return json.loads(data)
    loader = getattr(orjson, "loads", None)
    return loader(data) if loader is not None else json.loads(data)


def _process_task(payload: tuple[dict[str, Any], tuple[int, ...], bool]) -> dict[str, Any]:
    task, players, allow_partial = payload
    try:
        replay = _loads_replay(str(task["path"]))
        steps = list(replay.get("steps", []) or [])
        if not allow_partial and len(steps) < 720:
            raise ValueError(f"incomplete replay: {len(steps)} steps")
        references = list(task["references"])
        provenance = {
            "datasets": sorted({str(value["dataset"]) for value in references}),
            "replay_sources": references,
        }
        genomes = [
            extract_route_genome(
                replay,
                player,
                provenance=provenance,
            ).to_dict()
            for player in players
        ]
        return {"episode_id": task["episode_id"], "genomes": genomes, "error": None}
    except Exception as exc:
        return {
            "episode_id": task["episode_id"],
            "genomes": [],
            "error": {"type": type(exc).__name__, "message": str(exc), "path": task["path"]},
        }


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def _existing_records(path: Path) -> tuple[set[str], set[str], Counter[str], int]:
    source_ids: set[str] = set()
    genome_ids: set[str] = set()
    results: Counter[str] = Counter()
    records = 0
    if not path.exists():
        return source_ids, genome_ids, results, records
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid resume file at line {line_number}: {path}") from exc
            source_ids.add(str(value["source"]["source_id"]))
            genome_ids.add(str(value["genome_id"]))
            results[str(value["source"].get("result", "unknown"))] += 1
            records += 1
    return source_ids, genome_ids, results, records


def _iter_results(
    tasks: list[dict[str, Any]],
    players: tuple[int, ...],
    allow_partial: bool,
    workers: int,
) -> Iterable[dict[str, Any]]:
    payloads = ((task, players, allow_partial) for task in tasks)
    if workers <= 1:
        yield from map(_process_task, payloads)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_process_task, payloads, chunksize=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=_source,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Replay directory tree; repeat for multiple collections.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL or JSONL.GZ genome library")
    parser.add_argument("--summary", type=Path, help="Defaults to OUTPUT.summary.json")
    parser.add_argument("--errors", type=Path, help="Defaults to OUTPUT.errors.jsonl")
    parser.add_argument("--players", type=_players, default=(0, 1))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-replays", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-partial", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_replays is not None and args.max_replays <= 0:
        parser.error("--max-replays must be positive")
    output = args.output.resolve()
    summary_path = (args.summary or output.with_name(output.name + ".summary.json")).resolve()
    errors_path = (args.errors or output.with_name(output.name + ".errors.jsonl")).resolve()
    if output.exists() and not (args.resume or args.overwrite):
        parser.error(f"output exists; pass --resume or --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)

    started = time.time()
    tasks, discovery = discover_replays(list(args.source))
    if args.max_replays is not None:
        tasks = tasks[: args.max_replays]
    completed_ids, genome_ids, result_counts, record_count = _existing_records(output) if args.resume else (set(), set(), Counter(), 0)
    pending = [
        task
        for task in tasks
        if any(f"{task['episode_id']}:{player}" not in completed_ids for player in args.players)
    ]
    output_mode = "a" if args.resume and output.exists() else "w"
    error_mode = "a" if args.resume and errors_path.exists() else "w"
    error_count = 0
    processed_episodes = 0
    team_counts: Counter[str] = Counter()

    print(json.dumps({
        "status": "starting",
        **discovery,
        "selected_episodes": len(tasks),
        "pending_episodes": len(pending),
        "workers": args.workers,
        "orjson": _has_orjson(),
    }, ensure_ascii=True), flush=True)

    with _open_text(output, output_mode) as output_handle, errors_path.open(
        error_mode, encoding="utf-8", newline="\n"
    ) as error_handle:
        for result in _iter_results(pending, args.players, args.allow_partial, args.workers):
            processed_episodes += 1
            if result["error"] is not None:
                error_count += 1
                error_handle.write(json.dumps({
                    "episode_id": result["episode_id"],
                    **result["error"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
            else:
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
            if args.progress_every > 0 and processed_episodes % args.progress_every == 0:
                output_handle.flush()
                error_handle.flush()
                elapsed = max(time.time() - started, 1e-9)
                print(json.dumps({
                    "status": "running",
                    "processed_episodes": processed_episodes,
                    "pending_episodes": len(pending),
                    "genome_records": record_count,
                    "errors": error_count,
                    "episodes_per_second": processed_episodes / elapsed,
                }, ensure_ascii=True), flush=True)

    summary = {
        "schema_version": ROUTE_GENOME_SCHEMA_VERSION,
        "kind": f"route_genome_v{ROUTE_GENOME_SCHEMA_VERSION}_seed_library",
        "sources": [
            {"label": label, "root": str(path.resolve())}
            for label, path in args.source
        ],
        **discovery,
        "selected_episodes": len(tasks),
        "processed_this_run": processed_episodes,
        "genome_records": record_count,
        "unique_genome_ids": len(genome_ids),
        "duplicate_genome_records": record_count - len(genome_ids),
        "result_counts": dict(sorted(result_counts.items())),
        "team_counts_this_run": dict(team_counts.most_common()),
        "errors_this_run": error_count,
        "players": list(args.players),
        "allow_partial": args.allow_partial,
        "output": str(output),
        "errors": str(errors_path),
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "selected_episodes": summary["selected_episodes"],
        "genome_records": summary["genome_records"],
        "unique_genome_ids": summary["unique_genome_ids"],
        "errors_this_run": summary["errors_this_run"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "output": summary["output"],
        "summary": str(summary_path),
    }, ensure_ascii=True), flush=True)


def _has_orjson() -> bool:
    try:
        import orjson
    except ImportError:
        return False
    return callable(getattr(orjson, "loads", None))


if __name__ == "__main__":
    main()
