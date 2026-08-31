"""Manifest-filtered replay watcher with deterministic train/dev/sealed splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import watch_replay_evolution_pool as v1


SCHEMA_VERSION = 2
PUBLIC = "EPISODE_TYPE_PUBLIC"
VALIDATION = "EPISODE_TYPE_VALIDATION"


def split_episode(episode_id: int, salt: str) -> tuple[str, int]:
    digest = hashlib.sha256(f"{salt}:{int(episode_id)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1000
    return ("train" if bucket < 800 else "dev" if bucket < 900 else "sealed"), bucket


def load_manifest_snapshot(path: Path) -> tuple[bytes, dict[int, set[str]]]:
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    result: dict[int, set[str]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("episode_id"):
            result.setdefault(int(row["episode_id"]), set()).add(
                str(row.get("episode_type", ""))
            )
    return payload, result


def classify_task(
    task: Mapping[str, Any], manifest_types: Mapping[int, set[str]], salt: str,
) -> dict[str, Any]:
    episode = int(task["episode_id"])
    datasets = sorted({
        str(reference.get("dataset", "")) for reference in task.get("references", [])
    })
    types = sorted(manifest_types.get(episode, set()))
    if "top40" in datasets:
        if VALIDATION in types:
            return {
                "split": "excluded_source_validation",
                "bucket": None,
                "episode_types": types,
                "datasets": datasets,
            }
        if PUBLIC not in types:
            return {
                "split": "excluded_not_public_in_snapshot",
                "bucket": None,
                "episode_types": types,
                "datasets": datasets,
            }
    split, bucket = split_episode(episode, salt)
    return {
        "split": split,
        "bucket": bucket,
        "episode_types": types,
        "datasets": datasets,
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _record_nontrain_state(
    tasks: Sequence[Mapping[str, Any]], classifications: Mapping[int, Mapping[str, Any]],
    state: dict[str, Any], observed_at: float,
) -> None:
    for task in tasks:
        classification = classifications[int(task["episode_id"])]
        if classification["split"] == "train":
            continue
        path = Path(str(task["path"]))
        state["files"][str(path.resolve())] = {
            **v1._signature(path),
            "episode_id": int(task["episode_id"]),
            "last_seen": observed_at,
            "status": classification["split"],
            "bucket": classification["bucket"],
        }


def scan_once(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    manifest_payload, manifest_types = load_manifest_snapshot(args.top40_manifest)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    _atomic_bytes(args.manifest_snapshot, manifest_payload)
    tasks, discovery = v1._discover_replays_fast(list(args.source))
    classifications = {
        int(task["episode_id"]): classify_task(task, manifest_types, args.split_salt)
        for task in tasks
    }
    split_counts = Counter(
        value["split"] for value in classifications.values()
    )
    _record_nontrain_state(tasks, classifications, state, started)

    train_tasks = [
        task for task in tasks
        if classifications[int(task["episode_id"])]["split"] == "train"
    ]
    completed_ids, genome_ids, result_counts, record_count = v1._existing_records(
        args.pool
    )
    pending = v1._pending_tasks(
        train_tasks, completed_ids, args.players, state,
        args.stable_seconds, started,
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
    with v1._open_text(args.pool, output_mode) as output_handle, args.errors.open(
        error_mode, encoding="utf-8", newline="\n",
    ) as error_handle:
        pending_by_episode = {
            int(task["episode_id"]): task for task in pending
        }
        for result in v1._iter_results(
            pending, args.players, False, args.workers,
        ):
            task = pending_by_episode[int(result["episode_id"])]
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
                genome["source"]["ingestion_split"] = "train"
                genome["source"]["split_salt_sha256"] = hashlib.sha256(
                    args.split_salt.encode()
                ).hexdigest()
                output_handle.write(json.dumps(
                    genome, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")
                completed_ids.add(source_id)
                genome_ids.add(str(genome["genome_id"]))
                result_counts[str(genome["source"].get("result", "unknown"))] += 1
                team_counts[str(genome["source"].get("team_name", ""))] += 1
                record_count += 1
                new_genomes += 1
            state["files"][key].update({
                "status": "ingested_train",
                "last_attempt": time.time(),
                "error": None,
                "bucket": classifications[int(task["episode_id"])]["bucket"],
            })

    state["last_scan"] = time.time()
    state["last_new_genomes"] = new_genomes
    state["schema_version"] = 1
    v1._write_json_atomic(args.state, state)
    split_rows = []
    for task in tasks:
        episode = int(task["episode_id"])
        classification = classifications[episode]
        split_rows.append({
            "episode_id": episode,
            **classification,
            "path": str(task["path"]),
        })
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": state["last_scan"],
        "split_salt_sha256": hashlib.sha256(args.split_salt.encode()).hexdigest(),
        "bucket_contract": {"train": [0, 799], "dev": [800, 899], "sealed": [900, 999]},
        "top40_manifest_snapshot": str(args.manifest_snapshot.resolve()),
        "top40_manifest_sha256": manifest_sha,
        "counts": dict(sorted(split_counts.items())),
        "episodes": split_rows,
    }
    v1._write_json_atomic(args.split_manifest, split_manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "top40_manifest_sha256": manifest_sha,
        "split_counts": dict(sorted(split_counts.items())),
        **discovery,
        "eligible_pending_train_episodes": eligible_pending,
        "processed_train_episode_batch": len(pending),
        "new_train_genomes": new_genomes,
        "errors_this_scan": errors,
        "train_genome_records": record_count,
        "unique_train_genome_ids": len(genome_ids),
        "result_counts": dict(sorted(result_counts.items())),
        "new_team_counts": dict(team_counts.most_common()),
        "pool": str(args.pool.resolve()),
        "legacy_pool_reused": False,
        "validation_or_sealed_genomes_written": False,
        "elapsed_seconds": time.time() - started,
        "scanned_at": state["last_scan"],
    }
    v1._write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=v1._source, action="append", required=True)
    result.add_argument("--top40-manifest", type=Path, required=True)
    result.add_argument("--manifest-snapshot", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--state", type=Path, required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--errors", type=Path, required=True)
    result.add_argument("--split-salt", default="route-genome-v2-20260829")
    result.add_argument("--players", type=v1._players, default=(0, 1))
    result.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    result.add_argument("--stable-seconds", type=float, default=30.0)
    result.add_argument("--max-replays-per-scan", type=int, default=64)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.workers <= 0 or args.stable_seconds < 0 or args.max_replays_per_scan <= 0:
        raise SystemExit("workers/max-replays must be positive and stable-seconds non-negative")
    for name in (
        "top40_manifest", "manifest_snapshot", "pool", "state", "summary",
        "split_manifest", "errors",
    ):
        setattr(args, name, getattr(args, name).resolve())
    state = v1._load_state(args.state)
    scan_once(args, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
