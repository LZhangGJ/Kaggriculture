"""Sticky-split continuous wrapper for the manifest-filtered v2 replay watcher."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import watch_replay_evolution_pool as v1
import watch_replay_evolution_pool_v2 as v2


STICKY = {"train", "dev", "sealed"}


class StickyResolver:
    def __init__(self, split_manifest: Path, manifest_sha: str, split_salt: str):
        self.manifest_sha = manifest_sha
        self.salt_sha = hashlib.sha256(split_salt.encode()).hexdigest()
        self.previous_manifest_sha = manifest_sha
        self.previous: dict[int, dict[str, Any]] = {}
        if split_manifest.exists():
            value = json.loads(split_manifest.read_text(encoding="utf-8"))
            if int(value.get("schema_version", 0)) != v2.SCHEMA_VERSION:
                raise ValueError("unsupported split manifest schema")
            if value.get("split_salt_sha256") != self.salt_sha:
                raise ValueError("split salt changed after assignments were frozen")
            self.previous_manifest_sha = str(
                value.get("top40_manifest_sha256", manifest_sha)
            )
            self.previous = {
                int(row["episode_id"]): {
                    key: item for key, item in row.items()
                    if key not in {"episode_id", "path"}
                }
                for row in value.get("episodes", [])
            }

    def __call__(
        self, task: Mapping[str, Any], manifest_types: Mapping[int, set[str]],
        split_salt: str,
    ) -> dict[str, Any]:
        if hashlib.sha256(split_salt.encode()).hexdigest() != self.salt_sha:
            raise ValueError("split salt changed during scan")
        episode = int(task["episode_id"])
        fresh_classifier = getattr(v2, "classify_task_original", v2.classify_task)
        current = fresh_classifier(task, manifest_types, split_salt)
        old = self.previous.get(episode)
        old_split = None if old is None else str(old.get("split"))
        if old_split == "excluded_source_validation":
            selected = dict(old)
        elif current["split"] == "excluded_source_validation":
            if old_split in STICKY:
                raise ValueError(
                    f"episode {episode} changed from {old_split} to source validation"
                )
            selected = dict(current)
        elif old_split in STICKY:
            selected = dict(old)
            selected["datasets"] = current["datasets"]
            selected["current_episode_types"] = current["episode_types"]
        else:
            selected = dict(current)
        selected.setdefault(
            "assigned_manifest_sha256",
            self.previous_manifest_sha if old_split in STICKY else self.manifest_sha,
        )
        selected["assignment_sticky"] = selected["split"] in (
            STICKY | {"excluded_source_validation"}
        )
        return selected


def scan_once(args, state):
    manifest_sha = hashlib.sha256(args.top40_manifest.read_bytes()).hexdigest()
    resolver = StickyResolver(args.split_manifest, manifest_sha, args.split_salt)
    original = v2.classify_task
    v2.classify_task_original = original
    v2.classify_task = resolver
    try:
        summary = v2.scan_once(args, state)
    finally:
        v2.classify_task = original
        delattr(v2, "classify_task_original")
    if summary["top40_manifest_sha256"] != manifest_sha:
        raise RuntimeError("top40 manifest changed during sticky scan")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split["sticky_assignment"] = True
    v1._write_json_atomic(args.split_manifest, split)
    summary["sticky_split_assignment"] = True
    v1._write_json_atomic(args.summary, summary)
    return summary


def parser():
    result = v2.parser()
    result.add_argument("--poll-seconds", type=float, default=60.0)
    result.add_argument("--once", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.workers <= 0 or args.stable_seconds < 0
        or args.max_replays_per_scan <= 0 or args.poll_seconds <= 0
    ):
        raise SystemExit("workers/max-replays/poll must be positive")
    for name in (
        "top40_manifest", "manifest_snapshot", "pool", "state", "summary",
        "split_manifest", "errors",
    ):
        setattr(args, name, getattr(args, name).resolve())
    state = v1._load_state(args.state)
    while True:
        scan_once(args, state)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
