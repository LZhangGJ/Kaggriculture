"""Atomic records and a single coordinator lock."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ident(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("Invalid record ID")
    return value


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def locked(root):
    """Kernel lock releases on crash. Never expire a lock by wall-clock age."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".coordinator.lock", "a+b") as f:
        f.seek(0)
        f.write(b"0")
        f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise RuntimeError("Coordinator already running") from e
        else:
            import fcntl
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise RuntimeError("Coordinator already running") from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


DEFAULT_CONFIG = {
    "schema": 1, "timezone": "America/New_York", "daily_seeds": 128,
    "roster_cap": 20, "placement_seeds": 32, "placement_references": [],
    "workers": 1, "max_games_per_tick": 32, "placement_fraction": 0.2,
    "execution_enabled": False, "daily_enabled": False,
    "image": None, "memory_mb": 1024, "cpus_per_agent": 1,
    "game_timeout_seconds": 600, "turn_timeout_seconds": 5,
    "bootstrap": 500, "publication_enabled": False,
    "github": {"intake_enabled": False, "repository": "", "allowed_authors": []},
    "public_refresh_commands": [],
    "public_sources": [],
    "contract": {"engine": "kaggle-environments", "version": "1.32.7",
                 "game": "kaggriculture", "episode_steps": 720,
                 "protocol": "jsonl-observation-configuration-v1",
                 "agent_failure": "forfeit", "infrastructure_failure": "unresolved"},
}


def init(root):
    root = Path(root)
    if (root / "config.json").exists():
        raise ValueError("Workspace already initialized")
    for folder in ("agents", "submissions", "artifacts", "runs", "private", "events", "experiments", "site", "snapshots"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    write(root / "config.json", DEFAULT_CONFIG)
    write(root / "roster.json", [])
    write(root / "champion.json", {"agent": None, "evidence": None})
    write(root / "private/seeds.json", {})


def event(root, kind, key, detail):
    eid = digest([kind, key])
    path = Path(root) / "events" / (eid + ".json")
    if not path.exists():
        write(path, {"id": eid, "kind": kind, "key": key, "detail": detail, "created": now(), "acknowledged": False})


def records(root, folder):
    return [read(p) for p in sorted((Path(root) / folder).glob("*.json"))]
