#!/usr/bin/env python3
"""Safely synchronize the latest Kaggriculture submission and its replays.

The script shells out to the official Kaggle CLI.  It never prints or persists the
``KAGGLE_API_TOKEN`` value.  Kaggle CLI 2.2.x+ is required for simulation episode
commands; ``submission-download`` requires a build that exposes that subcommand.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

_TOKEN_PATTERN = re.compile(r"KGAT_[A-Za-z0-9_-]+")


def _sanitize(text: str) -> str:
    return _TOKEN_PATTERN.sub("KGAT_[REDACTED]", text)


def _run(
    command: Sequence[str],
    *,
    dry_run: bool,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"[sync] {_sanitize(printable)}")
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Kaggle CLI executable not found: {command[0]!r}. "
            "Install/upgrade it with `python -m pip install -U kaggle`."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {_sanitize(printable)}\n"
            f"stdout:\n{_sanitize(result.stdout)}\n"
            f"stderr:\n{_sanitize(result.stderr)}"
        )
    return result


def _csv_rows(text: str, required_any: Iterable[str]) -> list[dict[str, str]]:
    required = {key.lower() for key in required_any}
    lines = [line for line in text.splitlines() if line.strip()]
    header_index: int | None = None
    for index, line in enumerate(lines):
        try:
            columns = next(csv.reader([line]))
        except csv.Error:
            continue
        lower = {column.strip().lower() for column in columns}
        if lower & required:
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"could not find CSV header in Kaggle output; expected one of {sorted(required)}")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    return [
        {str(key).strip(): str(value or "").strip() for key, value in row.items() if key is not None}
        for row in reader
    ]


def _first_value(row: Mapping[str, str], names: Iterable[str]) -> str:
    lookup = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.lower(), "").strip()
        if value:
            return value
    return ""


def _submission_id(row: Mapping[str, str]) -> int | None:
    value = _first_value(row, ("ref", "id", "submissionId", "submission_id"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _date_key(row: Mapping[str, str]) -> tuple[str, int]:
    date = _first_value(row, ("date", "dateSubmitted", "submittedAt", "submissionDate"))
    return (date, _submission_id(row) or -1)


def select_latest_submission(rows: Sequence[Mapping[str, str]], allow_unscored: bool) -> tuple[int, Mapping[str, str]]:
    candidates: list[Mapping[str, str]] = []
    for row in rows:
        submission_id = _submission_id(row)
        if submission_id is None:
            continue
        status = _first_value(row, ("status", "state")).lower()
        if allow_unscored or not status or status in {"complete", "completed", "scored", "finished"}:
            candidates.append(row)
    if not candidates:
        raise RuntimeError("no eligible Kaggle submission was found")
    row = max(candidates, key=_date_key)
    submission_id = _submission_id(row)
    assert submission_id is not None
    return submission_id, row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> set[Path]:
    return {path.resolve() for path in root.rglob("*") if path.is_file()} if root.exists() else set()


def _new_files(root: Path, before: set[Path]) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.resolve() not in before),
        key=lambda path: path.as_posix(),
    )


def _episode_ids(rows: Sequence[Mapping[str, str]]) -> list[int]:
    result: list[int] = []
    for row in rows:
        value = _first_value(row, ("id", "episodeId", "episode_id", "ref"))
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result), reverse=True)


def _token_available() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return (Path.home() / ".kaggle" / "access_token").is_file() or (Path.home() / ".kaggle" / "kaggle.json").is_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(r"D:\Kaggriculture"))
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-replays", type=int, default=64, help="0 downloads every listed replay")
    parser.add_argument("--sleep-seconds", type=float, default=0.75)
    parser.add_argument("--kaggle-bin", default="kaggle")
    parser.add_argument("--allow-unscored", action="store_true")
    parser.add_argument("--skip-submission-download", action="store_true")
    parser.add_argument("--skip-replays", action="store_true")
    parser.add_argument("--download-logs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    output = (args.output or workspace / "live_assets" / "own_latest").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not _token_available():
        raise RuntimeError(
            "Kaggle authentication was not found. Set KAGGLE_API_TOKEN or create "
            "~/.kaggle/access_token. The token value is never written by this script."
        )

    version = _run([args.kaggle_bin, "--version"], dry_run=args.dry_run).stdout.strip()
    submissions_result = _run(
        [args.kaggle_bin, "competitions", "submissions", args.competition, "-v"],
        dry_run=args.dry_run,
    )
    if args.dry_run and args.submission_id is None:
        raise RuntimeError("--dry-run requires --submission-id because no Kaggle CSV is fetched")

    selected_row: Mapping[str, str] = {}
    if args.submission_id is None:
        rows = _csv_rows(submissions_result.stdout, ("ref", "id", "submissionId"))
        submission_id, selected_row = select_latest_submission(rows, args.allow_unscored)
    else:
        submission_id = args.submission_id

    submission_dir = output / f"submission_{submission_id}"
    replay_dir = submission_dir / "replays"
    log_dir = submission_dir / "logs"
    submission_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    if args.download_logs:
        log_dir.mkdir(parents=True, exist_ok=True)

    downloaded_submission_files: list[Path] = []
    if not args.skip_submission_download:
        before = _snapshot_files(submission_dir)
        command = [
            args.kaggle_bin,
            "competitions",
            "submission-download",
            str(submission_id),
            "-p",
            str(submission_dir),
        ]
        if args.force:
            command.append("-o")
        _run(command, dry_run=args.dry_run)
        downloaded_submission_files = _new_files(submission_dir, before)

    episode_rows: list[dict[str, str]] = []
    episode_ids: list[int] = []
    if not args.skip_replays:
        episodes_result = _run(
            [args.kaggle_bin, "competitions", "episodes", str(submission_id), "-v"],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            episode_rows = _csv_rows(episodes_result.stdout, ("id", "episodeId"))
            episode_ids = _episode_ids(episode_rows)
            if args.max_replays > 0:
                episode_ids = episode_ids[: args.max_replays]

        for episode_id in episode_ids:
            existing = list(replay_dir.glob(f"*{episode_id}*.json"))
            if existing and not args.force:
                continue
            _run(
                [
                    args.kaggle_bin,
                    "competitions",
                    "replay",
                    str(episode_id),
                    "-p",
                    str(replay_dir),
                ],
                dry_run=args.dry_run,
            )
            if args.download_logs:
                for agent_index in (0, 1):
                    _run(
                        [
                            args.kaggle_bin,
                            "competitions",
                            "logs",
                            str(episode_id),
                            str(agent_index),
                            "-p",
                            str(log_dir),
                        ],
                        dry_run=args.dry_run,
                    )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    files = sorted((path for path in submission_dir.rglob("*") if path.is_file()), key=lambda p: p.as_posix())
    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": args.competition,
        "submission_id": submission_id,
        "selected_submission_row": dict(selected_row),
        "kaggle_cli_version": _sanitize(version),
        "workspace": str(workspace),
        "output": str(output),
        "episode_count_listed": len(episode_rows),
        "episode_ids_selected": episode_ids,
        "downloaded_submission_files": [path.relative_to(submission_dir).as_posix() for path in downloaded_submission_files],
        "files": [
            {
                "path": path.relative_to(submission_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
            if path.name != "sync_receipt.json"
        ],
        "security": {
            "token_persisted": False,
            "token_printed": False,
        },
    }
    receipt_path = submission_dir / "sync_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sync] receipt: {receipt_path}")
    print(f"[sync] submission: {submission_id}; replays selected: {len(episode_ids)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # CLI boundary: present one concise actionable failure.
        print(f"ERROR: {_sanitize(str(exc))}", file=sys.stderr)
        raise SystemExit(2)
