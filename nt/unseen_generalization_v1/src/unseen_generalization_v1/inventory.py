"""Deterministic local inventory for public agents, replays, and submissions."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping
import zipfile

from .replay_features import looks_like_replay, summarize_replay_payload
from .schema import AssetKind, AssetRecord

_ALLOWED_SUFFIXES = {
    ".py",
    ".ipynb",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".zip",
    ".tgz",
    ".gz",
}
_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".jax_cache",
    ".uv-cache",
    "node_modules",
}
_SECRET_NAMES = {
    ".env",
    "access_token",
    "kaggle.json",
    "auth.json",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_submission_archive(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith((".tar.gz", ".tgz", ".zip")) and any(
        token in lower for token in ("submission", "agent")
    )


def _archive_metadata(path: Path) -> dict[str, Any]:
    members: list[str] = []
    main_py_sha256: str | None = None
    errors: list[str] = []
    try:
        if path.name.lower().endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    members.append(member.name)
                    normalized = member.name.replace("\\", "/").lstrip("./")
                    if normalized == "main.py":
                        extracted = archive.extractfile(member)
                        if extracted is not None:
                            main_py_sha256 = hashlib.sha256(extracted.read()).hexdigest()
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    members.append(name)
                    normalized = name.replace("\\", "/").lstrip("./")
                    if normalized == "main.py":
                        main_py_sha256 = hashlib.sha256(archive.read(name)).hexdigest()
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "member_count": len(members),
        "members_preview": sorted(members)[:50],
        "has_root_main_py": main_py_sha256 is not None,
        "main_py_sha256": main_py_sha256,
        "errors": errors,
    }


def _notebook_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        cells = payload.get("cells", []) if isinstance(payload, Mapping) else []
        code_sources: list[str] = []
        code_cells = 0
        markdown_cells = 0
        for cell in cells if isinstance(cells, list) else []:
            if not isinstance(cell, Mapping):
                continue
            cell_type = cell.get("cell_type")
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(str(part) for part in source)
            else:
                source_text = str(source)
            if cell_type == "code":
                code_cells += 1
                code_sources.append(source_text)
            elif cell_type == "markdown":
                markdown_cells += 1
        result.update(
            {
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
                "code_sha256": hashlib.sha256(
                    "\n\n# --- CELL ---\n\n".join(code_sources).encode("utf-8")
                ).hexdigest(),
                "contains_agent_function": any(
                    token in source
                    for source in code_sources
                    for token in ("def agent(", "def agent (", "agent =")
                ),
            }
        )
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        if isinstance(metadata, Mapping):
            kaggle = metadata.get("kaggle")
            if isinstance(kaggle, Mapping):
                result["kaggle_metadata"] = dict(kaggle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _json_metadata(path: Path, max_parse_bytes: int) -> tuple[AssetKind, dict[str, Any]]:
    if path.stat().st_size > max_parse_bytes:
        return AssetKind.JSON, {"parse_skipped": "file_too_large"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return AssetKind.JSON, {"error": f"{type(exc).__name__}: {exc}"}
    if looks_like_replay(payload):
        try:
            summary = summarize_replay_payload(payload, source_path=path.as_posix())
            return AssetKind.REPLAY, {
                "episode_id": summary.episode_id,
                "submission_ids": list(summary.submission_ids),
                "step_count": summary.step_count,
                "player_count": summary.player_count,
                "episode_seed": summary.episode_seed,
                "coarse_families": [row.coarse_family for row in summary.players],
                "behavior_signatures": [row.behavior_signature for row in summary.players],
                "parse_warnings": list(summary.parse_warnings),
            }
        except (TypeError, ValueError, KeyError) as exc:
            return AssetKind.REPLAY, {"error": f"{type(exc).__name__}: {exc}"}
    lower = path.name.lower()
    if "receipt" in lower or "manifest" in lower:
        return AssetKind.RECEIPT, {}
    return AssetKind.JSON, {}


def classify_asset(path: Path, max_parse_bytes: int = 128 * 1024 * 1024) -> tuple[AssetKind, dict[str, Any]]:
    lower = path.name.lower()
    if _is_submission_archive(path):
        return AssetKind.SUBMISSION_ARCHIVE, _archive_metadata(path)
    if lower == "main.py" and any(token in path.as_posix().lower() for token in ("submission", "agent")):
        return AssetKind.SUBMISSION_FILE, {}
    if path.suffix.lower() == ".ipynb":
        return AssetKind.NOTEBOOK, _notebook_metadata(path)
    if path.suffix.lower() == ".py":
        return AssetKind.PYTHON, {}
    if path.suffix.lower() == ".json":
        return _json_metadata(path, max_parse_bytes=max_parse_bytes)
    if path.suffix.lower() in {".csv", ".jsonl"}:
        return AssetKind.TABLE, {}
    if path.suffix.lower() == ".md":
        if any(token in lower for token in ("report", "summary", "audit", "readme")):
            return AssetKind.REPORT, {}
    return AssetKind.OTHER, {}


def _candidate_file(path: Path) -> bool:
    lower = path.name.lower()
    if lower in _SECRET_NAMES:
        return False
    if lower.endswith(".tar.gz"):
        return True
    return path.suffix.lower() in _ALLOWED_SUFFIXES


def _iter_files(root: Path, excluded_paths: tuple[Path, ...]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        resolved = path.resolve()
        if any(resolved == excluded or resolved.is_relative_to(excluded) for excluded in excluded_paths):
            continue
        if path.is_file() and _candidate_file(path):
            yield path


def scan_assets(
    roots: Mapping[str, str | Path],
    *,
    max_parse_bytes: int = 128 * 1024 * 1024,
    exclude_paths: Iterable[str | Path] = (),
) -> list[AssetRecord]:
    records: list[AssetRecord] = []
    excluded = tuple(Path(path).expanduser().resolve() for path in exclude_paths)
    for label, raw_root in sorted(roots.items()):
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"inventory root does not exist: {label}={root}")
        for path in sorted(_iter_files(root, excluded), key=lambda item: item.as_posix().lower()):
            stat = path.stat()
            kind, metadata = classify_asset(path, max_parse_bytes=max_parse_bytes)
            records.append(
                AssetRecord(
                    root_label=label,
                    relative_path=path.relative_to(root).as_posix(),
                    kind=kind,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=sha256_file(path),
                    metadata=metadata,
                )
            )
    return records


def build_asset_manifest(
    roots: Mapping[str, str | Path],
    *,
    max_parse_bytes: int = 128 * 1024 * 1024,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    records = scan_assets(
        roots,
        max_parse_bytes=max_parse_bytes,
        exclude_paths=exclude_paths,
    )
    kinds = Counter(record.kind.value for record in records)
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        hashes[record.sha256].append(f"{record.root_label}:{record.relative_path}")
    duplicates = {
        digest: sorted(paths)
        for digest, paths in sorted(hashes.items())
        if len(paths) > 1
    }
    replay_families: Counter[str] = Counter()
    replay_parse_errors = 0
    for record in records:
        if record.kind is AssetKind.REPLAY:
            families = record.metadata.get("coarse_families", [])
            if isinstance(families, list):
                replay_families.update(str(value) for value in families)
            if record.metadata.get("error"):
                replay_parse_errors += 1
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {label: str(Path(path).expanduser().resolve()) for label, path in sorted(roots.items())},
        "record_count": len(records),
        "kind_counts": dict(sorted(kinds.items())),
        "duplicate_content_groups": duplicates,
        "replay_family_counts": dict(sorted(replay_families.items())),
        "replay_parse_errors": replay_parse_errors,
        "records": [record.to_dict() for record in records],
    }
