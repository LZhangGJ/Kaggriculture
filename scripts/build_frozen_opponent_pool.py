"""Create an immutable, hash-deduplicated opponent pool from downloaded agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloaded-root", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"frozen pool already exists: {args.output_dir}")

    candidates: list[tuple[str, Path, str]] = []
    for root, kind in (
        (args.downloaded_root, "public_notebook_output"),
        (args.existing_root, "existing_local_agent"),
    ):
        for path in sorted(root.glob("**/main.py")):
            if args.output_dir in path.parents:
                continue
            source = path.read_text(encoding="utf-8")
            if "def agent(" not in source:
                continue
            compile(source, str(path), "exec")
            candidates.append((_candidate_name(path, root), path, kind))

    by_hash: dict[str, list[tuple[str, Path, str]]] = defaultdict(list)
    for name, path, kind in candidates:
        by_hash[_sha256(path)].append((name, path, kind))

    agents_dir = args.output_dir / "agents"
    entries = []
    used_names: set[str] = set()
    for digest, aliases in sorted(by_hash.items(), key=lambda item: item[1][0][0]):
        aliases.sort(key=lambda item: (item[2] != "public_notebook_output", len(item[1].parts)))
        canonical, source_path, source_kind = aliases[0]
        base = canonical
        suffix = 2
        while canonical in used_names:
            canonical = f"{base}_{suffix}"
            suffix += 1
        used_names.add(canonical)
        destination = agents_dir / canonical / "main.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_hash = _sha256(destination)
        if copied_hash != digest:
            raise RuntimeError(f"copy hash mismatch for {canonical}")
        entries.append(
            {
                "name": canonical,
                "sha256": digest,
                "bytes": destination.stat().st_size,
                "path": str(destination),
                "selected_source": str(source_path),
                "selected_source_kind": source_kind,
                "aliases": [
                    {"name": name, "path": str(path), "kind": kind}
                    for name, path, kind in aliases
                ],
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "downloaded_root": str(args.downloaded_root),
        "existing_root": str(args.existing_root),
        "agent_count": len(entries),
        "candidate_count": len(candidates),
        "duplicate_count": len(candidates) - len(entries),
        "agents": entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
