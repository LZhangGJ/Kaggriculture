"""Check the frozen handoff without rerunning the 11,800 matches."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> None:
    manifest = read(ROOT / "PACKAGE_MANIFEST.json")
    files = manifest["files"]
    for relative, expected in files.items():
        path = ROOT / relative
        assert path.is_file(), f"missing: {relative}"
        assert sha(path.read_bytes()) == expected, f"changed: {relative}"

    pool = read(ROOT / "evaluation/internal13/POOL.json")
    assert len(pool) == 13 and len({row["id"] for row in pool}) == 13
    for row in pool:
        assert manifest["original_archives"][row["id"]] == {
            "archive": row["archive"], "sha256": row["archive_sha256"]
        }
        for relative, expected in row["files"].items():
            assert sha((ROOT / "agents" / row["id"] / relative).read_bytes()) == expected, (
                row["id"], relative
            )

    public = read(ROOT / "evaluation/cashflow_public10/PROTOCOL.json")
    assert len(public["opponents"]) == 10
    for row in public["opponents"]:
        for relative, expected in row["files"].items():
            assert sha((ROOT / "opponents" / row["id"] / relative).read_bytes()) == expected, (
                row["id"], relative
            )
    assert sha((ROOT / "referee/official/kaggriculture.py").read_bytes()) == public["engine_sha256"]

    for panel, receipt in manifest["receipts"].items():
        data = gzip.decompress((ROOT / "evaluation" / panel / "games.jsonl.gz").read_bytes())
        assert sha(data) == receipt["sha256_uncompressed"], panel
        rows = [json.loads(line) for line in data.splitlines()]
        assert len(rows) == receipt["games"], panel
        assert all(row.get("error") is None and row.get("steps") == 719 for row in rows), panel

    print(f"PASS: {len(files)} files, 13 agents, 10 opponents, "
          f"{sum(x['games'] for x in manifest['receipts'].values())} complete games")


if __name__ == "__main__":
    verify()
