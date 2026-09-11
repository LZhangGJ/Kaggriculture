"""Verify the handoff's tracked file bytes against MANIFEST_SHA256.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST_SHA256.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for name, expected_hash in expected["files"].items():
        path = ROOT / name
        if not path.is_file():
            failures.append({"file": name, "error": "missing"})
        elif digest(path) != expected_hash:
            failures.append({"file": name, "error": "sha256 mismatch"})
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked": len(expected["files"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
