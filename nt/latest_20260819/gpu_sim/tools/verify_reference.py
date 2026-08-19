"""Verify the frozen Kaggriculture reference and installed package provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import kaggle_environments


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT / "reference" / "manifest.json"


def file_facts(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "lines": len(data.decode("utf-8").splitlines()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def check_file(
    *,
    label: str,
    path: Path,
    expected: dict[str, object],
    check_size_and_lines: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        row["ok"] = False
        return row

    actual = file_facts(path)
    row["actual"] = actual
    row["expected_sha256"] = expected["sha256"]
    ok = actual["sha256"] == expected["sha256"]
    if check_size_and_lines:
        row["expected_bytes"] = expected["bytes"]
        row["expected_lines"] = expected["lines"]
        ok = (
            ok
            and actual["bytes"] == expected["bytes"]
            and actual["lines"] == expected["lines"]
        )
    row["ok"] = ok
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT / "receipts" / "reference_verification.json",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen_dir = (
        PROJECT
        / "reference"
        / f"kaggle_environments_{manifest['version'].replace('.', '_')}"
        / "kaggriculture"
    )
    installed_version = importlib.metadata.version("kaggle-environments")
    package_dir = Path(kaggle_environments.__file__).resolve().parent
    site_packages = package_dir.parent

    checks: list[dict[str, object]] = []
    for expected in manifest["frozen_files"]:
        relative = Path(expected["path"])
        checks.append(
            check_file(
                label=f"frozen:{relative.as_posix()}",
                path=frozen_dir / relative.name,
                expected=expected,
                check_size_and_lines=True,
            )
        )
        checks.append(
            check_file(
                label=f"installed:{relative.as_posix()}",
                path=site_packages / relative,
                expected=expected,
                check_size_and_lines=True,
            )
        )

    for expected in manifest["framework_hashes"]:
        relative = Path(expected["path"])
        checks.append(
            check_file(
                label=f"installed:{relative.as_posix()}",
                path=site_packages / relative,
                expected=expected,
                check_size_and_lines=False,
            )
        )

    ok = (
        installed_version == manifest["version"]
        and all(bool(row["ok"]) for row in checks)
    )
    receipt = {
        "schema": "kaggriculture_reference_verification_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST_PATH),
        "expected_package_version": manifest["version"],
        "installed_package_version": installed_version,
        "checks": checks,
        "ok": ok,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
