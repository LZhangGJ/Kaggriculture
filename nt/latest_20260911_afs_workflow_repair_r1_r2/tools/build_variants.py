from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_one(root: Path, name: str, macros: list[str]) -> dict:
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    output = build / f"{name}.so"
    flags = json.loads((root / "references/BUILD_FLAGS.json").read_text())
    command = [
        "g++",
        *flags,
        *macros,
        "policy/bridge.cpp",
        "policy/executor/vendor/simulator.cpp",
        "-o",
        str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    row = {
        "name": name,
        "macros": macros,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode == 0:
        row["binary"] = str(output)
        row["sha256"] = sha256(output)
        row["bytes"] = output.stat().st_size
    (build / f"{name}.BUILD.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    variants = [
        ("workflow_off", []),
        ("workflow_audit", ["-DP16_AFS_WORKFLOW_AUDIT=1"]),
        ("workflow_on", ["-DP16_AFS_WORKFLOW_REPAIR=1"]),
    ]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda item: compile_one(root, *item), variants))
    result = {"status": "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL", "variants": rows}
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
