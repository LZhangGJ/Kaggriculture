"""Run the complete WSL test suite and persist an auditable JSON receipt."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax


PROJECT = Path(__file__).resolve().parents[1]
RECEIPT = PROJECT / "receipts" / "test_suite.json"


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q"]
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "kaggriculture_test_suite_v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "command": command,
        "python": sys.version,
        "jax": importlib.metadata.version("jax"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "output": result.stdout,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result.stdout, end="")
    print(f"receipt={RECEIPT}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

