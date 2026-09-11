"""Verify that the WSL runtime executes compiled JAX work on the RTX GPU."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp


PROJECT = Path(__file__).resolve().parents[1]
RECEIPT = PROJECT / "receipts" / "jax_runtime.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected a JAX GPU, found: {devices!r}")

    @jax.jit
    def workload(x: jax.Array) -> jax.Array:
        return jnp.tanh(x @ x + 0.25)

    x = jnp.ones((2048, 2048), dtype=jnp.float32)
    started = time.perf_counter()
    y = workload(x)
    y.block_until_ready()
    compile_and_first_s = time.perf_counter() - started

    timings = []
    for _ in range(12):
        started = time.perf_counter()
        y = workload(x)
        y.block_until_ready()
        timings.append(time.perf_counter() - started)

    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    lock = PROJECT / "requirements-wsl.lock.txt"
    receipt = {
        "schema": "kaggriculture_jax_runtime_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "gpu": gpu_line,
        "float32_matmul_2048": {
            "compile_and_first_s": compile_and_first_s,
            "steady_median_s": statistics.median(timings),
            "steady_min_s": min(timings),
            "runs": len(timings),
            "result_sample": float(y[0, 0]),
        },
        "lock_sha256": sha256(lock),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

