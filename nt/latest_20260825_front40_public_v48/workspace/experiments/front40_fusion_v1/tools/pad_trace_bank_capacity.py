"""Pad a Replay trace bank to a fixed route capacity without changing real routes.

Only the first ``source_route_count`` routes are eligible for screening.  Extra
rows copy the bootstrap route solely to keep JAX argument shapes identical
across player families and allow persistent compilation-cache reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as source:
        if "source_reward" not in source.files or "bootstrap_route_id" not in source.files:
            raise ValueError("source is not a supported trace bank")
        route_count = int(source["source_reward"].shape[0])
        capacity = int(args.capacity)
        if capacity < route_count:
            raise ValueError(f"capacity {capacity} is below route count {route_count}")
        bootstrap = int(np.asarray(source["bootstrap_route_id"]).item())
        if bootstrap < 0 or bootstrap >= route_count:
            raise ValueError(f"invalid bootstrap route {bootstrap}")

        padded: dict[str, np.ndarray] = {}
        for key in source.files:
            value = np.asarray(source[key])
            if value.ndim == 0:
                padded[key] = value.copy()
                continue
            if value.shape[0] != route_count:
                raise ValueError(
                    f"route-axis mismatch for {key}: {value.shape[0]} != {route_count}"
                )
            if capacity == route_count:
                padded[key] = value.copy()
                continue
            filler = np.repeat(value[bootstrap : bootstrap + 1], capacity - route_count, axis=0)
            padded[key] = np.concatenate((value, filler), axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **padded)

    with np.load(args.output, allow_pickle=False) as check:
        for key, original in padded.items():
            if not np.array_equal(np.asarray(check[key]), original):
                raise RuntimeError(f"written array differs: {key}")

    receipt = {
        "schema": "kaggriculture.front40_fusion.padded-trace-bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "source_route_count": route_count,
        "capacity": capacity,
        "padding_route_id": bootstrap,
        "eligible_route_ids": list(range(route_count)),
        "padding_rows_are_eligible": False,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "semantic_claim": "routes [0, source_route_count) are byte-identical to source",
    }
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "routes": route_count, "capacity": capacity}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
