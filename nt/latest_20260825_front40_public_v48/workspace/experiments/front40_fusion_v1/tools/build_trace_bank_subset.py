"""Build a compact trace bank while preserving global-route provenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--route-ids", required=True)
    parser.add_argument("--bootstrap-route-id", type=int, required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    route_ids = np.asarray([int(value) for value in args.route_ids.split(",") if value], np.int32)
    if route_ids.size == 0 or np.unique(route_ids).size != route_ids.size:
        raise ValueError("route IDs must be non-empty and unique")
    with np.load(args.source_bank.resolve(), allow_pickle=False) as source:
        route_count = int(source["unit_op"].shape[0])
        if np.any(route_ids < 0) or np.any(route_ids >= route_count):
            raise ValueError(f"route IDs outside source bank: {route_ids.tolist()}")
        if args.bootstrap_route_id not in route_ids:
            raise ValueError("bootstrap route must be included in route IDs")
        output = {}
        for name in source.files:
            value = np.asarray(source[name])
            if name == "bootstrap_route_id":
                output[name] = np.asarray(
                    int(np.flatnonzero(route_ids == args.bootstrap_route_id)[0]), dtype=np.int16
                )
            elif value.ndim > 0 and value.shape[0] == route_count:
                output[name] = value[route_ids]
            else:
                output[name] = value
    output["global_route_id"] = route_ids
    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bank, **output)
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-bank-subset.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_bank": str(args.source_bank),
        "source_route_count": route_count,
        "global_route_ids": route_ids.tolist(),
        "local_bootstrap_route_id": int(output["bootstrap_route_id"]),
        "output_bank": str(args.output_bank),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
