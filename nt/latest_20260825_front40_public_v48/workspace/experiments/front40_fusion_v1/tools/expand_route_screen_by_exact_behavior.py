"""Expand a compact route screen to routes with exactly identical runtime behavior.

Replay collections often contain the same controlled-agent program under
different public shop histories.  Those copies are useful as routing metadata,
but re-running every copy wastes Arena work.  This tool copies a screened
representative only after proving byte equality for every bank field that can
change the emitted controlled action in the forced-route evaluator:

* all eight raw unit/market action arrays;
* expected unit positions and activity masks used by local recovery.

Opponent summaries, source reward and source shop history are deliberately not
part of the behavior hash: forced-route execution does not consult them.  They
remain route-specific metadata in the expanded result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BEHAVIOR_FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
    "expected_unit_pos",
    "expected_unit_active",
)
ROUTE_MATRIX_FIELDS = (
    "cash",
    "opponent_cash",
    "margin",
    "invalid",
    "resync",
    "hard",
    "observed_shops",
)


def _behavior_hash(bank: np.lib.npyio.NpzFile, route_id: int) -> str:
    digest = hashlib.sha256()
    for field in BEHAVIOR_FIELDS:
        value = np.ascontiguousarray(bank[field][route_id])
        digest.update(field.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _assert_behavior_equal(
    bank: np.lib.npyio.NpzFile, left: int, right: int
) -> None:
    for field in BEHAVIOR_FIELDS:
        if not np.array_equal(bank[field][left], bank[field][right]):
            raise ValueError(
                f"behavior hash collision or incomplete match: routes {left}/{right}, field {field}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.trace_bank.resolve(), allow_pickle=False) as bank, np.load(
        args.screen.with_suffix(".npz").resolve(), allow_pickle=False
    ) as screen:
        route_count = int(bank["source_reward"].shape[0])
        screen_route_ids = np.asarray(screen["route_ids"], dtype=np.int32)
        if screen_route_ids.ndim != 1 or np.unique(screen_route_ids).size != screen_route_ids.size:
            raise ValueError("screen route_ids must be a one-dimensional unique list")
        if np.any(screen_route_ids < 0) or np.any(screen_route_ids >= route_count):
            raise ValueError("screen route_ids are outside trace bank")

        hashes = [_behavior_hash(bank, route_id) for route_id in range(route_count)]
        representative_by_hash: dict[str, int] = {}
        screen_column_by_hash: dict[str, int] = {}
        for column, route_id in enumerate(screen_route_ids.tolist()):
            behavior_hash = hashes[route_id]
            if behavior_hash in representative_by_hash:
                _assert_behavior_equal(bank, representative_by_hash[behavior_hash], route_id)
                continue
            representative_by_hash[behavior_hash] = route_id
            screen_column_by_hash[behavior_hash] = column

        missing_hashes = sorted(set(hashes) - set(screen_column_by_hash))
        if missing_hashes:
            missing_routes = [index for index, value in enumerate(hashes) if value in missing_hashes]
            raise ValueError(
                f"screen does not cover {len(missing_hashes)} behavior groups; routes={missing_routes}"
            )

        representative_routes = np.empty(route_count, dtype=np.int32)
        source_columns = np.empty(route_count, dtype=np.int32)
        exact_checks = 0
        for route_id, behavior_hash in enumerate(hashes):
            representative = representative_by_hash[behavior_hash]
            _assert_behavior_equal(bank, representative, route_id)
            exact_checks += len(BEHAVIOR_FIELDS)
            representative_routes[route_id] = representative
            source_columns[route_id] = screen_column_by_hash[behavior_hash]

        output: dict[str, np.ndarray] = {
            "seeds": np.asarray(screen["seeds"]),
            "first_shop": np.asarray(screen["first_shop"]),
        }
        for field in ROUTE_MATRIX_FIELDS:
            value = np.asarray(screen[field])
            if value.ndim < 3 or value.shape[2] != screen_route_ids.size:
                raise ValueError(f"unexpected route axis for {field}: {value.shape}")
            output[field] = np.take(value, source_columns, axis=2)
        output.update(
            source_episode_id=np.asarray(bank["source_episode_id"]),
            source_reward=np.asarray(bank["source_reward"]),
            source_first_shop=np.asarray(bank["source_shop_sequence"][:, 0]),
            route_ids=np.arange(route_count, dtype=np.int32),
        )

    if not np.array_equal(output["margin"], output["cash"] - output["opponent_cash"]):
        raise ValueError("expanded margin does not equal cash - opponent_cash")
    hard_total = int(np.sum(output["hard"], dtype=np.int64))
    input_receipt = json.loads(args.screen.read_text(encoding="utf-8"))
    all_done = bool(input_receipt.get("all_done", False))
    status = "PASS" if all_done and hard_total == 0 else "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **output)
    groups: dict[int, list[int]] = {}
    for route_id, representative in enumerate(representative_routes.tolist()):
        groups.setdefault(representative, []).append(route_id)
    payload = {
        "schema": "kaggriculture.front40_fusion.exact-behavior-screen-expansion.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "trace_bank": str(args.trace_bank),
        "input_screen": str(args.screen),
        "input_route_count": int(screen_route_ids.size),
        "output_route_count": route_count,
        "behavior_fields": list(BEHAVIOR_FIELDS),
        "behavior_group_count": len(groups),
        "representative_routes": sorted(groups),
        "group_sizes": {str(route): len(route_ids) for route, route_ids in sorted(groups.items())},
        "exact_field_checks": exact_checks,
        "all_done": all_done,
        "hard_counter_total": hard_total,
        "matrix": str(args.output.with_suffix(".npz")),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
