"""Summarize unique route-card instances stored in an M3.9 trace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


CARD_TYPES = {
    1: "FEED_TOUR",
    2: "CROP_HARVEST_TOUR",
    3: "ANIMAL_HARVEST_TOUR",
    4: "WATER_TOUR",
    5: "FERTILIZER_TOUR",
    6: "PLACE_TOUR",
    7: "CROP_FIELD_TOUR",
    8: "MIXED_FARM_TOUR",
}
ACTION_BITS = {
    1: "FEED",
    2: "CARE",
    4: "COLLECT_FERTILIZER",
    8: "HARVEST",
    16: "REPLANT_AND_WATER",
    32: "WATER",
    64: "FERTILIZE",
    128: "ANIMAL_HARVEST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    args = parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        card_type = np.asarray(data["internal_route_card_card_type"])
        status = np.asarray(data["internal_route_card_status"])
        start = np.asarray(data["internal_route_card_start_step"])
        length = np.asarray(data["internal_route_card_route_length"])
        cursor = np.asarray(data["internal_route_card_route_cursor"])
        actions = np.asarray(data["internal_route_card_action_masks"])

    if card_type.ndim != 3:
        raise ValueError(f"expected [step,batch,unit], got {card_type.shape}")
    instances: dict[tuple[int, int, int, int], dict[str, object]] = {}
    steps, batch_size, unit_count = card_type.shape
    for step in range(steps):
        for lane in range(batch_size):
            for unit in range(unit_count):
                kind = int(card_type[step, lane, unit])
                start_step = int(start[step, lane, unit])
                if kind <= 0 or start_step < 0:
                    continue
                key = (lane, unit, start_step, kind)
                route_length = int(length[step, lane, unit])
                route_actions = actions[step, lane, unit, :route_length]
                row = instances.setdefault(
                    key,
                    {
                        "lane": lane,
                        "unit": unit,
                        "start_step": start_step,
                        "card_type": CARD_TYPES.get(kind, f"UNKNOWN_{kind}"),
                        "route_length": route_length,
                        "max_cursor": 0,
                        "statuses": set(),
                        "action_counts": Counter(),
                    },
                )
                row["route_length"] = max(int(row["route_length"]), route_length)
                row["max_cursor"] = max(
                    int(row["max_cursor"]), int(cursor[step, lane, unit])
                )
                row["statuses"].add(int(status[step, lane, unit]))
                counts = Counter()
                for bit, name in ACTION_BITS.items():
                    counts[name] = int(np.count_nonzero(route_actions & bit))
                # The effect updater clears completed bits in-place.  Preserve
                # the maximum count seen over the card lifecycle, which is the
                # admitted route rather than its final all-cleared snapshot.
                previous = row["action_counts"]
                for name, value in counts.items():
                    previous[name] = max(int(previous[name]), int(value))

    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in instances.values():
        row["statuses"] = sorted(row["statuses"])
        row["action_counts"] = dict(row["action_counts"])
        by_type[str(row["card_type"])].append(row)

    type_summary = {}
    for name, rows in sorted(by_type.items()):
        totals = Counter()
        for row in rows:
            totals.update(row["action_counts"])
        type_summary[name] = {
            "instance_count": len(rows),
            "max_route_length": max(int(row["route_length"]) for row in rows),
            "max_cursor": max(int(row["max_cursor"]) for row in rows),
            "action_counts": dict(totals),
        }

    mixed_rows = by_type.get("MIXED_FARM_TOUR", [])
    payload = {
        "schema": "kaggriculture.m39_route_trace_summary.v1",
        "status": "PASS",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "steps": steps,
        "batch_size": batch_size,
        "unit_count": unit_count,
        "unique_card_instance_count": len(instances),
        "by_type": type_summary,
        "mixed_collect_then_fertilize_instance_count": sum(
            int(row["action_counts"].get("COLLECT_FERTILIZER", 0)) > 0
            and int(row["action_counts"].get("FERTILIZE", 0)) > 0
            for row in mixed_rows
        ),
        "boundary": "Trace lifecycle summary; economic value is reported by rollout receipts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
