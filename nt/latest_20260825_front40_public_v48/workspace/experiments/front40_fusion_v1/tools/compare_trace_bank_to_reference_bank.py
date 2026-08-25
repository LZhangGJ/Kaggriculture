"""Compare every route in one raw-action trace bank with a reference route bank.

This is a structural/provenance audit.  Similar actions do not establish that a
player copied a public route; they only identify a reusable executable skeleton.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}


def first_false(mask: np.ndarray) -> int:
    indices = np.flatnonzero(~mask)
    return int(indices[0]) if indices.size else int(mask.size)


def compare_one(candidate: dict[str, np.ndarray], route: int, reference: dict[str, np.ndarray]) -> list[dict]:
    unit = (
        np.all(reference["unit_op"] == candidate["unit_op"][route][None, ...], axis=2)
        & np.all(reference["unit_item"] == candidate["unit_item"][route][None, ...], axis=2)
        & np.all(reference["unit_amount"] == candidate["unit_amount"][route][None, ...], axis=2)
        & (reference["unit_count"] == candidate["unit_count"][route][None, ...])
    )
    market = (
        np.all(reference["market_op"] == candidate["market_op"][route][None, ...], axis=2)
        & np.all(reference["market_item"] == candidate["market_item"][route][None, ...], axis=2)
        & np.all(reference["market_amount"] == candidate["market_amount"][route][None, ...], axis=2)
        & (reference["market_count"] == candidate["market_count"][route][None, ...])
    )
    complete = unit & market
    rows = []
    for reference_route in range(complete.shape[0]):
        rows.append(
            {
                "reference_route_id": int(reference_route),
                "complete_steps": int(np.sum(complete[reference_route])),
                "unit_steps": int(np.sum(unit[reference_route])),
                "market_steps": int(np.sum(market[reference_route])),
                "exact_prefix_steps": first_false(complete[reference_route]),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["complete_steps"],
            -row["unit_steps"],
            -row["market_steps"],
            row["reference_route_id"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_npz(args.candidate_bank)
    reference = load_npz(args.reference_bank)
    for name in ACTION_FIELDS:
        if name not in candidate or name not in reference:
            raise ValueError(f"missing action field: {name}")
    manifest = json.loads(args.reference_manifest.resolve().read_text(encoding="utf-8"))
    skeletons = {int(row["skeleton_id"]): row for row in manifest["skeletons"]}
    route_count = int(candidate["unit_op"].shape[0])
    best_counter: Counter[int] = Counter()
    rows = []
    for route in range(route_count):
        matches = compare_one(candidate, route, reference)
        for match in matches[: args.top_k]:
            source = skeletons[int(match["reference_route_id"])]
            match["reference_opponent"] = source.get("opponent")
            match["reference_route"] = source.get("route")
            match["reference_source"] = source.get("source")
        best_counter[int(matches[0]["reference_route_id"])] += 1
        rows.append(
            {
                "candidate_route_id": route,
                "source_episode_id": int(candidate.get("source_episode_id", np.arange(route_count))[route]),
                "source_reward": int(candidate.get("source_reward", np.zeros(route_count, np.int32))[route]),
                "top_matches": matches[: args.top_k],
            }
        )

    best_reference_counts = []
    for reference_route, count in best_counter.most_common():
        source = skeletons[reference_route]
        best_reference_counts.append(
            {
                "reference_route_id": reference_route,
                "count": count,
                "reference_opponent": source.get("opponent"),
                "reference_route": source.get("route"),
                "reference_source": source.get("source"),
            }
        )
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-bank-reference-similarity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_bank": str(args.candidate_bank),
        "reference_bank": str(args.reference_bank),
        "candidate_routes": route_count,
        "reference_routes": int(reference["unit_op"].shape[0]),
        "best_reference_counts": best_reference_counts,
        "routes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "candidate_routes": route_count,
                "best_reference_counts": best_reference_counts[:10],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
