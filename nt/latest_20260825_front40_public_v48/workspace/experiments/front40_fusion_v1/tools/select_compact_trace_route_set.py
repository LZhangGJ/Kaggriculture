"""Select a compact Replay-route set that preserves counterfactual upside."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def metrics(margin: np.ndarray, selected: list[int]) -> dict:
    best = np.max(margin[:, selected], axis=1)
    return {
        "routes": list(selected),
        "route_count": len(selected),
        "oracle_wins": int(np.sum(best > 0)),
        "oracle_win_rate": float(np.mean(best > 0)),
        "oracle_mean_margin": float(np.mean(best)),
        "oracle_median_margin": float(np.median(best)),
    }


def greedy_select(margin: np.ndarray, count: int) -> list[int]:
    selected: list[int] = []
    remaining = set(range(margin.shape[1]))
    best = np.full(margin.shape[0], np.iinfo(np.int32).min, dtype=np.int64)
    for _ in range(min(count, margin.shape[1])):
        winner = max(
            remaining,
            key=lambda route: (
                int(np.sum(np.maximum(best, margin[:, route]) > 0)),
                float(np.mean(np.maximum(best, margin[:, route]))),
            ),
        )
        selected.append(int(winner))
        remaining.remove(winner)
        best = np.maximum(best, margin[:, winner])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sizes", default="4,8,12,16,24")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = np.load(args.matrix)
    margin3 = np.asarray(screen["margin"], dtype=np.int64)
    margin = margin3.reshape(-1, margin3.shape[-1])
    first_shop = np.asarray(screen["first_shop"], dtype=np.int8).reshape(-1)
    source_episode = np.asarray(screen["source_episode_id"], dtype=np.int64)
    sizes = sorted({int(value) for value in args.sizes.split(",") if value})

    rows = []
    for size in sizes:
        selected = greedy_select(margin, size)
        row = metrics(margin, selected)
        row["source_episode_ids"] = [int(source_episode[index]) for index in selected]
        row["by_first_shop"] = []
        for shop in range(8):
            mask = first_shop == shop
            shop_metrics = metrics(margin[mask], selected)
            shop_metrics["first_shop_id"] = shop
            shop_metrics["contexts"] = int(np.sum(mask))
            row["by_first_shop"].append(shop_metrics)
        rows.append(row)

    full = metrics(margin, list(range(margin.shape[1])))
    payload = {
        "schema": "kaggriculture.front40_fusion.compact-trace-route-set.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "contexts": int(margin.shape[0]),
        "available_routes": int(margin.shape[1]),
        "full_oracle": full,
        "greedy_frontier": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
