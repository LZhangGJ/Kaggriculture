"""Search a robust causal two-shop tree from multiple independent panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_FIELDS = (
    "unit_op", "unit_item", "unit_amount", "unit_count",
    "market_op", "market_item", "market_amount", "market_count",
)


def route_hash(bank, route: int, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for field in ACTION_FIELDS:
        digest.update(np.ascontiguousarray(bank[field][route, start:end]).tobytes())
    return digest.hexdigest()


def load_panel(path: Path, expected_routes: np.ndarray | None = None) -> dict:
    data = np.load(path)
    margin = np.asarray(data["margin"])
    route_ids = np.asarray(
        data["route_ids"] if "route_ids" in data else np.arange(margin.shape[-1]),
        dtype=np.int32,
    )
    if expected_routes is not None and not np.array_equal(route_ids, expected_routes):
        raise ValueError(f"route IDs differ: {path}")
    observed = np.asarray(data["observed_shops"])
    return {
        "path": str(path),
        "route_ids": route_ids,
        "margin": margin,
        "invalid": np.asarray(data["invalid"]),
        "hard": np.asarray(data["hard"]),
        "first": np.asarray(data["first_shop"])[0],
        "second": observed[0, :, 0, 1],
    }


def candidate_key(panel_values: list[np.ndarray], mode: str) -> tuple:
    rates = np.asarray([np.mean(values > 0) for values in panel_values])
    margins = np.asarray([np.mean(values) for values in panel_values])
    pooled = np.concatenate(panel_values)
    if mode == "minimax_win":
        return float(np.min(rates)), float(np.mean(rates)), float(np.min(margins)), float(np.mean(pooled))
    if mode == "stable_win":
        return float(np.mean(rates) - 0.5 * np.std(rates)), float(np.mean(rates)), float(np.mean(pooled))
    return float(np.mean(pooled > 0)), float(np.mean(pooled))


def select_route(
    panels: list[dict], first_shop: int, second_shop: int,
    eligible_local: np.ndarray, mode: str,
) -> int:
    scored = []
    for route in eligible_local:
        values = []
        for panel in panels:
            mask = (panel["first"] == first_shop) & (panel["second"] == second_shop)
            if np.any(mask):
                values.append(panel["margin"][:, mask, route].reshape(-1))
        if values:
            scored.append((candidate_key(values, mode), int(route)))
    return max(scored)[1] if scored else int(eligible_local[0])


def tree_values(panel: dict, base_map: np.ndarray, second_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = []
    hard = []
    for seat in range(2):
        for seed in range(panel["margin"].shape[1]):
            first_shop = int(panel["first"][seed])
            second_shop = int(panel["second"][seed])
            base = int(base_map[first_shop])
            route = int(second_map[base, second_shop])
            values.append(int(panel["margin"][seat, seed, route]))
            hard.append(int(panel["hard"][seat, seed, route]))
    return np.asarray(values), np.asarray(hard)


def stats(values: np.ndarray, hard: np.ndarray) -> dict:
    return {
        "games": int(values.size),
        "wins": int(np.sum(values > 0)),
        "win_rate": float(np.mean(values > 0)),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "hard_counter_total": int(np.sum(hard)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--dev-matrices", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-matrix", type=Path, required=True)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--switch-step", type=int, default=144)
    parser.add_argument("--mode", choices=("minimax_win", "stable_win", "pooled_win"), required=True)
    parser.add_argument("--suffix-scope", choices=("source_second", "compatible_all"), required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank = np.load(args.trace_bank)
    dev = [load_panel(path) for path in args.dev_matrices]
    route_ids = dev[0]["route_ids"]
    for panel in dev[1:]:
        if not np.array_equal(panel["route_ids"], route_ids):
            raise ValueError("dev route IDs differ")
    holdout = load_panel(args.holdout_matrix, route_ids)
    if route_ids.size > args.route_capacity:
        raise ValueError("route capacity too small")
    source_shops = np.asarray(bank["source_shop_sequence"])
    hashes = {
        int(route): route_hash(bank, int(route), args.prefix_start, args.switch_step)
        for route in route_ids
    }

    local_by_global = {int(route): index for index, route in enumerate(route_ids)}
    base_map_local = np.zeros(8, dtype=np.int16)
    second_local = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    rows = []
    for first_shop in range(8):
        candidates = []
        for base_local, base_global in enumerate(route_ids):
            compatible_global = np.asarray(
                [route for route in route_ids if hashes[int(route)] == hashes[int(base_global)]],
                dtype=np.int32,
            )
            selected_by_second = np.full(8, base_local, dtype=np.int16)
            for second_shop in range(8):
                eligible_global = compatible_global
                if args.suffix_scope == "source_second":
                    eligible_global = compatible_global[
                        source_shops[compatible_global, 1] == second_shop
                    ]
                eligible_global = np.unique(np.append(eligible_global, base_global)).astype(np.int32)
                eligible_local = np.asarray(
                    [local_by_global[int(route)] for route in eligible_global], dtype=np.int32
                )
                selected_by_second[second_shop] = select_route(
                    dev, first_shop, second_shop, eligible_local, args.mode
                )

            panel_values = []
            for panel in dev:
                mask = panel["first"] == first_shop
                values = []
                for seat in range(2):
                    for seed in np.flatnonzero(mask):
                        route = int(selected_by_second[int(panel["second"][seed])])
                        values.append(int(panel["margin"][seat, seed, route]))
                panel_values.append(np.asarray(values))
            candidates.append((candidate_key(panel_values, args.mode), base_local, selected_by_second))

        _, selected_base, selected_by_second = max(candidates, key=lambda item: item[0])
        base_map_local[first_shop] = selected_base
        second_local[selected_base] = selected_by_second
        rows.append(
            {
                "first_shop": first_shop,
                "base_route": int(route_ids[selected_base]),
                "second_routes": [int(route_ids[index]) for index in selected_by_second],
            }
        )

    base_map_global = route_ids[base_map_local]
    second_global = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    for base_local in np.unique(base_map_local):
        base_global = int(route_ids[base_local])
        second_global[base_global] = route_ids[second_local[base_local]]

    dev_stats = []
    for panel in dev:
        values, hard = tree_values(panel, base_map_local, second_local)
        dev_stats.append(stats(values, hard))
    hold_values, hold_hard = tree_values(holdout, base_map_local, second_local)
    result = {
        "schema": "kaggriculture.front40_fusion.robust-two-stage-route-tree.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "trace_bank": str(args.trace_bank),
        "dev_matrices": [str(path) for path in args.dev_matrices],
        "holdout_matrix": str(args.holdout_matrix),
        "mode": args.mode,
        "suffix_scope": args.suffix_scope,
        "prefix_range": [args.prefix_start, args.switch_step],
        "route_ids": base_map_global.tolist(),
        "second_route_ids": second_global.tolist(),
        "routes": rows,
        "dev": dev_stats,
        "holdout": stats(hold_values, hold_hard),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode": args.mode, "scope": args.suffix_scope, "holdout": result["holdout"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
