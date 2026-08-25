"""Search a causal first-shop plus compatible second-shop trace tree."""

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


def prefix_hash(bank: np.lib.npyio.NpzFile, route: int, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for field in ACTION_FIELDS:
        digest.update(np.ascontiguousarray(bank[field][route, start:end]).tobytes())
    return digest.hexdigest()


def best_route(margin: np.ndarray, invalid: np.ndarray, mask: np.ndarray, routes: np.ndarray) -> int:
    if not np.any(mask) or routes.size == 0:
        return -1
    values = margin[:, mask, :][:, :, routes].reshape(-1, routes.size)
    failures = invalid[:, mask, :][:, :, routes].reshape(-1, routes.size)
    wins = np.mean(values > 0, axis=0)
    means = np.mean(values, axis=0)
    invalid_means = np.mean(failures, axis=0)
    return int(routes[np.lexsort((-invalid_means, means, wins))[-1]])


def evaluate_tree(
    margin: np.ndarray,
    hard: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    base_map: np.ndarray,
    second_map: np.ndarray,
    seed_mask: np.ndarray,
) -> dict:
    values = []
    hard_values = []
    selected_routes = []
    for seat in range(margin.shape[0]):
        for seed in np.flatnonzero(seed_mask):
            base = int(base_map[int(first[seed])])
            selected = int(second_map[base, int(second[seed])])
            values.append(int(margin[seat, seed, selected]))
            hard_values.append(int(hard[seat, seed, selected]))
            selected_routes.append(selected)
    values_array = np.asarray(values, dtype=np.int64)
    return {
        "games": int(values_array.size),
        "wins": int(np.sum(values_array > 0)),
        "win_rate": float(np.mean(values_array > 0)),
        "mean_margin": float(np.mean(values_array)),
        "median_margin": float(np.median(values_array)),
        "hard_counter_total": int(np.sum(hard_values)),
        "unique_routes": int(np.unique(selected_routes).size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--switch-step", type=int, default=144)
    parser.add_argument("--suffix-fit-end", type=int, default=128)
    parser.add_argument("--base-score-end", type=int, default=192)
    parser.add_argument("--holdout-start", type=int, default=192)
    parser.add_argument(
        "--base-scope", choices=("same_first", "unrestricted"), default="unrestricted"
    )
    parser.add_argument(
        "--suffix-scope", choices=("source_second", "compatible_all"), default="source_second"
    )
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank = np.load(args.trace_bank)
    screen = np.load(args.matrix)
    margin = np.asarray(screen["margin"])
    invalid = np.asarray(screen["invalid"])
    hard = np.asarray(screen["hard"])
    observed = np.asarray(screen["observed_shops"])
    first = np.asarray(screen["first_shop"])[0]
    second = observed[0, :, 0, 1]
    source_shops = np.asarray(bank["source_shop_sequence"])
    routes = int(bank["unit_op"].shape[0])
    seeds = int(margin.shape[1])
    if routes > args.route_capacity or margin.shape[2] != routes:
        raise ValueError("route capacity or matrix mismatch")
    if not (0 < args.suffix_fit_end < args.base_score_end <= args.holdout_start < seeds):
        raise ValueError("invalid seed split")

    hashes = np.asarray(
        [prefix_hash(bank, route, args.prefix_start, args.switch_step) for route in range(routes)]
    )
    fit_seed = np.arange(seeds) < args.suffix_fit_end
    tune_seed = (np.arange(seeds) >= args.suffix_fit_end) & (
        np.arange(seeds) < args.base_score_end
    )
    train_seed = np.arange(seeds) < args.base_score_end
    holdout_seed = np.arange(seeds) >= args.holdout_start

    base_map = np.zeros(8, dtype=np.int16)
    second_map = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    rows = []
    all_routes = np.arange(routes, dtype=np.int32)
    for first_shop in range(8):
        base_candidates = (
            np.flatnonzero(source_shops[:, 0] == first_shop)
            if args.base_scope == "same_first"
            else all_routes
        )
        scored = []
        candidate_maps: dict[int, np.ndarray] = {}
        for base in base_candidates:
            compatible = np.flatnonzero(hashes == hashes[base])
            route_by_second = np.full(8, int(base), dtype=np.int16)
            for second_shop in range(8):
                eligible = compatible
                if args.suffix_scope == "source_second":
                    eligible = compatible[source_shops[compatible, 1] == second_shop]
                eligible = np.unique(np.append(eligible, base)).astype(np.int32)
                mask = fit_seed & (first == first_shop) & (second == second_shop)
                chosen = best_route(margin, invalid, mask, eligible)
                if chosen >= 0:
                    route_by_second[second_shop] = chosen
            candidate_maps[int(base)] = route_by_second

            values = []
            for seat in range(2):
                for seed in np.flatnonzero(tune_seed & (first == first_shop)):
                    selected = int(route_by_second[int(second[seed])])
                    values.append(int(margin[seat, seed, selected]))
            if values:
                values_array = np.asarray(values)
                scored.append(
                    (int(np.sum(values_array > 0)), float(np.mean(values_array)), int(base))
                )
        if not scored:
            raise RuntimeError(f"no tune contexts for first shop {first_shop}")
        _, _, selected_base = max(scored)
        base_map[first_shop] = selected_base
        second_map[selected_base] = candidate_maps[selected_base]
        rows.append(
            {
                "first_shop": first_shop,
                "base_route": selected_base,
                "second_routes": candidate_maps[selected_base].tolist(),
            }
        )

    result = {
        "schema": "kaggriculture.front40_fusion.two-stage-compatible-route-tree.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "trace_bank": str(args.trace_bank),
        "matrix": str(args.matrix),
        "prefix_range": [args.prefix_start, args.switch_step],
        "seed_split": {
            "suffix_fit": [0, args.suffix_fit_end - 1],
            "base_tune": [args.suffix_fit_end, args.base_score_end - 1],
            "holdout": [args.holdout_start, seeds - 1],
        },
        "base_scope": args.base_scope,
        "suffix_scope": args.suffix_scope,
        "route_ids": base_map.tolist(),
        "second_route_ids": second_map.tolist(),
        "routes": rows,
        "train": evaluate_tree(
            margin, hard, first, second, base_map, second_map, train_seed
        ),
        "holdout": evaluate_tree(
            margin, hard, first, second, base_map, second_map, holdout_seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "base_routes": base_map.tolist(), "holdout": result["holdout"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
