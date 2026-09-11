"""Compile daily per-unit task-chain modes from Replay traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


UNIT_FIELDS = ("unit_op", "unit_item", "unit_amount")
MARKET_FIELDS = ("market_op", "market_item", "market_amount", "market_count")
GLOBAL_EXPECTED = (
    "expected_money", "expected_self_summary", "expected_opponent_summary",
    "expected_shed", "expected_seeds", "expected_carried",
    "expected_market_price", "expected_market_inventory",
)
TIMED_FIELDS = UNIT_FIELDS + ("unit_count",) + MARKET_FIELDS + (
    "expected_unit_pos", "expected_unit_active",
) + GLOBAL_EXPECTED


def modal_route(rows: np.ndarray, group: np.ndarray, reward: np.ndarray) -> int:
    if group.size == 1:
        return int(group[0])
    if rows.shape[0] != group.size:
        raise ValueError("modal rows must already be restricted to the source group")
    _, inverse, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    best_count = int(np.max(counts))
    candidates = group[counts[inverse] == best_count]
    return int(max(candidates, key=lambda route: (int(reward[route]), -int(route))))


def synthesize(bank, group: np.ndarray, block_size: int) -> tuple[dict[str, np.ndarray], float]:
    reward = np.asarray(bank["source_reward"])
    result = {
        field: np.zeros(np.asarray(bank[field]).shape[1:], dtype=np.asarray(bank[field]).dtype)
        for field in TIMED_FIELDS
    }
    agreement = []
    for start in range(0, 719, block_size):
        end = min(start + block_size, 719)
        component_sources = []
        action_active = np.zeros((end - start, 33), dtype=bool)
        for unit in range(33):
            parts = [
                np.asarray(bank[field])[group, start:end, unit].reshape(group.size, -1).astype(np.int32)
                for field in UNIT_FIELDS
            ]
            active = (
                unit < np.asarray(bank["unit_count"])[group, start:end]
            ).astype(np.int32)
            parts.append(active.reshape(group.size, -1))
            source = modal_route(np.concatenate(parts, axis=1), group, reward)
            component_sources.append(source)
            for field in UNIT_FIELDS:
                result[field][start:end, unit] = bank[field][source, start:end, unit]
            result["expected_unit_pos"][start:end, unit] = bank[
                "expected_unit_pos"
            ][source, start:end, unit]
            result["expected_unit_active"][start:end, unit] = bank[
                "expected_unit_active"
            ][source, start:end, unit]
            action_active[:, unit] = unit < bank["unit_count"][source, start:end]

        result["unit_count"][start:end] = np.max(
            np.where(action_active, np.arange(1, 34)[None, :], 0), axis=1
        ).astype(
            result["unit_count"].dtype
        )
        market_rows = np.concatenate(
            [
                np.asarray(bank[field])[group, start:end].reshape(group.size, -1).astype(np.int32)
                for field in MARKET_FIELDS
            ],
            axis=1,
        )
        market_source = modal_route(market_rows, group, reward)
        component_sources.append(market_source)
        for field in MARKET_FIELDS:
            result[field][start:end] = bank[field][market_source, start:end]

        reference = Counter(component_sources).most_common()
        top_count = reference[0][1]
        reference_candidates = [route for route, count in reference if count == top_count]
        reference_source = max(
            reference_candidates, key=lambda route: (int(reward[route]), -int(route))
        )
        for field in GLOBAL_EXPECTED:
            result[field][start:end] = bank[field][reference_source, start:end]
        agreement.append(top_count / len(component_sources))
    return result, float(np.mean(agreement))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--switch-step", type=int, default=144)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--output-first-map", type=Path, required=True)
    parser.add_argument("--output-second-map", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    bank = np.load(args.trace_bank)
    shops = np.asarray(bank["source_shop_sequence"])
    reward = np.asarray(bank["source_reward"])
    base_groups = [np.flatnonzero(shops[:, 0] == value) for value in range(8)]
    if any(group.size == 0 for group in base_groups):
        raise RuntimeError("missing first-shop source group")
    synthesis_cache: dict[tuple[int, ...], tuple[dict[str, np.ndarray], float]] = {}

    def cached_synthesize(group: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        key = tuple(int(route) for route in group)
        if key not in synthesis_cache:
            synthesis_cache[key] = synthesize(bank, group, args.block_size)
        route, agreement = synthesis_cache[key]
        return {field: value.copy() for field, value in route.items()}, agreement

    base_routes = []
    base_agreement = []
    for group in base_groups:
        route, agreement = cached_synthesize(group)
        base_routes.append(route)
        base_agreement.append(agreement)

    routes = list(base_routes)
    agreements = list(base_agreement)
    groups = list(base_groups)
    prefixes = [(first, -1) for first in range(8)]
    for first in range(8):
        for second in range(8):
            group = np.flatnonzero((shops[:, 0] == first) & (shops[:, 1] == second))
            if group.size:
                route, agreement = cached_synthesize(group)
                for field in TIMED_FIELDS:
                    route[field][: args.switch_step] = base_routes[first][field][: args.switch_step]
            else:
                group = base_groups[first]
                route = {field: value.copy() for field, value in base_routes[first].items()}
                agreement = base_agreement[first]
            routes.append(route)
            agreements.append(agreement)
            groups.append(group)
            prefixes.append((first, second))

    output = {
        field: np.stack([route[field] for route in routes]) for field in TIMED_FIELDS
    }
    source_episode = []
    source_reward = []
    source_opp = []
    source_margin = []
    source_seat = []
    source_shops = []
    for group, prefix in zip(groups, prefixes, strict=True):
        source = int(group[np.argmax(reward[group])])
        source_episode.append(int(bank["source_episode_id"][source]))
        source_reward.append(int(bank["source_reward"][source]))
        source_opp.append(int(bank["source_opponent_reward"][source]))
        source_margin.append(int(bank["source_margin"][source]))
        source_seat.append(int(bank["source_seat"][source]))
        sequence = np.asarray(bank["source_shop_sequence"][source]).copy()
        sequence[0] = prefix[0]
        if prefix[1] >= 0:
            sequence[1] = prefix[1]
        source_shops.append(sequence)
    output.update(
        source_shop_sequence=np.asarray(source_shops, dtype=np.int8),
        source_episode_id=np.asarray(source_episode, dtype=np.int64),
        source_reward=np.asarray(source_reward, dtype=np.int32),
        source_opponent_reward=np.asarray(source_opp, dtype=np.int32),
        source_margin=np.asarray(source_margin, dtype=np.int32),
        source_seat=np.asarray(source_seat, dtype=np.int8),
        bootstrap_route_id=np.asarray(0, dtype=np.int16),
    )
    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bank, **output)
    args.output_first_map.write_text(
        json.dumps({"route_ids": list(range(8)), "status": "FROZEN_REPLAY_DERIVED"}, indent=2) + "\n",
        encoding="utf-8",
    )
    second_map = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    for first in range(8):
        for second in range(8):
            second_map[first, second] = 8 + first * 8 + second
    args.output_second_map.write_text(
        json.dumps(
            {"second_route_ids": second_map.tolist(), "switch_step": args.switch_step,
             "status": "FROZEN_REPLAY_DERIVED"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "kaggriculture.front40_fusion.daily-component-modal-bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_bank": str(args.trace_bank),
        "block_size": args.block_size,
        "switch_step": args.switch_step,
        "output_routes": len(routes),
        "first_group_sizes": [int(group.size) for group in base_groups],
        "mean_component_source_agreement": float(np.mean(agreements)),
        "min_component_source_agreement": float(np.min(agreements)),
        "output_bank": str(args.output_bank),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
