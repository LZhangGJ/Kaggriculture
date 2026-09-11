"""Compile a causal first/second-shop modal action program from Replay traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_FIELDS = (
    "unit_op", "unit_item", "unit_amount", "unit_count",
    "market_op", "market_item", "market_amount", "market_count",
)
TIMED_FIELDS = ACTION_FIELDS + (
    "expected_unit_pos", "expected_unit_active", "expected_money",
    "expected_self_summary", "expected_opponent_summary", "expected_shed",
    "expected_seeds", "expected_carried", "expected_market_price",
    "expected_market_inventory",
)


def representatives(bank, signatures: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reward = np.asarray(bank["source_reward"])
    selected = np.zeros(719, dtype=np.int32)
    consensus = np.zeros(719, dtype=np.float32)
    for step in range(719):
        step_signatures = [int(signatures[int(route), step]) for route in group]
        counts = Counter(step_signatures)
        best_count = max(counts.values())
        best_signatures = {signature for signature, count in counts.items() if count == best_count}
        candidates = [
            int(route) for route, signature in zip(group, step_signatures, strict=True)
            if signature in best_signatures
        ]
        selected[step] = max(candidates, key=lambda route: (int(reward[route]), -route))
        consensus[step] = best_count / len(group)
    return selected, consensus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--switch-step", type=int, default=144)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--output-first-map", type=Path, required=True)
    parser.add_argument("--output-second-map", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    args = parser.parse_args()

    bank = np.load(args.trace_bank)
    source_shops = np.asarray(bank["source_shop_sequence"])
    source_reward = np.asarray(bank["source_reward"])
    action_matrix = np.concatenate(
        [
            np.asarray(bank[field]).reshape(source_shops.shape[0], 719, -1).astype(np.int32)
            for field in ACTION_FIELDS
        ],
        axis=2,
    )
    _, inverse = np.unique(
        action_matrix.reshape(-1, action_matrix.shape[-1]), axis=0, return_inverse=True
    )
    signatures = inverse.reshape(source_shops.shape[0], 719)
    base_groups = [np.flatnonzero(source_shops[:, 0] == shop) for shop in range(8)]
    if any(group.size == 0 for group in base_groups):
        raise RuntimeError("at least one first shop has no Replay source")
    base_reps = []
    base_consensus = []
    for group in base_groups:
        rep, consensus = representatives(bank, signatures, group)
        base_reps.append(rep)
        base_consensus.append(consensus)

    route_reps = []
    route_consensus = []
    route_groups = []
    route_prefixes = []
    # Routes 0..7 are first-shop modal programs.
    for first_shop in range(8):
        route_reps.append(base_reps[first_shop])
        route_consensus.append(base_consensus[first_shop])
        route_groups.append(base_groups[first_shop])
        route_prefixes.append((first_shop, -1))
    # Routes 8..71 copy the causal first-shop prefix, then use a modal suffix
    # from sources matching both shops. Missing pairs safely back off.
    for first_shop in range(8):
        for second_shop in range(8):
            group = np.flatnonzero(
                (source_shops[:, 0] == first_shop)
                & (source_shops[:, 1] == second_shop)
            )
            if group.size:
                suffix_rep, suffix_consensus = representatives(bank, signatures, group)
            else:
                group = base_groups[first_shop]
                suffix_rep = base_reps[first_shop]
                suffix_consensus = base_consensus[first_shop]
            rep = suffix_rep.copy()
            consensus = suffix_consensus.copy()
            rep[: args.switch_step] = base_reps[first_shop][: args.switch_step]
            consensus[: args.switch_step] = base_consensus[first_shop][: args.switch_step]
            route_reps.append(rep)
            route_consensus.append(consensus)
            route_groups.append(group)
            route_prefixes.append((first_shop, second_shop))

    route_reps_array = np.stack(route_reps)
    output: dict[str, np.ndarray] = {}
    route_index = np.arange(route_reps_array.shape[0])[:, None]
    step_index = np.arange(719)[None, :]
    for field in TIMED_FIELDS:
        output[field] = np.asarray(bank[field])[route_reps_array, step_index]

    source_episode = []
    output_reward = []
    output_opp_reward = []
    output_margin = []
    output_seat = []
    output_shops = []
    for group, prefix in zip(route_groups, route_prefixes, strict=True):
        representative = int(group[np.argmax(source_reward[group])])
        source_episode.append(int(bank["source_episode_id"][representative]))
        output_reward.append(int(bank["source_reward"][representative]))
        output_opp_reward.append(int(bank["source_opponent_reward"][representative]))
        output_margin.append(int(bank["source_margin"][representative]))
        output_seat.append(int(bank["source_seat"][representative]))
        shops = np.asarray(bank["source_shop_sequence"][representative]).copy()
        shops[0] = prefix[0]
        if prefix[1] >= 0:
            shops[1] = prefix[1]
        output_shops.append(shops)
    output["source_shop_sequence"] = np.asarray(output_shops, dtype=np.int8)
    output["source_episode_id"] = np.asarray(source_episode, dtype=np.int64)
    output["source_reward"] = np.asarray(output_reward, dtype=np.int32)
    output["source_opponent_reward"] = np.asarray(output_opp_reward, dtype=np.int32)
    output["source_margin"] = np.asarray(output_margin, dtype=np.int32)
    output["source_seat"] = np.asarray(output_seat, dtype=np.int8)
    output["bootstrap_route_id"] = np.asarray(0, dtype=np.int16)

    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bank, **output)
    first_map = {
        "schema": "kaggriculture.front40_fusion.modal-first-shop-map.v1",
        "route_ids": list(range(8)),
        "status": "FROZEN_REPLAY_DERIVED",
    }
    args.output_first_map.write_text(
        json.dumps(first_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    second = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    for first_shop in range(8):
        for second_shop in range(8):
            second[first_shop, second_shop] = 8 + first_shop * 8 + second_shop
    second_payload = {
        "schema": "kaggriculture.front40_fusion.modal-second-shop-map.v1",
        "second_route_ids": second.tolist(),
        "switch_step": args.switch_step,
        "status": "FROZEN_REPLAY_DERIVED",
    }
    args.output_second_map.write_text(
        json.dumps(second_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    consensus_array = np.stack(route_consensus)
    receipt = {
        "schema": "kaggriculture.front40_fusion.modal-shop-prefix-bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_bank": str(args.trace_bank),
        "source_routes": int(source_shops.shape[0]),
        "output_routes": int(route_reps_array.shape[0]),
        "switch_step": args.switch_step,
        "first_group_sizes": [int(group.size) for group in base_groups],
        "pair_group_sizes": [int(group.size) for group in route_groups[8:]],
        "mean_action_consensus": float(np.mean(consensus_array)),
        "min_route_action_consensus": float(np.min(np.mean(consensus_array, axis=1))),
        "output_bank": str(args.output_bank),
        "first_map": str(args.output_first_map),
        "second_map": str(args.output_second_map),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
