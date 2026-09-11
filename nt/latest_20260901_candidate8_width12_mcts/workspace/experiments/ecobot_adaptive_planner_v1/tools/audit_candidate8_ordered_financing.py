#!/usr/bin/env python3
"""Prove a composite project candidate executes SELL before financed purchases."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][index]["values"])
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--minimum-decision-day", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"])
        state_id = np.asarray(data["state_id"])
        prefix_seed = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        gain = np.asarray(data["gain_vs_keep"])
        signature = np.asarray(data["signature"])

    composite = np.flatnonzero(
        (features[:, 12] == 4) & np.any(features[:, :8] > 0, axis=1)
    )
    if not len(composite):
        raise RuntimeError("dataset has no financed positive project candidate")
    selected = int(composite[np.argmax(gain[composite])])
    state = int(state_id[selected])
    state_members = np.flatnonzero(state_id == state)
    local_rank = int(np.flatnonzero(state_members == selected)[0])
    current_seed = int(prefix_seed[selected])
    current_seat = int(seat[selected])

    keep = bundle.adaptive_executor.play_candidate8(
        genome, opponent, current_seed, current_seat, 0,
        args.minimum_decision_day, True, True,
    )
    edited = bundle.adaptive_executor.play_candidate8(
        genome, opponent, current_seed, current_seat, local_rank,
        args.minimum_decision_day, True, True,
    )
    decision_step = int(edited["first_candidate8_day"]) * 24
    ordered_step = None
    ordered_market = None
    for step in range(decision_step, len(edited["trace"])):
        market = edited["trace"][step][current_seat]["market"]
        sell_positions = [
            index for index, order in enumerate(market) if order[0] == "SELL"
        ]
        purchase_positions = [
            index
            for index, order in enumerate(market)
            if order[0].startswith("BUY_") or order[0] in {"HIRE", "BUY_LAND"}
        ]
        if sell_positions and purchase_positions and min(sell_positions) < max(purchase_positions):
            ordered_step = step
            ordered_market = market
            break

    candidate_features = [int(value) for value in features[selected, :20]]
    hard_errors = {
        "avoidable_crop_losses": int(edited["avoidable_crop_losses"]),
        "avoidable_animal_losses": int(edited["avoidable_animal_losses"]),
        "end_overflow": int(edited["end_overflow"]),
    }
    status = (
        "PASS"
        if ordered_step is not None
        and candidate_features[12] == 4
        and any(value > 0 for value in candidate_features[:8])
        and int(edited["first_candidate8_signature"]) == int(signature[selected])
        and not any(hard_errors.values())
        else "FAIL"
    )
    payload = {
        "schema": "kaggriculture.candidate8_ordered_financing.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "opponent": args.opponent,
        "prefix_seed": current_seed,
        "seat": current_seat,
        "state_id": state,
        "feasible_rank": local_rank,
        "signature": int(signature[selected]),
        "candidate_features": candidate_features,
        "ordered_transaction_step": ordered_step,
        "ordered_market": ordered_market,
        "dataset_expected_gain_vs_keep": float(gain[selected]),
        "realized_gain_vs_keep": float(
            edited["rewards"][current_seat] - keep["rewards"][current_seat]
        ),
        "hard_errors": hard_errors,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "genomes": args.genomes,
                "dataset": args.dataset,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
