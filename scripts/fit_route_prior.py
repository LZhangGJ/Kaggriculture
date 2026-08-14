"""Fit per-step PolicyV2 route logits from structured BC shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kaggriculture_lab.gpu_policy import MAX_UNITS, UNIT_ACTIONS
from kaggriculture_lab.policy_v2 import (
    MARKET_TOKENS,
    MAX_MARKET_ORDERS,
    QUANTITY_CLASSES,
    StructuredKaggriculturePolicy,
)


def _counts(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros((720, 2, *shape), dtype=np.int32)


def _set_majority_logits(parameter: torch.Tensor, counts: np.ndarray, strength: float) -> int:
    winners = counts.argmax(axis=-1)
    observed = counts.sum(axis=-1) > 0
    logits = np.zeros(counts.shape, dtype=np.float32)
    coordinates = np.nonzero(observed)
    logits[(*coordinates, winners[coordinates])] = strength
    parameter.copy_(torch.as_tensor(logits, device=parameter.device))
    return int(observed.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=20.0)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = StructuredKaggriculturePolicy(
        int(checkpoint["hidden_size"]), route_prior=True
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        name for name, _ in model.named_parameters() if "route_logits" in name
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible source checkpoint: {incompatible}")

    unit = _counts((MAX_UNITS, len(UNIT_ACTIONS)))
    unit_quantity = _counts((MAX_UNITS, QUANTITY_CLASSES))
    market = _counts((MAX_MARKET_ORDERS, len(MARKET_TOKENS)))
    market_quantity = _counts((MAX_MARKET_ORDERS, QUANTITY_CLASSES))
    examples = 0
    splits = ("train", "validation") if args.include_validation else ("train",)
    shards = sorted(
        shard
        for dataset_dir in args.dataset_dir
        for split in splits
        for shard in (dataset_dir / split).glob("*.npz")
    )
    if not shards:
        parser.error("no structured shards found")

    for shard in shards:
        with np.load(shard) as data:
            step = np.clip(
                np.rint(data["features"][:, 0].astype(np.float32) * 720.0).astype(np.int64),
                0,
                719,
            )
            seat = np.clip(
                np.rint(data["features"][:, 3].astype(np.float32)).astype(np.int64), 0, 1
            )
            examples += len(step)
            for slot in range(MAX_UNITS):
                active = data["unit_context"][:, slot, 2] > 0.5
                np.add.at(
                    unit,
                    (step[active], seat[active], slot, data["unit_targets"][active, slot]),
                    1,
                )
                quantity_active = active & data["unit_quantity_active"][:, slot]
                np.add.at(
                    unit_quantity,
                    (
                        step[quantity_active],
                        seat[quantity_active],
                        slot,
                        data["unit_quantity_targets"][quantity_active, slot],
                    ),
                    1,
                )
            for slot in range(MAX_MARKET_ORDERS):
                np.add.at(
                    market,
                    (step, seat, slot, data["market_targets"][:, slot]),
                    1,
                )
                quantity_active = data["market_quantity_active"][:, slot]
                np.add.at(
                    market_quantity,
                    (
                        step[quantity_active],
                        seat[quantity_active],
                        slot,
                        data["market_quantity_targets"][quantity_active, slot],
                    ),
                    1,
                )

    with torch.no_grad():
        observed = {
            "unit": _set_majority_logits(model.unit_route_logits, unit, args.strength),
            "unit_quantity": _set_majority_logits(
                model.unit_quantity_route_logits, unit_quantity, args.strength
            ),
            "market": _set_majority_logits(model.market_route_logits, market, args.strength),
            "market_quantity": _set_majority_logits(
                model.market_quantity_route_logits, market_quantity, args.strength
            ),
        }

    output = dict(checkpoint)
    output.update(
        {
            "model": model.state_dict(),
            "route_prior": True,
            "route_fit": {
                "datasets": [str(path) for path in args.dataset_dir],
                "examples": examples,
                "include_validation": args.include_validation,
                "observed_routes": observed,
                "strength": args.strength,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shards": len(shards),
                "examples": examples,
                "strength": args.strength,
                "observed_routes": observed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
