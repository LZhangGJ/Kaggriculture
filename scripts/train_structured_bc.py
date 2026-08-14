"""Train PolicyV2 on structured teacher/replay BC shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kaggriculture_lab.policy_v2 import StructuredKaggriculturePolicy


ARRAY_DTYPES = {
    "features": None,
    "unit_context": None,
    "unit_active": torch.bool,
    "unit_targets": torch.long,
    "unit_quantity_targets": torch.long,
    "unit_quantity_active": torch.bool,
    "market_targets": torch.long,
    "market_quantity_targets": torch.long,
    "market_quantity_active": torch.bool,
    "market_order_active": torch.bool,
    "value_targets": None,
}


def load_batch(
    data: np.lib.npyio.NpzFile, indices: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    result = {}
    for key, dtype in ARRAY_DTYPES.items():
        if key == "unit_active":
            # Early PolicyV2 shards accidentally stored this array twice per step.
            # The third unit-context feature is the canonical active flag and lets
            # those shards be consumed without rewriting data on disk.
            values = data["unit_context"][indices, :, 2] > 0.5
        else:
            values = data[key][indices]
        result[key] = torch.as_tensor(values, device=device, dtype=dtype)
    return result


def batch_loss(
    model: StructuredKaggriculturePolicy,
    batch: dict[str, torch.Tensor],
    amp: bool,
    value_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.autocast(
        device_type=batch["features"].device.type,
        dtype=torch.bfloat16,
        enabled=amp,
    ):
        model_kwargs = {}
        if model.autoregressive_market:
            model_kwargs = {
                "market_teacher_tokens": batch["market_targets"],
                "market_teacher_quantities": batch["market_quantity_targets"],
            }
        unit_logits, unit_quantity_logits, market_logits, market_quantity_logits, values = model(
            batch["features"], batch["unit_context"], **model_kwargs
        )
        unit_active = batch["unit_active"]
        unit_loss = F.cross_entropy(unit_logits[unit_active], batch["unit_targets"][unit_active])

        unit_quantity_active = batch["unit_quantity_active"] & unit_active
        if unit_quantity_active.any():
            unit_quantity_loss = F.cross_entropy(
                unit_quantity_logits[unit_quantity_active],
                batch["unit_quantity_targets"][unit_quantity_active],
            )
        else:
            unit_quantity_loss = unit_logits.sum() * 0.0

        market_token_losses = F.cross_entropy(
            market_logits.flatten(0, 1),
            batch["market_targets"].flatten(),
            reduction="none",
        ).view_as(batch["market_targets"])
        market_weights = torch.where(
            batch["market_order_active"],
            torch.full_like(market_token_losses, 4.0),
            torch.ones_like(market_token_losses),
        )
        market_loss = (market_token_losses * market_weights).sum() / market_weights.sum()

        market_quantity_active = batch["market_quantity_active"]
        if market_quantity_active.any():
            market_quantity_loss = F.cross_entropy(
                market_quantity_logits[market_quantity_active],
                batch["market_quantity_targets"][market_quantity_active],
            )
        else:
            market_quantity_loss = market_logits.sum() * 0.0

        value_loss = F.mse_loss(values.float(), batch["value_targets"].float())
        loss = (
            unit_loss
            + market_loss
            + 0.35 * unit_quantity_loss
            + 0.35 * market_quantity_loss
            + value_weight * value_loss
        )

    with torch.no_grad():
        unit_prediction = unit_logits.argmax(-1)
        unit_quantity_prediction = unit_quantity_logits.argmax(-1)
        market_prediction = market_logits.argmax(-1)
        market_quantity_prediction = market_quantity_logits.argmax(-1)
        unit_rows_exact = ((unit_prediction == batch["unit_targets"]) | ~unit_active).all(dim=1)
        market_token_rows_exact = (market_prediction == batch["market_targets"]).all(dim=1)
        market_quantity_rows_exact = (
            (market_quantity_prediction == batch["market_quantity_targets"])
            | ~market_quantity_active
        ).all(dim=1)
        exact = unit_rows_exact & market_token_rows_exact & market_quantity_rows_exact
        metrics = {
            "loss": float(loss.detach()),
            "unit_loss": float(unit_loss.detach()),
            "unit_quantity_loss": float(unit_quantity_loss.detach()),
            "market_loss": float(market_loss.detach()),
            "market_quantity_loss": float(market_quantity_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "unit_correct": float(((unit_prediction == batch["unit_targets"]) & unit_active).sum()),
            "unit_count": float(unit_active.sum()),
            "market_order_correct": float(
                ((market_prediction == batch["market_targets"]) & batch["market_order_active"]).sum()
            ),
            "market_order_count": float(batch["market_order_active"].sum()),
            "market_slot_correct": float((market_prediction == batch["market_targets"]).sum()),
            "market_slot_count": float(batch["market_targets"].numel()),
            "market_quantity_correct": float(
                (
                    (market_quantity_prediction == batch["market_quantity_targets"])
                    & market_quantity_active
                ).sum()
            ),
            "market_quantity_count": float(market_quantity_active.sum()),
            "exact_actions": float(exact.sum()),
            "examples": float(len(exact)),
        }
    return loss, metrics


METRIC_KEYS = (
    "loss",
    "unit_loss",
    "unit_quantity_loss",
    "market_loss",
    "market_quantity_loss",
    "value_loss",
    "unit_correct",
    "unit_count",
    "market_order_correct",
    "market_order_count",
    "market_slot_correct",
    "market_slot_count",
    "market_quantity_correct",
    "market_quantity_count",
    "exact_actions",
    "examples",
)


def summarize(totals: dict[str, float], batches: int) -> dict[str, float]:
    for key in (
        "loss",
        "unit_loss",
        "unit_quantity_loss",
        "market_loss",
        "market_quantity_loss",
        "value_loss",
    ):
        totals[key] /= max(1, batches)
    totals["unit_accuracy"] = totals["unit_correct"] / max(1.0, totals["unit_count"])
    totals["market_order_accuracy"] = totals["market_order_correct"] / max(
        1.0, totals["market_order_count"]
    )
    totals["market_slot_accuracy"] = totals["market_slot_correct"] / max(
        1.0, totals["market_slot_count"]
    )
    totals["market_quantity_accuracy"] = totals["market_quantity_correct"] / max(
        1.0, totals["market_quantity_count"]
    )
    totals["exact_action_rate"] = totals["exact_actions"] / max(1.0, totals["examples"])
    return totals


@torch.no_grad()
def evaluate(
    model: StructuredKaggriculturePolicy,
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    value_weight: float = 0.1,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in METRIC_KEYS}
    batches = 0
    for shard in shards:
        with np.load(shard) as data:
            for start in range(0, len(data["features"]), batch_size):
                indices = np.arange(start, min(start + batch_size, len(data["features"])))
                _, metrics = batch_loss(
                    model,
                    load_batch(data, indices, device),
                    device.type == "cuda",
                    value_weight,
                )
                for key, value in metrics.items():
                    totals[key] += value
                batches += 1
    return summarize(totals, batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        help="repeat to mix teacher, replay, and DAgger shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\Kaggriculture\artifacts\teacher_bc_v2.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--value-weight",
        type=float,
        default=0.1,
        help="BC value-loss coefficient; use 0 for DAgger policy-only distillation",
    )
    parser.add_argument("--route-prior", action="store_true")
    parser.add_argument(
        "--components",
        choices=("all", "market"),
        default="all",
        help="market freezes the trunk, unit heads, and value head during BC/DAgger",
    )
    parser.add_argument(
        "--canonical-seat",
        action="store_true",
        help="encode public farms as own/opponent and suppress the absolute seat embedding",
    )
    parser.add_argument(
        "--autoregressive-market",
        action="store_true",
        help="decode market slots causally with teacher forcing during BC",
    )
    parser.add_argument("--route-lr", type=float, default=3e-2)
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="also save output-stem.epochN.pt for gameplay-based checkpoint selection",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dataset_dirs = args.dataset_dir or [Path(r"D:\Kaggriculture\data\processed\teacher_bc_v2")]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_shards = sorted(
        shard for dataset_dir in dataset_dirs for shard in (dataset_dir / "train").glob("*.npz")
    )
    validation_shards = sorted(
        shard
        for dataset_dir in dataset_dirs
        for shard in (dataset_dir / "validation").glob("*.npz")
    )
    if not train_shards or not validation_shards:
        parser.error("dataset must contain train and validation shards")

    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        args.hidden_size = int(initial_checkpoint["hidden_size"])
        args.route_prior = args.route_prior or bool(initial_checkpoint.get("route_prior", False))
        args.canonical_seat = args.canonical_seat or bool(
            initial_checkpoint.get("canonical_seat", False)
        )
        args.autoregressive_market = args.autoregressive_market or bool(
            initial_checkpoint.get("autoregressive_market", False)
        )
    model = StructuredKaggriculturePolicy(
        args.hidden_size,
        route_prior=args.route_prior,
        canonical_seat=args.canonical_seat,
        autoregressive_market=args.autoregressive_market,
    ).to(device)
    if initial_checkpoint:
        incompatible = model.load_state_dict(initial_checkpoint["model"], strict=False)
        allowed_missing = {
            name
            for name, _ in model.named_parameters()
            if "route_logits" in name or name.startswith("market_ar_")
        }
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"incompatible initial checkpoint: {incompatible}")
    if args.components == "market":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("market_token_head")
                or name.startswith("market_quantity_head")
                or name.startswith("market_ar_")
                or name.startswith("market_route_logits")
                or name.startswith("market_quantity_route_logits")
            )
    route_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "route_logits" in name
    ]
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "route_logits" not in name
    ]
    parameter_groups = [{"params": base_parameters, "lr": args.lr}]
    if route_parameters:
        parameter_groups.append(
            {"params": route_parameters, "lr": args.route_lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=1e-4)
    initial = evaluate(
        model, validation_shards, args.batch_size, device, args.value_weight
    )
    print("validation_epoch=0 " + json.dumps(initial, sort_keys=True))
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_shards)
        totals = {key: 0.0 for key in METRIC_KEYS}
        batches = 0
        for shard in train_shards:
            with np.load(shard) as data:
                order = np.random.permutation(len(data["features"]))
                for start in range(0, len(order), args.batch_size):
                    indices = order[start : start + args.batch_size]
                    batch = load_batch(data, indices, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = batch_loss(
                        model, batch, device.type == "cuda", args.value_weight
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    for key, value in metrics.items():
                        totals[key] += value
                    batches += 1
        train_metrics = summarize(totals, batches)
        validation = evaluate(
            model, validation_shards, args.batch_size, device, args.value_weight
        )
        print(f"train_epoch={epoch} " + json.dumps(train_metrics, sort_keys=True))
        print(f"validation_epoch={epoch} " + json.dumps(validation, sort_keys=True))
        checkpoint_payload = {
            "model": model.state_dict(),
            "hidden_size": args.hidden_size,
            "schema_version": 2,
            "route_prior": args.route_prior,
            "canonical_seat": args.canonical_seat,
            "autoregressive_market": args.autoregressive_market,
            "bc_components": args.components,
            "loss_weights": {"value": args.value_weight},
            "datasets": [str(path) for path in dataset_dirs],
            "epoch": epoch,
            "validation": validation,
        }
        if args.save_every_epoch:
            epoch_output = args.output.with_name(
                f"{args.output.stem}.epoch{epoch}{args.output.suffix}"
            )
            torch.save(checkpoint_payload, epoch_output)
            print(f"saved_epoch={epoch_output}")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint_payload, args.output)
            print(f"saved={args.output}")


if __name__ == "__main__":
    main()
