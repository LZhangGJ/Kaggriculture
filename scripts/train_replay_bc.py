"""Train the BC-v0 policy on preprocessed official replay shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kaggriculture_lab.gpu_policy import KaggriculturePolicy


def load_batch(data: np.lib.npyio.NpzFile, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(data["features"][indices], device=device),
        "unit_context": torch.as_tensor(data["unit_context"][indices], device=device),
        "unit_masks": torch.as_tensor(data["unit_masks"][indices], device=device),
        "market_masks": torch.as_tensor(data["market_masks"][indices], device=device),
        "unit_targets": torch.as_tensor(data["unit_targets"][indices], device=device, dtype=torch.long),
        "market_targets": torch.as_tensor(data["market_targets"][indices], device=device, dtype=torch.long),
        "unit_valid": torch.as_tensor(data["unit_valid"][indices], device=device),
        "market_valid": torch.as_tensor(data["market_valid"][indices], device=device),
        "value_targets": torch.as_tensor(data["value_targets"][indices], device=device),
    }


def batch_loss(model: KaggriculturePolicy, batch: dict[str, torch.Tensor], amp: bool) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.autocast(device_type=batch["features"].device.type, dtype=torch.bfloat16, enabled=amp):
        unit_logits, market_logits, values = model(batch["features"], batch["unit_context"])
        # Official actions may deliberately be no-ops, and market orders can become
        # executable after unit DROP actions earlier in the same turn. A mask made
        # from the pre-action observation therefore rejects valid expert labels.
        unit_active = batch["unit_context"][..., 2] > 0
        market_active = torch.ones_like(batch["market_targets"], dtype=torch.bool)
        unit_loss = F.cross_entropy(unit_logits[unit_active], batch["unit_targets"][unit_active])
        market_loss = F.cross_entropy(market_logits[market_active], batch["market_targets"][market_active])
        value_loss = F.mse_loss(values.float(), batch["value_targets"].float())
        loss = unit_loss + market_loss + 0.25 * value_loss
    with torch.no_grad():
        unit_correct = (unit_logits[unit_active].argmax(-1) == batch["unit_targets"][unit_active]).sum().item()
        market_correct = (market_logits[market_active].argmax(-1) == batch["market_targets"][market_active]).sum().item()
    return loss, {
        "loss": float(loss.detach()),
        "unit_loss": float(unit_loss.detach()),
        "market_loss": float(market_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "unit_correct": unit_correct,
        "unit_count": int(unit_active.sum().item()),
        "market_correct": market_correct,
        "market_count": int(market_active.sum().item()),
    }


@torch.no_grad()
def evaluate(model: KaggriculturePolicy, shards: list[Path], batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("loss", "unit_loss", "market_loss", "value_loss", "unit_correct", "unit_count", "market_correct", "market_count")}
    batches = 0
    for shard in shards:
        with np.load(shard) as data:
            for start in range(0, len(data["features"]), batch_size):
                indices = np.arange(start, min(start + batch_size, len(data["features"])))
                _, metrics = batch_loss(model, load_batch(data, indices, device), device.type == "cuda")
                for key, value in metrics.items():
                    totals[key] += value
                batches += 1
    for key in ("loss", "unit_loss", "market_loss", "value_loss"):
        totals[key] /= max(1, batches)
    totals["unit_accuracy"] = totals["unit_correct"] / max(1.0, totals["unit_count"])
    totals["market_accuracy"] = totals["market_correct"] / max(1.0, totals["market_count"])
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\Kaggriculture\data\processed\bc_v0"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\Kaggriculture\artifacts\bc_v0.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_shards = sorted((args.dataset_dir / "train").glob("*.npz"))
    validation_shards = sorted((args.dataset_dir / "validation").glob("*.npz"))
    if not train_shards or not validation_shards:
        parser.error("dataset must contain train and validation shards")

    model = KaggriculturePolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    initial = evaluate(model, validation_shards, args.batch_size, device)
    print("validation_epoch=0 " + json.dumps(initial, sort_keys=True))
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_shards)
        totals = {key: 0.0 for key in ("loss", "unit_loss", "market_loss", "value_loss", "unit_correct", "unit_count", "market_correct", "market_count")}
        batches = 0
        for shard in train_shards:
            with np.load(shard) as data:
                order = np.random.permutation(len(data["features"]))
                for start in range(0, len(order), args.batch_size):
                    indices = order[start : start + args.batch_size]
                    batch = load_batch(data, indices, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = batch_loss(model, batch, device.type == "cuda")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    for key, value in metrics.items():
                        totals[key] += value
                    batches += 1
        for key in ("loss", "unit_loss", "market_loss", "value_loss"):
            totals[key] /= max(1, batches)
        totals["unit_accuracy"] = totals["unit_correct"] / max(1.0, totals["unit_count"])
        totals["market_accuracy"] = totals["market_correct"] / max(1.0, totals["market_count"])
        validation = evaluate(model, validation_shards, args.batch_size, device)
        print(f"train_epoch={epoch} " + json.dumps(totals, sort_keys=True))
        print(f"validation_epoch={epoch} " + json.dumps(validation, sort_keys=True))
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden_size": args.hidden_size,
                    "schema_version": 1,
                    "dataset": str(args.dataset_dir),
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output,
            )
            print(f"saved={args.output}")


if __name__ == "__main__":
    main()
