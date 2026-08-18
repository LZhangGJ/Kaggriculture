"""Train a multi-head route-value network from counterfactual JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from kaggriculture_lab.route_learning import (
    PLAN_FEATURE_NAMES,
    ROUTE_CONTEXT_FEATURE_NAMES,
    RouteNormalizer,
    RouteValueNetwork,
    evaluate_route_model,
    read_counterfactual_records,
    records_to_batch,
    route_value_loss,
    scenario_batches,
    split_records_by_scenario,
    validate_counterfactual_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-scenarios", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = read_counterfactual_records(args.dataset)
    if not records:
        raise ValueError("The dataset is empty")
    validate_counterfactual_groups(records)
    training, validation = split_records_by_scenario(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not training:
        raise ValueError("No training scenarios remain after the split")
    normalizer = RouteNormalizer.fit(training)
    device = torch.device(args.device)
    model = RouteValueNetwork(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        pair_count = 0
        for indices in scenario_batches(
            training,
            scenarios_per_batch=args.batch_scenarios,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            batch = records_to_batch(training, indices, normalizer, device=device)
            predictions = model(batch.context, batch.plan)
            losses = route_value_loss(predictions, batch)
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses.total.detach().cpu()))
            pair_count += losses.pair_count

        train_metrics = evaluate_route_model(
            model,
            training,
            normalizer,
            device=device,
            scenarios_per_batch=max(args.batch_scenarios, 64),
        )
        validation_metrics = evaluate_route_model(
            model,
            validation,
            normalizer,
            device=device,
            scenarios_per_batch=max(args.batch_scenarios, 64),
        ) if validation else train_metrics
        selection_metric = float(validation_metrics["top1"])
        row = {
            "epoch": epoch,
            "train_batch_loss": float(np.mean(totals)),
            "train_top1": float(train_metrics["top1"]),
            "validation_top1": selection_metric,
            "validation_loss": float(validation_metrics["loss"]),
            "pairs": pair_count,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if selection_metric > best_metric + 1e-8:
            best_metric = selection_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early_stop epoch={epoch} best_validation_top1={best_metric:.4f}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_train = evaluate_route_model(model, training, normalizer, device=device)
    final_validation = (
        evaluate_route_model(model, validation, normalizer, device=device)
        if validation
        else final_train
    )
    route_names = sorted({record.route_name for record in records})
    checkpoint = {
        "schema": "kaggriculture-route-selector-v1",
        "model": best_state,
        "context_dim": len(ROUTE_CONTEXT_FEATURE_NAMES),
        "plan_dim": len(PLAN_FEATURE_NAMES),
        "hidden_size": args.hidden_size,
        "normalizer": normalizer.to_mapping(),
        "context_feature_names": ROUTE_CONTEXT_FEATURE_NAMES,
        "plan_feature_names": PLAN_FEATURE_NAMES,
        "route_names": route_names,
        "training_scenarios": len({record.scenario_id for record in training}),
        "validation_scenarios": len({record.scenario_id for record in validation}),
        "metrics": {"train": final_train, "validation": final_validation},
        "history": history,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(checkpoint["metrics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output}")
    print(json.dumps(checkpoint["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
