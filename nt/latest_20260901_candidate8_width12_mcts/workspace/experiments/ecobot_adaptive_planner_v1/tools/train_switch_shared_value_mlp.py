#!/usr/bin/env python3
"""Train a small shared candidate-value MLP with pairwise and cash losses.

The input NPZ files contain the deterministic public-state/candidate features
prepared by ``generalized_engineered_features``.  One network scores every
candidate independently.  Pairwise preferences are therefore represented as
``score(left) - score(right)`` and are automatically transitive.  A small
Huber auxiliary loss can anchor the score to counterfactual delta cash.

The validation corpus selects architecture/loss/checkpoint and the KEEP safety
threshold.  The diagnostic test corpus is evaluated only after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


VALUE_SCALE = 5000.0
KEEP_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
KEEP_BONUSES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class LossVariant:
    name: str
    soft_label_scale: float
    cash_aux_weight: float


VARIANTS = (
    LossVariant("hard_pair_only", 0.0, 0.0),
    LossVariant("soft500_cash010", 500.0, 0.10),
    LossVariant("soft1000_cash020", 1000.0, 0.20),
    LossVariant("soft2000_cash035", 2000.0, 0.35),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def group_keys(data: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.asarray(data["source_dataset_index"], dtype=np.int64) * 10**10
        + np.asarray(data["prefix_seed"], dtype=np.int64) * 2
        + np.asarray(data["seat"], dtype=np.int64)
    )


def pair_structure(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    keys = group_keys(data)
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    std = np.asarray(data["future_std"], dtype=np.float64)
    count = np.asarray(data["future_sample_count"], dtype=np.float64)
    standard_error = std / np.sqrt(np.maximum(1.0, count))
    left: list[int] = []
    right: list[int] = []
    difference: list[float] = []
    weight: list[float] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        for left_position in range(len(rows)):
            for right_position in range(left_position + 1, len(rows)):
                left_row = int(rows[left_position])
                right_row = int(rows[right_position])
                gap = float(actual[left_row] - actual[right_row])
                if gap == 0.0:
                    continue
                pair_error = math.hypot(
                    float(standard_error[left_row]),
                    float(standard_error[right_row]),
                )
                confidence = 1.0 / (1.0 + (pair_error / 2500.0) ** 2)
                importance = 1.0 + min(abs(gap) / 3000.0, 3.0)
                left.append(left_row)
                right.append(right_row)
                difference.append(gap)
                weight.append(float(np.clip(confidence * importance, 0.05, 4.0)))
    return (
        np.asarray(left, dtype=np.int32),
        np.asarray(right, dtype=np.int32),
        np.asarray(difference, dtype=np.float32),
        np.asarray(weight, dtype=np.float32),
    )


def fit_preprocessor(train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = np.asarray(train["x"], dtype=np.float64)
    categorical = np.asarray(train["categorical"], dtype=np.int64)
    continuous = np.asarray(
        [index for index in range(x.shape[1]) if index not in set(categorical)],
        dtype=np.int64,
    )
    mean = np.mean(x[:, continuous], axis=0)
    scale = np.std(x[:, continuous], axis=0)
    scale[scale < 1e-6] = 1.0
    # source/destination include KEEP=-1; seat is binary; edit_family is the
    # generic source/destination/count encoding and currently spans 0..64.
    offsets = np.asarray([1, 1, 0, 0], dtype=np.int64)
    sizes = np.asarray([9, 9, 2, 65], dtype=np.int64)
    return {
        "categorical": categorical,
        "continuous": continuous,
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "category_offsets": offsets,
        "category_sizes": sizes,
    }


def transform(
    data: dict[str, np.ndarray], preprocessor: dict[str, np.ndarray]
) -> np.ndarray:
    raw = np.asarray(data["x"], dtype=np.float64)
    continuous = np.asarray(preprocessor["continuous"], dtype=np.int64)
    normalized = (
        raw[:, continuous] - np.asarray(preprocessor["mean"])
    ) / np.asarray(preprocessor["scale"])
    normalized = np.clip(normalized, -10.0, 10.0)
    encoded = [normalized]
    for index, offset, size in zip(
        preprocessor["categorical"],
        preprocessor["category_offsets"],
        preprocessor["category_sizes"],
        strict=True,
    ):
        values = np.asarray(np.rint(raw[:, int(index)]), dtype=np.int64) + int(offset)
        if values.size and (values.min() < 0 or values.max() >= int(size)):
            raise ValueError(
                f"categorical value outside frozen range at column {index}: "
                f"[{values.min()}, {values.max()}] vs size={size}"
            )
        encoded.append(np.eye(int(size), dtype=np.float64)[values])
    matrix = np.concatenate(encoded, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite transformed MLP feature")
    return matrix


def init_params(key: jax.Array, width: int, hidden: tuple[int, int]):
    sizes = (width, hidden[0], hidden[1], 1)
    keys = jax.random.split(key, 3)
    layers = []
    for layer, (source, destination) in enumerate(zip(sizes[:-1], sizes[1:])):
        limit = math.sqrt(6.0 / max(1, source))
        weight = jax.random.uniform(
            keys[layer], (source, destination), minval=-limit, maxval=limit
        )
        layers.append({"w": weight, "b": jnp.zeros((destination,))})
    return tuple(layers)


def score(params, x: jax.Array) -> jax.Array:
    value = x
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["w"] + layer["b"])
    final = params[-1]
    return (value @ final["w"] + final["b"]).squeeze(-1)


def pair_metrics(
    difference: np.ndarray, predicted_difference: np.ndarray
) -> list[dict[str, float | int]]:
    rows = []
    for gap in (0.0, 500.0, 1000.0, 2000.0):
        selected = np.abs(difference) >= max(1e-9, gap)
        correct = (
            (predicted_difference[selected] >= 0.0)
            == (difference[selected] > 0.0)
        )
        rows.append({
            "minimum_expected_value_gap": gap,
            "pairs": int(np.sum(selected)),
            "accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        })
    return rows


def ranking_metrics(
    data: dict[str, np.ndarray], scores: np.ndarray
) -> dict[str, object]:
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    keys = group_keys(data)
    hits = np.zeros(5, dtype=np.int64)
    selected = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        order = np.argsort(-scores[rows], kind="stable")
        oracle = int(np.argmax(actual[rows]))
        for top_k in range(1, 6):
            hits[top_k - 1] += oracle in order[:top_k]
        selected.append(float(actual[rows[int(order[0])]]))
    values = np.asarray(selected, dtype=np.float64)
    groups = len(values)
    return {
        "groups": groups,
        "mean_realized_delta": float(np.mean(values)),
        "negative_choice_rate": float(np.mean(values < 0.0)),
        "topk_oracle_recall": {
            str(k): float(hits[k - 1] / groups) for k in range(1, 6)
        },
    }


def safety_metrics(
    data: dict[str, np.ndarray], scores: np.ndarray,
    threshold: float, keep_bonus: float,
) -> dict[str, object]:
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    rank = np.asarray(data["candidate_rank"], dtype=np.int64)
    keys = group_keys(data)
    chosen = []
    switches = 0
    top3_hits = 0
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        local_rank = rank[rows]
        keep_rows = np.flatnonzero(local_rank < 0)
        if len(keep_rows) != 1:
            raise ValueError(f"group {key} has {len(keep_rows)} KEEP rows")
        keep = int(keep_rows[0])
        local_scores = np.asarray(scores[rows], dtype=np.float64).copy()
        beat_keep = 1.0 / (
            1.0 + np.exp(np.clip(local_scores[keep] - local_scores, -50.0, 50.0))
        )
        allowed = (local_rank < 0) | (beat_keep >= threshold)
        local_scores[~allowed] = -np.inf
        local_scores[keep] += keep_bonus
        order = np.argsort(-local_scores, kind="stable")
        selected = int(order[0])
        value = actual[rows[selected]]
        chosen.append(float(value))
        switches += int(local_rank[selected] >= 0)
        oracle = int(np.argmax(actual[rows]))
        finite = [int(i) for i in order if np.isfinite(local_scores[i])]
        top3_hits += oracle in finite[:3]
    values = np.asarray(chosen, dtype=np.float64)
    groups = len(values)
    return {
        "groups": groups,
        "keep_win_threshold": threshold,
        "keep_bonus": keep_bonus,
        "mean_realized_delta": float(np.mean(values)),
        "median_realized_delta": float(np.median(values)),
        "p10_realized_delta": float(np.quantile(values, 0.10)),
        "min_realized_delta": float(np.min(values)),
        "positive_choice_rate": float(np.mean(values > 0.0)),
        "negative_choice_rate": float(np.mean(values < 0.0)),
        "switch_rate": switches / groups,
        "top3_oracle_recall_with_keep": top3_hits / groups,
    }


def safety_key(metrics: dict[str, object]) -> tuple[float, ...]:
    switch_rate = float(metrics["switch_rate"])
    negative_rate = float(metrics["negative_choice_rate"])
    p10 = float(metrics["p10_realized_delta"])
    mean = float(metrics["mean_realized_delta"])
    strong = switch_rate >= 0.10 and negative_rate <= 0.10 and p10 >= 0.0
    weak = switch_rate >= 0.10 and negative_rate <= 0.15 and p10 >= -500.0
    return (
        float(strong), float(weak), mean, p10, -negative_rate,
        float(metrics["top3_oracle_recall_with_keep"]), switch_rate,
    )


def w3_key(pairwise: list[dict[str, object]], ranking: dict[str, object]):
    topk = ranking["topk_oracle_recall"]
    return (
        float(float(topk["4"]) >= 0.90),
        float(pairwise[0]["accuracy"]),
        float(pairwise[2]["accuracy"]),
        float(topk["4"]),
        float(topk["5"]),
        float(ranking["mean_realized_delta"]),
    )


def copy_params(params):
    return jax.tree.map(lambda value: np.asarray(value), params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seeds", default="7701,7702")
    parser.add_argument("--hidden", default="192,96")
    args = parser.parse_args()

    train = load(args.train)
    validation = load(args.validation)
    test = load(args.test)
    if not np.array_equal(train["feature_names"], validation["feature_names"]):
        raise ValueError("train/validation engineered feature schema mismatch")
    if not np.array_equal(train["feature_names"], test["feature_names"]):
        raise ValueError("train/test engineered feature schema mismatch")
    preprocessor = fit_preprocessor(train)
    train_x = transform(train, preprocessor)
    validation_x = transform(validation, preprocessor)
    test_x = transform(test, preprocessor)
    hidden = tuple(int(value) for value in args.hidden.split(","))
    if len(hidden) != 2:
        raise ValueError("--hidden must contain exactly two widths")
    seeds = [int(value) for value in args.seeds.split(",") if value]

    train_left, train_right, train_difference, train_weight = pair_structure(train)
    val_left, val_right, val_difference, _ = pair_structure(validation)
    test_left, test_right, test_difference, _ = pair_structure(test)
    train_target_cash = np.asarray(train["expected_delta"], dtype=np.float32) / VALUE_SCALE
    batch_size = min(args.batch_size, len(train_left))
    steps_per_epoch = int(math.ceil(len(train_left) / batch_size))
    train_x_device = jax.device_put(train_x)

    def loss_function(
        params, left, right, pair_difference, pair_weight,
        soft_label_scale, cash_aux_weight,
    ):
        left_score = score(params, train_x_device[left])
        right_score = score(params, train_x_device[right])
        if soft_label_scale > 0.0:
            target = jax.nn.sigmoid(pair_difference / soft_label_scale)
            logit_scale = VALUE_SCALE / soft_label_scale
        else:
            target = (pair_difference > 0.0).astype(jnp.float32)
            logit_scale = 1.0
        logits = (left_score - right_score) * logit_scale
        pair_loss = optax.sigmoid_binary_cross_entropy(logits, target)
        pair_loss = jnp.sum(pair_loss * pair_weight) / jnp.sum(pair_weight)
        left_target = jnp.asarray(train_target_cash)[left]
        right_target = jnp.asarray(train_target_cash)[right]
        cash_loss = 0.5 * (
            jnp.mean(optax.huber_loss(left_score, left_target, delta=1.0))
            + jnp.mean(optax.huber_loss(right_score, right_target, delta=1.0))
        )
        return pair_loss + cash_aux_weight * cash_loss

    audits = []
    best_key = None
    best_params = None
    best_description = None
    for variant_index, variant in enumerate(VARIANTS):
        for seed in seeds:
            params = init_params(jax.random.key(seed), train_x.shape[1], hidden)
            optimizer = optax.adamw(1e-3, weight_decay=1e-5)
            optimizer_state = optimizer.init(params)

            @jax.jit
            def train_step(params, optimizer_state, left, right, difference, weight):
                loss, gradients = jax.value_and_grad(loss_function)(
                    params, left, right, difference, weight,
                    variant.soft_label_scale, variant.cash_aux_weight,
                )
                updates, optimizer_state = optimizer.update(
                    gradients, optimizer_state, params
                )
                return optax.apply_updates(params, updates), optimizer_state, loss

            rng = np.random.default_rng(seed)
            run_best_key = None
            run_best = None
            history = []
            for epoch in range(1, args.epochs + 1):
                order = rng.permutation(len(train_left))
                losses = []
                for step_index in range(steps_per_epoch):
                    begin = step_index * batch_size
                    selection = order[begin:begin + batch_size]
                    params, optimizer_state, batch_loss = train_step(
                        params,
                        optimizer_state,
                        jnp.asarray(train_left[selection]),
                        jnp.asarray(train_right[selection]),
                        jnp.asarray(train_difference[selection]),
                        jnp.asarray(train_weight[selection]),
                    )
                    losses.append(float(batch_loss))
                if epoch % args.eval_every != 0 and epoch != args.epochs:
                    continue
                validation_scores = np.asarray(score(params, validation_x))
                predicted_difference = (
                    validation_scores[val_left] - validation_scores[val_right]
                )
                pairwise = pair_metrics(val_difference, predicted_difference)
                ranking = ranking_metrics(validation, validation_scores)
                key = w3_key(pairwise, ranking)
                history.append({
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "pairwise": pairwise,
                    "ranking": ranking,
                })
                if run_best_key is None or key > run_best_key:
                    run_best_key = key
                    run_best = {
                        "params": copy_params(params),
                        "epoch": epoch,
                        "pairwise": pairwise,
                        "ranking": ranking,
                    }
            assert run_best is not None
            run = {
                "variant_index": variant_index,
                "variant": variant.__dict__,
                "seed": seed,
                "selected_epoch": run_best["epoch"],
                "validation_pairwise": run_best["pairwise"],
                "validation_ranking": run_best["ranking"],
                "history": history,
            }
            audits.append(run)
            if best_key is None or run_best_key > best_key:
                best_key = run_best_key
                best_params = run_best["params"]
                best_description = run

    assert best_params is not None and best_description is not None
    validation_scores = np.asarray(score(best_params, validation_x))
    safety_grid = [
        safety_metrics(validation, validation_scores, threshold, bonus)
        for threshold in KEEP_THRESHOLDS
        for bonus in KEEP_BONUSES
    ]
    selected_safety = max(safety_grid, key=safety_key)
    test_scores = np.asarray(score(best_params, test_x))
    test_pairwise = pair_metrics(
        test_difference, test_scores[test_left] - test_scores[test_right]
    )
    test_ranking = ranking_metrics(test, test_scores)
    test_safety = safety_metrics(
        test,
        test_scores,
        float(selected_safety["keep_win_threshold"]),
        float(selected_safety["keep_bonus"]),
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {}
    for layer_index, layer in enumerate(best_params):
        model_payload[f"layer_{layer_index}_weight"] = np.asarray(layer["w"])
        model_payload[f"layer_{layer_index}_bias"] = np.asarray(layer["b"])
    model_payload.update({
        "continuous": preprocessor["continuous"],
        "categorical": preprocessor["categorical"],
        "mean": preprocessor["mean"],
        "scale": preprocessor["scale"],
        "category_offsets": preprocessor["category_offsets"],
        "category_sizes": preprocessor["category_sizes"],
        "keep_win_threshold": np.asarray(
            selected_safety["keep_win_threshold"], dtype=np.float32
        ),
        "keep_bonus": np.asarray(selected_safety["keep_bonus"], dtype=np.float32),
    })
    np.savez_compressed(args.model_output, **model_payload)
    payload = {
        "schema": "kaggriculture.switch-shared-value-mlp.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC",
        "invariants": {
            "validation_only_model_selection": True,
            "test_used_for_selection": False,
            "candidate_score_is_shared": True,
            "pair_preference_is_transitive_score_difference": True,
            "opponent_identity_feature": False,
        },
        "device": str(jax.devices()[0]),
        "train": {"path": str(args.train), "sha256": sha256(args.train),
                  "rows": len(train_x), "pairs": len(train_left)},
        "validation": {"path": str(args.validation),
                       "sha256": sha256(args.validation),
                       "rows": len(validation_x), "pairs": len(val_left)},
        "test": {"path": str(args.test), "sha256": sha256(args.test),
                 "rows": len(test_x), "pairs": len(test_left)},
        "input_dim": int(train_x.shape[1]),
        "hidden": list(hidden),
        "parameter_count": int(sum(value.size for layer in best_params
                                    for value in layer.values())),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "seeds": seeds,
        "audits": audits,
        "selected": best_description,
        "selected_validation_safety": selected_safety,
        "test_pairwise_not_used_for_selection": test_pairwise,
        "test_ranking_not_used_for_selection": test_ranking,
        "test_safety_not_used_for_selection": test_safety,
        "model": {"path": str(args.model_output),
                  "sha256": sha256(args.model_output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_variant": best_description["variant"],
        "selected_seed": best_description["seed"],
        "selected_epoch": best_description["selected_epoch"],
        "validation_pairwise": best_description["validation_pairwise"],
        "validation_ranking": best_description["validation_ranking"],
        "validation_safety": selected_safety,
        "test_pairwise": test_pairwise,
        "test_ranking": test_ranking,
        "test_safety": test_safety,
        "model": str(args.model_output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
