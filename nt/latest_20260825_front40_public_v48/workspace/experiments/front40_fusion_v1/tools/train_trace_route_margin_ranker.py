"""Train a visible-state route ranker from a counterfactual screen matrix.

The terminal margins are training labels only.  Runtime features are restricted
to public current state plus the controlled player's private inventory and
static candidate-route descriptors available in the frozen Replay bank.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


NUM_KINDS = 6
NUM_CROPS = 5
NUM_ANIMALS = 3
NUM_PRODUCTS = 9
NUM_SHED_ITEMS = 12
NUM_SHOPS = 8


def add(parts: list[np.ndarray], names: list[str], value: np.ndarray, prefix: str) -> None:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    flat = value.reshape(value.shape[0], -1)
    parts.append(flat)
    names.extend(f"{prefix}_{index}" for index in range(flat.shape[1]))


def categorical_counts(value: np.ndarray, size: int) -> np.ndarray:
    flat = value.reshape(value.shape[0], -1)
    return np.stack([(flat == index).sum(axis=1) for index in range(size)], axis=1)


def flag_counts(flags: np.ndarray) -> np.ndarray:
    flat = flags.reshape(flags.shape[0], -1)
    return np.stack([((flat & bit) != 0).sum(axis=1) for bit in (1, 2, 4, 8)], axis=1)


def shop_one_hot(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.int16)
    return np.concatenate(
        [(sequence[:, position : position + 1] == np.arange(NUM_SHOPS)[None, :]).astype(np.float32)
         for position in range(sequence.shape[1])],
        axis=1,
    )


def build_context_features(context: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    seats, seeds = context["candidate_seat"].shape
    rows = seats * seeds
    parts: list[np.ndarray] = []
    names: list[str] = []

    def flat(name: str) -> np.ndarray:
        value = np.asarray(context[name])
        return value.reshape((rows, *value.shape[2:]))

    add(parts, names, flat("candidate_seat"), "ctx_seat")
    add(parts, names, flat("own_money"), "ctx_own_money")
    add(parts, names, flat("opponent_money"), "ctx_opp_money")
    for side in ("own", "opponent"):
        kind = flat(f"{side}_tile_kind")
        crop = flat(f"{side}_tile_crop")
        animal = flat(f"{side}_tile_animal")
        tile_yield = flat(f"{side}_tile_yield")
        neglect = flat(f"{side}_tile_neglect")
        flags = flat(f"{side}_tile_flags")
        add(parts, names, categorical_counts(kind, NUM_KINDS), f"ctx_{side}_kind_count")
        add(parts, names, categorical_counts(crop, NUM_CROPS), f"ctx_{side}_crop_count")
        add(parts, names, categorical_counts(animal, NUM_ANIMALS), f"ctx_{side}_animal_count")
        yield_features = []
        for item in range(NUM_CROPS):
            yield_features.append(np.where(crop == item, tile_yield, 0).sum(axis=(1, 2)))
        for item in range(NUM_ANIMALS):
            yield_features.append(np.where(animal == item, tile_yield, 0).sum(axis=(1, 2)))
        add(parts, names, np.stack(yield_features, axis=1), f"ctx_{side}_yield")
        add(parts, names, neglect.reshape(rows, -1).sum(axis=1), f"ctx_{side}_neglect_sum")
        add(parts, names, neglect.reshape(rows, -1).max(axis=1), f"ctx_{side}_neglect_max")
        add(parts, names, flag_counts(flags), f"ctx_{side}_flag_count")
        # Spatial weed pattern is the main first-day stochastic disturbance.
        add(parts, names, (kind == 2).astype(np.float32), f"ctx_{side}_weed_map")

    for side in ("own", "opponent"):
        add(parts, names, flat(f"{side}_unit_active"), f"ctx_{side}_unit_active")
        add(parts, names, flat(f"{side}_unit_pos"), f"ctx_{side}_unit_pos")
    own_inventory = flat("own_unit_inventory").sum(axis=1)
    add(parts, names, own_inventory, "ctx_own_carried")
    for field in (
        "own_shed", "own_seeds", "own_hires_today", "opponent_hires_today",
        "own_unlocked_count", "opponent_unlocked_count", "market_inventory",
        "market_price", "town_count",
    ):
        add(parts, names, flat(field), f"ctx_{field}")
    town = flat("town_shops")
    add(parts, names, shop_one_hot(town), "ctx_shop_position")
    return np.concatenate(parts, axis=1).astype(np.float32), names


def planned_route_features(bank: np.lib.npyio.NpzFile, start: int) -> tuple[np.ndarray, list[str]]:
    routes = bank["unit_op"].shape[0]
    parts: list[np.ndarray] = []
    names: list[str] = []

    def route_add(value: np.ndarray, prefix: str) -> None:
        add(parts, names, value, prefix)

    route_add(bank["source_reward"], "route_source_reward")
    route_add(bank["source_margin"], "route_source_margin")
    route_add(bank["source_seat"], "route_source_seat")
    route_add(bank["expected_self_summary"][:, start], "route_own_summary")
    route_add(bank["expected_opponent_summary"][:, start], "route_opp_summary")
    route_add(bank["expected_shed"][:, start], "route_shed")
    route_add(bank["expected_seeds"][:, start], "route_seeds")
    route_add(bank["expected_carried"][:, start], "route_carried")
    route_add(bank["expected_market_price"][:, start], "route_market_price")
    route_add(bank["expected_market_inventory"][:, start], "route_market_inventory")
    route_add(bank["expected_unit_active"][:, start], "route_unit_active")
    route_add(bank["expected_unit_pos"][:, start], "route_unit_pos")
    route_add(shop_one_hot(bank["source_shop_sequence"]), "route_shop_position")
    route_add(np.eye(routes, dtype=np.float32), "route_id")

    unit_op = np.asarray(bank["unit_op"])
    unit_item = np.asarray(bank["unit_item"])
    unit_count = np.asarray(bank["unit_count"])
    market_op = np.asarray(bank["market_op"])
    market_item = np.asarray(bank["market_item"])
    market_amount = np.asarray(bank["market_amount"])
    market_count = np.asarray(bank["market_count"])
    for window_start, window_end in ((start, 144), (144, 240), (240, 480), (480, 719)):
        unit_slot = np.arange(unit_op.shape[2])[None, None, :]
        unit_active = unit_slot < unit_count[:, window_start:window_end, None]
        op_counts = np.stack(
            [np.sum(unit_active & (unit_op[:, window_start:window_end] == op), axis=(1, 2))
             for op in range(18)],
            axis=1,
        )
        route_add(op_counts, f"route_w{window_start}_unit_op_count")
        plant_counts = np.stack(
            [np.sum(
                unit_active
                & (unit_op[:, window_start:window_end] == 8)
                & (unit_item[:, window_start:window_end] == crop),
                axis=(1, 2),
            ) for crop in range(NUM_CROPS)],
            axis=1,
        )
        route_add(plant_counts, f"route_w{window_start}_plant_count")
        market_slot = np.arange(market_op.shape[2])[None, None, :]
        market_active = market_slot < market_count[:, window_start:window_end, None]
        route_add(
            np.stack(
                [np.sum(market_active & (market_op[:, window_start:window_end] == op), axis=(1, 2))
                 for op in range(7)],
                axis=1,
            ),
            f"route_w{window_start}_market_op_count",
        )
        for op, size, label, offset in (
            (3, NUM_CROPS, "seed", 0),
            (4, NUM_PRODUCTS, "product", 0),
            (5, NUM_ANIMALS, "animal", NUM_PRODUCTS),
            (6, NUM_PRODUCTS, "sell", 0),
        ):
            quantity = []
            for item in range(size):
                match = (
                    market_active
                    & (market_op[:, window_start:window_end] == op)
                    & (market_item[:, window_start:window_end] == item + offset)
                )
                quantity.append(
                    np.sum(np.where(match, market_amount[:, window_start:window_end], 0), axis=(1, 2))
                )
            route_add(np.stack(quantity, axis=1), f"route_w{window_start}_{label}_quantity")
    return np.concatenate(parts, axis=1).astype(np.float32), names


def build_difference_features(
    context: np.lib.npyio.NpzFile, bank: np.lib.npyio.NpzFile, step: int
) -> tuple[np.ndarray, list[str]]:
    seats, seeds = context["candidate_seat"].shape
    rows = seats * seeds
    routes = bank["unit_op"].shape[0]

    def flat(name: str) -> np.ndarray:
        value = np.asarray(context[name])
        return value.reshape((rows, *value.shape[2:]))

    own_kind = flat("own_tile_kind")
    opp_kind = flat("opponent_tile_kind")
    own_crop = flat("own_tile_crop")
    opp_crop = flat("opponent_tile_crop")
    own_animal = flat("own_tile_animal")
    opp_animal = flat("opponent_tile_animal")
    own_summary = np.concatenate(
        (
            categorical_counts(own_animal, NUM_ANIMALS),
            categorical_counts(own_crop, NUM_CROPS),
            np.sum(own_kind != 1, axis=(1, 2))[:, None],
            flat("own_unit_active").sum(axis=1)[:, None] - 1,
            flat("own_money")[:, None] if flat("own_money").ndim == 1 else flat("own_money"),
        ),
        axis=1,
    )
    opp_summary = np.concatenate(
        (
            categorical_counts(opp_animal, NUM_ANIMALS),
            categorical_counts(opp_crop, NUM_CROPS),
            np.sum(opp_kind != 1, axis=(1, 2))[:, None],
            flat("opponent_unit_active").sum(axis=1)[:, None] - 1,
            flat("opponent_money")[:, None] if flat("opponent_money").ndim == 1 else flat("opponent_money"),
        ),
        axis=1,
    )
    actual = np.concatenate(
        (
            own_summary,
            opp_summary,
            flat("own_shed"),
            flat("own_seeds"),
            flat("own_unit_inventory").sum(axis=1),
            flat("market_price"),
            flat("market_inventory"),
        ),
        axis=1,
    ).astype(np.float32)
    expected = np.concatenate(
        (
            bank["expected_self_summary"][:, step],
            bank["expected_opponent_summary"][:, step],
            bank["expected_shed"][:, step],
            bank["expected_seeds"][:, step],
            bank["expected_carried"][:, step],
            bank["expected_market_price"][:, step],
            bank["expected_market_inventory"][:, step],
        ),
        axis=1,
    ).astype(np.float32)
    signed = actual[:, None, :] - expected[None, :, :]
    features = np.concatenate((signed, np.abs(signed)), axis=2)
    names = [f"diff_signed_{index}" for index in range(actual.shape[1])]
    names += [f"diff_abs_{index}" for index in range(actual.shape[1])]
    return features.reshape(rows * routes, -1).astype(np.float32), names


def policy_metrics(margin: np.ndarray, score: np.ndarray, context_mask: np.ndarray) -> dict:
    route = np.argmax(score[context_mask], axis=1)
    selected = margin[context_mask, route]
    return {
        "contexts": int(np.sum(context_mask)),
        "wins": int(np.sum(selected > 0)),
        "win_rate": float(np.mean(selected > 0)),
        "mean_margin": float(np.mean(selected)),
        "median_margin": float(np.median(selected)),
        "unique_routes": int(np.unique(route).size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--decision-step", type=int, default=72)
    parser.add_argument("--validation-seeds", type=int, default=64)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument(
        "--route-ids",
        default="",
        help="Optional comma-separated global route IDs retained for training.",
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    context = np.load(args.context)
    bank = np.load(args.route_bank)
    screen = np.load(args.matrix)
    margin3_screen = np.asarray(screen["margin"], dtype=np.float32)
    seats, seeds, screen_routes = margin3_screen.shape
    bank_routes = int(bank["unit_op"].shape[0])
    screen_route_ids = np.asarray(
        screen["route_ids"] if "route_ids" in screen else np.arange(screen_routes),
        dtype=np.int32,
    )
    if screen_route_ids.shape != (screen_routes,):
        raise ValueError("screen route ID shape mismatch")
    route_ids = np.asarray(
        [int(value) for value in args.route_ids.split(",") if value]
        if args.route_ids
        else screen_route_ids.tolist(),
        dtype=np.int32,
    )
    if route_ids.size == 0 or np.any(route_ids < 0) or np.any(route_ids >= bank_routes):
        raise ValueError(f"invalid route IDs: {route_ids.tolist()}")
    if np.unique(route_ids).size != route_ids.size:
        raise ValueError("route IDs must be unique")
    screen_lookup = {int(route): index for index, route in enumerate(screen_route_ids)}
    if any(int(route) not in screen_lookup for route in route_ids):
        raise ValueError("requested route ID is absent from screen matrix")
    screen_columns = np.asarray(
        [screen_lookup[int(route)] for route in route_ids], dtype=np.int32
    )
    margin3 = margin3_screen[:, :, screen_columns]
    routes = int(route_ids.size)
    if context["candidate_seat"].shape != (seats, seeds):
        raise ValueError("context shape mismatch")

    context_features, context_names = build_context_features(context)
    route_features_full, route_names = planned_route_features(bank, args.decision_step)
    route_features = route_features_full[route_ids]
    difference_features_full, difference_names = build_difference_features(
        context, bank, args.decision_step
    )
    contexts = seats * seeds
    difference_features = difference_features_full.reshape(
        contexts, bank_routes, -1
    )[:, route_ids].reshape(contexts * routes, -1)
    matrix = np.concatenate(
        (
            np.repeat(context_features, routes, axis=0),
            np.tile(route_features, (contexts, 1)),
            difference_features,
        ),
        axis=1,
    ).astype(np.float32)
    feature_names = context_names + route_names + difference_names
    margin = margin3.reshape(contexts, routes)
    labels = margin.reshape(-1)
    win_labels = (labels > 0).astype(np.int8)

    seed_index = np.tile(np.arange(seeds), seats)
    validation_seed_mask = seed_index >= seeds - args.validation_seeds
    train_context = ~validation_seed_mask
    validation_context = validation_seed_mask
    train_rows = np.repeat(train_context, routes)
    validation_rows = np.repeat(validation_context, routes)

    margin_model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        random_state=20260824,
        n_jobs=args.threads,
        verbosity=-1,
    )
    margin_model.fit(
        matrix[train_rows],
        labels[train_rows],
        eval_set=[(matrix[validation_rows], labels[validation_rows])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    win_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        random_state=20260825,
        n_jobs=args.threads,
        verbosity=-1,
    )
    win_model.fit(
        matrix[train_rows],
        win_labels[train_rows],
        eval_set=[(matrix[validation_rows], win_labels[validation_rows])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    predicted_margin = margin_model.predict(matrix).reshape(contexts, routes)
    predicted_win = win_model.predict_proba(matrix)[:, 1].reshape(contexts, routes)

    # Group-relative relevance directly optimizes route order.  Absolute cash
    # regression can be good globally while still picking the wrong route in a
    # particular context, so retain both objectives and select their blend on
    # seed-grouped validation only.
    relevance = np.zeros_like(margin, dtype=np.int32)
    relevance[margin > -10_000] = 1
    relevance[margin > 0] = 2
    order = np.argsort(margin, axis=1)
    for depth, value in ((10, 3), (3, 4), (1, 5)):
        selected = order[:, -depth:]
        relevance[np.arange(contexts)[:, None], selected] = np.maximum(
            relevance[np.arange(contexts)[:, None], selected], value
        )
    rank_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 3, 5),
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=40,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=20260826,
        n_jobs=args.threads,
        verbosity=-1,
    )
    rank_model.fit(
        matrix[train_rows],
        relevance[train_context].reshape(-1),
        group=np.full(int(np.sum(train_context)), routes, dtype=np.int32),
        eval_set=[(matrix[validation_rows], relevance[validation_context].reshape(-1))],
        eval_group=[np.full(int(np.sum(validation_context)), routes, dtype=np.int32)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    predicted_rank = rank_model.predict(matrix).reshape(contexts, routes)

    blends = []
    best = None
    scale = max(float(np.std(predicted_margin[train_context])), 1.0)
    rank_scale = max(float(np.std(predicted_rank[train_context])), 1e-6)
    for win_weight in (0.0, 1.0, 2.0, 4.0):
        for rank_weight in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            if win_weight == 0.0 and rank_weight == 0.0:
                continue
            score = (
                predicted_margin
                + win_weight * scale * predicted_win
                + rank_weight * scale * predicted_rank / rank_scale
            )
            valid_metrics = policy_metrics(margin, score, validation_context)
            row = {
                "win_probability_weight": win_weight,
                "rank_weight": rank_weight,
                "validation": valid_metrics,
            }
            blends.append(row)
            key = (valid_metrics["win_rate"], valid_metrics["mean_margin"])
            if best is None or key > best[0]:
                best = (key, (win_weight, rank_weight), score)
    assert best is not None
    best_win_weight, best_rank_weight = best[1]
    best_score = best[2]
    train_metrics = policy_metrics(margin, best_score, train_context)
    validation_metrics = policy_metrics(margin, best_score, validation_context)
    fixed_route = int(np.argmax(np.mean(margin[train_context], axis=0)))
    fixed_validation = margin[validation_context, fixed_route]
    oracle_validation = np.max(margin[validation_context], axis=1)

    artifact = {
        "margin_model": margin_model,
        "win_model": win_model,
        "rank_model": rank_model,
        "win_probability_weight": best_win_weight,
        "rank_weight": best_rank_weight,
        "score_scale": scale,
        "rank_scale": rank_scale,
        "decision_step": args.decision_step,
        "feature_names": feature_names,
        "route_count": routes,
        "route_ids": route_ids,
        "route_bank": str(args.route_bank),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model)
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-route-margin-ranker.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "runtime_boundary": "current public state plus own private inventory; terminal margin is label only",
        "context": str(args.context),
        "route_bank": str(args.route_bank),
        "screen_matrix": str(args.matrix),
        "decision_step": args.decision_step,
        "contexts": contexts,
        "routes": routes,
        "route_ids": route_ids.tolist(),
        "training_rows": int(np.sum(train_rows)),
        "validation_rows": int(np.sum(validation_rows)),
        "features": matrix.shape[1],
        "seed_group_split": {
            "train_seeds": seeds - args.validation_seeds,
            "validation_seeds": args.validation_seeds,
            "both_seats_kept_together": True,
        },
        "selected_blend": {
            "win_probability_weight": best_win_weight,
            "rank_weight": best_rank_weight,
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "fixed_route_validation": {
            "route_id": int(route_ids[fixed_route]),
            "win_rate": float(np.mean(fixed_validation > 0)),
            "mean_margin": float(np.mean(fixed_validation)),
        },
        "oracle_validation": {
            "win_rate": float(np.mean(oracle_validation > 0)),
            "mean_margin": float(np.mean(oracle_validation)),
        },
        "blend_search": blends,
        "model": str(args.output_model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "train": train_metrics, "validation": validation_metrics, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
