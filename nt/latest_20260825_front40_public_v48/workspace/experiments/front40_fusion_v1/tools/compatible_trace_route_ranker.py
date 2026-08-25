"""Causal feature assembly for compatible Replay-route ranking.

Only routes whose already-executed raw action prefix is byte-identical to the
active first-shop route are eligible.  At day six, non-baseline routes must
also have been authored under the actually observed second town shop.  The
terminal screen is used only as an offline label; runtime features are the
current public state, the controlled player's own private state, and frozen
route descriptors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train_trace_route_margin_ranker import (
    build_context_features,
    build_difference_features,
    planned_route_features,
)


ACTION_FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
)


@dataclass
class CandidateDataset:
    features: np.ndarray
    feature_names: list[str]
    margin: np.ndarray
    hard: np.ndarray
    invalid: np.ndarray
    resync: np.ndarray
    group_sizes: np.ndarray
    group_offsets: np.ndarray
    route_id: np.ndarray
    base_row: np.ndarray
    legacy_row: np.ndarray
    panel_index: np.ndarray
    seat: np.ndarray
    seed: np.ndarray
    first_shop: np.ndarray
    second_shop: np.ndarray


def route_hash(bank: np.lib.npyio.NpzFile, route: int, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for field in ACTION_FIELDS:
        digest.update(np.ascontiguousarray(bank[field][route, start:end]).tobytes())
    return digest.hexdigest()


def load_route_tree(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = np.asarray(payload["route_ids"], dtype=np.int32)
    if first.shape != (8,):
        raise ValueError("route tree must contain eight first-shop routes")
    if "second_route_ids_by_first_shop" in payload:
        second = np.asarray(payload["second_route_ids_by_first_shop"], dtype=np.int32)
    else:
        legacy = np.asarray(payload["second_route_ids"], dtype=np.int32)
        second = np.stack([legacy[int(first[index])] for index in range(8)], axis=0)
    if second.shape != (8, 8):
        raise ValueError("route tree second-shop map must have shape (8, 8)")
    return first, second


def _flatten_context(value: np.ndarray) -> np.ndarray:
    return value.reshape((value.shape[0] * value.shape[1], *value.shape[2:]))


def _validate_context_screen(
    context: np.lib.npyio.NpzFile,
    screen: np.lib.npyio.NpzFile,
) -> tuple[int, int, np.ndarray, dict[int, int]]:
    margin = np.asarray(screen["margin"])
    if margin.ndim != 3:
        raise ValueError("screen margin must have shape (seat, seed, route)")
    seats, seeds, routes = margin.shape
    if context["candidate_seat"].shape != (seats, seeds):
        raise ValueError("context and screen seat/seed shapes differ")
    if not np.array_equal(np.asarray(context["seeds"]), np.asarray(screen["seeds"])):
        raise ValueError("context and screen seeds differ")
    expected_seats = np.broadcast_to(np.arange(seats, dtype=np.int8)[:, None], (seats, seeds))
    if not np.array_equal(np.asarray(context["candidate_seat"]), expected_seats):
        raise ValueError("context candidate seat ordering differs")
    route_ids = np.asarray(
        screen["route_ids"] if "route_ids" in screen else np.arange(routes),
        dtype=np.int32,
    )
    if route_ids.shape != (routes,) or np.unique(route_ids).size != routes:
        raise ValueError("screen route IDs are malformed")
    return seats, seeds, route_ids, {
        int(route): index for index, route in enumerate(route_ids.tolist())
    }


def build_candidate_dataset(
    context_paths: list[Path],
    matrix_paths: list[Path],
    route_bank_path: Path,
    route_tree_path: Path,
    candidate_route_ids: np.ndarray,
    decision_step: int = 144,
    prefix_start: int = 72,
) -> CandidateDataset:
    if len(context_paths) != len(matrix_paths) or not context_paths:
        raise ValueError("context and matrix lists must be paired and non-empty")
    bank = np.load(route_bank_path, allow_pickle=False)
    bank_routes = int(bank["unit_op"].shape[0])
    candidate_route_ids = np.asarray(candidate_route_ids, dtype=np.int32)
    if (
        candidate_route_ids.ndim != 1
        or candidate_route_ids.size == 0
        or np.unique(candidate_route_ids).size != candidate_route_ids.size
        or np.any(candidate_route_ids < 0)
        or np.any(candidate_route_ids >= bank_routes)
    ):
        raise ValueError("candidate route IDs are invalid")
    first_map, second_map = load_route_tree(route_tree_path)
    if np.any(first_map < 0) or np.any(first_map >= bank_routes):
        raise ValueError("first-shop route exceeds route bank")
    hashes = {
        int(route): route_hash(bank, int(route), prefix_start, decision_step)
        for route in np.unique(np.concatenate((candidate_route_ids, first_map)))
    }
    source_shops = np.asarray(bank["source_shop_sequence"], dtype=np.int16)
    route_features_full, route_names = planned_route_features(bank, decision_step)

    feature_blocks: list[np.ndarray] = []
    margin_blocks: list[np.ndarray] = []
    hard_blocks: list[np.ndarray] = []
    invalid_blocks: list[np.ndarray] = []
    resync_blocks: list[np.ndarray] = []
    route_blocks: list[np.ndarray] = []
    group_sizes: list[int] = []
    base_rows: list[int] = []
    legacy_rows: list[int] = []
    panel_indices: list[int] = []
    seats_out: list[int] = []
    seeds_out: list[int] = []
    first_out: list[int] = []
    second_out: list[int] = []
    feature_names: list[str] | None = None
    row_offset = 0

    for panel_index, (context_path, matrix_path) in enumerate(
        zip(context_paths, matrix_paths, strict=True)
    ):
        context = np.load(context_path, allow_pickle=False)
        screen = np.load(matrix_path, allow_pickle=False)
        seats, seeds, screen_routes, screen_lookup = _validate_context_screen(context, screen)
        missing = [int(route) for route in candidate_route_ids if int(route) not in screen_lookup]
        if missing:
            raise ValueError(f"candidate routes absent from {matrix_path}: {missing}")
        context_features, context_names = build_context_features(context)
        difference_full, difference_names = build_difference_features(
            context, bank, decision_step
        )
        contexts = seats * seeds
        difference = difference_full.reshape(contexts, bank_routes, -1)
        names = context_names + route_names + difference_names
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("feature schema differs between panels")

        margin_all = np.asarray(screen["margin"], dtype=np.float32).reshape(
            contexts, screen_routes.size
        )
        hard_all = np.asarray(screen["hard"], dtype=np.int32).reshape(
            contexts, screen_routes.size
        )
        invalid_all = np.asarray(screen["invalid"], dtype=np.int32).reshape(
            contexts, screen_routes.size
        )
        resync_all = np.asarray(screen["resync"], dtype=np.int32).reshape(
            contexts, screen_routes.size
        )
        town = _flatten_context(np.asarray(context["town_shops"], dtype=np.int16))
        town_count = _flatten_context(np.asarray(context["town_count"], dtype=np.int16))
        context_seat = _flatten_context(np.asarray(context["candidate_seat"], dtype=np.int8))
        context_seed = np.tile(np.asarray(context["seeds"], dtype=np.int32), seats)

        for index in range(contexts):
            count = int(np.asarray(town_count[index]).reshape(-1)[0])
            if count < 2:
                raise RuntimeError("second shop is not publicly observable at decision step")
            first_shop = int(town[index, 0])
            second_shop = int(town[index, 1])
            if not (0 <= first_shop < 8 and 0 <= second_shop < 8):
                raise RuntimeError("observed shop ID is invalid")
            base = int(first_map[first_shop])
            eligible = [
                int(route)
                for route in candidate_route_ids
                if hashes[int(route)] == hashes[base]
                and int(source_shops[int(route), 1]) == second_shop
            ]
            if base not in eligible:
                eligible.append(base)
            eligible = sorted(set(eligible))
            legacy = int(second_map[first_shop, second_shop])
            if legacy not in eligible:
                raise RuntimeError(
                    f"legacy route {legacy} is not causally compatible with ({first_shop}, {second_shop})"
                )
            missing_eligible = [route for route in eligible if route not in screen_lookup]
            if missing_eligible:
                raise RuntimeError(
                    f"eligible routes absent from {matrix_path}: {missing_eligible}"
                )
            route_array = np.asarray(eligible, dtype=np.int32)
            columns = np.asarray([screen_lookup[route] for route in eligible], dtype=np.int32)
            block = np.concatenate(
                (
                    np.repeat(context_features[index : index + 1], route_array.size, axis=0),
                    route_features_full[route_array],
                    difference[index, route_array],
                ),
                axis=1,
            ).astype(np.float32)
            feature_blocks.append(block)
            margin_blocks.append(margin_all[index, columns])
            hard_blocks.append(hard_all[index, columns])
            invalid_blocks.append(invalid_all[index, columns])
            resync_blocks.append(resync_all[index, columns])
            route_blocks.append(route_array)
            group_sizes.append(int(route_array.size))
            base_rows.append(row_offset + eligible.index(base))
            legacy_rows.append(row_offset + eligible.index(legacy))
            panel_indices.append(panel_index)
            seats_out.append(int(np.asarray(context_seat[index]).reshape(-1)[0]))
            seeds_out.append(int(context_seed[index]))
            first_out.append(first_shop)
            second_out.append(second_shop)
            row_offset += int(route_array.size)

    assert feature_names is not None
    sizes = np.asarray(group_sizes, dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(sizes, dtype=np.int64)))
    return CandidateDataset(
        features=np.concatenate(feature_blocks, axis=0),
        feature_names=feature_names,
        margin=np.concatenate(margin_blocks).astype(np.float32),
        hard=np.concatenate(hard_blocks).astype(np.int32),
        invalid=np.concatenate(invalid_blocks).astype(np.int32),
        resync=np.concatenate(resync_blocks).astype(np.int32),
        group_sizes=sizes,
        group_offsets=offsets,
        route_id=np.concatenate(route_blocks).astype(np.int32),
        base_row=np.asarray(base_rows, dtype=np.int64),
        legacy_row=np.asarray(legacy_rows, dtype=np.int64),
        panel_index=np.asarray(panel_indices, dtype=np.int16),
        seat=np.asarray(seats_out, dtype=np.int8),
        seed=np.asarray(seeds_out, dtype=np.int32),
        first_shop=np.asarray(first_out, dtype=np.int8),
        second_shop=np.asarray(second_out, dtype=np.int8),
    )


def group_argmax_rows(score: np.ndarray, dataset: CandidateDataset) -> np.ndarray:
    selected = np.empty(dataset.group_sizes.size, dtype=np.int64)
    for group, (start, end) in enumerate(
        zip(dataset.group_offsets[:-1], dataset.group_offsets[1:], strict=True)
    ):
        selected[group] = int(start + np.argmax(score[start:end]))
    return selected


def apply_conservative_switch(
    score: np.ndarray,
    dataset: CandidateDataset,
    threshold: float,
) -> np.ndarray:
    best = group_argmax_rows(score, dataset)
    legacy = dataset.legacy_row
    gain = score[best] - score[legacy]
    return np.where(gain > threshold, best, legacy).astype(np.int64)


def selection_metrics(dataset: CandidateDataset, rows: np.ndarray) -> dict[str, object]:
    margin = dataset.margin[rows]
    hard = dataset.hard[rows]
    invalid = dataset.invalid[rows]
    resync = dataset.resync[rows]
    result: dict[str, object] = {
        "games": int(rows.size),
        "wins": int(np.sum(margin > 0)),
        "win_rate": float(np.mean(margin > 0)),
        "mean_margin": float(np.mean(margin)),
        "median_margin": float(np.median(margin)),
        "hard_total": int(np.sum(hard)),
        "invalid_total": int(np.sum(invalid)),
        "resync_total": int(np.sum(resync)),
        "unique_routes": int(np.unique(dataset.route_id[rows]).size),
    }
    by_panel = []
    for panel in np.unique(dataset.panel_index):
        mask = dataset.panel_index == panel
        values = margin[mask]
        by_panel.append(
            {
                "panel_index": int(panel),
                "games": int(values.size),
                "wins": int(np.sum(values > 0)),
                "win_rate": float(np.mean(values > 0)),
                "mean_margin": float(np.mean(values)),
            }
        )
    result["by_panel"] = by_panel
    by_seat = []
    for seat in np.unique(dataset.seat):
        mask = dataset.seat == seat
        values = margin[mask]
        by_seat.append(
            {
                "seat": int(seat),
                "games": int(values.size),
                "wins": int(np.sum(values > 0)),
                "win_rate": float(np.mean(values > 0)),
                "mean_margin": float(np.mean(values)),
            }
        )
    result["by_seat"] = by_seat
    return result


def oracle_rows(dataset: CandidateDataset) -> np.ndarray:
    return group_argmax_rows(dataset.margin, dataset)

