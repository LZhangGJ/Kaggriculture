#!/usr/bin/env python3
"""Prepare and run the minimal continuation-contract multi-tail oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_block_mvp_routed_intent_v2 as routed_v2
from extract_route_genomes import _loads_replay
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_plan import normalized_action
from meta_agent.src.route_switch_features import (
    ANIMAL_COST,
    CROPS,
    HORIZONS,
    LAND_COST,
    SEED_COST,
    RouteSwitchHistory,
    route_switch_vector,
)
from meta_agent.src.teammate_expanded_routes import load_action_tapes


HORIZON = 719
SWITCH_STEP = 216
ANCHORS = tuple(range(216, 697, 24))
RUN_CHECKPOINTS = (216, 240, 264)
FEATURE_DIM = 147
PLAN_START = 129
MARKET_START, MARKET_STOP = 86, 104
LAYOUT_SIZE = 100
QUADRANTS = ("NW", "NE", "SW", "SE")
DEFAULT_POOL = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\live_replay_pool\route_genomes_live.jsonl"
)
DEFAULT_POOL_START_BYTE = 506_333_706
DEFAULT_POOL_END_BYTE = 510_184_144
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\block_mvp_continuation_multitail_v1"
)

LAYOUT_CODE = {
    "EMPTY": 0, "LOCKED": 1, "WEED": 2,
    "WHEAT": 3, "CARROT": 4, "TOMATO": 5,
    "STRAWBERRY": 6, "MELON": 7,
    "SOIL": 8, "COOP": 9, "PASTURE": 10,
    "GOOSE": 11, "COW": 12, "SHEEP": 13,
}
STATE_DISTANCE_FEATURES = np.asarray(
    [*range(MARKET_START), *range(MARKET_STOP, PLAN_START)], dtype=np.int64
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _actions(replay: Mapping[str, Any], player: int) -> list[dict[str, Any]]:
    steps = list(replay.get("steps", []) or [])
    if len(steps) < HORIZON + 1:
        raise ValueError(f"incomplete replay: {len(steps)} steps")
    return [
        normalized_action(steps[step + 1][player].get("action") or {})
        for step in range(HORIZON)
    ]


def fixed_commitments(
    observation: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Future capital using only fixed-cost seed/animal/land/hire orders."""

    step = int(observation.get("step", 0) or 0)
    player = int(observation.get("player", 0) or 0)
    farms = list(observation.get("farms", []) or [])
    farm = farms[player] if player < len(farms) else {}
    money = float(farm.get("money", 0) or 0)
    values: list[float] = []
    for horizon in HORIZONS:
        expense = requirement = cumulative = 0.0
        hires = int(farm.get("hires_today", 0) or 0)
        land = len(farm.get("unlocked_quadrants", []) or [])
        hires_planned = lands_planned = animals_planned = 0
        previous_day = step // 24
        for future in range(step, min(len(actions), step + horizon)):
            day = future // 24
            if day != previous_day:
                hires = 0
                previous_day = day
            for raw in list((actions[future] or {}).get("market", []) or []):
                order = list(raw or [])
                if not order:
                    continue
                operation = str(order[0])
                item = str(order[1]) if len(order) >= 2 else ""
                quantity = max(1, int(order[2] or 0)) if len(order) >= 3 else 1
                spend = 0.0
                if operation == "HIRE":
                    left = right = 1
                    for _ in range(max(0, hires)):
                        left, right = right, left + right
                    spend = float(left)
                    hires += 1
                    hires_planned += 1
                elif operation == "BUY_LAND":
                    extra = max(0, land - 1)
                    if extra < len(LAND_COST):
                        spend = float(LAND_COST[extra])
                        land += 1
                        lands_planned += 1
                elif operation == "BUY_SEED" and item in SEED_COST:
                    spend = float(quantity * SEED_COST[item])
                elif operation == "BUY_ANIMAL" and item in ANIMAL_COST:
                    spend = float(quantity * ANIMAL_COST[item])
                    animals_planned += quantity
                expense += spend
                cumulative += spend
                requirement = max(requirement, cumulative)
        values.extend((
            expense, requirement, money - requirement,
            float(hires_planned), float(lands_planned), float(animals_planned),
        ))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (18,) or not np.isfinite(result).all():
        raise ValueError("invalid fixed commitment vector")
    return result


def own_layout(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    player = int(observation.get("player", 0) or 0)
    farms = list(observation.get("farms", []) or [])
    farm = farms[player] if player < len(farms) else {}
    result = np.zeros(LAYOUT_SIZE, dtype=np.uint8)
    rows = list(farm.get("tiles", []) or [])
    for y in range(10):
        row = list(rows[y] or []) if y < len(rows) else []
        for x in range(10):
            tile = row[x] if x < len(row) else "LOCKED"
            if tile is None:
                label = "EMPTY"
            elif isinstance(tile, str):
                label = tile
            else:
                animal = str(tile.get("animal") or "")
                crop = str(tile.get("crop") or "")
                label = animal or crop or str(tile.get("kind") or "EMPTY")
            result[y * 10 + x] = LAYOUT_CODE.get(label, LAYOUT_CODE["EMPTY"])
    unlocked = set(farm.get("unlocked_quadrants", []) or [])
    return result, np.asarray([name in unlocked for name in QUADRANTS], np.uint8)


def replay_contracts(
    replay: Mapping[str, Any], player: int,
    actions: Sequence[Mapping[str, Any]], anchors: Sequence[int] = ANCHORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = list(replay.get("steps", []) or [])
    wanted = set(map(int, anchors))
    history = RouteSwitchHistory()
    vectors: list[np.ndarray] = []
    layouts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for step in range(max(wanted) + 1):
        observation = dict(steps[step][player].get("observation") or {})
        observation["step"] = step
        observation["player"] = player
        history.update(observation)
        if step not in wanted:
            continue
        vector = route_switch_vector(observation, history, actions).copy()
        vector[PLAN_START:] = fixed_commitments(observation, actions)
        layout, mask = own_layout(observation)
        vectors.append(vector)
        layouts.append(layout)
        masks.append(mask)
    if len(vectors) != len(anchors):
        raise ValueError("missing replay contract anchor")
    return np.stack(vectors), np.stack(layouts), np.stack(masks)


def _pool_segment(path: Path, start: int, stop: int) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if not (0 <= start < stop <= size):
        raise ValueError(f"invalid pool byte range {start}:{stop} for {size} bytes")
    with path.open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                raise ValueError("pool-start-byte is not a JSONL boundary")
        handle.seek(start)
        payload = handle.read(stop - start)
    if not payload.endswith(b"\n"):
        raise ValueError("pool-end-byte is not a JSONL boundary")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _pool_segment_sha256(path: Path, start: int, stop: int) -> str:
    with path.open("rb") as handle:
        handle.seek(start)
        return hashlib.sha256(handle.read(stop - start)).hexdigest()


def _replay_path(record: Mapping[str, Any]) -> Path:
    source = dict(record.get("source") or {})
    references = list(source.get("replay_sources", []) or [])
    for reference in references:
        path = Path(str(reference.get("absolute_path") or ""))
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no live replay path for {source.get('source_id')}")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    pool = args.live_pool.resolve()
    records = _pool_segment(pool, args.pool_start_byte, args.pool_end_byte)
    by_episode: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        source = dict(record.get("source") or {})
        episode = int(source["episode_id"])
        player = int(source["player_index"])
        if player in by_episode[episode]:
            raise ValueError(f"duplicate frozen source {episode}:{player}")
        by_episode[episode][player] = record
    complete = {episode: sides for episode, sides in by_episode.items() if set(sides) == {0, 1}}
    if len(records) != 2 * args.expected_episodes or len(complete) != args.expected_episodes:
        raise ValueError(
            f"frozen segment is {len(records)} rows/{len(complete)} complete episodes; "
            f"expected {2 * args.expected_episodes}/{args.expected_episodes}"
        )

    route_actions: dict[str, list[dict[str, Any]]] = {}
    route_rows: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    contracts: list[np.ndarray] = []
    layouts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for episode in sorted(complete):
        # One episode is loaded at most once per distinct replica, then released.
        replay_cache: dict[Path, Mapping[str, Any]] = {}
        for player in (0, 1):
            record = complete[episode][player]
            source = dict(record.get("source") or {})
            replay_path = _replay_path(record)
            if replay_path not in replay_cache:
                replay_cache[replay_path] = _loads_replay(str(replay_path))
            replay = replay_cache[replay_path]
            rewards = list(replay.get("rewards", []) or [])
            if len(rewards) < 2 or not all(math.isfinite(float(value)) for value in rewards):
                raise ValueError(f"non-finite replay rewards: {replay_path}")
            tape = _actions(replay, player)
            tail_sha = _sha256(tape[SWITCH_STEP:])
            route_id = f"MTV1_{tail_sha}"
            if route_id not in route_actions:
                route_actions[route_id] = tape
                route_rows[route_id] = {
                    "route_id": route_id, "tail_sha256_step216": tail_sha,
                    "representative_source_id": str(source["source_id"]),
                    "provenance_ids": [],
                }
            provenance_id = str(source["source_id"])
            route_rows[route_id]["provenance_ids"].append(provenance_id)
            vector, layout, mask = replay_contracts(replay, player, tape, args.anchors)
            contracts.append(vector)
            layouts.append(layout)
            masks.append(mask)
            provenance.append({
                "provenance_id": provenance_id,
                "route_id": route_id,
                "episode_id": episode,
                "player_index": player,
                "team_name": str(source.get("team_name", "")),
                "opponent_team_name": str(source.get("opponent_team_name", "")),
                "final_reward": float(source.get("final_reward", rewards[player]) or 0),
                "opponent_reward": float(source.get("opponent_reward", rewards[1-player]) or 0),
                "seed": int((replay.get("info") or {}).get("seed") or 0),
                "replay_path": str(replay_path),
                "replay_bytes": replay_path.stat().st_size,
            })

    output = args.output_root.resolve() / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    action_path = output / "tail_actions.json.zlib"
    action_path.write_bytes(zlib.compress(_canonical_bytes(route_actions), level=9))
    contract_path = output / "contracts.npz"
    np.savez_compressed(
        contract_path,
        contracts=np.stack(contracts).astype(np.float32),
        layouts=np.stack(layouts).astype(np.uint8),
        unlocked_masks=np.stack(masks).astype(np.uint8),
        provenance_ids=np.asarray([row["provenance_id"] for row in provenance]),
        route_ids=np.asarray([row["route_id"] for row in provenance]),
        anchors=np.asarray(args.anchors, dtype=np.int16),
    )
    manifest = {
        "schema": "block-mvp-continuation-multitail-v1",
        "status": "prepared",
        "pool": str(pool),
        "pool_byte_watermark_before": args.pool_start_byte,
        "pool_byte_watermark_after": args.pool_end_byte,
        "pool_delta_bytes": args.pool_end_byte - args.pool_start_byte,
        "pool_segment_sha256": _pool_segment_sha256(
            pool, args.pool_start_byte, args.pool_end_byte
        ),
        "pool_bytes_observed": pool.stat().st_size,
        "frozen_rows": len(records),
        "frozen_episodes": len(complete),
        "anchors": list(args.anchors),
        "logical_anchor_records": len(provenance) * len(args.anchors),
        "provenance_count": len(provenance),
        "unique_tail_count": len(route_actions),
        "tail_deduplicated_provenance": len(provenance) - len(route_actions),
        "actions_file": action_path.name,
        "actions_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
        "contracts_file": contract_path.name,
        "contracts_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "routes": [route_rows[key] for key in sorted(route_rows)],
        "provenance": provenance,
        "contract": {
            "feature_dim": FEATURE_DIM,
            "layout_dim": LAYOUT_SIZE,
            "unlocked_mask_dim": len(QUADRANTS),
            "distance_excluded_feature_indices": list(range(MARKET_START, MARKET_STOP)),
            "opponent_private_included": False,
            "future_capital": "fixed BUY_SEED/BUY_ANIMAL/BUY_LAND/HIRE only",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)
    return manifest


def contract_topk(
    query_feature: np.ndarray, query_layout: np.ndarray, query_mask: np.ndarray,
    source_features: np.ndarray, source_layouts: np.ndarray,
    source_masks: np.ndarray, source_route_ids: Sequence[str],
    scale: np.ndarray, k: int = 8,
) -> tuple[list[str], dict[str, float]]:
    state = np.mean(
        np.abs(source_features[:, STATE_DISTANCE_FEATURES]
               - query_feature[STATE_DISTANCE_FEATURES])
        / scale[STATE_DISTANCE_FEATURES],
        axis=1,
    )
    plan = np.mean(
        np.abs(source_features[:, PLAN_START:] - query_feature[PLAN_START:])
        / scale[PLAN_START:],
        axis=1,
    )
    spatial = (
        np.sum(source_layouts != query_layout, axis=1)
        + np.sum(source_masks != query_mask, axis=1)
    ) / float(source_layouts.shape[1] + source_masks.shape[1])
    total = (state + spatial + plan) / 3.0
    best: dict[str, float] = {}
    for route_id, distance in zip(source_route_ids, total):
        best[str(route_id)] = min(best.get(str(route_id), float("inf")), float(distance))
    ordered = sorted(best, key=lambda route_id: (best[route_id], route_id))
    return ordered[:min(k, len(ordered))], best


def deterministic_hash_topk(route_ids: Sequence[str], query_key: str, k: int = 8) -> list[str]:
    return sorted(
        map(str, route_ids),
        key=lambda route_id: (
            hashlib.sha256(f"{query_key}|{route_id}".encode()).digest(), route_id,
        ),
    )[:min(k, len(route_ids))]


def _best(
    route_ids: Sequence[str], outcome: Mapping[str, int], margin: Mapping[str, float],
    baseline_id: str, baseline_outcome: int, baseline_margin: float,
) -> tuple[str, int, float]:
    choice, rank = str(baseline_id), (int(baseline_outcome), float(baseline_margin))
    for route_id in route_ids:
        candidate = (int(outcome[route_id]), float(margin[route_id]))
        if candidate > rank:
            choice, rank = route_id, candidate
    return choice, rank[0], rank[1]


def _capture_layouts(
    bundle: NativeTeammateBundle, opening: int, opponent: int,
    seed: int, seat: int, checkpoints: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from fast_kaggriculture import Config, FastEnv, NativeAgentState

    env = FastEnv(Config(), int(seed))
    states = [NativeAgentState(), NativeAgentState()]
    wanted = set(map(int, checkpoints))
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    while not env.done and int(env.step_count) <= max(wanted):
        step = int(env.step_count)
        if step in wanted:
            observation = dict(env.observation(seat))
            observation["step"] = step
            observation["player"] = seat
            result[step] = own_layout(observation)
        if step == max(wanted):
            break
        indices = (opening, opponent) if seat == 0 else (opponent, opening)
        actions = [
            bundle.executor.action_at(env, player, indices[player], states[player])
            for player in (0, 1)
        ]
        env.step(actions)
    if set(result) != wanted:
        raise RuntimeError("native prefix did not reach every checkpoint")
    return result


def load_frozen_v1_openings(
    root: Path,
) -> tuple[str, list[dict[str, Any]], str, list[dict[str, Any]], Any, dict[str, Any]]:
    """Load immutable opening artifacts without re-signing the mutable live pool."""

    root = root.resolve()
    v1_args, manifest = routed_v2._v1_inputs(root)
    if (
        manifest.get("schema") != "block-mvp-96-216-candidates-v1"
        or int(manifest.get("start_step", -1)) != routed_v2.v1.START
        or int(manifest.get("end_step", -1)) != routed_v2.v1.STOP
    ):
        raise ValueError("invalid frozen v1 candidate manifest")
    action_path = root / "block_actions.json.zlib"
    digest = hashlib.sha256(action_path.read_bytes()).hexdigest()
    if digest != str(manifest.get("actions_sha256", "")):
        raise ValueError("frozen v1 action archive digest mismatch")
    actions = load_action_tapes(action_path)
    rows = [
        json.loads(line)
        for line in (root / "blocks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_rows = [row for row in rows if str(row.get("kind")) == "baseline"]
    c02_rows = [row for row in rows if str(row.get("block_id")) == "RB96_C02"]
    if len(baseline_rows) != 1 or len(c02_rows) != 1:
        raise ValueError("frozen v1 artifacts require one baseline and RB96_C02")
    baseline_id = str(baseline_rows[0]["block_id"])
    c02_id = str(c02_rows[0]["block_id"])
    for route_id in (baseline_id, c02_id):
        if route_id not in actions or len(actions[route_id]) != HORIZON:
            raise ValueError(f"frozen v1 route is not a 719-step tape: {route_id}")
    return baseline_id, actions[baseline_id], c02_id, actions[c02_id], v1_args, manifest


def load_prepared_bank(
    root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    root = root.resolve()
    manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "block-mvp-continuation-multitail-v1":
        raise ValueError("invalid continuation candidate manifest schema")
    action_path = root / str(manifest["actions_file"])
    contract_path = root / str(manifest["contracts_file"])
    if hashlib.sha256(action_path.read_bytes()).hexdigest() != str(
        manifest.get("actions_sha256", "")
    ):
        raise ValueError("continuation action archive digest mismatch")
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != str(
        manifest.get("contracts_sha256", "")
    ):
        raise ValueError("continuation contract archive digest mismatch")
    bank = load_action_tapes(action_path)
    if not bank or any(len(tape) != HORIZON for tape in bank.values()):
        raise ValueError("continuation bank requires complete 719-step tapes")
    with np.load(contract_path, allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    provenance = int(manifest.get("provenance_count", -1))
    expected = {
        "contracts": (provenance, len(ANCHORS), FEATURE_DIM),
        "layouts": (provenance, len(ANCHORS), LAYOUT_SIZE),
        "unlocked_masks": (provenance, len(ANCHORS), len(QUADRANTS)),
        "provenance_ids": (provenance,),
        "route_ids": (provenance,),
        "anchors": (len(ANCHORS),),
    }
    if set(arrays) != set(expected) or any(
        arrays[name].shape != shape for name, shape in expected.items()
    ):
        raise ValueError("continuation contract arrays have an invalid fixed shape")
    if (
        tuple(map(int, arrays["anchors"])) != ANCHORS
        or not np.isfinite(arrays["contracts"]).all()
        or set(map(str, arrays["route_ids"])) != set(bank)
    ):
        raise ValueError("continuation contract arrays disagree with the route bank")
    return manifest, bank, arrays


def _prefix_trace_checks(
    bundle: NativeTeammateBundle, opening_ids: Sequence[str],
    baseline_id: str, candidate_id: str, opponent_id: str,
    seed: int, checkpoints: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    opponent = bundle.index(opponent_id)
    baseline = bundle.index(baseline_id)
    candidate = bundle.index(candidate_id)
    for opening_id in opening_ids:
        opening = bundle.index(opening_id)
        for checkpoint in checkpoints:
            for seat in (0, 1):
                if seat == 0:
                    common = (opening, opponent, seed, checkpoint)
                    left = bundle.executor.play(*common, baseline, -1, -1, True)
                    right = bundle.executor.play(*common, candidate, -1, -1, True)
                else:
                    common = (opponent, opening, seed, -1, -1, checkpoint)
                    left = bundle.executor.play(*common, baseline, True)
                    right = bundle.executor.play(*common, candidate, True)
                left_hash = _sha256(list(left["trace"])[:checkpoint])
                right_hash = _sha256(list(right["trace"])[:checkpoint])
                left_turns = len(left["trace"])
                right_turns = len(right["trace"])
                rows.append({
                    "opening": opening_id, "checkpoint": int(checkpoint),
                    "seat": seat, "left_sha256": left_hash,
                    "right_sha256": right_hash,
                    "left_turns": left_turns, "right_turns": right_turns,
                    "exact": (
                        left_hash == right_hash
                        and left_turns == HORIZON and right_turns == HORIZON
                    ),
                })
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for opening in sorted({str(row["opening"]) for row in rows}):
        result[opening] = {}
        for checkpoint in sorted({int(row["checkpoint"]) for row in rows}):
            part = [
                row for row in rows
                if row["opening"] == opening and int(row["checkpoint"]) == checkpoint
            ]
            if not part:
                continue
            baseline_losses = sum(int(row["baseline_outcome"]) == 0 for row in part)
            result[opening][str(checkpoint)] = {
                "queries": len(part),
                "baseline_raw_win_rate": sum(
                    int(row["baseline_outcome"]) == 2 for row in part
                ) / len(part),
                "baseline_mean_margin": float(np.mean([
                    float(row["baseline_margin"]) for row in part
                ])),
                "oracles": {
                    name: {
                        "raw_win_rate": sum(
                            int(row[name]["outcome"]) == 2 for row in part
                        ) / len(part),
                        "mean_margin": float(np.mean([
                            float(row[name]["margin"]) for row in part
                        ])),
                        "repaired_baseline_losses": sum(
                            int(row["baseline_outcome"]) == 0
                            and int(row[name]["outcome"]) > 0
                            for row in part
                        ),
                        "baseline_loss_repair_fraction": sum(
                            int(row["baseline_outcome"]) == 0
                            and int(row[name]["outcome"]) > 0
                            for row in part
                        ) / max(1, baseline_losses),
                    }
                    for name in ("full_bank", "contract_top8", "hash_top8")
                },
            }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = args.prepared_root.resolve()
    manifest, bank, arrays = load_prepared_bank(prepared)
    anchors = [int(value) for value in arrays["anchors"]]
    anchor_index = {value: index for index, value in enumerate(anchors)}
    if any(value not in anchor_index for value in args.checkpoints):
        raise ValueError("run checkpoint is absent from prepared contracts")

    (
        baseline_id, baseline_tape, c02_id, c02_tape, v1_args, v1_manifest,
    ) = load_frozen_v1_openings(args.v1_root)
    splits, lineages = routed_v2.v1._opponent_splits(
        v1_args.base_metadata, v1_args.random_seed
    )
    opponents = list(splits[args.split])[:args.opponents]
    seeds = [args.seed_base + index for index in range(args.seeds)]
    additional = {baseline_id: baseline_tape, c02_id: c02_tape, **bank}
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*opponents, *additional)),
    )
    opening_ids = [baseline_id, c02_id]
    route_ids = sorted(bank)
    target_ids = [baseline_id, *route_ids]
    native = bundle.executor.switch_search(
        [bundle.index(value) for value in opening_ids],
        [bundle.index(value) for value in target_ids],
        list(args.checkpoints), seeds,
        [bundle.index(value) for value in opponents],
    )
    outcomes = np.asarray(native["outcome"], np.uint8)
    margins = np.asarray(native["margin"], np.float32)
    states = np.asarray(native["states"], np.float32)
    all_finite = bool(np.isfinite(margins).all() and np.isfinite(states).all())
    prefix_checks = _prefix_trace_checks(
        bundle, opening_ids, baseline_id, route_ids[0], opponents[0],
        seeds[0], args.checkpoints,
    )
    prefix_exact = all(bool(row["exact"]) for row in prefix_checks)

    source_features = np.asarray(arrays["contracts"], np.float32)
    source_layouts = np.asarray(arrays["layouts"], np.uint8)
    source_masks = np.asarray(arrays["unlocked_masks"], np.uint8)
    source_route_ids = [str(value) for value in arrays["route_ids"]]
    action_by_route = {str(key): value for key, value in bank.items()}
    rows: list[dict[str, Any]] = []
    group_names = ("full_bank", "contract_top8", "hash_top8")
    repair = {name: 0 for name in group_names}
    wins = {name: 0 for name in group_names}
    scores = {name: 0.0 for name in group_names}
    margin_sum = {name: 0.0 for name in group_names}
    baseline_losses = baseline_wins = 0
    baseline_margin_sum = 0.0
    total_queries = 0
    for oi, opening_id in enumerate(opening_ids):
        opening_index = bundle.index(opening_id)
        for pi, opponent_id in enumerate(opponents):
            opponent_index = bundle.index(opponent_id)
            for si, seed in enumerate(seeds):
                for seat in (0, 1):
                    captured = _capture_layouts(
                        bundle, opening_index, opponent_index, seed, seat,
                        args.checkpoints,
                    )
                    for ci, checkpoint in enumerate(args.checkpoints):
                        total_queries += 1
                        baseline_outcome = int(outcomes[oi, ci, 0, pi, si, seat])
                        baseline_margin = float(margins[oi, ci, 0, pi, si, seat])
                        baseline_losses += int(baseline_outcome == 0)
                        baseline_wins += int(baseline_outcome == 2)
                        baseline_margin_sum += baseline_margin
                        candidate_outcome = {
                            route_id: int(outcomes[oi, ci, index + 1, pi, si, seat])
                            for index, route_id in enumerate(route_ids)
                        }
                        candidate_margin = {
                            route_id: float(margins[oi, ci, index + 1, pi, si, seat])
                            for index, route_id in enumerate(route_ids)
                        }
                        anchor = anchor_index[checkpoint]
                        scale = np.maximum(
                            np.std(source_features[:, anchor], axis=0), 1.0,
                        )
                        layout, unlocked = captured[checkpoint]
                        query_core = states[oi, ci, pi, si, seat].copy()
                        # Each route has different fixed future commitments; score separately.
                        route_distance: dict[str, float] = {}
                        for route_id in route_ids:
                            query = query_core.copy()
                            observation = {
                                "step": checkpoint, "player": seat,
                                "farms": [{"money": query_core[0]}, {"money": query_core[27]}],
                            }
                            if seat == 1:
                                observation["farms"] = list(reversed(observation["farms"]))
                            # fixed_commitments only needs own money/land/hires. Recover those
                            # from the own farm prefix in the 147-vector.
                            own_offset = 0
                            own = {
                                "money": float(query_core[own_offset]),
                                "hands": [None] * int(query_core[own_offset + 1]),
                                "unlocked_quadrants": list(QUADRANTS[:int(query_core[own_offset + 2])]),
                                "hires_today": int(query_core[own_offset + 3]),
                            }
                            observation["farms"][seat] = own
                            query[PLAN_START:] = fixed_commitments(
                                observation, action_by_route[route_id]
                            )
                            selected, distances = contract_topk(
                                query, layout, unlocked,
                                source_features[:, anchor], source_layouts[:, anchor],
                                source_masks[:, anchor], source_route_ids, scale,
                                k=len(route_ids),
                            )
                            del selected
                            route_distance[route_id] = distances[route_id]
                        contract_ids = sorted(
                            route_ids, key=lambda value: (route_distance[value], value)
                        )[:min(args.topk, len(route_ids))]
                        query_key = (
                            f"{opening_id}|{checkpoint}|{opponent_id}|{seed}|{seat}"
                        )
                        hash_ids = deterministic_hash_topk(route_ids, query_key, args.topk)
                        choices = {
                            "full_bank": route_ids,
                            "contract_top8": contract_ids,
                            "hash_top8": hash_ids,
                        }
                        result_row = {
                            "opening": opening_id, "checkpoint": checkpoint,
                            "opponent": opponent_id, "seed": seed, "seat": seat,
                            "baseline_outcome": baseline_outcome,
                            "baseline_margin": baseline_margin,
                            "contract_top8_candidates": contract_ids,
                            "hash_top8_candidates": hash_ids,
                        }
                        for name, candidates in choices.items():
                            choice, outcome, margin = _best(
                                candidates, candidate_outcome, candidate_margin,
                                opening_id, baseline_outcome, baseline_margin,
                            )
                            wins[name] += int(outcome == 2)
                            scores[name] += .5 * outcome
                            margin_sum[name] += margin
                            repair[name] += int(baseline_outcome == 0 and outcome > 0)
                            result_row[name] = {
                                "choice": choice, "outcome": outcome, "margin": margin,
                            }
                        rows.append(result_row)

    retention = repair["contract_top8"] / max(1, repair["full_bank"])
    full_non_degrading = all(
        (int(row["full_bank"]["outcome"]), float(row["full_bank"]["margin"]))
        >= (int(row["baseline_outcome"]), float(row["baseline_margin"]))
        for row in rows
    )
    b0_rows = [row for row in rows if row["opening"] == baseline_id]
    b0_queries = len(b0_rows)
    b0_baseline_wins = sum(
        int(row["baseline_outcome"]) == 2 for row in b0_rows
    )
    b0_baseline_losses = sum(
        int(row["baseline_outcome"]) == 0 for row in b0_rows
    )
    b0_full_wins = sum(
        int(row["full_bank"]["outcome"]) == 2 for row in b0_rows
    )
    b0_repairs = sum(
        int(row["baseline_outcome"]) == 0
        and int(row["full_bank"]["outcome"]) > 0
        for row in b0_rows
    )
    b0_raw_win_gain_pp = 100.0 * (
        b0_full_wins - b0_baseline_wins
    ) / max(1, b0_queries)
    b0_mean_margin_delta = float(np.mean([
        float(row["full_bank"]["margin"]) - float(row["baseline_margin"])
        for row in b0_rows
    ])) if b0_rows else 0.0
    b0_feasibility = {
        "queries": b0_queries,
        "baseline_losses": b0_baseline_losses,
        "repaired_baseline_losses": b0_repairs,
        "raw_win_gain_vs_baseline_pp": b0_raw_win_gain_pp,
        "mean_margin_delta_vs_baseline": b0_mean_margin_delta,
        "passed": (
            (b0_repairs >= 2 or b0_raw_win_gain_pp >= 6.25)
            and b0_mean_margin_delta > 0.0
        ),
    }
    nonbaseline_full_choices = [
        str(row["full_bank"]["choice"]) for row in rows
        if str(row["full_bank"]["choice"]) != str(row["opening"])
    ]
    retrieval = {
        "full_bank_nonbaseline_choice_count": len(nonbaseline_full_choices),
        "contract_top8_exact_choice_recall": sum(
            str(row["full_bank"]["choice"]) in row["contract_top8_candidates"]
            for row in rows
            if str(row["full_bank"]["choice"]) != str(row["opening"])
        ) / max(1, len(nonbaseline_full_choices)),
        "hash_top8_exact_choice_recall": sum(
            str(row["full_bank"]["choice"]) in row["hash_top8_candidates"]
            for row in rows
            if str(row["full_bank"]["choice"]) != str(row["opening"])
        ) / max(1, len(nonbaseline_full_choices)),
        "gating": False,
    }
    gates = {
        "G0_all_native_outputs_finite": all_finite,
        "G1_sampled_pre_switch_prefix_exact": prefix_exact,
        "G2_full_bank_oracle_includes_non_degrading_B0_fallback": full_non_degrading,
        "G3_B0_full_bank_repairs_2_or_gains_6_25pp_and_improves_margin": (
            b0_feasibility["passed"]
        ),
        "G4_contract_top8_retains_80pct_full_repairs": (
            repair["full_bank"] > 0 and retention >= .80
        ),
        "G5_contract_top8_within_6_25pp_full_and_not_below_hash8": (
            100.0 * (wins["full_bank"] - wins["contract_top8"])
            / max(1, total_queries) <= 6.25
            and repair["contract_top8"] >= repair["hash_top8"]
            and wins["contract_top8"] >= wins["hash_top8"]
            and margin_sum["contract_top8"] >= margin_sum["hash_top8"]
            and (
                wins["contract_top8"] > wins["hash_top8"]
                or repair["contract_top8"] > repair["hash_top8"]
                or margin_sum["contract_top8"] > margin_sum["hash_top8"]
            )
        ),
    }
    summary = {
        "schema": "block-mvp-continuation-multitail-v1",
        "status": (
            "inconclusive_saturated_baseline" if b0_baseline_losses == 0
            else "passed_minimal_feasibility" if all(gates.values())
            else "falsified_minimal_feasibility"
        ),
        "engine": "NativeTeammateExecutor.switch_search",
        "single_independent_switch": True,
        "openings": opening_ids,
        "evaluation_split": args.split,
        "checkpoints": list(args.checkpoints),
        "opponents": opponents,
        "opponent_lineages": lineages[args.split],
        "seeds": seeds,
        "states_per_opening_checkpoint": len(opponents) * len(seeds) * 2,
        "queries": total_queries,
        "unique_tails": len(route_ids),
        "v1_actions_sha256": str(v1_manifest["actions_sha256"]),
        "games": int(outcomes.size),
        "all_finite": all_finite,
        "prefix_trace_checks": prefix_checks,
        "baseline": {
            "raw_win_rate": baseline_wins / total_queries,
            "losses": baseline_losses,
            "mean_margin": baseline_margin_sum / total_queries,
        },
        "oracles": {
            name: {
                "raw_win_rate": wins[name] / total_queries,
                "score_rate": scores[name] / total_queries,
                "mean_margin": margin_sum[name] / total_queries,
                "repaired_baseline_losses": repair[name],
                "baseline_loss_repair_fraction": repair[name] / max(1, baseline_losses),
                "raw_win_gain_vs_baseline_pp": 100.0 * (
                    wins[name] - baseline_wins
                ) / total_queries,
            }
            for name in group_names
        },
        "contract_repair_retention_vs_full": retention,
        "B0_feasibility": b0_feasibility,
        "retrieval": retrieval,
        "stratified": _summarize_rows(rows),
        "gates": gates,
        "selector_trained": False,
        "scope": [
            "Contract distance is frozen equal-weight retrieval; no weights or selector were trained.",
            f"{args.split} is a pre-frozen feasibility diagnostic, not a holdout or deployment conclusion.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = args.output_root.resolve() / "run"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return summary


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--live-pool", type=Path, default=DEFAULT_POOL)
    prep.add_argument("--pool-start-byte", type=int, default=DEFAULT_POOL_START_BYTE)
    prep.add_argument("--pool-end-byte", type=int, default=DEFAULT_POOL_END_BYTE)
    prep.add_argument("--expected-episodes", type=int, default=64)
    prep.add_argument("--anchors", type=_int_csv, default=ANCHORS)
    prep.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    execute = commands.add_parser("run")
    execute.add_argument("--prepared-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    execute.add_argument("--v1-root", type=Path, default=routed_v2.DEFAULT_V1_ROOT)
    execute.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    execute.add_argument("--checkpoints", type=_int_csv, default=RUN_CHECKPOINTS)
    execute.add_argument("--split", choices=("validation", "championship"), default="validation")
    execute.add_argument("--opponents", type=int, default=4)
    execute.add_argument("--seeds", type=int, default=2)
    execute.add_argument("--seed-base", type=int, default=2026082900)
    execute.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.expected_episodes <= 0 or len(args.anchors) != 21 or tuple(args.anchors) != ANCHORS:
            parser.error("prepare requires 64-ish episodes and the 21 day anchors 216..696")
        prepare(args)
    else:
        if min(args.opponents, args.seeds) <= 0 or args.topk != 8:
            parser.error("opponents/seeds must be positive and this MVP requires topk=8")
        run(args)


if __name__ == "__main__":
    main()
