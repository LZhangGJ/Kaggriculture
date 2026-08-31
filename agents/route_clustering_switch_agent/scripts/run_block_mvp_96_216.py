#!/usr/bin/env python3
"""Run the falsifiable BLOCK-MVP-96-216-v1 experiment end to end."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_macro_route_library import _snapshot
from evolve_native_route_library import _apply_evolution_genes, _donor_atoms
from extract_route_genomes import _loads_replay
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_genome import MACRO_KEYS, extract_route_genome
from meta_agent.src.route_plan import normalized_action
from meta_agent.src.route_switch_features import TRADE_ITEMS
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _candidate_genes


START = 96
STOP = 216
HORIZON = 719
BIN_WIDTH = 24
BLOCK_DIM = 102
STATE_DIM = 129
COMPAT_DIM = 12
PAIR_DIM = STATE_DIM + BLOCK_DIM + COMPAT_DIM
DESCRIPTOR_SCHEMA = "plan-only-action-timing-v2"
ROLE_NAMES = ("crop", "animal", "builder", "logistics", "recovery_other")
ENTRY_NAMES = (
    "wheat", "carrot", "tomato", "strawberry", "melon",
    "goose", "cow", "sheep", "coop", "pasture", "land", "hands",
)
EVENT_KEYS = (
    "BUILD_COOP", "BUILD_PASTURE", "GOOSE", "COW", "SHEEP", "BUY_LAND", "HIRE",
)


def block_feature_names() -> list[str]:
    names = [
        f"bin_{index}_{key.lower()}"
        for index in range((STOP - START) // BIN_WIDTH)
        for key in MACRO_KEYS
    ]
    names.extend(f"planned_{name}" for name in ENTRY_NAMES)
    names.extend((
        "planned_buy_units", "planned_sell_units",
        "planned_buy_mean_step", "planned_sell_mean_step",
        "planned_buy_first_step", "planned_sell_first_step",
    ))
    names.extend(f"worker_{name}_share" for name in ROLE_NAMES)
    for item in TRADE_ITEMS:
        names.extend((f"market_{item.lower()}_buy", f"market_{item.lower()}_sell"))
    names.append("duration")
    if len(names) != BLOCK_DIM:
        raise AssertionError(f"block schema has {len(names)} features")
    return names


@dataclass
class Block:
    block_id: str
    kind: str
    parent_id: str
    cluster_id: str
    source: dict[str, Any]
    entry: np.ndarray
    vector: np.ndarray
    event_profile: np.ndarray
    tape: list[dict[str, Any]]
    genes: list[dict[str, Any]]

    def row(self) -> dict[str, Any]:
        return {
            "schema": "route-block-96-216-v1",
            "block_id": self.block_id,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "cluster_id": self.cluster_id,
            "start_step": START,
            "end_step": STOP,
            "entry": dict(zip(ENTRY_NAMES, self.entry.astype(float).tolist())),
            "vector": self.vector.astype(float).tolist(),
            "event_profile": self.event_profile.astype(float).tolist(),
            "genes": self.genes,
            "source": self.source,
        }


def _farm(steps: Sequence[Any], recorded_step: int, player: int) -> dict[str, Any]:
    observation = steps[recorded_step][player].get("observation") or {}
    farms = list(observation.get("farms", ()) or ())
    return dict(farms[player]) if player < len(farms) else {}


def _tape(steps: Sequence[Any], player: int) -> list[dict[str, Any]]:
    if len(steps) < HORIZON + 1:
        raise ValueError(f"incomplete replay with {len(steps)} recorded steps")
    return [
        normalized_action(steps[step + 1][player].get("action") or {})
        for step in range(HORIZON)
    ]


def _unit_role(order: Sequence[Any]) -> int:
    op = str(order[0]) if order else "PASS"
    if op in {"DIG", "PLANT", "WATER", "HARVEST"}:
        return 0
    if op in {"PLACE", "FEED", "TEND", "CARE"}:
        return 1
    if op in {"BUILD_COOP", "BUILD_PASTURE"}:
        return 2
    if op in {"PICKUP", "DROP"}:
        return 3
    return 4


def _action_parts(
    tape: Sequence[Mapping[str, Any]], start: int = START, stop: int = STOP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    macro = np.zeros(((stop - start) // BIN_WIDTH, len(MACRO_KEYS)), np.float32)
    market = np.zeros((len(TRADE_ITEMS), 2), np.float32)
    roles = np.zeros(len(ROLE_NAMES), np.float32)
    first = np.full(len(EVENT_KEYS), stop - start, np.float32)
    event_count = np.zeros(len(EVENT_KEYS), np.float32)
    macro_index = {name: index for index, name in enumerate(MACRO_KEYS)}
    event_index = {name: index for index, name in enumerate(EVENT_KEYS)}
    trade_index = {name: index for index, name in enumerate(TRADE_ITEMS)}
    for step in range(start, stop):
        action = tape[step]
        bin_index = min(len(macro) - 1, (step - start) // BIN_WIDTH)
        unit_orders = [action.get("farmer"), *(action.get("hands", ()) or ())]
        for raw in unit_orders:
            order = list(raw or ["PASS"])
            roles[_unit_role(order)] += 1
            op = str(order[0]) if order else ""
            item = str(order[1]) if len(order) >= 2 else ""
            key = item if op == "PLANT" and item in macro_index else ""
            if op in {"BUILD_COOP", "BUILD_PASTURE"}:
                key = op
            elif op == "PLACE" and item in {"GOOSE", "COW", "SHEEP"}:
                key = item
            if key:
                macro[bin_index, macro_index[key]] += 1
                if key in event_index:
                    position = event_index[key]
                    first[position] = min(first[position], step - start)
                    event_count[position] += 1
        for raw in action.get("market", ()) or ():
            order = list(raw or ())
            if not order:
                continue
            op = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            try:
                quantity = max(0, int(order[2])) if len(order) >= 3 else 1
            except (TypeError, ValueError):
                quantity = 0
            if op in {"BUY_LAND", "HIRE"}:
                macro[bin_index, macro_index[op]] += 1
                position = event_index[op]
                first[position] = min(first[position], step - start)
                event_count[position] += 1
            if item in trade_index:
                if op in {"BUY_SEED", "BUY_PRODUCT"}:
                    market[trade_index[item], 0] += quantity
                elif op == "SELL":
                    market[trade_index[item], 1] += quantity
    roles /= max(1.0, float(roles.sum()))
    return macro.reshape(-1), roles, market.reshape(-1), np.concatenate((first, event_count))


def _total_facilities(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, np.float32).copy()
    result[8] += result[5]
    result[9] += result[6] + result[7]
    return result


def _plan_vector(
    tape: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    macro, roles, market, events = _action_parts(tape)
    totals = macro.reshape(-1, len(MACRO_KEYS)).sum(axis=0)
    planned = np.asarray((
        *totals[:5], *totals[7:10], totals[5], totals[6], totals[10], totals[11],
    ), np.float32)
    trades = market.reshape(-1, 2)
    timing = {name: [] for name in ("buy_step", "buy_weight", "sell_step", "sell_weight")}
    for step in range(START, STOP):
        for raw in tape[step].get("market", ()) or ():
            order = list(raw or ())
            if not order:
                continue
            op = str(order[0])
            try:
                quantity = max(0, int(order[2])) if len(order) >= 3 else 1
            except (TypeError, ValueError):
                quantity = 0
            if op in {
                "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND", "HIRE",
            }:
                timing["buy_step"].append(step - START)
                timing["buy_weight"].append(max(1, quantity))
            elif op == "SELL":
                timing["sell_step"].append(step - START)
                timing["sell_weight"].append(max(1, quantity))
    duration = STOP - START
    def time_summary(kind: str) -> tuple[float, float]:
        steps = np.asarray(timing[f"{kind}_step"], np.float32)
        weights = np.asarray(timing[f"{kind}_weight"], np.float32)
        if not len(steps):
            return float(duration), float(duration)
        return float(np.average(steps, weights=weights)), float(np.min(steps))
    buy_mean, buy_first = time_summary("buy")
    sell_mean, sell_first = time_summary("sell")
    summary = np.asarray((
        trades[:, 0].sum(), trades[:, 1].sum(),
        buy_mean, sell_mean, buy_first, sell_first,
    ), np.float32)
    vector = np.concatenate((
        macro, planned, summary, roles, market,
        np.asarray([STOP - START], np.float32),
    ))
    if vector.shape != (BLOCK_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid plan-only block vector {vector.shape}")
    return vector, events


def _block_from_record(record: Mapping[str, Any], block_id: str) -> Block:
    source = dict(record["source"])
    references = list(source.get("replay_sources", ()) or ())
    paths = [Path(str(value["absolute_path"])) for value in references]
    replay_path = next((path for path in paths if path.is_file()), paths[0])
    replay = _loads_replay(replay_path)
    steps = list(replay.get("steps", ()) or ())
    player = int(source["player_index"])
    tape = _tape(steps, player)
    entry, _ = _snapshot(_farm(steps, START, player))
    entry = _total_facilities(entry)
    window = dict(replay)
    window["steps"] = steps[START:STOP + 1]
    genome = extract_route_genome(
        window, player, provenance={"dataset": "block-mvp"},
        anchor_steps=(), phase_width=BIN_WIDTH, horizon=STOP - START,
    )
    macro = np.asarray([
        float(phase["counts"].get(key, 0))
        for phase in genome.phase_macro_counts for key in MACRO_KEYS
    ], np.float32)
    vector, events = _plan_vector(tape)
    if not np.array_equal(macro, vector[:60]):
        raise AssertionError("window macro extraction disagrees with tape extraction")
    money = np.asarray([
        float(_farm(steps, step, player).get("money", 0.0) or 0.0)
        for step in range(START, STOP + 1)
    ], np.float32)
    replay_stat = replay_path.stat()
    metadata = {
        "source_id": str(source["source_id"]),
        "episode_id": int(source["episode_id"]),
        "player_index": player,
        "team_name": str(source.get("team_name", "")),
        "opponent_team_name": str(source.get("opponent_team_name", "")),
        "result": str(source.get("result", "")),
        "final_reward": float(source.get("final_reward", 0.0) or 0.0),
        "opponent_reward": float(source.get("opponent_reward", 0.0) or 0.0),
        "minimum_cash": float(np.min(money)),
        "datasets": list(source.get("datasets", ()) or ()),
        "replay_path": str(replay_path.resolve()),
        "replay_mtime_ns": int(replay_stat.st_mtime_ns),
        "genome_id": str(record.get("genome_id", "")),
    }
    return Block(
        block_id=block_id, kind="replay", parent_id=block_id, cluster_id="",
        source=metadata, entry=entry.astype(np.float32), vector=vector,
        event_profile=events, tape=tape, genes=[],
    )


def _minimal_record(row: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(row.get("source", {}))
    return {
        "genome_id": str(row.get("genome_id", "")),
        "cash_profile": dict(row.get("cash_profile", {})),
        "source": source,
    }


def _file_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _input_signatures(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        name: _file_signature(Path(getattr(args, name)))
        for name in (
            "records", "source", "opening_actions", "opening_metadata",
            "base_actions", "base_metadata",
        )
    }


def _generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "opening": args.opening,
        "prefilter": args.prefilter,
        "carriers": args.carriers,
        "random_seed": args.random_seed,
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "feature_names": block_feature_names(),
    }


def _quality(record: Mapping[str, Any]) -> tuple[Any, ...]:
    source = record["source"]
    own = float(source.get("final_reward", 0.0) or 0.0)
    other = float(source.get("opponent_reward", 0.0) or 0.0)
    datasets = set(map(str, source.get("datasets", ()) or ()))
    return (
        "our_latest" in datasets,
        float(record.get("cash_profile", {}).get("minimum", -1.0) or -1.0) >= 0,
        own - other, own, int(source.get("episode_id", 0) or 0),
    )


def select_replay_records(path: Path, limit: int) -> list[dict[str, Any]]:
    import orjson

    best_by_genome: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = orjson.loads(line)
            source = row.get("source", {})
            if (
                str(source.get("result", "")) != "win"
                or int(source.get("replay_steps", 0) or 0) < HORIZON + 1
                or not source.get("replay_sources")
            ):
                continue
            genome_id = str(row.get("genome_id", ""))
            candidate = _minimal_record(row)
            if genome_id not in best_by_genome or _quality(candidate) > _quality(
                best_by_genome[genome_id]
            ):
                best_by_genome[genome_id] = candidate
    values = sorted(best_by_genome.values(), key=_quality, reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    team_counts: Counter[str] = Counter()
    best_by_team: dict[str, dict[str, Any]] = {}
    for row in values:
        team = str(row["source"].get("team_name", ""))
        best_by_team.setdefault(team, row)
    for row in sorted(best_by_team.values(), key=_quality, reverse=True)[: limit // 2]:
        selected.append(row)
        seen.add(str(row["genome_id"]))
        team_counts[str(row["source"].get("team_name", ""))] += 1
    for row in values:
        team = str(row["source"].get("team_name", ""))
        if len(selected) >= limit:
            break
        if str(row["genome_id"]) in seen or team_counts[team] >= 4:
            continue
        selected.append(row)
        seen.add(str(row["genome_id"]))
        team_counts[team] += 1
    return selected


def _scaled(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    scale = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.std(values, axis=0))
    return (values - median) / np.where(scale > 1e-6, scale, 1.0)


def _cluster_features(blocks: Sequence[Block]) -> np.ndarray:
    vectors = np.stack([block.vector for block in blocks]).astype(np.float64)
    events = np.stack([block.event_profile for block in blocks]).astype(np.float64)
    groups = (
        (_scaled(np.log1p(vectors[:, :60])), .30),
        (_scaled(np.sign(vectors[:, 60:72]) * np.log1p(np.abs(vectors[:, 60:72]))), .20),
        (_scaled(events), .20),
        (_scaled(np.sign(vectors[:, 72:78]) * np.log1p(np.abs(vectors[:, 72:78]))), .15),
        (_scaled(np.log1p(vectors[:, 83:101])), .10),
        (_scaled(vectors[:, 78:83]), .05),
    )
    return np.concatenate([
        value * math.sqrt(weight / max(1, value.shape[1]))
        for value, weight in groups
    ], axis=1).astype(np.float32)


def select_carriers(blocks: Sequence[Block], count: int) -> list[Block]:
    if len(blocks) < count:
        raise ValueError(f"only {len(blocks)} compatible blocks for {count} carriers")
    features = _cluster_features(blocks)
    quality = np.asarray([
        float(block.source["final_reward"] - block.source["opponent_reward"])
        + .01 * float(block.source["minimum_cash"])
        for block in blocks
    ])
    quality = (quality - np.min(quality)) / max(1e-9, float(np.ptp(quality)))
    chosen = [int(np.argmax(quality))]
    distance = np.sum((features - features[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        score = distance + .05 * quality
        score[chosen] = -np.inf
        index = int(np.argmax(score))
        chosen.append(index)
        distance = np.minimum(distance, np.sum((features - features[index]) ** 2, axis=1))
    centers = features[chosen]
    assignment = np.argmin(
        np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
    )
    result = []
    for cluster, index in enumerate(chosen):
        block = blocks[index]
        block.block_id = f"RB96_C{cluster:02d}"
        block.parent_id = block.block_id
        block.cluster_id = f"C{cluster:02d}"
        block.source["cluster_support"] = int(np.sum(assignment == cluster))
        result.append(block)
    return result


def hybrid_tape(
    base: Sequence[Mapping[str, Any]], donor: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(base) != HORIZON or len(donor) != HORIZON:
        raise ValueError("BLOCK-MVP requires two 719-step tapes")
    result = copy.deepcopy(list(base))
    result[START:STOP] = copy.deepcopy(list(donor[START:STOP]))
    if result[:START] != list(base[:START]) or result[STOP:] != list(base[STOP:]):
        raise AssertionError("block splice changed the frozen prefix or continuation")
    return result


def _shift_structural_market(
    tape: Sequence[Mapping[str, Any]], delta: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    result = copy.deepcopy(list(tape))
    for step in range(START, STOP):
        market = result[step].get("market", []) or []
        for index, raw in enumerate(market):
            order = list(raw or ())
            if not order or str(order[0]) not in {"HIRE", "BUY_LAND"}:
                continue
            target = min(STOP - 1, max(START, step + delta))
            moved = market.pop(index)
            candidates = sorted(
                range(START, STOP), key=lambda value: (abs(value - target), value)
            )
            destination = next(
                value for value in candidates
                if len(result[value].get("market", []) or []) < 10
            )
            result[destination].setdefault("market", []).append(moved)
            return result, {
                "operator": "shift_structural_market", "operation": str(order[0]),
                "source_step": step, "target_step": destination, "delta": delta,
            }
    return None


def _clamped_atom(gene: Mapping[str, Any]) -> dict[str, Any] | None:
    result = dict(gene)
    result["start"] = max(START, int(result.get("start", START)))
    result["stop"] = min(STOP, int(result.get("stop", STOP)))
    return result if result["start"] < result["stop"] else None


def _variant_vector(
    parent: Block, tape: Sequence[Mapping[str, Any]], parent_tape: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    del parent, parent_tape
    return _plan_vector(tape)


def _descriptor_key(block: Block) -> tuple[bytes, bytes]:
    return (
        np.asarray(block.entry, np.float32).tobytes(),
        np.asarray(block.vector, np.float32).tobytes(),
    )


def _block_tape_digest(block: Block) -> str:
    return hashlib.sha256(json.dumps(
        block.tape[START:STOP], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _block_quality(block: Block) -> tuple[float, float, int]:
    return (
        block.source["final_reward"] - block.source["opponent_reward"],
        block.source["minimum_cash"], block.source["episode_id"],
    )


def generate_variants(
    carriers: Sequence[Block], baseline: Block, seed: int,
) -> list[Block]:
    rng = random.Random(seed)
    base_tape = baseline.tape
    parent_tapes = {block.block_id: block.tape for block in carriers}
    descriptor_seen = {
        _descriptor_key(block) for block in (baseline, *carriers)
    }
    tape_seen = {
        _block_tape_digest(block) for block in (baseline, *carriers)
    }
    result: list[Block] = []
    for parent in carriers:
        pool: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        for genes in _candidate_genes(parent.tape):
            if len(genes) != 1:
                continue
            atom = _clamped_atom(genes[0])
            if atom is None:
                continue
            pool.append(([atom], _apply_genes(parent.tape, [atom])))
        donor_atoms = [
            atom for atom in _donor_atoms(
                parent.block_id, tuple(parent_tapes), parent_tapes
            )
            if int(atom["start"]) >= START and int(atom["stop"]) <= STOP
        ]
        for atom in donor_atoms:
            pool.append(([atom], _apply_evolution_genes(
                parent.tape, [atom], parent_tapes
            )))
        for delta in (-24, -12, 12, 24):
            shifted = _shift_structural_market(parent.tape, delta)
            if shifted is not None:
                tape, gene = shifted
                pool.append(([gene], tape))
        simple_atoms = [genes[0] for genes, _ in pool if len(genes) == 1]
        for _ in range(min(32, len(simple_atoms) * 2)):
            if len(simple_atoms) < 2:
                break
            genes = rng.sample(simple_atoms, 2)
            if any(str(gene["operator"]) == "shift_structural_market" for gene in genes):
                continue
            pool.append((genes, _apply_evolution_genes(
                parent.tape, genes, parent_tapes
            )))
        def operator(genes: Sequence[Mapping[str, Any]]) -> str:
            return str(genes[0].get("operator", "")) if len(genes) == 1 else "combo"

        categories = (
            lambda genes: operator(genes) == "scale_quantity"
            and int(genes[0]["numerator"]) < int(genes[0]["denominator"]),
            lambda genes: operator(genes) == "scale_quantity"
            and int(genes[0]["numerator"]) > int(genes[0]["denominator"]),
            lambda genes: operator(genes) == "shift_market"
            and int(genes[0]["delta"]) < 0,
            lambda genes: operator(genes) == "shift_market"
            and int(genes[0]["delta"]) > 0,
            lambda genes: operator(genes) in {"shift_structural_market", "reduce_hires"},
            lambda genes: operator(genes) == "replace_market_phase",
            lambda genes: operator(genes) == "replace_worker_phase",
        )
        ordered_pool = []
        used_rows: set[int] = set()
        for predicate in categories:
            matches = [
                index for index, (genes, _) in enumerate(pool)
                if index not in used_rows and predicate(genes)
            ]
            rng.shuffle(matches)
            if matches:
                used_rows.add(matches[0])
                ordered_pool.append(pool[matches[0]])
        remaining = [
            row for index, row in enumerate(pool) if index not in used_rows
        ]
        rng.shuffle(remaining)
        pool = [*ordered_pool, *remaining]
        accepted = 0
        for genes, tape in pool:
            tape = hybrid_tape(base_tape, tape)
            digest = hashlib.sha256(json.dumps(
                tape[START:STOP], ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if digest in tape_seen:
                continue
            vector, events = _variant_vector(parent, tape, parent.tape)
            descriptor = (
                np.asarray(parent.entry, np.float32).tobytes(),
                np.asarray(vector, np.float32).tobytes(),
            )
            if descriptor in descriptor_seen:
                continue
            tape_seen.add(digest)
            descriptor_seen.add(descriptor)
            block_id = f"GB96_{parent.cluster_id}_V{accepted:02d}"
            result.append(Block(
                block_id=block_id, kind="ga", parent_id=parent.block_id,
                cluster_id=parent.cluster_id, source=dict(parent.source),
                entry=parent.entry.copy(), vector=vector, event_profile=events,
                tape=tape, genes=genes,
            ))
            accepted += 1
            if accepted == 7:
                break
        if accepted != 7:
            raise ValueError(
                f"only {accepted} distinct variants for {parent.block_id}; expected 7"
            )
    return result


def _baseline_record(metadata: Mapping[str, Any], family: str) -> dict[str, Any]:
    row = next(value for value in metadata["opponent_routes"] if value["family"] == family)
    provenance = dict(row.get("provenance", {}))
    reward = float(provenance.get("historical_reward", 0.0) or 0.0)
    margin = float(provenance.get("historical_margin", 0.0) or 0.0)
    source_id = str(provenance["source_id"])
    episode, player = source_id.split(":")
    return {
        "genome_id": str(provenance.get("genome_id", "")),
        "cash_profile": {},
        "source": {
            "source_id": source_id,
            "episode_id": int(episode),
            "player_index": int(player),
            "team_name": str(row.get("team", "")),
            "opponent_team_name": "",
            "result": "win",
            "final_reward": reward,
            "opponent_reward": reward - margin,
            "replay_steps": HORIZON + 1,
            "datasets": list(provenance.get("datasets", ()) or ()),
            "replay_sources": [{"absolute_path": str(provenance["replay_path"])}],
        },
    }


def prepare_blocks(args: argparse.Namespace) -> tuple[Block, list[Block], list[Block]]:
    input_signatures = _input_signatures(args)
    opening_metadata = json.loads(args.opening_metadata.read_text(encoding="utf-8"))
    opening_row = next(
        row for row in opening_metadata["opponent_routes"]
        if row["family"] == args.opening
    )
    baseline = _block_from_record(
        _baseline_record(opening_metadata, args.opening), "B0_KEEP"
    )
    baseline.tape = load_action_tapes(args.opening_actions)[
        str(opening_row["route_id"])
    ]
    baseline.vector, baseline.event_profile = _plan_vector(baseline.tape)
    baseline.kind = "baseline"
    baseline.parent_id = baseline.block_id
    baseline.cluster_id = "BASE"
    candidates = select_replay_records(args.records, args.prefilter)
    eligible: list[Block] = []
    errors: list[dict[str, str]] = []
    for index, record in enumerate(candidates):
        try:
            block = _block_from_record(record, f"TMP{index:04d}")
        except Exception as exc:
            errors.append({
                "source_id": str(record["source"].get("source_id", "")),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if np.array_equal(block.entry[8:], baseline.entry[8:]):
            eligible.append(block)
    compatible = []
    seen_descriptors = {_descriptor_key(baseline)}
    seen_tapes = {_block_tape_digest(baseline)}
    for block in sorted(eligible, key=_block_quality, reverse=True):
        descriptor = _descriptor_key(block)
        tape_digest = _block_tape_digest(block)
        if descriptor in seen_descriptors or tape_digest in seen_tapes:
            continue
        seen_descriptors.add(descriptor)
        seen_tapes.add(tape_digest)
        compatible.append(block)
    carriers = select_carriers(compatible, args.carriers)
    for block in carriers:
        block.tape = hybrid_tape(baseline.tape, block.tape)
    variants = generate_variants(carriers, baseline, args.random_seed)
    all_blocks = [baseline, *carriers, *variants]
    if len({_descriptor_key(block) for block in all_blocks}) != len(all_blocks):
        raise AssertionError("candidate selector descriptors are not unique")
    if len({_block_tape_digest(block) for block in all_blocks}) != len(all_blocks):
        raise AssertionError("candidate block tapes are not unique")
    if input_signatures != _input_signatures(args):
        raise RuntimeError("BLOCK-MVP inputs changed during candidate preparation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    blocks = all_blocks
    with (args.output_root / "blocks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for block in blocks:
            handle.write(json.dumps(
                block.row(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")
    actions = {block.block_id: block.tape for block in blocks}
    packed = zlib.compress(json.dumps(
        actions, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8"), level=9)
    (args.output_root / "block_actions.json.zlib").write_bytes(packed)
    np.savez_compressed(
        args.output_root / "block_features.npz",
        block_ids=np.asarray([block.block_id for block in blocks]),
        kinds=np.asarray([block.kind for block in blocks]),
        parent_ids=np.asarray([block.parent_id for block in blocks]),
        cluster_ids=np.asarray([block.cluster_id for block in blocks]),
        vectors=np.stack([block.vector for block in blocks]).astype(np.float32),
        entries=np.stack([block.entry for block in blocks]).astype(np.float32),
        feature_names=np.asarray(block_feature_names()),
    )
    manifest = {
        "schema": "block-mvp-96-216-candidates-v1",
        "records": str(args.records.resolve()),
        "records_bytes": args.records.stat().st_size,
        "inputs": input_signatures,
        "generation_contract": _generation_contract(args),
        "opening": args.opening,
        "start_step": START,
        "end_step": STOP,
        "prefilter_records": len(candidates),
        "compatible_records": len(compatible),
        "parse_errors": errors,
        "carrier_count": len(carriers),
        "ga_variant_count": len(variants),
        "candidate_count_excluding_baseline": len(carriers) + len(variants),
        "block_dimension": BLOCK_DIM,
        "feature_names": block_feature_names(),
        "structural_entry_compatibility": list(ENTRY_NAMES[8:]),
        "descriptor_uses_source_market_or_cash_outcomes": False,
        "descriptor_is_plan_only": True,
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
        "blocks": [block.row() for block in blocks],
    }
    (args.output_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return baseline, carriers, variants


def _opponent_splits(
    path: Path, random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = {}
    for row in metadata["opponent_routes"]:
        if not bool(row.get("selected", True)):
            continue
        family = str(row["family"])
        lineage = str(row.get("team") or f"family:{family}")
        grouped.setdefault(lineage, []).append(family)
    order = sorted(grouped, key=lambda value: hashlib.sha256(
        f"{random_seed}:lineage:{value}".encode("utf-8")
    ).digest())
    targets = (
        ("evolution", 16), ("train", 20), ("validation", 10),
        ("championship", 20), ("ood", 10),
    )
    splits: dict[str, list[str]] = {}
    lineages: dict[str, list[str]] = {}
    for name, target in targets:
        families: list[str] = []
        used: list[str] = []
        while len(families) < target and order:
            lineage = order.pop(0)
            choices = sorted(
                dict.fromkeys(grouped[lineage]),
                key=lambda value: hashlib.sha256(
                    f"{random_seed}:{lineage}:{value}".encode("utf-8")
                ).digest(),
            )
            families.extend(choices[:min(2, target - len(families))])
            used.append(lineage)
        if len(families) != target:
            raise ValueError(f"only {len(families)} opponents for {name}; need {target}")
        splits[name] = families
        lineages[name] = used
    return splits, lineages


def evaluate_panel(
    args: argparse.Namespace,
    blocks: Sequence[Block],
    opponents: Sequence[str],
    seeds: Sequence[int],
    name: str,
    opening: Block | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    opening = opening or blocks[0]
    additional = {block.block_id: block.tape for block in blocks}
    additional[opening.block_id] = opening.tape
    bundle = NativeTeammateBundle(
        args.source, args.base_actions, args.base_metadata,
        additional_routes=additional, included_families=tuple(opponents),
    )
    result = bundle.executor.switch_search(
        [bundle.index(opening.block_id)],
        [bundle.index(block.block_id) for block in blocks],
        [START], list(map(int, seeds)),
        [bundle.index(opponent) for opponent in opponents],
    )
    outcome = np.asarray(result["outcome"])[0, 0].reshape(len(blocks), -1)
    margin = np.asarray(result["margin"])[0, 0].reshape(len(blocks), -1)
    states = np.asarray(result["states"])[0, 0].reshape(-1, 147)
    snapshot_opponents = np.repeat(
        np.asarray(opponents), len(seeds) * 2
    )
    snapshot_seeds = np.tile(np.repeat(np.asarray(seeds, np.int64), 2), len(opponents))
    snapshot_seats = np.tile(np.asarray((0, 1), np.int8), len(opponents) * len(seeds))
    elapsed = time.perf_counter() - started
    path = args.output_root / f"{name}.npz"
    np.savez_compressed(
        path,
        block_ids=np.asarray([block.block_id for block in blocks]),
        kinds=np.asarray([block.kind for block in blocks]),
        parent_ids=np.asarray([block.parent_id for block in blocks]),
        cluster_ids=np.asarray([block.cluster_id for block in blocks]),
        vectors=np.stack([block.vector for block in blocks]).astype(np.float32),
        entries=np.stack([block.entry for block in blocks]).astype(np.float32),
        states=states.astype(np.float32),
        outcome=outcome.astype(np.uint8),
        margin=margin.astype(np.float32),
        opponents=snapshot_opponents,
        seeds=snapshot_seeds,
        seats=snapshot_seats,
    )
    return {
        "name": name,
        "path": str(path.resolve()),
        "blocks": [block.block_id for block in blocks],
        "opening": opening.block_id,
        "opponents": list(opponents),
        "seeds": list(map(int, seeds)),
        "snapshots": int(states.shape[0]),
        "games": int(outcome.size),
        "elapsed_seconds": elapsed,
        "games_per_second": float(outcome.size / max(elapsed, 1e-9)),
        "outcome": outcome,
        "margin": margin,
        "states": states,
    }


def audit_composability(
    args: argparse.Namespace,
    blocks: Sequence[Block],
    opponents: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    additional = {block.block_id: block.tape for block in blocks}
    bundle = NativeTeammateBundle(
        args.source, args.base_actions, args.base_metadata,
        additional_routes=additional, included_families=tuple(opponents),
    )
    seeds = list(map(int, seeds))
    samples = [
        (opponent, seed, seat)
        for opponent in opponents for seed in seeds for seat in (0, 1)
    ]
    tasks = np.empty((len(blocks) * len(samples), 7), dtype=np.int64)
    row = 0
    opening_index = bundle.index(blocks[0].block_id)
    for block in blocks:
        target = bundle.index(block.block_id)
        for opponent, seed, seat in samples:
            other = bundle.index(opponent)
            if seat == 0:
                tasks[row] = (
                    opening_index, other, seed, START, target, -1, -1,
                )
            else:
                tasks[row] = (
                    other, opening_index, seed, -1, -1, START, target,
                )
            row += 1
    rewards, raw_audit = bundle.executor.play_audit_batch(tasks)
    rewards = np.asarray(rewards, np.float64).reshape(
        len(blocks), len(samples), 2
    )
    raw_audit = np.asarray(raw_audit, np.int32).reshape(
        len(blocks), len(samples), 2, 3
    )
    seats = np.asarray([seat for _, _, seat in samples], np.int64)
    columns = np.arange(len(samples))
    own_audit = raw_audit[:, columns, seats, :]
    baseline_total = own_audit[0, :, 0] + own_audit[0, :, 1]
    rows = []
    for index, block in enumerate(blocks):
        total = own_audit[index, :, 0] + own_audit[index, :, 1]
        excess = np.maximum(0, total - baseline_total)
        first = own_audit[index, :, 2]
        rows.append({
            "block_id": block.block_id,
            "kind": block.kind,
            "completion_rate": float(np.mean(
                np.isfinite(rewards[index]).all(axis=1)
            )),
            "macro_unit_failure_rate": float(np.mean(
                own_audit[index, :, 0] > 0
            )),
            "macro_market_failure_rate": float(np.mean(
                own_audit[index, :, 1] > 0
            )),
            "mean_macro_unit_failures": float(np.mean(
                own_audit[index, :, 0]
            )),
            "mean_macro_market_failures": float(np.mean(
                own_audit[index, :, 1]
            )),
            "first_failure_in_block_rate": float(np.mean(
                (first >= START) & (first < STOP)
            )),
            "new_failure_vs_baseline_rate": float(np.mean(excess > 0)),
            "mean_excess_failures_vs_baseline": float(np.mean(excess)),
        })
    trace_lengths = []
    for block_index in range(len(blocks)):
        for sample_index in (0, 1):
            task = tasks[block_index * len(samples) + sample_index]
            trace_lengths.append(
                len(bundle.executor.play(*task.tolist(), True)["trace"])
            )
    active_rows = rows[1:]
    gate = bool(
        all(row["completion_rate"] == 1.0 for row in rows)
        and all(length == HORIZON for length in trace_lengths)
        and all(
            row["new_failure_vs_baseline_rate"] <= .05
            and row["mean_excess_failures_vs_baseline"] <= .25
            for row in active_rows
        )
    )
    return {
        "scope": "paired whole-match audit; first-failure window is diagnostic",
        "snapshots_per_block": len(samples),
        "trace_length_checks": trace_lengths,
        "thresholds": {
            "max_new_failure_vs_baseline_rate": .05,
            "max_mean_excess_failures_vs_baseline": .25,
        },
        "blocks": rows,
        "gate_passed": gate,
    }


def _utility(outcome: np.ndarray, margin: np.ndarray) -> np.ndarray:
    return outcome.astype(np.float32) * .5 + .02 * np.tanh(margin / 10000.0)


def greedy_order(
    outcome: np.ndarray, margin: np.ndarray, count: int,
) -> list[int]:
    utility = _utility(outcome, margin)
    current = utility[0].copy()
    available = set(range(1, len(utility)))
    selected: list[int] = []
    while available and len(selected) < count:
        best = max(available, key=lambda index: (
            float(np.mean(np.maximum(current, utility[index])) - np.mean(current)),
            float(np.mean(utility[index])),
            -index,
        ))
        selected.append(best)
        available.remove(best)
        current = np.maximum(current, utility[best])
    return selected


def portfolio_metrics(
    outcome: np.ndarray, margin: np.ndarray, indices: Sequence[int],
    reference_index: int | None = None,
) -> dict[str, Any]:
    rows = np.asarray(indices, np.int32)
    utility = _utility(outcome[rows], margin[rows])
    constant_local = int(np.argmax(np.mean(utility, axis=1)))
    constant = int(rows[constant_local]) if reference_index is None else int(reference_index)
    if constant not in rows:
        raise ValueError("reference index must be part of the portfolio")
    oracle_local = np.argmax(utility, axis=0)
    chosen_outcome = outcome[rows[oracle_local], np.arange(outcome.shape[1])]
    chosen_margin = margin[rows[oracle_local], np.arange(outcome.shape[1])]
    constant_outcome = outcome[constant]
    losses = constant_outcome == 0
    fixed = losses & (chosen_outcome == 2)
    return {
        "block_ids": rows.astype(int).tolist(),
        "best_constant_index": constant,
        "best_constant_raw_win_rate": float(np.mean(constant_outcome == 2)),
        "best_constant_score_rate": float(np.mean(constant_outcome) * .5),
        "best_constant_mean_margin": float(np.mean(margin[constant])),
        "oracle_raw_win_rate": float(np.mean(chosen_outcome == 2)),
        "oracle_score_rate": float(np.mean(chosen_outcome) * .5),
        "oracle_mean_margin": float(np.mean(chosen_margin)),
        "oracle_gain_pp": float(
            100 * (np.mean(chosen_outcome == 2) - np.mean(constant_outcome == 2))
        ),
        "constant_loss_states": int(np.sum(losses)),
        "constant_loss_fix_fraction": float(np.sum(fixed) / max(1, np.sum(losses))),
    }


def _panel_json(panel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in panel.items()
        if key not in {"outcome", "margin", "states"}
    }


def search_blocks(
    args: argparse.Namespace, baseline: Block, carriers: list[Block], variants: list[Block],
) -> tuple[list[Block], dict[str, Any], dict[str, list[str]]]:
    split, split_lineages = _opponent_splits(
        args.base_metadata, args.random_seed
    )
    candidates = [*carriers, *variants]
    stages = []
    stage1 = evaluate_panel(
        args, [baseline, *candidates], split["evolution"][:8],
        range(args.seed_base, args.seed_base + 8), "search_stage1_64x128",
    )
    keep = greedy_order(stage1["outcome"], stage1["margin"], 32)
    candidates = [[baseline, *candidates][index] for index in keep]
    stages.append(_panel_json(stage1))
    stage2 = evaluate_panel(
        args, [baseline, *candidates], split["evolution"][:12],
        range(args.seed_base, args.seed_base + 16), "search_stage2_32x384",
    )
    keep = greedy_order(stage2["outcome"], stage2["margin"], 16)
    candidates = [[baseline, *candidates][index] for index in keep]
    stages.append(_panel_json(stage2))
    stage3 = evaluate_panel(
        args, [baseline, *candidates], split["evolution"],
        range(args.seed_base, args.seed_base + 32), "search_stage3_16x1024",
    )
    keep = greedy_order(stage3["outcome"], stage3["margin"], 8)
    active = [[baseline, *candidates][index] for index in keep]
    stages.append(_panel_json(stage3))

    union = list({block.block_id: block for block in [baseline, *carriers, *active]}.values())
    championship = evaluate_panel(
        args, union, split["championship"],
        range(args.seed_base + 3000, args.seed_base + 3050),
        "search_stage4_championship",
    )
    by_id = {block.block_id: index for index, block in enumerate(union)}
    baseline_index = by_id[baseline.block_id]
    replay_indices = [baseline_index, *[
        by_id[block.block_id] for block in carriers
    ]]
    combined_indices = list(range(len(union)))
    replay_metrics = portfolio_metrics(
        championship["outcome"], championship["margin"], replay_indices,
        reference_index=baseline_index,
    )
    combined_metrics = portfolio_metrics(
        championship["outcome"], championship["margin"], combined_indices,
        reference_index=baseline_index,
    )
    h1 = (
        replay_metrics["oracle_gain_pp"] >= 3.0
        or replay_metrics["constant_loss_fix_fraction"] >= .20
    )
    extra_oracle_pp = 100 * (
        combined_metrics["oracle_raw_win_rate"] - replay_metrics["oracle_raw_win_rate"]
    )
    replay_utility = _utility(
        championship["outcome"][replay_indices],
        championship["margin"][replay_indices],
    )
    combined_utility = _utility(
        championship["outcome"][combined_indices],
        championship["margin"][combined_indices],
    )
    columns = np.arange(championship["outcome"].shape[1])
    replay_oracle = championship["outcome"][
        np.asarray(replay_indices)[np.argmax(replay_utility, axis=0)], columns
    ]
    combined_oracle = championship["outcome"][
        np.asarray(combined_indices)[np.argmax(combined_utility, axis=0)], columns
    ]
    replay_unsolved = replay_oracle != 2
    extra_fix = float(np.sum(
        replay_unsolved & (combined_oracle == 2)
    ) / max(1, np.sum(replay_unsolved)))
    h3 = extra_oracle_pp >= 1.0 or extra_fix >= .05
    execution_audit = audit_composability(
        args, [baseline, *active], split["championship"],
        range(args.seed_base + 3000, args.seed_base + 3050),
    )
    h4 = bool(execution_audit["gate_passed"])
    report = {
        "schema": "block-mvp-96-216-search-v1",
        "opponent_splits": split,
        "opponent_split_lineages": split_lineages,
        "seed_base": args.seed_base,
        "stages": stages,
        "active_block_ids": [block.block_id for block in active],
        "active_kinds": [block.kind for block in active],
        "active_lineages": [block.cluster_id for block in active],
        "championship": _panel_json(championship),
        "replay_carrier_oracle": replay_metrics,
        "replay_plus_ga_oracle": combined_metrics,
        "ga_extra_oracle_pp": extra_oracle_pp,
        "ga_extra_loss_fix_fraction": extra_fix,
        "execution_audit": execution_audit,
        "gates": {
            "H1_block_library_value": h1,
            "H3_ga_value": h3,
            "H4_composable_execution": h4,
        },
    }
    (args.output_root / "search_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return active, report, split


def _state_entry(states: np.ndarray) -> np.ndarray:
    return np.column_stack((
        states[:, 6:11], states[:, 11:14], states[:, 15], states[:, 16],
        states[:, 2], states[:, 1],
    )).astype(np.float32)


def _pair_features(
    states: np.ndarray, blocks: Sequence[Block],
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(states[:, :STATE_DIM], np.float32)
    vectors = np.stack([block.vector for block in blocks]).astype(np.float32)
    entries = np.stack([block.entry for block in blocks]).astype(np.float32)
    compatibility = _state_entry(states)[:, None, :] - entries[None, :, :]
    repeated_state = np.repeat(state[:, None, :], len(blocks), axis=1)
    repeated_blocks = np.repeat(vectors[None, :, :], len(states), axis=0)
    pairs = np.concatenate((repeated_state, repeated_blocks, compatibility), axis=2)
    if pairs.shape[2] != PAIR_DIM:
        raise AssertionError(f"pair feature dimension {pairs.shape[2]}")
    hard_compatible = np.all(compatibility[:, :, 8:] == 0, axis=2)
    hard_compatible[:, 0] = True
    return pairs.astype(np.float32), hard_compatible


def _advantage(outcome: np.ndarray, margin: np.ndarray) -> np.ndarray:
    utility = _utility(outcome, margin)
    return utility - utility[0:1]


def _selector_choice(
    model: ExtraTreesRegressor,
    pairs: np.ndarray,
    compatible: np.ndarray,
    threshold: float,
    agreement: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, blocks, _ = pairs.shape
    flat = pairs.reshape(-1, pairs.shape[-1])
    mean = model.predict(flat).reshape(samples, blocks)
    positive = np.mean(np.stack([
        estimator.predict(flat).reshape(samples, blocks) > 0
        for estimator in model.estimators_
    ]), axis=0)
    candidate = mean.copy()
    candidate[:, 0] = -np.inf
    candidate[~compatible] = -np.inf
    best = np.argmax(candidate, axis=1)
    rows = np.arange(samples)
    accept = (
        np.isfinite(candidate[rows, best])
        & (candidate[rows, best] > threshold)
        & (positive[rows, best] >= agreement)
    )
    choice = np.where(accept, best, 0).astype(np.int32)
    return choice, mean, positive


def _choice_metrics(
    outcome: np.ndarray,
    margin: np.ndarray,
    choices: np.ndarray,
    opponents: np.ndarray,
) -> dict[str, Any]:
    rows = np.arange(len(choices))
    selected_outcome = outcome[choices, rows]
    selected_margin = margin[choices, rows]
    utility = _utility(outcome, margin)
    constant = int(np.argmax(np.mean(utility, axis=1)))
    constant_wins = outcome[constant] == 2
    keep_wins = outcome[0] == 2
    oracle = np.argmax(utility, axis=0)
    oracle_wins = outcome[oracle, rows] == 2
    selector_rate = float(np.mean(selected_outcome == 2))
    constant_rate = float(np.mean(constant_wins))
    oracle_rate = float(np.mean(oracle_wins))
    denominator = oracle_rate - constant_rate
    opponent_rates = {
        str(opponent): float(np.mean(selected_outcome[opponents == opponent] == 2))
        for opponent in np.unique(opponents)
    }
    paired = (selected_outcome == 2).reshape(-1, 2)
    return {
        "raw_win_rate": selector_rate,
        "score_rate": float(np.mean(selected_outcome) * .5),
        "mean_margin": float(np.mean(selected_margin)),
        "keep_rate": float(np.mean(choices == 0)),
        "keep_raw_win_rate": float(np.mean(keep_wins)),
        "best_constant_index": constant,
        "best_constant_raw_win_rate": constant_rate,
        "oracle_raw_win_rate": oracle_rate,
        "selector_gain_pp": 100 * (selector_rate - constant_rate),
        "selector_gain_vs_keep_pp": 100 * (
            selector_rate - float(np.mean(keep_wins))
        ),
        "oracle_gap_closure": float(
            (selector_rate - constant_rate) / denominator
        ) if denominator > 1e-9 else 0.0,
        "both_seats_win_rate": float(np.mean(np.all(paired, axis=1))),
        "minimum_opponent_raw_win_rate": float(min(opponent_rates.values())),
        "opponent_raw_win_rates": opponent_rates,
    }


def _fit_pair_model(
    pairs: np.ndarray,
    advantage: np.ndarray,
    *,
    leaf: int,
    max_features: float,
    estimators: int,
    random_state: int,
    block_mask: np.ndarray | None = None,
) -> ExtraTreesRegressor:
    samples, blocks, dimensions = pairs.shape
    keep = np.ones((samples, blocks), bool)
    if block_mask is not None:
        keep &= block_mask[None, :]
    return ExtraTreesRegressor(
        n_estimators=estimators, min_samples_leaf=leaf,
        max_features=max_features, n_jobs=-1, random_state=random_state,
    ).fit(pairs.reshape(-1, dimensions)[keep.reshape(-1)],
          advantage.T.reshape(-1)[keep.reshape(-1)])


def train_selector(
    args: argparse.Namespace,
    baseline: Block,
    active: Sequence[Block],
    opponent_split: Mapping[str, list[str]],
) -> tuple[dict[str, Any], ExtraTreesRegressor]:
    blocks = [baseline, *active]
    train = evaluate_panel(
        args, blocks, opponent_split["train"],
        range(args.seed_base + 1000, args.seed_base + 1050), "selector_train_2000",
    )
    validation = evaluate_panel(
        args, blocks, opponent_split["validation"],
        range(args.seed_base + 2000, args.seed_base + 2025), "selector_validation_500",
    )
    championship = evaluate_panel(
        args, blocks, opponent_split["championship"],
        range(args.seed_base + 3000, args.seed_base + 3050), "selector_championship_2000",
    )
    train_pairs, _ = _pair_features(train["states"], blocks)
    valid_pairs, valid_compat = _pair_features(validation["states"], blocks)
    championship_pairs, championship_compat = _pair_features(
        championship["states"], blocks
    )
    train_advantage = _advantage(train["outcome"], train["margin"])
    trials = []
    fitted: dict[tuple[int, float], ExtraTreesRegressor] = {}
    for leaf in (4, 8, 16):
        for max_features in (.5, 1.0):
            model = _fit_pair_model(
                train_pairs, train_advantage, leaf=leaf,
                max_features=max_features, estimators=args.estimators,
                random_state=args.random_seed,
            )
            fitted[(leaf, max_features)] = model
            for threshold in (0.0, .002, .005, .01):
                for agreement in (.5, .6, .7):
                    choices, _, _ = _selector_choice(
                        model, valid_pairs, valid_compat, threshold, agreement
                    )
                    metrics = _choice_metrics(
                        validation["outcome"], validation["margin"], choices,
                        np.repeat(np.asarray(opponent_split["validation"]), 25 * 2),
                    )
                    trials.append({
                        "leaf": leaf, "max_features": max_features,
                        "threshold": threshold, "agreement": agreement,
                        "validation": metrics,
                    })
    trials.sort(key=lambda row: (
        -row["validation"]["raw_win_rate"],
        -row["validation"]["oracle_gap_closure"],
        -row["validation"]["mean_margin"],
        -row["threshold"], -row["agreement"],
        row["leaf"], row["max_features"],
    ))
    selected = trials[0]
    model = fitted[(int(selected["leaf"]), float(selected["max_features"]))]
    championship_choices, _, _ = _selector_choice(
        model, championship_pairs, championship_compat,
        float(selected["threshold"]), float(selected["agreement"]),
    )
    championship_metrics = _choice_metrics(
        championship["outcome"], championship["margin"], championship_choices,
        np.repeat(np.asarray(opponent_split["championship"]), 50 * 2),
    )
    lineage_counts = Counter(block.cluster_id for block in active)
    held_lineage = max(sorted(lineage_counts), key=lambda key: lineage_counts[key])
    block_mask = np.asarray([
        block.cluster_id != held_lineage or block.block_id == baseline.block_id
        for block in blocks
    ])
    seen_indices = np.flatnonzero(block_mask)
    lineage_trials = []
    lineage_models: dict[tuple[int, float], ExtraTreesRegressor] = {}
    for leaf in (4, 8, 16):
        for max_features in (.5, 1.0):
            lineage_model = _fit_pair_model(
                train_pairs, train_advantage,
                leaf=leaf, max_features=max_features,
                estimators=args.estimators, random_state=args.random_seed + 1,
                block_mask=block_mask,
            )
            lineage_models[(leaf, max_features)] = lineage_model
            for threshold in (0.0, .002, .005, .01):
                for agreement in (.5, .6, .7):
                    choices, _, _ = _selector_choice(
                        lineage_model, valid_pairs[:, seen_indices],
                        valid_compat[:, seen_indices], threshold, agreement,
                    )
                    metrics = _choice_metrics(
                        validation["outcome"][seen_indices],
                        validation["margin"][seen_indices], choices,
                        np.repeat(
                            np.asarray(opponent_split["validation"]), 25 * 2
                        ),
                    )
                    lineage_trials.append({
                        "leaf": leaf, "max_features": max_features,
                        "threshold": threshold, "agreement": agreement,
                        "validation_seen_lineages": metrics,
                    })
    lineage_trials.sort(key=lambda row: (
        -row["validation_seen_lineages"]["raw_win_rate"],
        -row["validation_seen_lineages"]["oracle_gap_closure"],
        -row["validation_seen_lineages"]["mean_margin"],
        -row["threshold"], -row["agreement"],
        row["leaf"], row["max_features"],
    ))
    lineage_selected = lineage_trials[0]
    lineage_model = lineage_models[(
        int(lineage_selected["leaf"]),
        float(lineage_selected["max_features"]),
    )]
    hidden_indices = np.asarray([
        0, *[
            index for index, block in enumerate(blocks)
            if index and block.cluster_id == held_lineage
        ],
    ], np.int32)
    lineage_choices, _, _ = _selector_choice(
        lineage_model, championship_pairs[:, hidden_indices],
        championship_compat[:, hidden_indices],
        float(lineage_selected["threshold"]),
        float(lineage_selected["agreement"]),
    )
    lineage_metrics = _choice_metrics(
        championship["outcome"][hidden_indices],
        championship["margin"][hidden_indices], lineage_choices,
        np.repeat(np.asarray(opponent_split["championship"]), 50 * 2),
    )
    hidden_choice_rate = float(np.mean(lineage_choices != 0))
    h2 = (
        championship_metrics["selector_gain_pp"] >= 2.0
        and championship_metrics["oracle_gap_closure"] >= .30
        and lineage_metrics["oracle_gap_closure"] >= .15
        and hidden_choice_rate > 0.0
    )
    model_path = args.output_root / "block_pair_selector.joblib"
    joblib.dump({
        "schema": "block-pair-extra-trees-v1",
        "model": model,
        "block_ids": [block.block_id for block in blocks],
        "block_vectors": np.stack([block.vector for block in blocks]),
        "block_entries": np.stack([block.entry for block in blocks]),
        "feature_dimensions": {
            "state": STATE_DIM, "block": BLOCK_DIM,
            "compatibility": COMPAT_DIM, "pair": PAIR_DIM,
        },
        "gate": {
            "threshold": selected["threshold"],
            "positive_tree_fraction": selected["agreement"],
        },
    }, model_path, compress=3)
    report = {
        "schema": "block-mvp-96-216-selector-v1",
        "train": _panel_json(train),
        "validation": _panel_json(validation),
        "championship": _panel_json(championship),
        "block_ids": [block.block_id for block in blocks],
        "selected": selected,
        "championship_metrics": championship_metrics,
        "unseen_lineage": {
            "held_lineage": held_lineage,
            "hidden_block_ids": [
                block.block_id for block in blocks
                if block.cluster_id == held_lineage
            ],
            "selection": lineage_selected,
            "hidden_route_choice_rate": hidden_choice_rate,
            "metrics": lineage_metrics,
            "trials": lineage_trials,
        },
        "model": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "gate_H2_selector_learnability": h2,
        "trials": trials,
    }
    (args.output_root / "selector_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, model


def _perturbed_openings(baseline: Block) -> list[Block]:
    result = []
    for quantity in (1, 3):
        tape = copy.deepcopy(baseline.tape)
        destination = next(
            step for step in range(START - 1, START - 13, -1)
            if len(tape[step].get("market", []) or []) < 10
        )
        tape[destination].setdefault("market", []).append(
            ["BUY_SEED", "STRAWBERRY", quantity]
        )
        result.append(Block(
            block_id=f"OOD_CASH_DOWN_{100 * quantity}", kind="perturbation",
            parent_id=baseline.block_id, cluster_id="OOD",
            source={"perturbation": f"extra strawberry seed purchase x{quantity}"},
            entry=baseline.entry.copy(), vector=baseline.vector.copy(),
            event_profile=baseline.event_profile.copy(), tape=tape, genes=[],
        ))
    tape = copy.deepcopy(baseline.tape)
    changed = False
    for step in range(START - 1, START - 13, -1):
        orders = [tape[step].get("farmer"), *(tape[step].get("hands", ()) or ())]
        for actor, raw in enumerate(orders):
            order = list(raw or ["PASS"])
            if str(order[0]) == "PASS":
                continue
            if actor == 0:
                tape[step]["farmer"] = ["PASS"]
            else:
                tape[step]["hands"][actor - 1] = ["PASS"]
            changed = True
            break
        if changed:
            break
    if changed:
        result.append(Block(
            block_id="OOD_WORKER_DELAY", kind="perturbation",
            parent_id=baseline.block_id, cluster_id="OOD",
            source={"perturbation": "one late-prefix unit action replaced with PASS"},
            entry=baseline.entry.copy(), vector=baseline.vector.copy(),
            event_profile=baseline.event_profile.copy(), tape=tape, genes=[],
        ))
    tape = copy.deepcopy(baseline.tape)
    removed = None
    for step in range(START - 1, START - 25, -1):
        for index, raw in enumerate(tape[step].get("market", ()) or ()):
            order = list(raw or ())
            if order and str(order[0]) in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                removed = tape[step]["market"].pop(index)
                break
        if removed is not None:
            break
    if removed is not None:
        result.append(Block(
            block_id="OOD_CASH_UP", kind="perturbation",
            parent_id=baseline.block_id, cluster_id="OOD",
            source={"perturbation": f"removed late-prefix purchase {removed}"},
            entry=baseline.entry.copy(), vector=baseline.vector.copy(),
            event_profile=baseline.event_profile.copy(), tape=tape, genes=[],
        ))
    return result


def evaluate_ood(
    args: argparse.Namespace,
    baseline: Block,
    active: Sequence[Block],
    model: ExtraTreesRegressor,
    selector_report: Mapping[str, Any],
    opponent_split: Mapping[str, list[str]],
) -> dict[str, Any]:
    blocks = [baseline, *active]
    seeds = range(args.seed_base + 4000, args.seed_base + 4025)
    opponents = opponent_split["ood"]
    gate = selector_report["selected"]
    regular = evaluate_panel(
        args, blocks, opponents, seeds, "ood_regular_500", opening=baseline,
    )

    def metrics(panel: Mapping[str, Any]) -> dict[str, Any]:
        pairs, compatible = _pair_features(panel["states"], blocks)
        choices, _, _ = _selector_choice(
            model, pairs, compatible,
            float(gate["threshold"]), float(gate["agreement"]),
        )
        return _choice_metrics(
            panel["outcome"], panel["margin"], choices,
            np.repeat(np.asarray(opponents), 25 * 2),
        )

    regular_metrics = metrics(regular)
    perturbations = []
    openings = _perturbed_openings(baseline)
    h5 = bool(
        np.isfinite(regular["margin"]).all()
        and regular_metrics["selector_gain_vs_keep_pp"] >= 0.0
        and len(openings) == 4
    )
    for opening in openings:
        panel = evaluate_panel(
            args, blocks, opponents, seeds,
            f"ood_{opening.block_id.lower()}", opening=opening,
        )
        value = metrics(panel)
        drop_pp = 100 * (
            regular_metrics["raw_win_rate"] - value["raw_win_rate"]
        )
        finite = bool(np.isfinite(panel["margin"]).all())
        state_changed = np.any(
            np.abs(panel["states"][:, :STATE_DIM]
                   - regular["states"][:, :STATE_DIM]) > 1e-6,
            axis=1,
        )
        state_change_rate = float(np.mean(state_changed))
        h5 &= bool(
            finite and drop_pp <= 2.0 and state_change_rate >= .50
            and value["selector_gain_vs_keep_pp"] >= 0.0
        )
        perturbations.append({
            "opening": opening.block_id,
            "description": opening.source["perturbation"],
            "mean_cash_delta_at_96": float(
                np.mean(panel["states"][:, 0] - regular["states"][:, 0])
            ),
            "raw_win_drop_pp": drop_pp,
            "state_change_rate_at_96": state_change_rate,
            "finite_rewards": finite,
            "metrics": value,
            "panel": _panel_json(panel),
        })
    weed_mask = (
        (regular["states"][:, 4] >= 1) & (regular["states"][:, 4] <= 3)
    )
    weed = None
    if np.any(weed_mask):
        pairs, compatible = _pair_features(regular["states"][weed_mask], blocks)
        choices, _, _ = _selector_choice(
            model, pairs, compatible,
            float(gate["threshold"]), float(gate["agreement"]),
        )
        weed = _choice_metrics(
            regular["outcome"][:, weed_mask],
            regular["margin"][:, weed_mask],
            choices,
            np.repeat(np.asarray(opponents), 25 * 2)[weed_mask],
        )
        weed["snapshots"] = int(np.sum(weed_mask))
        h5 &= weed["selector_gain_vs_keep_pp"] >= 0.0
    else:
        h5 = False
    completion_rate = float(np.mean([
        np.isfinite(regular["margin"]).all(),
        *[row["finite_rewards"] for row in perturbations],
    ]))
    report = {
        "schema": "block-mvp-96-216-ood-v1",
        "regular": regular_metrics,
        "weed_1_to_3": weed,
        "perturbations": perturbations,
        "completion_rate": completion_rate,
        "required_perturbations_present": len(openings) == 4,
        "uses_exact_state_key": False,
        "gate_H5_ood_robustness": h5,
    }
    (args.output_root / "ood_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_blocks(
    root: Path, args: argparse.Namespace,
) -> tuple[Block, list[Block], list[Block]]:
    manifest = json.loads(
        (root / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema") != "block-mvp-96-216-candidates-v1"
        or int(manifest.get("start_step", -1)) != START
        or int(manifest.get("end_step", -1)) != STOP
        or manifest.get("inputs") != _input_signatures(args)
        or manifest.get("generation_contract") != _generation_contract(args)
    ):
        raise ValueError(
            "candidate inputs changed; rerun without --resume-candidates"
        )
    rows = [
        json.loads(line)
        for line in (root / "blocks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_path = root / "block_actions.json.zlib"
    if hashlib.sha256(action_path.read_bytes()).hexdigest() != manifest.get(
        "actions_sha256"
    ):
        raise ValueError("candidate action archive digest mismatch")
    actions = load_action_tapes(action_path)
    blocks = [
        Block(
            block_id=str(row["block_id"]),
            kind=str(row["kind"]),
            parent_id=str(row["parent_id"]),
            cluster_id=str(row["cluster_id"]),
            source=dict(row["source"]),
            entry=np.asarray([row["entry"][name] for name in ENTRY_NAMES], np.float32),
            vector=np.asarray(row["vector"], np.float32),
            event_profile=np.asarray(row["event_profile"], np.float32),
            tape=actions[str(row["block_id"])],
            genes=list(row.get("genes", ())),
        )
        for row in rows
    ]
    baseline = next(block for block in blocks if block.kind == "baseline")
    return (
        baseline,
        [block for block in blocks if block.kind == "replay"],
        [block for block in blocks if block.kind == "ga"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path,
        default=Path(r"D:\Kaggriculture\cpp_route_experiments_20260827\live_replay_pool\route_genomes_live.jsonl"),
    )
    parser.add_argument(
        "--source", type=Path,
        default=CODE_ROOT / "runtime" / "teammate_base.py",
    )
    parser.add_argument(
        "--opening-actions", type=Path,
        default=CODE_ROOT / "runtime" / "route_actions.json.zlib",
    )
    parser.add_argument(
        "--opening-metadata", type=Path,
        default=CODE_ROOT / "runtime" / "route_library.json",
    )
    parser.add_argument(
        "--base-actions", type=Path,
        default=Path(r"D:\Kaggriculture\cpp_route_experiments_20260827\causal_tree_v5\route_actions.json.zlib"),
    )
    parser.add_argument(
        "--base-metadata", type=Path,
        default=Path(r"D:\Kaggriculture\cpp_route_experiments_20260827\causal_tree_v5\route_library.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(r"D:\Kaggriculture\cpp_route_experiments_20260827\block_mvp_96_216_v1"),
    )
    parser.add_argument("--opening", default="NR295")
    parser.add_argument("--prefilter", type=int, default=128)
    parser.add_argument("--carriers", type=int, default=8)
    parser.add_argument("--estimators", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=2026082800)
    parser.add_argument("--random-seed", type=int, default=20260828)
    parser.add_argument("--resume-candidates", action="store_true")
    parser.add_argument(
        "--stop-after", choices=("prepare", "search", "selector", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.prefilter < args.carriers or args.carriers != 8 or args.estimators <= 0:
        parser.error("prefilter must cover exactly 8 carriers and estimators must be positive")
    for name in (
        "records", "source", "opening_actions", "opening_metadata",
        "base_actions", "base_metadata",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
    args.output_root = args.output_root.resolve()
    started = time.perf_counter()
    if args.resume_candidates and (args.output_root / "blocks.jsonl").is_file():
        baseline, carriers, variants = load_blocks(args.output_root, args)
    else:
        baseline, carriers, variants = prepare_blocks(args)
    if args.stop_after == "prepare":
        print(json.dumps({
            "status": "prepared", "carriers": len(carriers),
            "variants": len(variants), "output_root": str(args.output_root),
        }, indent=2), flush=True)
        return
    active, search_report, opponent_split = search_blocks(
        args, baseline, carriers, variants
    )
    if args.stop_after == "search":
        print(json.dumps({
            "status": "searched", "gates": search_report["gates"],
            "active_block_ids": [block.block_id for block in active],
        }, indent=2), flush=True)
        return
    selector_report, model = train_selector(
        args, baseline, active, opponent_split
    )
    if args.stop_after == "selector":
        print(json.dumps({
            "status": "selector_trained",
            "gate_H2": selector_report["gate_H2_selector_learnability"],
            "metrics": selector_report["championship_metrics"],
        }, indent=2), flush=True)
        return
    ood_report = evaluate_ood(
        args, baseline, active, model, selector_report, opponent_split
    )
    gates = {
        **search_report["gates"],
        "H2_selector_learnability": bool(
            selector_report["gate_H2_selector_learnability"]
        ),
        "H5_ood_robustness": bool(ood_report["gate_H5_ood_robustness"]),
    }
    passed = all(gates.values())
    final = {
        "schema": "block-mvp-96-216-v1",
        "status": "passed" if passed else "falsified",
        "gates": gates,
        "active_block_ids": [block.block_id for block in active],
        "championship_selector": selector_report["championship_metrics"],
        "replay_carrier_oracle": search_report["replay_carrier_oracle"],
        "replay_plus_ga_oracle": search_report["replay_plus_ga_oracle"],
        "ood": {
            "regular_raw_win_rate": ood_report["regular"]["raw_win_rate"],
            "perturbation_drops_pp": {
                row["opening"]: row["raw_win_drop_pp"]
                for row in ood_report["perturbations"]
            },
        },
        "deployment_attempted": False,
        "next_action": (
            "integrate one-switch pair selector and run official-engine gate"
            if passed
            else "stop expansion and fix the first failed hypothesis"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "candidates": str((args.output_root / "candidate_manifest.json").resolve()),
            "search": str((args.output_root / "search_report.json").resolve()),
            "selector": str((args.output_root / "selector_report.json").resolve()),
            "ood": str((args.output_root / "ood_report.json").resolve()),
        },
    }
    (args.output_root / "FINAL_REPORT.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
