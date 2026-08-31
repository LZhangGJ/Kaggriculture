#!/usr/bin/env python3
"""Test replay-trace reassignment as a path-only residual on a frozen genome.

The experiment keeps every route gene and every market order frozen.  At the
first four turns after each 24-step anchor it may splice one or two actor slots
from a heldout-lineage-filtered replay-block donor.  It compares four-step
score/execute open-loop commits with one-step score/execute replanning (MPC1).

This is deliberately a proxy experiment over native route opponents.  It is
not a packaged Kaggle agent and does not claim the full-pool championship gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fast_kaggriculture
import fast_kaggriculture._fast_kaggriculture as native_extension
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_compiler import _inventory, _positions
from meta_agent.src.route_plan import CarrierRoute
from meta_agent.src.route_switch_features import (
    RouteSwitchHistory,
    route_switch_feature_names,
    route_switch_vector,
)
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_route_residual_adapter_v0 as residual


SCHEMA = "multi-farmer-path-residual-v1"
HORIZON = residual.HORIZON
ANCHORS = residual.ANCHORS
STOPS = residual.STOPS
PLAN_HORIZON = 4
COMMIT4_OVERRIDE_HORIZON = PLAN_HORIZON
MPC1_OVERRIDE_HORIZON = 1
MAX_ACTORS = 16
MAX_PAIRS = 3
MAX_ACTIVE_DONORS = 8
WINDOW_OFFSETS = tuple(range(1, PLAN_HORIZON + 1))
DECISION_STEPS = frozenset(
    anchor + offset for anchor in ANCHORS for offset in WINDOW_OFFSETS
    if anchor + offset < HORIZON
)
MOVE_OPS = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})
KIND_CODES = ("KEEP", "SINGLE", "PAIR", "ALL")
KIND_INDEX = {name: index for index, name in enumerate(KIND_CODES)}
ITEM_FEATURES = (*residual.ITEMS, "NONE")

TRAIN_OPPONENTS = (
    "NT0056", "NT0051", "NT0071", "NT0070", "NT0301", "NT0284",
)
HELDOUT_OPPONENTS = (
    "NT0557", "NT0550", "NT0001", "NT0568", "NT0588", "NT0576",
)
TRAIN_SEEDS = tuple(range(2026086300, 2026086308))
VALIDATION_SEEDS = tuple(range(2026086320, 2026086328))
DEFAULT_FROZEN_RUN = residual.DEFAULT_FROZEN_RUN
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1\run"
)
DEFAULT_BLOCK_LIBRARY = adaptive.DEFAULT_BLOCK_LIBRARY
DEFAULT_HELDOUT_TEAMS = ("tetsuya", "Ryo Hasegawa", "Crop Dusta")


@dataclass(frozen=True)
class DonorBlock:
    block_id: str
    anchor: int
    cluster_id: str
    source_provenance_id: str
    source_route_id: str
    support: int
    actions: tuple[Mapping[str, Any], ...]
    team_name: str


@dataclass(frozen=True)
class PathCandidate:
    code: str
    kind: str
    units: tuple[tuple[tuple[Any, ...], ...], ...]
    assignments: tuple[tuple[int, int], ...]
    donor_id: str | None
    donor_cluster: str | None
    donor_support: int
    donor_rank: int
    trace_novelty: int
    aliases: tuple[str, ...]
    raw_sha256: str

    @property
    def keep(self) -> bool:
        return self.kind == "KEEP"


@dataclass
class PathPanel:
    features: list[np.ndarray]
    target: list[float]
    delta_margin: list[float]
    rewards: list[tuple[float, float]]
    seed: list[int]
    seat: list[int]
    opponent: list[int]
    step: list[int]
    edit: list[int]
    decision: list[int]
    selected: list[bool]
    genome: list[int]
    candidate_sha256: list[str]

    @classmethod
    def empty(cls) -> "PathPanel":
        return cls(*([] for _ in range(13)))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.features:
            raise ValueError("path training panel is empty")
        return {
            "features": np.stack(self.features).astype(np.float32),
            "target_signed_log_margin": np.asarray(self.target, np.float32),
            "delta_margin": np.asarray(self.delta_margin, np.float64),
            "terminal_rewards": np.asarray(self.rewards, np.float64),
            "seed": np.asarray(self.seed, np.int64),
            "seat": np.asarray(self.seat, np.int8),
            "opponent": np.asarray(self.opponent, np.int16),
            "step": np.asarray(self.step, np.int16),
            "edit": np.asarray(self.edit, np.int8),
            "decision": np.asarray(self.decision, np.int32),
            "selected_by_oracle": np.asarray(self.selected, np.bool_),
            "genome": np.asarray(self.genome, np.int8),
            "candidate_sha256": np.asarray(self.candidate_sha256),
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unit(raw: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(residual._action(raw))


def _unit_trace(
    tape: Sequence[Mapping[str, Any]], step: int, actor_count: int,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    trace = []
    for offset in range(PLAN_HORIZON):
        at = min(HORIZON - 1, step + offset)
        raw = residual.normalize_player_action(tape[at])
        units = [_unit(raw["farmer"]), *(_unit(value) for value in raw["hands"])]
        units.extend(("PASS",) for _ in range(max(0, actor_count - len(units))))
        trace.append(tuple(units))
    return tuple(trace)


def _select_active_donor_rows(
    manifest: Mapping[str, Any],
    provenance_teams: Mapping[str, str],
    heldout_teams: Iterable[str],
    top_k: int,
) -> dict[int, list[Mapping[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k donors must be positive")
    excluded = set(map(str, heldout_teams))
    result = {}
    for anchor in ANCHORS:
        rows = [
            row for row in manifest.get("active_representatives", {}).get(str(anchor), ())
            if provenance_teams.get(str(row["source_provenance_id"]), "") not in excluded
        ]
        rows.sort(key=lambda row: (-int(row.get("support", 0)), str(row["block_id"])))
        if not rows:
            raise ValueError(f"heldout-lineage filter removed every donor at {anchor}")
        result[int(anchor)] = rows[:top_k]
    return result


def load_filtered_active_donors(
    root: Path,
    heldout_teams: Iterable[str] = DEFAULT_HELDOUT_TEAMS,
    *,
    top_k: int = MAX_ACTIVE_DONORS,
) -> tuple[dict[int, tuple[DonorBlock, ...]], dict[str, Any]]:
    """Load the support-ranked active subset after exact team-lineage removal."""

    root = root.resolve()
    manifest_path = root / "block_library_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "adaptive-tail-block-library-v0":
        raise ValueError("invalid adaptive block library schema")
    actions_path = root / str(manifest["actions_file"])
    if residual._sha256_file(actions_path) != str(manifest["actions_sha256"]):
        raise ValueError("adaptive block action archive digest mismatch")
    actions = continuation.load_action_tapes(actions_path)
    provenance_teams: dict[str, str] = {}
    provenance_inputs = []
    for prepared in manifest.get("prepared_roots", ()):
        prepared_path = Path(str(prepared["path"])).resolve() / "candidate_manifest.json"
        expected = str(prepared.get("candidate_manifest_sha256", ""))
        if expected and residual._sha256_file(prepared_path) != expected:
            raise ValueError("prepared provenance manifest digest mismatch")
        raw = json.loads(prepared_path.read_text(encoding="utf-8"))
        provenance_teams.update({
            str(row["provenance_id"]): str(row.get("team_name", ""))
            for row in raw.get("provenance", ())
        })
        provenance_inputs.append({
            "path": str(prepared_path), "sha256": residual._sha256_file(prepared_path),
        })
    selected = _select_active_donor_rows(
        manifest, provenance_teams, heldout_teams, top_k,
    )
    donors = {}
    for anchor, rows in selected.items():
        values = []
        for row in rows:
            block_id = str(row["block_id"])
            block_actions = tuple(actions[block_id])
            if len(block_actions) < PLAN_HORIZON + 1:
                raise ValueError(f"donor block is too short: {block_id}")
            provenance_id = str(row["source_provenance_id"])
            values.append(DonorBlock(
                block_id=block_id,
                anchor=anchor,
                cluster_id=str(row["cluster_id"]),
                source_provenance_id=provenance_id,
                source_route_id=str(row["source_route_id"]),
                support=int(row.get("support", 0)),
                actions=block_actions,
                team_name=provenance_teams.get(provenance_id, ""),
            ))
        donors[anchor] = tuple(values)
    return donors, {
        "schema": "filtered-active-donor-subset-v1",
        "block_manifest": {
            "path": str(manifest_path), "sha256": residual._sha256_file(manifest_path),
        },
        "actions": {
            "path": str(actions_path), "sha256": residual._sha256_file(actions_path),
        },
        "provenance_manifests": provenance_inputs,
        "heldout_team_lineages": sorted(set(map(str, heldout_teams))),
        "selection": "existing active medoids, lineage filter, retain support-ordered allowed set",
        "top_k": int(top_k),
        "per_anchor_allowed_before_top_k": {
            str(anchor): sum(
                provenance_teams.get(str(row["source_provenance_id"]), "")
                not in set(map(str, heldout_teams))
                for row in manifest["active_representatives"][str(anchor)]
            )
            for anchor in ANCHORS
        },
        "per_anchor_selected": {str(anchor): len(values) for anchor, values in donors.items()},
    }


def _block_unit_trace(
    actions: Sequence[Mapping[str, Any]], start_offset: int,
    actor_count: int, horizon: int,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    result = []
    for offset in range(start_offset, start_offset + horizon):
        if offset >= len(actions):
            raise ValueError("donor block does not cover the requested H4 window")
        raw = residual.normalize_player_action(actions[offset])
        units = [_unit(raw["farmer"]), *(_unit(value) for value in raw["hands"])]
        units = units[:actor_count]
        units.extend(("PASS",) for _ in range(max(0, actor_count - len(units))))
        result.append(tuple(units))
    return tuple(result)


def _donor_patch(
    base: Sequence[Sequence[Sequence[Any]]],
    donor: Sequence[Sequence[Sequence[Any]]],
    actors: Sequence[int],
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    actor_set = set(map(int, actors))
    return tuple(tuple(
        tuple(donor[offset][actor]) if actor in actor_set else tuple(base[offset][actor])
        for actor in range(len(base[offset]))
    ) for offset in range(len(base)))


def _donor_variants(
    base: Sequence[Sequence[Sequence[Any]]],
    donor: DonorBlock,
    actor_count: int,
    horizon: int,
    start_offset: int,
    donor_rank: int,
    include_all: bool,
) -> list[tuple[
    str, str, tuple[tuple[int, int], ...], str | None, str | None,
    int, int, int, tuple[tuple[tuple[Any, ...], ...], ...],
]]:
    trace = _block_unit_trace(donor.actions, start_offset, actor_count, horizon)
    novelty = []
    movement = []
    for actor in range(min(actor_count, MAX_ACTORS)):
        changed = 0
        moved = False
        for offset in range(horizon):
            before, after = _unit(base[offset][actor]), _unit(trace[offset][actor])
            if before == after:
                continue
            changed += 1
            moved |= str(before[0]) in MOVE_OPS or str(after[0]) in MOVE_OPS
        novelty.append(changed)
        movement.append(moved)
    actors = sorted(
        (
            actor for actor, count in enumerate(novelty)
            if count >= 2 and movement[actor]
        ),
        key=lambda actor: (-novelty[actor], actor),
    )
    pair_choices = sorted(
        combinations(actors, 2),
        key=lambda pair: (-(novelty[pair[0]] + novelty[pair[1]]), pair),
    )[:2]
    single_actors = sorted(
        {actor for pair in pair_choices for actor in pair}
        or set(actors[:2]),
        key=lambda actor: (-novelty[actor], actor),
    )
    values = []
    for actor in single_actors:
        assignments = ((actor, actor),)
        values.append((
            f"SINGLE_A{actor}_{donor.block_id}", "SINGLE", assignments,
            donor.block_id, donor.cluster_id, donor.support, donor_rank,
            novelty[actor], _donor_patch(base, trace, (actor,)),
        ))
    for pair in pair_choices:
        assignments = tuple((actor, actor) for actor in pair)
        values.append((
            f"PAIR_A{pair[0]}_A{pair[1]}_{donor.block_id}", "PAIR",
            assignments, donor.block_id, donor.cluster_id, donor.support,
            donor_rank, sum(novelty[actor] for actor in pair),
            _donor_patch(base, trace, pair),
        ))
    if include_all and len(actors) > 2:
        assignments = tuple((actor, actor) for actor in actors)
        values.append((
            f"ALL_{donor.block_id}", "ALL", assignments, donor.block_id,
            donor.cluster_id, donor.support, donor_rank, sum(novelty),
            _donor_patch(base, trace, actors),
        ))
    return values


def _variant_coverage(
    base: Sequence[Sequence[Sequence[Any]]],
    donors: Sequence[DonorBlock],
    actor_count: int,
    horizon: int,
    start_offset: int,
    include_all: bool,
) -> int:
    return len({
        _canonical_bytes(row[-1])
        for donor_rank, donor in enumerate(donors)
        for row in _donor_variants(
            base, donor, actor_count, horizon, start_offset, donor_rank, include_all,
        )
    })


def select_diverse_donors(
    base_tape: Sequence[Mapping[str, Any]],
    donor_blocks: Sequence[DonorBlock],
    anchor: int,
    actor_count: int,
    *,
    top_k: int = MAX_PAIRS,
    horizon: int = PLAN_HORIZON,
    decision_step: int | None = None,
    include_all: bool = True,
) -> tuple[tuple[DonorBlock, ...], dict[str, Any]]:
    """Use the same budget as support-top-k but maximize unique H4 residuals."""

    if top_k < 1:
        raise ValueError("top_k donors must be positive")
    ordered = tuple(sorted(donor_blocks, key=lambda donor: donor.block_id))
    count = min(top_k, len(ordered))
    step = anchor + 1 if decision_step is None else int(decision_step)
    start_offset = step - anchor
    if start_offset < 1:
        raise ValueError("donor residual starts only after the anchor action")
    base = _unit_trace(base_tape, step, actor_count)
    support = tuple(sorted(
        ordered, key=lambda donor: (-donor.support, donor.block_id),
    )[:count])
    best: tuple[DonorBlock, ...] | None = None
    best_score = (-1, -1)
    for values in combinations(ordered, count):
        score = (
            _variant_coverage(
                base, values, actor_count, horizon, start_offset, include_all,
            ),
            sum(donor.support for donor in values),
        )
        if score > best_score:
            best, best_score = tuple(values), score
    if best is None:
        raise ValueError(f"no allowed donor blocks at anchor {anchor}")
    selected = tuple(sorted(best, key=lambda donor: donor.block_id))
    return selected, {
        "anchor": int(anchor), "allowed": len(ordered), "budget": count,
        "support_top_ids": [donor.block_id for donor in support],
        "support_top_unique_residuals": _variant_coverage(
            base, support, actor_count, horizon, start_offset, include_all,
        ),
        "diversity_ids": [donor.block_id for donor in selected],
        "diversity_unique_residuals": best_score[0],
        "diversity_support": best_score[1],
    }


def donor_candidates(
    base_tape: Sequence[Mapping[str, Any]],
    donor_blocks: Sequence[DonorBlock],
    anchor: int,
    actor_count: int,
    *,
    horizon: int = PLAN_HORIZON,
    decision_step: int | None = None,
    include_all: bool = True,
) -> list[PathCandidate]:
    """Splice donor unit traces into identity-preserving actor slots."""

    if horizon != PLAN_HORIZON:
        raise ValueError("v1 uses a fixed four-step native plan horizon")
    if anchor not in ANCHORS or actor_count < 1:
        raise ValueError("invalid donor anchor or actor count")
    step = anchor + 1 if decision_step is None else int(decision_step)
    start_offset = step - anchor
    base = _unit_trace(base_tape, step, actor_count)
    selected, _ = select_diverse_donors(
        base_tape, donor_blocks, anchor, actor_count, top_k=MAX_PAIRS,
        horizon=horizon, decision_step=step, include_all=include_all,
    )
    raw_values: list[tuple[
        str, str, tuple[tuple[int, int], ...], str | None, str | None,
        int, int, int, tuple[tuple[tuple[Any, ...], ...], ...],
    ]] = [("KEEP", "KEEP", tuple(), None, None, 0, -1, 0, base)]
    for donor_rank, donor in enumerate(selected):
        if donor.anchor != anchor:
            raise ValueError("donor block anchor mismatch")
        raw_values.extend(_donor_variants(
            base, donor, actor_count, horizon, start_offset, donor_rank, include_all,
        ))

    canonical: dict[bytes, list[Any]] = {}
    order: list[bytes] = []
    for code, kind, assignments, donor_id, cluster, support, rank, novelty, units in raw_values:
        payload = _canonical_bytes(units)
        if payload not in canonical:
            canonical[payload] = [
                code, kind, assignments, donor_id, cluster, support, rank,
                novelty, units, [code],
            ]
            order.append(payload)
        else:
            canonical[payload][-1].append(code)
    result = []
    for payload in order:
        (code, kind, assignments, donor_id, cluster, support, rank,
         novelty, units, aliases) = canonical[payload]
        result.append(PathCandidate(
            code=code,
            kind=kind,
            units=units,
            assignments=assignments,
            donor_id=donor_id,
            donor_cluster=cluster,
            donor_support=int(support),
            donor_rank=int(rank),
            trace_novelty=int(novelty),
            aliases=tuple(aliases),
            raw_sha256=hashlib.sha256(payload).hexdigest(),
        ))
    if not result or not result[0].keep or result[0].code != "KEEP":
        raise AssertionError("KEEP must be the first canonical path candidate")
    return result


def effective_candidates(
    candidates: Sequence[PathCandidate], override_horizon: int,
) -> list[PathCandidate]:
    """Drop candidate actions that the controller will never execute."""

    if not candidates or not candidates[0].keep:
        raise ValueError("effective candidates require KEEP first")
    if not 1 <= override_horizon <= PLAN_HORIZON:
        raise ValueError("override horizon must be between one and four")
    if override_horizon == PLAN_HORIZON:
        return list(candidates)
    base = candidates[0].units[:override_horizon]
    canonical: dict[bytes, PathCandidate] = {}
    aliases: dict[bytes, list[str]] = {}
    order: list[bytes] = []
    for candidate in candidates:
        units = candidate.units[:override_horizon]
        payload = _canonical_bytes(units)
        if payload not in canonical:
            novelty = sum(
                _unit(action) != _unit(
                    base[offset][actor] if actor < len(base[offset]) else ("PASS",)
                )
                for offset, actions in enumerate(units)
                for actor, action in enumerate(actions)
            )
            canonical[payload] = PathCandidate(
                code=candidate.code,
                kind=candidate.kind,
                units=units,
                assignments=candidate.assignments,
                donor_id=candidate.donor_id,
                donor_cluster=candidate.donor_cluster,
                donor_support=candidate.donor_support,
                donor_rank=candidate.donor_rank,
                trace_novelty=novelty,
                aliases=(),
                raw_sha256=hashlib.sha256(_canonical_bytes({
                    "override_horizon": override_horizon, "units": units,
                })).hexdigest(),
            )
            aliases[payload] = []
            order.append(payload)
        aliases[payload].extend(candidate.aliases)
    return [
        PathCandidate(**{
            **canonical[payload].__dict__,
            "aliases": tuple(dict.fromkeys(aliases[payload])),
        })
        for payload in order
    ]


def pack_unit_plans(
    candidates: Sequence[PathCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("candidate batch is empty")
    max_units = max(
        1,
        max(len(units) for candidate in candidates for units in candidate.units),
    )
    packed = np.zeros(
        (len(candidates), PLAN_HORIZON, max_units, 3), dtype=np.int32,
    )
    counts = np.full((len(candidates), PLAN_HORIZON), -1, dtype=np.int32)
    for candidate_index, candidate in enumerate(candidates):
        for offset, units in enumerate(candidate.units):
            if candidate.keep:
                continue
            counts[candidate_index, offset] = len(units)
            for actor, action in enumerate(units):
                packed[candidate_index, offset, actor] = residual._packed_action(action)
    return packed, counts


def _assignment_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_active",
        *(f"{prefix}_runtime_actor_{actor}" for actor in range(MAX_ACTORS)),
        *(f"{prefix}_source_actor_{actor}" for actor in range(MAX_ACTORS)),
    ]


def feature_names() -> tuple[str, ...]:
    names = [*route_switch_feature_names()]
    names.extend([
        *(f"self_tile_{index}" for index in range(100)),
        *(f"self_unlock_{index}" for index in range(4)),
        *(f"opponent_tile_{index}" for index in range(100)),
        *(f"opponent_unlock_{index}" for index in range(4)),
    ])
    for actor in range(MAX_ACTORS):
        names.extend((
            f"actor_{actor}_current_x", f"actor_{actor}_current_y",
            f"actor_{actor}_current_valid", f"actor_{actor}_source_x",
            f"actor_{actor}_source_y", f"actor_{actor}_source_valid",
        ))
        names.extend(
            f"actor_{actor}_inventory_{item.lower()}" for item in residual.ITEMS
        )
    for offset in range(PLAN_HORIZON):
        for actor in range(MAX_ACTORS):
            prefix = f"plan_{offset}_actor_{actor}"
            names.extend(f"{prefix}_op_{op.lower()}" for op in residual.OPS[:18])
            names.extend(f"{prefix}_item_{item.lower()}" for item in ITEM_FEATURES)
            names.extend((f"{prefix}_quantity", f"{prefix}_changed"))
    names.extend(f"kind_{kind.lower()}" for kind in KIND_CODES)
    names.extend(_assignment_names("assignment_0"))
    names.extend(_assignment_names("assignment_1"))
    names.extend((
        "donor_support_log", *(f"donor_rank_{index}" for index in range(MAX_PAIRS)),
        *(f"donor_cluster_c{index:02d}" for index in range(8)),
        "trace_novelty", "changed_actor_count", "runtime_actor_count",
        "source_actor_count", "seat", "segment", "segment_offset",
    ))
    return tuple(names)


FEATURE_NAMES = feature_names()


def _actor_features(
    observation: Mapping[str, Any],
    source_positions: Sequence[tuple[int, int] | None],
) -> np.ndarray:
    current = _positions(observation)
    values: list[float] = []
    for actor in range(MAX_ACTORS):
        if actor < len(current):
            values.extend((current[actor][0] / 9.0, current[actor][1] / 9.0, 1.0))
        else:
            values.extend((-1.0, -1.0, 0.0))
        source = source_positions[actor] if actor < len(source_positions) else None
        if source is None:
            values.extend((-1.0, -1.0, 0.0))
        else:
            values.extend((source[0] / 9.0, source[1] / 9.0, 1.0))
        inventory = _inventory(observation, actor)
        values.extend(np.log1p(float(inventory.get(item, 0))) for item in residual.ITEMS)
    return np.asarray(values, np.float32)


def _trace_features(
    candidate: PathCandidate,
    base: Sequence[Sequence[Sequence[Any]]],
) -> np.ndarray:
    values: list[float] = []
    for offset in range(PLAN_HORIZON):
        for actor in range(MAX_ACTORS):
            baseline = base[offset][actor] if actor < len(base[offset]) else ("PASS",)
            if offset < len(candidate.units):
                actions = candidate.units[offset]
                action = actions[actor] if actor < len(actions) else ("PASS",)
            else:
                action = baseline
            op, item, quantity = residual._packed_action(action)
            op_values = [0.0] * 18
            op_values[op] = 1.0
            item_values = [0.0] * len(ITEM_FEATURES)
            item_values[item if item >= 0 else len(ITEM_FEATURES) - 1] = 1.0
            values.extend(op_values)
            values.extend(item_values)
            values.extend((np.log1p(quantity), float(_unit(action) != _unit(baseline))))
    return np.asarray(values, np.float32)


def _candidate_metadata(
    candidate: PathCandidate,
    runtime_count: int,
    source_count: int,
    seat: int,
    step: int,
) -> np.ndarray:
    values = [float(index == KIND_INDEX[candidate.kind]) for index in range(len(KIND_CODES))]
    for assignment_index in range(2):
        active = assignment_index < len(candidate.assignments)
        values.append(float(active))
        target, source = candidate.assignments[assignment_index] if active else (-1, -1)
        values.extend(float(active and actor == target) for actor in range(MAX_ACTORS))
        values.extend(float(active and actor == source) for actor in range(MAX_ACTORS))
    segment = int(np.searchsorted(STOPS, step, side="right"))
    values.append(np.log1p(candidate.donor_support))
    values.extend(float(candidate.donor_rank == index) for index in range(MAX_PAIRS))
    values.extend(
        float(candidate.donor_cluster == f"C{index:02d}") for index in range(8)
    )
    values.extend((
        candidate.trace_novelty / float(2 * max(1, len(candidate.units))),
        len(candidate.assignments) / 2.0,
        min(runtime_count, MAX_ACTORS) / float(MAX_ACTORS),
        min(source_count, MAX_ACTORS) / float(MAX_ACTORS),
        float(seat),
        segment / max(1, len(ANCHORS) - 1),
        (step - ANCHORS[segment]) / 23.0,
    ))
    return np.asarray(values, np.float32)


def candidate_feature_rows(
    observation: Mapping[str, Any],
    history: RouteSwitchHistory,
    plan_tape: Sequence[Mapping[str, Any]],
    route_tape: Sequence[Mapping[str, Any]],
    source_positions: Sequence[tuple[int, int] | None],
    candidates: Sequence[PathCandidate],
    seat: int,
    step: int,
) -> np.ndarray:
    if int(observation.get("step", -1)) != int(step):
        raise ValueError("observation and path feature step disagree")
    if not candidates:
        raise ValueError("cannot encode an empty path candidate set")
    runtime = _positions(observation)
    base = _unit_trace(route_tape, step, max(1, len(runtime)))
    common = np.concatenate((
        route_switch_vector(observation, history, plan_tape),
        residual._layouts(observation, seat),
        _actor_features(observation, source_positions),
    )).astype(np.float32)
    rows = [
        np.concatenate((
            common,
            _trace_features(candidate, base),
            _candidate_metadata(
                candidate, len(runtime), len(source_positions), seat, step,
            ),
        ))
        for candidate in candidates
    ]
    result = np.stack(rows).astype(np.float32)
    if result.shape != (len(candidates), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("invalid fixed-slot path feature rows")
    return result


def _rollout_unit_plans(
    bundle: NativeTeammateBundle,
    env: Any,
    states: Sequence[Any],
    schedule: np.ndarray,
    stops: np.ndarray,
    seat: int,
    candidates: Sequence[PathCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    units, counts = pack_unit_plans(candidates)
    raw = bundle.executor.rollout_schedule_unit_override_sequence_batch(
        env, states[0], states[1], schedule, stops, seat, units, counts,
    )
    if isinstance(raw, tuple):
        rewards, market_diff = raw
    else:
        rewards, market_diff = raw, np.zeros(len(candidates), np.int32)
    rewards = np.asarray(rewards, np.float64)
    market_diff = np.asarray(market_diff, np.int32)
    if rewards.shape != (len(candidates), 2) or not np.isfinite(rewards).all():
        raise RuntimeError("native path rollout returned invalid rewards")
    if market_diff.shape != (len(candidates),) or np.any(market_diff):
        raise RuntimeError("unit-only path rollout changed a market action")
    return rewards, market_diff


def _candidate_units(candidate: PathCandidate, offset: int) -> dict[str, Any] | None:
    if candidate.keep:
        return None
    units = candidate.units[offset]
    return {
        "farmer": list(units[0]),
        "hands": [list(action) for action in units[1:]],
    }


def _commit_units(
    bundle: NativeTeammateBundle,
    env: Any,
    states: Sequence[Any],
    route_pair: Sequence[int],
    seat: int,
    candidate: PathCandidate,
    offset: int,
) -> None:
    actions = []
    for player in (0, 1):
        if player == seat:
            actions.append(bundle.executor.action_at_with_unit_override(
                env, player, int(route_pair[player]), states[player],
                _candidate_units(candidate, offset),
            ))
        else:
            actions.append(bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            ))
    env.step(actions)


def _strict_best(rewards: np.ndarray, seat: int) -> int:
    best = 0
    rank = adaptive.reward_rank(rewards[0], seat)
    for index in range(1, len(rewards)):
        candidate_rank = adaptive.reward_rank(rewards[index], seat)
        if candidate_rank > rank:
            best, rank = index, candidate_rank
    return best


def _pair_synergy(
    candidates: Sequence[PathCandidate], rewards: np.ndarray, seat: int,
) -> list[str]:
    ranks = [adaptive.reward_rank(rewards[index], seat) for index in range(len(candidates))]
    keep_rank = ranks[0]
    singles = {
        (candidate.donor_id, candidate.assignments[0][0]): ranks[index]
        for index, candidate in enumerate(candidates)
        if candidate.kind == "SINGLE" and len(candidate.assignments) == 1
    }
    result = []
    for index, candidate in enumerate(candidates):
        if candidate.kind != "PAIR":
            continue
        actor_ranks = [
            singles.get((candidate.donor_id, actor))
            for actor, _ in candidate.assignments
        ]
        if all(rank is not None for rank in actor_ranks) and ranks[index] > max(
            keep_rank, *(rank for rank in actor_ranks if rank is not None),
        ):
            result.append(candidate.code)
    return result


def _source_positions(
    carrier: CarrierRoute | None, step: int,
) -> tuple[tuple[int, int] | None, ...]:
    return tuple() if carrier is None else tuple(carrier.actor_positions[step])


def _scenario_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["genome_id"]), str(row["opponent"]),
        int(row["seed"]), int(row["seat"]),
    )


def _paired_rows(
    arm: Sequence[Mapping[str, Any]], fixed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = {_scenario_key(row): row for row in fixed}
    result = []
    for row in arm:
        base = reference[_scenario_key(row)]
        value = dict(row)
        value.update({
            "fixed_outcome": int(base["outcome"]),
            "fixed_margin": float(base["margin"]),
            "delta_margin_vs_fixed": float(row["margin"]) - float(base["margin"]),
            "loss_repaired_to_win": int(base["outcome"]) == 0 and int(row["outcome"]) == 2,
            "outcome_regression_vs_fixed": int(row["outcome"]) < int(base["outcome"]),
        })
        result.append(value)
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return residual._summary(rows)


def _selected_genome_ids(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(dict.fromkeys(filter(None, (
        str(args.base_genome_id),
        str(args.composition_genome_id) if args.composition_genome_id else None,
    ))))


def _load_carriers(
    prepared_roots: Sequence[Path],
    route_ids: Iterable[str],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, CarrierRoute | None], dict[str, Any]]:
    representative: dict[str, Mapping[str, Any]] = {}
    signatures = []
    for root in prepared_roots:
        manifest_path = root.resolve() / "candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        routes = {str(row["route_id"]): row for row in manifest.get("routes", ())}
        provenance = {
            str(row["provenance_id"]): row for row in manifest.get("provenance", ())
        }
        for route_id, row in routes.items():
            source_id = str(row["representative_source_id"])
            if source_id in provenance:
                representative.setdefault(route_id, provenance[source_id])
        signatures.append({
            "path": str(manifest_path),
            "sha256": residual._sha256_file(manifest_path),
        })
    carriers: dict[str, CarrierRoute | None] = {}
    carrier_manifest = {}
    for route_id in sorted(set(map(str, route_ids))):
        row = representative.get(route_id)
        if row is None:
            carriers[route_id] = None
            carrier_manifest[route_id] = {"kind": "no_replay_provenance"}
            continue
        replay_path = Path(str(row["replay_path"])).resolve()
        if replay_path.stat().st_size != int(row["replay_bytes"]):
            raise ValueError(f"representative replay byte size changed: {route_id}")
        carrier = CarrierRoute.from_replay(replay_path, int(row["player_index"]))
        if _sha256(list(carrier.actions[ANCHORS[0]:])) != _sha256(
            list(tapes[route_id][ANCHORS[0]:]),
        ):
            raise ValueError(f"representative replay tape disagrees with route: {route_id}")
        carriers[route_id] = carrier
        carrier_manifest[route_id] = {
            "provenance_id": str(row["provenance_id"]),
            "replay_path": str(replay_path),
            "replay_bytes": replay_path.stat().st_size,
            "player_index": int(row["player_index"]),
        }
    return carriers, {"prepared_manifests": signatures, "routes": carrier_manifest}


def load_experiment(args: argparse.Namespace) -> tuple[
    NativeTeammateBundle, str, Sequence[Mapping[str, Any]],
    dict[str, Sequence[Mapping[str, Any]]], dict[str, tuple[str, ...]],
    dict[str, CarrierRoute | None], dict[int, tuple[DonorBlock, ...]],
    dict[str, Any], dict[str, Any],
]:
    genome_ids = _selected_genome_ids(args)
    shadow = argparse.Namespace(
        frozen_run=args.frozen_run,
        prepared_root=args.prepared_root,
        v1_root=args.v1_root,
        opponent=list(genome_ids),
    )
    _, baseline_id, baseline_tape, tapes, genomes, inputs, source = (
        residual.load_experiment(shadow)
    )
    prepared = args.prepared_root or [
        adaptive.DEFAULT_OLD_PREPARED, adaptive.DEFAULT_NEW_PREPARED,
    ]
    selected_routes = {
        route_id for genome in genomes.values() for route_id in genome
    }
    carriers, carrier_inputs = _load_carriers(
        [Path(value) for value in prepared], selected_routes, tapes,
    )
    donors, donor_inputs = load_filtered_active_donors(
        args.block_library,
        args.heldout_team or DEFAULT_HELDOUT_TEAMS,
        top_k=MAX_ACTIVE_DONORS,
    )
    _, _, _, _, v1_args, _ = continuation.load_frozen_v1_openings(args.v1_root)
    opponents = tuple(dict.fromkeys((
        *(args.train_opponent or TRAIN_OPPONENTS),
        *(args.validation_opponent or HELDOUT_OPPONENTS),
    )))
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=tapes,
        included_families=tuple((*opponents, *tapes)),
    )
    for method in (
        "action_at_with_unit_override",
        "rollout_schedule_unit_override_sequence_batch",
        "rollout_schedule_batch",
    ):
        if not hasattr(bundle.executor, method):
            raise RuntimeError(f"native extension is missing {method}")
    inputs = dict(inputs)
    inputs.update({
        "carrier_replays": carrier_inputs,
        "donor_library": donor_inputs,
        "native_extension": {
            "path": str(Path(native_extension.__file__).resolve()),
            "sha256": residual._sha256_file(Path(native_extension.__file__).resolve()),
        },
        "fast_package": {
            "path": str(Path(fast_kaggriculture.__file__).resolve()),
            "sha256": residual._sha256_file(Path(fast_kaggriculture.__file__).resolve()),
        },
    })
    return (
        bundle, baseline_id, baseline_tape, tapes, genomes, carriers, donors,
        inputs, source,
    )


def _fixed(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    opponent: str,
    genome_id: str,
    genome: Sequence[str],
    seed: int,
    seat: int,
    split: str,
) -> dict[str, Any]:
    row = residual.fixed_result(
        bundle, baseline_id, opponent, genome, _sha256(list(genome)), seed, seat,
    )
    row.update(genome_id=genome_id, split=split)
    return row


def _route_context(
    genome: Sequence[str], tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, CarrierRoute | None], observation: Mapping[str, Any],
    donors: Mapping[int, Sequence[DonorBlock]], step: int,
    override_horizon: int = COMMIT4_OVERRIDE_HORIZON,
) -> tuple[int, str, Sequence[Mapping[str, Any]], CarrierRoute | None,
           tuple[tuple[int, int] | None, ...], list[PathCandidate]]:
    segment = int(np.searchsorted(STOPS, step, side="right"))
    route_id = str(genome[segment])
    tape = tapes[route_id]
    carrier = carriers.get(route_id)
    source = _source_positions(carrier, step)
    anchor = int(ANCHORS[segment])
    candidates = effective_candidates(donor_candidates(
        tape, donors[anchor], anchor, max(1, len(_positions(observation))),
        decision_step=step,
    ), override_horizon)
    return segment, route_id, tape, carrier, source, candidates


def training_label_contract(
    step: int, max_windows: int,
) -> tuple[tuple[str, int], ...]:
    anchors = ANCHORS[:max_windows]
    values: list[tuple[str, int]] = []
    if step in {anchor + 1 for anchor in anchors}:
        values.append(("commit4_h4", COMMIT4_OVERRIDE_HORIZON))
    if step in {
        anchor + offset for anchor in anchors for offset in WINDOW_OFFSETS
        if anchor + offset < HORIZON
    }:
        values.append(("mpc1_h1", MPC1_OVERRIDE_HORIZON))
    return tuple(values)


def collect_frozen_labels(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, CarrierRoute | None],
    donors: Mapping[int, Sequence[DonorBlock]],
    opponent: str,
    opponent_index: int,
    genome_id: str,
    genome_index: int,
    genome: Sequence[str],
    seed: int,
    seat: int,
    h4_panel: PathPanel,
    h1_panel: PathPanel,
    h4_decision_start: int,
    h1_decision_start: int,
    max_windows: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, Any],
]:
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(baseline_tape, tapes, genome)
    h4_decisions: list[dict[str, Any]] = []
    h1_decisions: list[dict[str, Any]] = []
    audits = {
        "commit4_h4": {
            "override_horizon": COMMIT4_OVERRIDE_HORIZON,
            "keep_equivalence_checks": 0,
            "market_unchanged_checks": 0,
            "pair_synergy_events": 0,
            "candidate_rollouts": 0,
            "label_decisions": 0,
        },
        "mpc1_h1": {
            "override_horizon": MPC1_OVERRIDE_HORIZON,
            "keep_equivalence_checks": 0,
            "market_unchanged_checks": 0,
            "pair_synergy_events": 0,
            "candidate_rollouts": 0,
            "label_decisions": 0,
        },
    }
    h4_decision_id = h4_decision_start
    h1_decision_id = h1_decision_start

    def collect_at_horizon(
        observation: Mapping[str, Any], step: int, override_horizon: int,
        panel: PathPanel, decision_id: int, controller: str,
    ) -> dict[str, Any]:
        segment, route_id, tape, _, source, candidates = _route_context(
            genome, tapes, carriers, observation, donors, step, override_horizon,
        )
        rewards, market_diff = _rollout_unit_plans(
            bundle, env, states, schedule[segment:], STOPS[segment:], seat,
            candidates,
        )
        reference = np.asarray(bundle.executor.rollout_schedule_batch(
            env, states[0], states[1], schedule[segment:][None, :, :],
            STOPS[segment:],
        ), np.float64)[0]
        if not np.array_equal(rewards[0], reference):
            raise RuntimeError(f"{controller} KEEP disagrees with frozen schedule")
        best = _strict_best(rewards, seat)
        keep_margin = adaptive.reward_rank(rewards[0], seat)[1]
        feature_rows = candidate_feature_rows(
            observation, history, plan_tape, tape, source, candidates, seat, step,
        )
        synergy_codes = _pair_synergy(candidates, rewards, seat)
        for index, candidate in enumerate(candidates):
            margin = adaptive.reward_rank(rewards[index], seat)[1]
            delta = margin - keep_margin
            panel.features.append(feature_rows[index])
            panel.target.append(residual._signed_log(delta))
            panel.delta_margin.append(delta)
            panel.rewards.append(tuple(map(float, rewards[index])))
            panel.seed.append(int(seed))
            panel.seat.append(int(seat))
            panel.opponent.append(int(opponent_index))
            panel.step.append(int(step))
            panel.edit.append(KIND_INDEX[candidate.kind])
            panel.decision.append(decision_id)
            panel.selected.append(index == best)
            panel.genome.append(int(genome_index))
            panel.candidate_sha256.append(candidate.raw_sha256)
        audit = audits[controller]
        audit["keep_equivalence_checks"] += 1
        audit["market_unchanged_checks"] += len(market_diff)
        audit["pair_synergy_events"] += len(synergy_codes)
        audit["candidate_rollouts"] += len(candidates)
        audit["label_decisions"] += 1
        return {
            "split": "train_labels_only", "label_controller": controller,
            "target_override_horizon": override_horizon,
            "genome_id": genome_id, "opponent": opponent,
            "seed": int(seed), "seat": int(seat), "step": int(step),
            "segment": segment, "route_id": route_id,
            "candidate_codes": [candidate.code for candidate in candidates],
            "candidate_aliases": [list(candidate.aliases) for candidate in candidates],
            "candidate_sha256": [candidate.raw_sha256 for candidate in candidates],
            "terminal_rewards": rewards.tolist(), "selected_index": best,
            "selected_code": candidates[best].code,
            "pair_synergy_codes": synergy_codes, "committed": "KEEP",
        }

    for step in range(ANCHORS[0], HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(STOPS, step, side="right"))
        for controller, override_horizon in training_label_contract(
            step, max_windows,
        ):
            if controller == "commit4_h4":
                h4_decisions.append(collect_at_horizon(
                    observation, step, override_horizon,
                    h4_panel, h4_decision_id, controller,
                ))
                h4_decision_id += 1
            else:
                h1_decisions.append(collect_at_horizon(
                    observation, step, override_horizon,
                    h1_panel, h1_decision_id, controller,
                ))
                h1_decision_id += 1
        route_pair = schedule[segment]
        actions = [
            bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            )
            for player in (0, 1)
        ]
        env.step(actions)
    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("frozen label trajectory did not complete")
    return h4_decisions, h1_decisions, h4_decision_id, h1_decision_id, {
        "genome_id": genome_id, "opponent": opponent, "seed": int(seed),
        "seat": int(seat), "controllers": audits, "completed": True,
    }


def _choose_model(
    model: Any,
    calibration: Mapping[str, Any],
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    return residual.model_choice(model, values, calibration)


def _validate_ranker_contract(mode: str, manifest: Mapping[str, Any]) -> int:
    expected = {
        "commit4": ("commit4_h4", COMMIT4_OVERRIDE_HORIZON),
        "mpc1": ("mpc1_h1", MPC1_OVERRIDE_HORIZON),
    }
    if mode not in expected:
        raise ValueError("learned path mode must be commit4 or mpc1")
    controller, horizon = expected[mode]
    if (
        str(manifest.get("controller")) != controller
        or int(manifest.get("training_override_horizon", -1)) != horizon
    ):
        raise ValueError(f"{mode} ranker does not match its controller horizon")
    return horizon


def learned_scenario(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, CarrierRoute | None],
    donors: Mapping[int, Sequence[DonorBlock]],
    opponent: str,
    genome_id: str,
    genome: Sequence[str],
    seed: int,
    seat: int,
    model: Any,
    calibration: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    mode: str,
    max_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranker_target_horizon = _validate_ranker_contract(mode, model_manifest)
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(baseline_tape, tapes, genome)
    active_steps = frozenset(
        anchor + offset
        for anchor in ANCHORS[:max_windows]
        for offset in WINDOW_OFFSETS
        if anchor + offset < HORIZON
    )
    window_starts = frozenset(anchor + 1 for anchor in ANCHORS[:max_windows])
    pending: tuple[PathCandidate, int] | None = None
    decisions = []
    edits = edited_steps = 0
    for step in range(ANCHORS[0], HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(STOPS, step, side="right"))
        candidate = None
        commit_offset = 0
        decide = step in active_steps if mode == "mpc1" else step in window_starts
        if decide:
            override_horizon = (
                MPC1_OVERRIDE_HORIZON if mode == "mpc1"
                else COMMIT4_OVERRIDE_HORIZON
            )
            segment, route_id, tape, _, source, candidates = _route_context(
                genome, tapes, carriers, observation, donors, step,
                override_horizon,
            )
            values = candidate_feature_rows(
                observation, history, plan_tape, tape, source, candidates, seat, step,
            )
            chosen, mean, std, positive = _choose_model(model, calibration, values)
            candidate = candidates[chosen]
            edits += int(not candidate.keep)
            decisions.append({
                "genome_id": genome_id, "opponent": opponent, "seed": int(seed),
                "seat": int(seat), "step": int(step), "mode": mode,
                "ranker_target_horizon": ranker_target_horizon,
                "ranker_model_sha256": str(model_manifest["model_sha256"]),
                "score_override_horizon": override_horizon,
                "execute_horizon": (
                    MPC1_OVERRIDE_HORIZON if mode == "mpc1"
                    else COMMIT4_OVERRIDE_HORIZON
                ),
                "route_id": route_id,
                "candidate_codes": [value.code for value in candidates],
                "selected_code": candidate.code,
                "selected_sha256": candidate.raw_sha256,
                "prediction_mean": mean.tolist(), "prediction_std": std.tolist(),
                "positive_tree_fraction": positive.tolist(),
            })
            pending = (candidate, 0) if mode == "commit4" and not candidate.keep else None
        elif mode == "commit4" and pending is not None:
            candidate, commit_offset = pending
        route_pair = schedule[segment]
        if candidate is None or candidate.keep:
            actions = [
                bundle.executor.action_at(
                    env, player, int(route_pair[player]), states[player],
                )
                for player in (0, 1)
            ]
            env.step(actions)
        else:
            _commit_units(
                bundle, env, states, route_pair, seat, candidate, commit_offset,
            )
            edited_steps += 1
        if mode == "commit4" and pending is not None:
            pending = None if pending[1] + 1 >= PLAN_HORIZON else (pending[0], pending[1] + 1)
    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("learned path trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    outcome, margin = adaptive.reward_rank(rewards, seat)
    return ({
        "arm": f"learned_{mode}", "split": "validation",
        "genome_id": genome_id, "opponent": opponent, "seed": int(seed),
        "seat": int(seat), "rewards": list(rewards), "outcome": outcome,
        "margin": margin, "edits": edits, "edited_steps": edited_steps,
        "genome_sha256": _sha256(list(genome)), "route_change_count": 0,
        "completed": True,
    }, decisions)


def oracle_scenario(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, CarrierRoute | None],
    donors: Mapping[int, Sequence[DonorBlock]],
    opponent: str,
    genome_id: str,
    genome: Sequence[str],
    seed: int,
    seat: int,
    fixed: Mapping[str, Any],
    max_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    active_steps = frozenset(
        anchor + offset
        for anchor in ANCHORS[:max_windows]
        for offset in WINDOW_OFFSETS
        if anchor + offset < HORIZON
    )
    decisions = []
    edits = keep_checks = market_checks = synergy = 0
    for step in range(ANCHORS[0], HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(STOPS, step, side="right"))
        candidate = None
        if step in active_steps:
            segment, route_id, _, _, _, candidates = _route_context(
                genome, tapes, carriers, observation, donors, step,
                MPC1_OVERRIDE_HORIZON,
            )
            rewards, market_diff = _rollout_unit_plans(
                bundle, env, states, schedule[segment:], STOPS[segment:], seat,
                candidates,
            )
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                STOPS[segment:],
            ), np.float64)[0]
            if not np.array_equal(rewards[0], reference):
                raise RuntimeError("oracle KEEP disagrees with frozen schedule")
            best = _strict_best(rewards, seat)
            candidate = candidates[best]
            synergy_codes = _pair_synergy(candidates, rewards, seat)
            synergy += len(synergy_codes)
            edits += int(not candidate.keep)
            keep_checks += 1
            market_checks += len(market_diff)
            decisions.append({
                "split": "validation_oracle_only", "genome_id": genome_id,
                "opponent": opponent, "seed": int(seed), "seat": int(seat),
                "step": int(step), "route_id": route_id,
                "score_override_horizon": MPC1_OVERRIDE_HORIZON,
                "execute_horizon": MPC1_OVERRIDE_HORIZON,
                "candidate_codes": [value.code for value in candidates],
                "terminal_rewards": rewards.tolist(), "selected_index": best,
                "selected_code": candidate.code,
                "pair_synergy_codes": synergy_codes,
            })
        route_pair = schedule[segment]
        if candidate is None or candidate.keep:
            actions = [
                bundle.executor.action_at(
                    env, player, int(route_pair[player]), states[player],
                )
                for player in (0, 1)
            ]
            env.step(actions)
        else:
            _commit_units(bundle, env, states, route_pair, seat, candidate, 0)
    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("oracle MPC path trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    outcome, margin = adaptive.reward_rank(rewards, seat)
    if (outcome, margin) < (int(fixed["outcome"]), float(fixed["margin"])):
        raise RuntimeError("strict MPC oracle regressed against frozen route")
    return ({
        "arm": "oracle_mpc1", "split": "validation_oracle_only",
        "genome_id": genome_id, "opponent": opponent, "seed": int(seed),
        "seat": int(seat), "rewards": list(rewards), "outcome": outcome,
        "margin": margin, "edits": edits, "genome_sha256": _sha256(list(genome)),
        "route_change_count": 0, "keep_equivalence_checks": keep_checks,
        "market_unchanged_checks": market_checks, "pair_synergy_events": synergy,
        "completed": True,
    }, decisions)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    residual._atomic_text(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    return residual._artifact(path, rows)


def _fit_ranker_artifacts(
    output: Path,
    panel: PathPanel,
    controller: str,
    override_horizon: int,
    decision_offsets: Sequence[int],
    args: argparse.Namespace,
    split_path: Path,
    vocabulary_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], tuple[Path, ...]]:
    contracts = {
        "commit4_h4": COMMIT4_OVERRIDE_HORIZON,
        "mpc1_h1": MPC1_OVERRIDE_HORIZON,
    }
    if controller not in contracts:
        raise ValueError("unknown ranker controller")
    suffix = controller
    expected_horizon = contracts[controller]
    if override_horizon != expected_horizon:
        raise ValueError("ranker artifact controller and target horizon disagree")
    arrays = panel.arrays()
    dataset_path = output / f"training_panel_{suffix}.npz"
    model_path = output / f"path_ranker_{suffix}.joblib"
    oof_path = output / f"oof_predictions_{suffix}.npz"
    manifest_path = output / f"model_manifest_{suffix}.json"
    _atomic_npz(dataset_path, arrays)
    model_seed = int(args.random_seed) + int(override_horizon == MPC1_OVERRIDE_HORIZON)
    model, calibration, oof = residual.fit_ranker(
        arrays, args.trees, args.cv_trees, args.cv_folds, model_seed,
    )
    _atomic_joblib(model_path, model)
    _atomic_npz(oof_path, oof)
    manifest = {
        "schema": "multi-farmer-path-ranker-v1",
        "controller": controller,
        "model": "ExtraTreesRegressor candidate-conditioned signed-log terminal margin delta",
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "training_rows": len(arrays["features"]),
        "training_decisions": int(len(np.unique(arrays["decision"]))),
        "training_override_horizon": int(override_horizon),
        "training_decision_offsets": list(map(int, decision_offsets)),
        "target_contract": (
            "terminal margin delta after exactly training_override_horizon "
            "unit-only actions, then the frozen route schedule"
        ),
        "model_random_seed": model_seed,
        "calibration": calibration,
        "model_sha256": residual._sha256_file(model_path),
        "training_panel_sha256": residual._sha256_file(dataset_path),
        "oof_predictions_sha256": residual._sha256_file(oof_path),
        "split_manifest_sha256": residual._sha256_file(split_path),
        "vocabulary_sha256": residual._sha256_file(vocabulary_path),
        "frozen_before_validation_oracle": True,
    }
    residual._atomic_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return model, manifest, arrays, (
        dataset_path, model_path, oof_path, manifest_path,
    )


def _report_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    lines = []
    for arm in ("fixed", "commit4", "mpc1", "oracle"):
        row = validation[arm]
        states = int(row["states"])
        wins = round(states * float(row.get("raw_win_rate", 0.0)))
        lines.append(
            f"| {arm} | {wins}/{states} ({100 * float(row.get('raw_win_rate', 0.0)):.2f}%) "
            f"| {float(row.get('mean_margin', 0.0)):.2f} "
            f"| {float(row.get('mean_delta_margin_vs_fixed', 0.0)):+.2f} "
            f"| {int(row.get('edits', 0))} "
            f"| {int(row.get('outcome_regressions_vs_fixed', 0))} |"
        )
    gates = "\n".join(
        f"- [{'x' if value else ' '}] `{name}`"
        for name, value in report["gates"].items()
    )
    return (
        "# MULTI-FARMER-PATH-RESIDUAL-v1\n\n"
        f"Status: `{report['status']}`\n\n"
        "This proxy freezes the 21-gene route and its market actions, then "
        "splices same-identity actor traces from filtered replay-block donors. It compares "
        "open-loop score-4/execute-4 commit4 with score-1/execute-1/replan MPC1.\n\n"
        "Commit4 is trained only on H4 labels at anchor+1; MPC1 has an independent "
        "H1 model trained at anchor+1 through anchor+4.\n\n"
        "| arm | wins | mean margin | delta vs fixed | edits | regressions |\n"
        "|---|---:|---:|---:|---:|---:|\n" + "\n".join(lines) + "\n\n"
        "## Gates\n\n" + gates + "\n\n"
        "## Evidence boundary\n\n"
        "These are native simulator proxy opponents, not submitted-agent-vs-agent "
        "matches. The result cannot establish the requested full-pool win-rate gate. "
        "Validation opponents and seeds are held out, and the model artifacts are "
        "hashed before the validation oracle is opened.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    train_opponents = tuple(args.train_opponent or TRAIN_OPPONENTS)
    validation_opponents = tuple(args.validation_opponent or HELDOUT_OPPONENTS)
    train_seeds = tuple(range(args.train_seed_start, args.train_seed_start + args.train_seed_count))
    validation_seeds = tuple(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_seed_count,
    ))
    if set(train_opponents) & set(validation_opponents):
        raise ValueError("train and validation opponents must be disjoint")
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("train and validation seeds must be disjoint")
    if (set(train_seeds) | set(validation_seeds)) & residual.SEALED_SEEDS:
        raise ValueError("path experiment cannot open sealed route-search seeds")
    genome_ids = _selected_genome_ids(args)
    split = {
        "schema": "multi-farmer-path-residual-split-v1",
        "train_opponents": list(train_opponents),
        "validation_opponents": list(validation_opponents),
        "heldout_replay_team_lineages": list(args.heldout_team or DEFAULT_HELDOUT_TEAMS),
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "genome_ids": list(genome_ids), "seats": [0, 1],
        "grouping": "seed keeps every train opponent, seat, window offset and candidate together",
        "model_selection": "train OOF only; validation oracle opened after frozen model",
        "training_label_contract": {
            "commit4_h4": "H4 at anchor+1 only",
            "mpc1_h1": "H1 at anchor+1 through anchor+4",
        },
        "sealed_test_executed": False,
    }
    split["sha256"] = _sha256(split)
    vocabulary = {
        "schema": "multi-farmer-path-residual-vocabulary-v1",
        "horizon": PLAN_HORIZON, "max_actor_slots": MAX_ACTORS,
        "max_ranked_pairs": MAX_PAIRS,
        "codes": {
            "KEEP": "no unit override",
            "SINGLE_Ai_DONOR": "actor i takes the donor block's same-identity H4 trace",
            "PAIR_Ai_Aj_DONOR": "actors i and j jointly take the donor H4 traces",
            "ALL_DONOR": "all eligible decision-time actors take donor H4 traces",
        },
        "admission": "every replaced actor differs on >=2 offsets and includes a movement change",
        "donor_retrieval": "heldout-team filter, then diversity-maximizing K<=3 active medoids",
        "pair_ranking": "top two eligible actor pairs by joint H4 novelty",
        "market": "not representable by this vocabulary or native API",
        "candidate_dedup": "canonical effective trace per controller horizon; KEEP wins identical aliases",
        "controller_horizons": {
            "commit4": {"score": 4, "execute": 4},
            "mpc1": {"score": 1, "execute": 1, "replan": 1},
        },
    }
    (bundle, baseline_id, baseline_tape, tapes, genomes, carriers, donors,
     inputs, source) = load_experiment(args)
    output.mkdir(parents=True, exist_ok=False)
    split_path = output / "split_manifest.json"
    vocabulary_path = output / "path_residual_vocabulary.json"
    residual._atomic_text(split_path, json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(vocabulary_path, json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n")

    h4_panel = PathPanel.empty()
    h1_panel = PathPanel.empty()
    train_fixed: list[dict[str, Any]] = []
    train_h4_decisions: list[dict[str, Any]] = []
    train_h1_decisions: list[dict[str, Any]] = []
    train_audits: list[dict[str, Any]] = []
    h4_decision_id = 0
    h1_decision_id = 0
    train_scenarios = [
        (genome_id, opponent, seed, seat)
        for genome_id in genome_ids for seed in train_seeds
        for opponent in train_opponents for seat in (0, 1)
    ]
    for index, (genome_id, opponent, seed, seat) in enumerate(train_scenarios, 1):
        genome = genomes[genome_id]
        fixed = _fixed(
            bundle, baseline_id, opponent, genome_id, genome, seed, seat, "train",
        )
        (h4_decisions, h1_decisions, h4_decision_id, h1_decision_id,
         audit) = collect_frozen_labels(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors,
            opponent, train_opponents.index(opponent), genome_id,
            genome_ids.index(genome_id), genome, seed, seat, h4_panel, h1_panel,
            h4_decision_id, h1_decision_id, args.max_windows,
        )
        train_fixed.append(fixed)
        train_h4_decisions.extend(h4_decisions)
        train_h1_decisions.extend(h1_decisions)
        train_audits.append(audit)
        print(json.dumps({
            "event": "train_frozen_labels_complete", "index": index,
            "total": len(train_scenarios), "genome": genome_id,
            "opponent": opponent, "seed": seed, "seat": seat,
            "h4_decisions": len(h4_decisions),
            "h1_decisions": len(h1_decisions),
        }, sort_keys=True), flush=True)

    (h4_model, h4_manifest, h4_arrays,
     h4_paths) = _fit_ranker_artifacts(
        output, h4_panel, "commit4_h4", COMMIT4_OVERRIDE_HORIZON, (1,),
        args, split_path, vocabulary_path,
    )
    (h1_model, h1_manifest, h1_arrays,
     h1_paths) = _fit_ranker_artifacts(
        output, h1_panel, "mpc1_h1", MPC1_OVERRIDE_HORIZON,
        WINDOW_OFFSETS, args, split_path, vocabulary_path,
    )
    frozen_paths = (*h4_paths, *h1_paths, split_path, vocabulary_path)
    frozen_identity = {
        str(path): (residual._sha256_file(path), path.stat().st_mtime_ns)
        for path in frozen_paths
    }

    validation_fixed: list[dict[str, Any]] = []
    validation_commit4: list[dict[str, Any]] = []
    validation_mpc1: list[dict[str, Any]] = []
    commit4_decisions: list[dict[str, Any]] = []
    mpc1_decisions: list[dict[str, Any]] = []
    validation_scenarios = [
        (genome_id, opponent, seed, seat)
        for genome_id in genome_ids for seed in validation_seeds
        for opponent in validation_opponents for seat in (0, 1)
    ]
    for index, (genome_id, opponent, seed, seat) in enumerate(validation_scenarios, 1):
        genome = genomes[genome_id]
        fixed = _fixed(
            bundle, baseline_id, opponent, genome_id, genome, seed, seat,
            "validation",
        )
        commit4, commit4_rows = learned_scenario(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors, opponent,
            genome_id, genome, seed, seat, h4_model,
            h4_manifest["calibration"]["chosen"], h4_manifest,
            "commit4", args.max_windows,
        )
        mpc1, mpc1_rows = learned_scenario(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors, opponent,
            genome_id, genome, seed, seat, h1_model,
            h1_manifest["calibration"]["chosen"], h1_manifest,
            "mpc1", args.max_windows,
        )
        validation_fixed.append(fixed)
        validation_commit4.append(commit4)
        validation_mpc1.append(mpc1)
        commit4_decisions.extend(commit4_rows)
        mpc1_decisions.extend(mpc1_rows)
        print(json.dumps({
            "event": "learned_validation_complete", "index": index,
            "total": len(validation_scenarios), "genome": genome_id,
            "opponent": opponent, "seed": seed, "seat": seat,
            "commit4_edits": commit4["edits"], "mpc1_edits": mpc1["edits"],
        }, sort_keys=True), flush=True)

    validation_oracle: list[dict[str, Any]] = []
    oracle_decisions: list[dict[str, Any]] = []
    fixed_by_key = {_scenario_key(row): row for row in validation_fixed}
    for index, (genome_id, opponent, seed, seat) in enumerate(validation_scenarios, 1):
        genome = genomes[genome_id]
        fixed = fixed_by_key[(genome_id, opponent, seed, seat)]
        oracle, decisions = oracle_scenario(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors, opponent,
            genome_id, genome, seed, seat, fixed, args.max_windows,
        )
        validation_oracle.append(oracle)
        oracle_decisions.extend(decisions)
        print(json.dumps({
            "event": "validation_oracle_complete", "index": index,
            "total": len(validation_scenarios), "genome": genome_id,
            "opponent": opponent, "seed": seed, "seat": seat,
            "edits": oracle["edits"],
        }, sort_keys=True), flush=True)
    after_identity = {
        str(path): (residual._sha256_file(path), path.stat().st_mtime_ns)
        for path in frozen_paths
    }
    if after_identity != frozen_identity:
        raise RuntimeError("frozen model artifacts changed during validation oracle")

    commit4_paired = _paired_rows(validation_commit4, validation_fixed)
    mpc1_paired = _paired_rows(validation_mpc1, validation_fixed)
    oracle_paired = _paired_rows(validation_oracle, validation_fixed)
    rows_by_file = {
        "train_fixed.jsonl": train_fixed,
        "train_oracle_labels_commit4_h4.jsonl": train_h4_decisions,
        "train_oracle_labels_mpc1_h1.jsonl": train_h1_decisions,
        "train_audit.jsonl": train_audits,
        "validation_fixed.jsonl": validation_fixed,
        "validation_commit4.jsonl": commit4_paired,
        "validation_commit4_decisions.jsonl": commit4_decisions,
        "validation_mpc1.jsonl": mpc1_paired,
        "validation_mpc1_decisions.jsonl": mpc1_decisions,
        "validation_oracle.jsonl": oracle_paired,
        "validation_oracle_decisions.jsonl": oracle_decisions,
    }
    artifacts = {}
    for name, rows in rows_by_file.items():
        path = output / name
        _write_jsonl(path, rows)
        artifacts[name] = _artifact(path, len(rows))
    for path in (*h4_paths, *h1_paths, split_path, vocabulary_path):
        artifacts[path.name] = _artifact(path)

    fixed_summary = _summary(validation_fixed)
    commit4_summary = _summary(commit4_paired)
    mpc1_summary = _summary(mpc1_paired)
    oracle_summary = _summary(oracle_paired)
    def train_audit(controller: str, metric: str) -> int:
        return sum(
            int(row["controllers"][controller][metric]) for row in train_audits
        )

    h4_keep_checks = train_audit("commit4_h4", "keep_equivalence_checks")
    h1_keep_checks = train_audit("mpc1_h1", "keep_equivalence_checks")
    keep_checks = h4_keep_checks + h1_keep_checks
    keep_checks += sum(int(row["keep_equivalence_checks"]) for row in validation_oracle)
    h4_market_checks = train_audit("commit4_h4", "market_unchanged_checks")
    h1_market_checks = train_audit("mpc1_h1", "market_unchanged_checks")
    market_checks = h4_market_checks + h1_market_checks
    market_checks += sum(int(row["market_unchanged_checks"]) for row in validation_oracle)
    h4_synergy = train_audit("commit4_h4", "pair_synergy_events")
    h1_synergy = train_audit("mpc1_h1", "pair_synergy_events")
    synergy_events = h4_synergy + h1_synergy
    synergy_events += sum(int(row["pair_synergy_events"]) for row in validation_oracle)
    mpc_horizon_rows = [*mpc1_decisions, *oracle_decisions]
    gates = {
        "G0_KEEP_equals_frozen_schedule": keep_checks > 0,
        "G1_path_only_market_unchanged": market_checks > 0,
        "G2_validation_oracle_has_headroom": (
            oracle_summary["mean_delta_margin_vs_fixed"] > 0
            and oracle_summary["edits"] > 0
        ),
        "G3_donor_pair_has_strict_synergy": synergy_events > 0,
        "G4_learned_adapter_fires": (
            commit4_summary["edits"] + mpc1_summary["edits"] > 0
        ),
        "G5_learned_arms_have_no_outcome_regression": (
            commit4_summary["outcome_regressions_vs_fixed"] == 0
            and mpc1_summary["outcome_regressions_vs_fixed"] == 0
        ),
        "G6_all_trajectories_complete": all(
            bool(row["completed"]) for row in (
                *train_fixed, *train_audits, *validation_fixed,
                *validation_commit4, *validation_mpc1, *validation_oracle,
            )
        ),
        "G7_model_frozen_before_validation_oracle": after_identity == frozen_identity,
        "G8_MPC1_candidate_score_execute_horizon_aligned": (
            bool(mpc_horizon_rows) and all(
                int(row["score_override_horizon"]) == MPC1_OVERRIDE_HORIZON
                and int(row["execute_horizon"]) == MPC1_OVERRIDE_HORIZON
                for row in mpc_horizon_rows
            )
        ),
        "G9_MPC1_ranker_target_horizon_aligned": (
            str(h1_manifest["controller"]) == "mpc1_h1"
            and int(h1_manifest["training_override_horizon"])
            == MPC1_OVERRIDE_HORIZON
            and list(h1_manifest["training_decision_offsets"])
            == list(WINDOW_OFFSETS)
            and bool(mpc1_decisions)
            and all(
                int(row["ranker_target_horizon"]) == MPC1_OVERRIDE_HORIZON
                and str(row["ranker_model_sha256"])
                == str(h1_manifest["model_sha256"])
                for row in mpc1_decisions
            )
        ),
        "G10_COMMIT4_ranker_target_horizon_aligned": (
            str(h4_manifest["controller"]) == "commit4_h4"
            and int(h4_manifest["training_override_horizon"])
            == COMMIT4_OVERRIDE_HORIZON
            and list(h4_manifest["training_decision_offsets"]) == [1]
            and bool(commit4_decisions)
            and all(
                int(row["ranker_target_horizon"])
                == COMMIT4_OVERRIDE_HORIZON
                and str(row["ranker_model_sha256"])
                == str(h4_manifest["model_sha256"])
                for row in commit4_decisions
            )
        ),
    }
    if (
        not gates["G9_MPC1_ranker_target_horizon_aligned"]
        or not gates["G10_COMMIT4_ranker_target_horizon_aligned"]
    ):
        status = "controller_ranker_horizon_contract_failed"
    elif not gates["G2_validation_oracle_has_headroom"]:
        status = "path_vocabulary_insufficient"
    elif not gates["G3_donor_pair_has_strict_synergy"]:
        status = "single_donor_residual_only"
    elif not gates["G4_learned_adapter_fires"]:
        status = "oracle_only_model_failed"
    elif not gates["G5_learned_arms_have_no_outcome_regression"]:
        status = "learned_but_unsafe"
    elif all(gates.values()):
        status = "multi_farmer_path_residual_feasible_proxy"
    else:
        status = "partial_path_residual_feasibility"
    report = {
        "schema": SCHEMA, "status": status,
        "claim": "frozen route/market with learned replay-block actor-slot residuals",
        "not_claimed": "not real submitted-agent evaluation and not the full-pool 90/80 gate",
        "source": source, "inputs": inputs, "splits": split,
        "baseline_route_id": baseline_id,
        "genomes": {
            genome_id: {
                "route_ids": list(genomes[genome_id]),
                "sha256": _sha256(list(genomes[genome_id])),
            }
            for genome_id in genome_ids
        },
        "vocabulary": vocabulary,
        "models": {"commit4_h4": h4_manifest, "mpc1_h1": h1_manifest},
        "train": {
            "fixed": _summary(train_fixed),
            "commit4_h4": {
                "label_decisions": len(train_h4_decisions),
                "candidate_rows": len(h4_arrays["features"]),
            },
            "mpc1_h1": {
                "label_decisions": len(train_h1_decisions),
                "candidate_rows": len(h1_arrays["features"]),
            },
            "labels_committed": False,
        },
        "validation": {
            "fixed": fixed_summary, "commit4": commit4_summary,
            "mpc1": mpc1_summary, "oracle": oracle_summary,
        },
        "audits": {
            "keep_equivalence_checks": keep_checks,
            "market_unchanged_checks": market_checks,
            "pair_synergy_events": synergy_events,
            "training_by_controller": {
                "commit4_h4": {
                    "keep_equivalence_checks": h4_keep_checks,
                    "market_unchanged_checks": h4_market_checks,
                    "pair_synergy_events": h4_synergy,
                },
                "mpc1_h1": {
                    "keep_equivalence_checks": h1_keep_checks,
                    "market_unchanged_checks": h1_market_checks,
                    "pair_synergy_events": h1_synergy,
                },
            },
        },
        "gates": gates, "sealed_test_executed": False,
        "artifacts": artifacts, "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(markdown_path, _report_markdown(report))
    report["artifacts"][report_path.name] = _artifact(report_path)
    report["artifacts"][markdown_path.name] = _artifact(markdown_path)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--block-library", type=Path, default=DEFAULT_BLOCK_LIBRARY)
    result.add_argument("--heldout-team", action="append", default=None)
    result.add_argument("--base-genome-id", default="NR020")
    result.add_argument("--composition-genome-id", default=None)
    result.add_argument("--train-opponent", action="append", default=None)
    result.add_argument("--validation-opponent", action="append", default=None)
    result.add_argument("--train-seed-start", type=int, default=TRAIN_SEEDS[0])
    result.add_argument("--train-seed-count", type=int, default=len(TRAIN_SEEDS))
    result.add_argument("--validation-seed-start", type=int, default=VALIDATION_SEEDS[0])
    result.add_argument("--validation-seed-count", type=int, default=len(VALIDATION_SEEDS))
    result.add_argument("--max-windows", type=int, default=len(ANCHORS))
    result.add_argument("--trees", type=int, default=96)
    result.add_argument("--cv-trees", type=int, default=24)
    result.add_argument("--cv-folds", type=int, default=5)
    result.add_argument("--random-seed", type=int, default=2026082901)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        args.train_opponent = [
            (args.train_opponent or list(TRAIN_OPPONENTS))[0],
        ]
        args.validation_opponent = [
            (args.validation_opponent or list(HELDOUT_OPPONENTS))[0],
        ]
        args.train_seed_count = min(args.train_seed_count, 2)
        args.validation_seed_count = min(args.validation_seed_count, 1)
        args.max_windows = min(args.max_windows, 2)
        args.trees = min(args.trees, 8)
        args.cv_trees = min(args.cv_trees, 4)
        args.cv_folds = min(args.cv_folds, 2)
    if (
        args.train_seed_count < 2 or args.validation_seed_count < 1
        or args.max_windows < 1 or args.max_windows > len(ANCHORS)
        or args.trees < 8 or args.cv_trees < 4 or args.cv_folds < 2
    ):
        raise ValueError("invalid MULTI-FARMER-PATH-RESIDUAL-v1 budget")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
