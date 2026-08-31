#!/usr/bin/env python3
"""Terminal-oracle A/B for identity slots versus actor-contract matching.

This is an oracle vocabulary test on frozen NR020 trajectories, not a trained
agent.  Both arms use the same replay-donor shortlist and per-kind candidate
caps.  Arm A keeps the existing i->i splice.  Arm B fixes farmer 0 and matches
hands with a rectangular Hungarian assignment over position and inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual
import scan_multi_farmer_path_oracle_v1 as oracle_scan


SCHEMA = "actor-contract-matching-oracle-ab-v0"
DEFAULT_AUDIT = CODE_ROOT.parents[2] / "actor_contract_audit_20260829.json"
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1\actor_contract_matching_oracle_ab_20260829"
)
DECISION_STEPS = tuple(int(anchor + 1) for anchor in path_v1.ANCHORS)


def match_actor_contract(
    runtime_positions: Sequence[tuple[int, int]],
    runtime_inventories: Sequence[Mapping[str, int]],
    donor_positions: Sequence[tuple[int, int]],
    donor_inventories: Sequence[Mapping[str, int]],
) -> dict[int, int]:
    """Return runtime->donor actor slots; farmer 0 never participates in matching."""

    if not runtime_positions or not donor_positions:
        return {}
    result = {0: 0}
    if len(runtime_positions) == 1 or len(donor_positions) == 1:
        return result
    runtime_inventory = np.asarray([
        [np.log1p(float((runtime_inventories[index] if index < len(runtime_inventories) else {}).get(item, 0)))
         for item in residual.ITEMS]
        for index in range(1, len(runtime_positions))
    ], np.float64)
    donor_inventory = np.asarray([
        [np.log1p(float((donor_inventories[index] if index < len(donor_inventories) else {}).get(item, 0)))
         for item in residual.ITEMS]
        for index in range(1, len(donor_positions))
    ], np.float64)
    cost = np.empty((len(runtime_inventory), len(donor_inventory)), np.float64)
    for row, runtime_xy in enumerate(runtime_positions[1:]):
        for column, donor_xy in enumerate(donor_positions[1:]):
            distance = (
                abs(int(runtime_xy[0]) - int(donor_xy[0]))
                + abs(int(runtime_xy[1]) - int(donor_xy[1]))
            ) / 18.0
            inventory = float(np.mean(np.abs(
                runtime_inventory[row] - donor_inventory[column]
            )))
            cost[row, column] = distance + inventory
    rows, columns = linear_sum_assignment(cost)
    result.update({int(row + 1): int(column + 1) for row, column in zip(rows, columns)})
    return result


def patch_matched_units(
    base: Sequence[Sequence[Sequence[Any]]],
    donor: Sequence[Sequence[Sequence[Any]]],
    assignments: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    """Copy only matched source actors; unmatched runtime actors stay on base."""

    actor_map = dict(assignments)
    return tuple(tuple(
        tuple(donor[offset][actor_map[target]])
        if target in actor_map and actor_map[target] < len(donor[offset])
        else tuple(action)
        for target, action in enumerate(base[offset])
    ) for offset in range(len(base)))


def _contract_from_observation(
    observation: Mapping[str, Any], seat: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[dict[str, int], ...]]:
    farms = list(observation.get("farms", []) or [])
    farm = dict(farms[seat] if seat < len(farms) else {})
    raw_positions = [farm.get("farmer"), *(farm.get("hands", []) or [])]
    positions = tuple((int(value[0]), int(value[1])) for value in raw_positions)
    private = dict(observation.get("private", {}) or {})
    raw_inventories = list(private.get("inventories", []) or [])
    inventories = tuple(
        {
            str(item): max(0, int(quantity or 0))
            for item, quantity in dict(
                raw_inventories[index] if index < len(raw_inventories) else {}
            ).items()
        }
        for index in range(len(positions))
    )
    return positions, inventories


def _load_provenance(block_library: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    manifest_path = block_library.resolve() / "block_library_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Mapping[str, Any]] = {}
    inputs = []
    for prepared in manifest.get("prepared_roots", ()):
        path = Path(str(prepared["path"])).resolve() / "candidate_manifest.json"
        expected = str(prepared.get("candidate_manifest_sha256", ""))
        if expected and residual._sha256_file(path) != expected:
            raise ValueError("prepared provenance manifest digest mismatch")
        raw = json.loads(path.read_text(encoding="utf-8"))
        result.update({str(row["provenance_id"]): row for row in raw.get("provenance", ())})
        inputs.append({"path": str(path), "sha256": residual._sha256_file(path)})
    return result, {"block_manifest": str(manifest_path), "prepared_manifests": inputs}


def _donor_contract(
    donor: path_v1.DonorBlock,
    step: int,
    provenance: Mapping[str, Mapping[str, Any]],
    replay_cache: dict[tuple[str, int], dict[int, tuple[Any, Any]]],
) -> tuple[tuple[tuple[int, int], ...], tuple[dict[str, int], ...]]:
    row = provenance.get(donor.source_provenance_id)
    if row is None:
        raise ValueError(f"missing replay provenance for {donor.block_id}")
    replay_path = Path(str(row["replay_path"])).resolve()
    if replay_path.stat().st_size != int(row["replay_bytes"]):
        raise ValueError(f"source replay byte size changed: {replay_path}")
    player = int(row["player_index"])
    key = (str(replay_path), player)
    if key not in replay_cache:
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = list(raw.get("steps", []) or [])
        replay_cache[key] = {
            at: _contract_from_observation(
                dict(steps[at][player].get("observation") or {}), player,
            )
            for at in DECISION_STEPS
        }
    return replay_cache[key][int(step)]


def _candidate(
    code: str,
    kind: str,
    units: tuple[tuple[tuple[Any, ...], ...], ...],
    assignments: tuple[tuple[int, int], ...],
    donor: path_v1.DonorBlock,
    donor_rank: int,
    novelty: int,
) -> path_v1.PathCandidate:
    payload = path_v1._canonical_bytes(units)
    return path_v1.PathCandidate(
        code=code,
        kind=kind,
        units=units,
        assignments=assignments,
        donor_id=donor.block_id,
        donor_cluster=donor.cluster_id,
        donor_support=donor.support,
        donor_rank=donor_rank,
        trace_novelty=novelty,
        aliases=(code,),
        raw_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _matched_variants(
    base: tuple[tuple[tuple[Any, ...], ...], ...],
    donor: path_v1.DonorBlock,
    donor_trace: tuple[tuple[tuple[Any, ...], ...], ...],
    actor_map: Mapping[int, int],
    donor_rank: int,
    budget: Mapping[str, int],
) -> list[path_v1.PathCandidate]:
    novelty: dict[int, int] = {}
    movement: dict[int, bool] = {}
    for target, source in actor_map.items():
        changed = 0
        moved = False
        for offset in range(path_v1.PLAN_HORIZON):
            if target >= len(base[offset]) or source >= len(donor_trace[offset]):
                continue
            before = path_v1._unit(base[offset][target])
            after = path_v1._unit(donor_trace[offset][source])
            if before != after:
                changed += 1
                moved |= str(before[0]) in path_v1.MOVE_OPS or str(after[0]) in path_v1.MOVE_OPS
        if changed >= 2 and moved:
            novelty[int(target)] = changed
            movement[int(target)] = moved
    actors = sorted(novelty, key=lambda actor: (-novelty[actor], actor))
    pairs = sorted(
        combinations(actors, 2),
        key=lambda pair: (-(novelty[pair[0]] + novelty[pair[1]]), pair),
    )[:2]
    singles = sorted(
        {actor for pair in pairs for actor in pair} or set(actors[:2]),
        key=lambda actor: (-novelty[actor], actor),
    )
    specs: dict[str, list[tuple[tuple[int, int], ...]]] = {
        "SINGLE": [((target, int(actor_map[target])),) for target in singles],
        "PAIR": [tuple((target, int(actor_map[target])) for target in pair) for pair in pairs],
        "ALL": [tuple((target, int(actor_map[target])) for target in actors)] if len(actors) > 2 else [],
    }
    result = []
    for kind in ("SINGLE", "PAIR", "ALL"):
        for assignments in specs[kind][:int(budget.get(kind, 0))]:
            name = "_".join(f"R{target}S{source}" for target, source in assignments)
            units = patch_matched_units(base, donor_trace, assignments)
            result.append(_candidate(
                f"MATCH_{kind}_{name}_{donor.block_id}", kind, units, assignments,
                donor, donor_rank, sum(novelty[target] for target, _ in assignments),
            ))
    return result


def _deduplicate(
    candidates: Sequence[path_v1.PathCandidate],
) -> list[path_v1.PathCandidate]:
    unique: dict[bytes, path_v1.PathCandidate] = {}
    aliases: dict[bytes, list[str]] = defaultdict(list)
    order = []
    for candidate in candidates:
        key = path_v1._canonical_bytes(candidate.units)
        if key not in unique:
            unique[key] = candidate
            order.append(key)
        aliases[key].extend(candidate.aliases)
    return [
        replace(unique[key], aliases=tuple(dict.fromkeys(aliases[key])))
        for key in order
    ]


def cap_by_kind(
    candidates: Sequence[path_v1.PathCandidate],
    budget: Mapping[str, int],
) -> list[path_v1.PathCandidate]:
    """Keep deterministic order while matching A's post-dedup kind budget."""

    used: Counter[str] = Counter()
    result = []
    for candidate in candidates:
        if used[candidate.kind] >= int(budget.get(candidate.kind, 0)):
            continue
        used[candidate.kind] += 1
        result.append(candidate)
    return result


def matched_candidates(
    tape: Sequence[Mapping[str, Any]],
    selected_donors: Sequence[path_v1.DonorBlock],
    anchor: int,
    step: int,
    runtime_contract: tuple[Sequence[tuple[int, int]], Sequence[Mapping[str, int]]],
    provenance: Mapping[str, Mapping[str, Any]],
    replay_cache: dict[tuple[str, int], dict[int, tuple[Any, Any]]],
) -> tuple[list[path_v1.PathCandidate], dict[str, Any]]:
    runtime_positions, runtime_inventories = runtime_contract
    actor_count = len(runtime_positions)
    base = path_v1._unit_trace(tape, step, actor_count)
    keep = path_v1.PathCandidate(
        code="KEEP", kind="KEEP", units=base, assignments=tuple(), donor_id=None,
        donor_cluster=None, donor_support=0, donor_rank=-1, trace_novelty=0,
        aliases=("KEEP",), raw_sha256=path_v1._sha256(base),
    )
    values = [keep]
    budget_counts: Counter[str] = Counter()
    mapping_rows = []
    for donor_rank, donor in enumerate(selected_donors):
        identity = path_v1._donor_variants(
            base, donor, actor_count, path_v1.PLAN_HORIZON,
            step - anchor, donor_rank, True,
        )
        budget = Counter(str(row[1]) for row in identity)
        budget_counts.update(budget)
        donor_positions, donor_inventories = _donor_contract(
            donor, step, provenance, replay_cache,
        )
        actor_map = match_actor_contract(
            runtime_positions, runtime_inventories,
            donor_positions, donor_inventories,
        )
        donor_trace = path_v1._block_unit_trace(
            donor.actions, step - anchor, len(donor_positions), path_v1.PLAN_HORIZON,
        )
        variants = _matched_variants(
            base, donor, donor_trace, actor_map, donor_rank, budget,
        )
        values.extend(variants)
        produced = Counter(value.kind for value in variants)
        shortfall = {
            kind: int(count - produced.get(kind, 0))
            for kind, count in budget.items()
            if count > produced.get(kind, 0)
        }
        mapping_rows.append({
            "donor_id": donor.block_id,
            "runtime_actor_count": actor_count,
            "donor_actor_count": len(donor_positions),
            "mapping": [[target, source] for target, source in sorted(actor_map.items())],
            "unmatched_runtime": sorted(set(range(actor_count)) - set(actor_map)),
            "identity_budget": dict(sorted(budget.items())),
            "matched_counts": dict(sorted(produced.items())),
            "budget_shortfall": dict(sorted(shortfall.items())),
            "matched_variants_before_dedup": len(variants),
        })
    return _deduplicate(values), {
        "identity_budget": dict(sorted(budget_counts.items())),
        "mappings": mapping_rows,
        "before_dedup": len(values),
    }


def _arm(
    bundle: Any,
    env: Any,
    states: Sequence[Any],
    schedule: np.ndarray,
    stops: np.ndarray,
    seat: int,
    candidates: Sequence[path_v1.PathCandidate],
) -> dict[str, Any]:
    rewards, market_diff = oracle_scan._rollout(
        bundle, env, states, schedule, stops, seat, candidates,
    )
    best = oracle_scan.select_path_pure_index(rewards, market_diff, seat)
    keep_rank = adaptive.reward_rank(rewards[0], seat)
    best_rank = adaptive.reward_rank(rewards[best], seat)
    selected = candidates[best]
    return {
        "candidate_count": len(candidates),
        "candidate_kind_counts": dict(sorted(Counter(value.kind for value in candidates).items())),
        "path_pure_candidate_count": int(np.count_nonzero(market_diff == 0)),
        "market_diff_candidate_count": int(np.count_nonzero(market_diff)),
        "market_diff_step_count": int(market_diff.sum()),
        "keep_rewards": rewards[0].tolist(),
        "keep_rank": [int(keep_rank[0]), float(keep_rank[1])],
        "best_rewards": rewards[best].tolist(),
        "best_rank": [int(best_rank[0]), float(best_rank[1])],
        "best_index": int(best),
        "best_code": selected.code,
        "best_kind": selected.kind,
        "best_donor_id": selected.donor_id,
        "best_assignments": [list(value) for value in selected.assignments],
        "delta_outcome": int(best_rank[0] - keep_rank[0]),
        "best_delta_margin": float(best_rank[1] - keep_rank[1]),
        "positive": bool(best_rank > keep_rank),
        "loss_to_win_repair": bool(keep_rank[0] == 0 and best_rank[0] == 2),
    }


def _paired_state(
    arm_a: Mapping[str, Any], arm_b: Mapping[str, Any],
) -> dict[str, Any]:
    rank_a = (int(arm_a["best_rank"][0]), float(arm_a["best_rank"][1]))
    rank_b = (int(arm_b["best_rank"][0]), float(arm_b["best_rank"][1]))
    comparison = (rank_b > rank_a) - (rank_b < rank_a)
    return {
        "b_vs_a_rank": int(comparison),
        "b_minus_a_best_margin": float(rank_b[1] - rank_a[1]),
        "gain_retained": bool(arm_a["positive"] and arm_b["positive"]),
        "full_gain_retained": bool(arm_a["positive"] and rank_b >= rank_a),
        "new_positive": bool(not arm_a["positive"] and arm_b["positive"]),
        "lost_positive": bool(arm_a["positive"] and not arm_b["positive"]),
        "repair_retained": bool(
            arm_a["loss_to_win_repair"] and arm_b["loss_to_win_repair"]
        ),
        "new_repair": bool(
            not arm_a["loss_to_win_repair"] and arm_b["loss_to_win_repair"]
        ),
        "lost_repair": bool(
            arm_a["loss_to_win_repair"] and not arm_b["loss_to_win_repair"]
        ),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a_positive = sum(bool(row["A"]["positive"]) for row in rows)
    b_positive = sum(bool(row["B"]["positive"]) for row in rows)
    a_repairs = sum(bool(row["A"]["loss_to_win_repair"]) for row in rows)
    b_repairs = sum(bool(row["B"]["loss_to_win_repair"]) for row in rows)
    return {
        "states": len(rows),
        "A_candidate_rollouts": sum(int(row["A"]["candidate_count"]) for row in rows),
        "B_candidate_rollouts": sum(int(row["B"]["candidate_count"]) for row in rows),
        "A_positive_states": a_positive,
        "A_positive_coverage": a_positive / len(rows) if rows else 0.0,
        "B_positive_states": b_positive,
        "B_positive_coverage": b_positive / len(rows) if rows else 0.0,
        "positive_coverage_delta": (b_positive - a_positive) / len(rows) if rows else 0.0,
        "A_gain_retained": sum(bool(row["paired"]["gain_retained"]) for row in rows),
        "A_full_gain_retained": sum(bool(row["paired"]["full_gain_retained"]) for row in rows),
        "A_lost_positive": sum(bool(row["paired"]["lost_positive"]) for row in rows),
        "B_new_positive": sum(bool(row["paired"]["new_positive"]) for row in rows),
        "A_loss_to_win_repairs": a_repairs,
        "B_loss_to_win_repairs": b_repairs,
        "repairs_retained": sum(bool(row["paired"]["repair_retained"]) for row in rows),
        "B_new_repairs": sum(bool(row["paired"]["new_repair"]) for row in rows),
        "A_repairs_lost": sum(bool(row["paired"]["lost_repair"]) for row in rows),
        "B_beats_A": sum(int(row["paired"]["b_vs_a_rank"]) > 0 for row in rows),
        "B_ties_A": sum(int(row["paired"]["b_vs_a_rank"]) == 0 for row in rows),
        "B_loses_A": sum(int(row["paired"]["b_vs_a_rank"]) < 0 for row in rows),
        "mean_B_minus_A_best_margin": float(np.mean([
            float(row["paired"]["b_minus_a_best_margin"]) for row in rows
        ])) if rows else 0.0,
        "max_B_minus_A_best_margin": max((
            float(row["paired"]["b_minus_a_best_margin"]) for row in rows
        ), default=0.0),
        "min_B_minus_A_best_margin": min((
            float(row["paired"]["b_minus_a_best_margin"]) for row in rows
        ), default=0.0),
        "A_market_diff_candidates": sum(int(row["A"]["market_diff_candidate_count"]) for row in rows),
        "B_market_diff_candidates": sum(int(row["B"]["market_diff_candidate_count"]) for row in rows),
        "keep_equivalence_failures": sum(not bool(row["keep_equivalent"]) for row in rows),
        "B_budget_cap_violations": sum(
            int(row["B_nominal_before_dedup"]) > int(row["A_nominal_before_dedup"])
            for row in rows
        ),
        "B_budget_shortfall_states": sum(
            int(row["B_nominal_before_dedup"]) < int(row["A_nominal_before_dedup"])
            for row in rows
        ),
        "B_budget_shortfall_candidates": sum(
            int(row["A_nominal_before_dedup"]) - int(row["B_nominal_before_dedup"])
            for row in rows
        ),
        "B_after_dedup_shortfall_states": sum(bool(
            row["candidate_budget"]["after_dedup_shortfall_by_kind"]
        ) for row in rows),
        "B_after_dedup_shortfall_candidates": sum(
            sum(int(value) for value in row["candidate_budget"]["after_dedup_shortfall_by_kind"].values())
            for row in rows
        ),
        "B_after_dedup_surplus_before_cap_states": sum(bool(
            row["candidate_budget"]["after_dedup_surplus_before_cap_by_kind"]
        ) for row in rows),
        "B_after_dedup_surplus_before_cap_candidates": sum(
            sum(int(value) for value in row["candidate_budget"]["after_dedup_surplus_before_cap_by_kind"].values())
            for row in rows
        ),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_anchor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
        by_anchor[str(row["anchor"])].append(row)
    return {
        "overall": _group_summary(rows),
        "by_split": {key: _group_summary(value) for key, value in sorted(by_split.items())},
        "by_anchor": {key: _group_summary(value) for key, value in sorted(by_anchor.items(), key=lambda item: int(item[0]))},
    }


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    provenance: Mapping[str, Mapping[str, Any]],
    replay_cache: dict[tuple[str, int], dict[int, tuple[Any, Any]]],
    spec: Mapping[str, Any],
    max_windows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opponent, seed, seat = str(spec["opponent"]), int(spec["seed"]), int(spec["seat"])
    env, states, _ = residual.prefix_with_history(bundle, baseline_id, opponent, seed, seat)
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    active = frozenset(int(anchor + 1) for anchor in path_v1.ANCHORS[:max_windows])
    rows = []
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        if step in active:
            anchor = int(path_v1.ANCHORS[segment])
            route_id = str(genome[segment])
            tape = tapes[route_id]
            observation = dict(env.observation(seat))
            observation.update(step=step, player=seat)
            runtime_contract = _contract_from_observation(observation, seat)
            actor_count = len(runtime_contract[0])
            selected, selection = path_v1.select_diverse_donors(
                tape, donors[anchor], anchor, actor_count,
                decision_step=step, top_k=path_v1.MAX_PAIRS,
            )
            candidates_a = path_v1.effective_candidates(path_v1.donor_candidates(
                tape, selected, anchor, actor_count, decision_step=step,
            ), path_v1.PLAN_HORIZON)
            generated_b, matching = matched_candidates(
                tape, selected, anchor, step, runtime_contract,
                provenance, replay_cache,
            )
            a_after = Counter(candidate.kind for candidate in candidates_a)
            b_generated_after = Counter(candidate.kind for candidate in generated_b)
            candidates_b = cap_by_kind(generated_b, a_after)
            arm_a = _arm(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, candidates_a,
            )
            arm_b = _arm(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, candidates_b,
            )
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                path_v1.STOPS[segment:],
            ), np.float64)[0]
            keep_equivalent = bool(
                np.array_equal(np.asarray(arm_a["keep_rewards"]), reference)
                and np.array_equal(np.asarray(arm_b["keep_rewards"]), reference)
            )
            a_before = Counter({"KEEP": 1})
            b_before = Counter({"KEEP": 1})
            for mapping in matching["mappings"]:
                a_before.update({
                    kind: int(count)
                    for kind, count in mapping["identity_budget"].items()
                })
                b_before.update({
                    kind: int(count)
                    for kind, count in mapping["matched_counts"].items()
                })
            b_after = Counter(candidate.kind for candidate in candidates_b)
            kinds = sorted(
                set(a_before) | set(b_before) | set(a_after)
                | set(b_generated_after) | set(b_after)
            )
            budget = {
                "A_before_dedup_by_kind": dict(sorted(a_before.items())),
                "B_before_dedup_by_kind": dict(sorted(b_before.items())),
                "before_dedup_shortfall_by_kind": {
                    kind: max(0, a_before[kind] - b_before[kind]) for kind in kinds
                    if a_before[kind] > b_before[kind]
                },
                "A_after_dedup_by_kind": dict(sorted(a_after.items())),
                "B_after_dedup_before_cap_by_kind": dict(sorted(b_generated_after.items())),
                "after_dedup_surplus_before_cap_by_kind": {
                    kind: b_generated_after[kind] - a_after[kind] for kind in kinds
                    if b_generated_after[kind] > a_after[kind]
                },
                "B_after_dedup_by_kind": dict(sorted(b_after.items())),
                "after_dedup_shortfall_by_kind": {
                    kind: max(0, a_after[kind] - b_after[kind]) for kind in kinds
                    if a_after[kind] > b_after[kind]
                },
                "after_dedup_surplus_by_kind": {
                    kind: max(0, b_after[kind] - a_after[kind]) for kind in kinds
                    if b_after[kind] > a_after[kind]
                },
            }
            paired = _paired_state(arm_a, arm_b)
            rows.append({
                "schema": SCHEMA,
                "split": str(spec["split"]),
                "genome_id": "NR020",
                "opponent": opponent,
                "seed": seed,
                "seat": seat,
                "step": int(step),
                "anchor": anchor,
                "route_id": route_id,
                "runtime_actor_count": actor_count,
                "selected_donor_ids": [donor.block_id for donor in selected],
                "selection": selection,
                "A_nominal_before_dedup": int(sum(a_before.values())),
                "B_nominal_before_dedup": int(matching["before_dedup"]),
                "candidate_budget": budget,
                "matching": matching,
                "A": arm_a,
                "B": arm_b,
                "paired": paired,
                "keep_equivalent": keep_equivalent,
                "committed": "KEEP",
            })
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(env, player, int(route_pair[player]), states[player])
            for player in (0, 1)
        ])
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("frozen A/B trajectory did not complete")
    return rows, {
        "split": str(spec["split"]), "opponent": opponent, "seed": seed,
        "seat": seat, "rewards": list(map(float, env.rewards)),
        "decisions": len(rows), "completed": True,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    residual._atomic_text(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _markdown(report: Mapping[str, Any]) -> str:
    row = report["summary"]["overall"]
    return (
        "# ACTOR-CONTRACT-MATCHING-ORACLE-AB-v0\n\n"
        f"Status: `{report['status']}`\n\n"
        "This is a terminal-oracle candidate-vocabulary A/B on frozen NR020 "
        "states. It is not a trained selector or deployable agent.\n\n"
        "| arm | positive states | positive coverage | loss->win repairs | candidate rollouts | market-diff excluded |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| A: same index | {row['A_positive_states']} | {100 * row['A_positive_coverage']:.2f}% | {row['A_loss_to_win_repairs']} | {row['A_candidate_rollouts']} | {row['A_market_diff_candidates']} |\n"
        f"| B: actor contract | {row['B_positive_states']} | {100 * row['B_positive_coverage']:.2f}% | {row['B_loss_to_win_repairs']} | {row['B_candidate_rollouts']} | {row['B_market_diff_candidates']} |\n\n"
        f"B beats/ties/loses A on {row['B_beats_A']}/{row['B_ties_A']}/{row['B_loses_A']} states. "
        f"It retains {row['A_gain_retained']} A-positive states, adds {row['B_new_positive']} new positives, "
        f"and adds {row['B_new_repairs']} new loss-to-win repairs.\n\n"
        f"Decision: {report['decision']}\n\n"
        "Both arms use the same diversity-selected donor shortlist and the B arm is capped by A's per-donor, per-kind SINGLE/PAIR/ALL budget. "
        "All reported best arms pass the native H4-local market-equality filter. Opponents are fixed-route native proxies, not submitted agents.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    audit_path = args.actor_contract_audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "actor-contract-audit-v0":
        raise ValueError("invalid actor-contract audit schema")
    scenarios = list(audit.get("scenarios", ()))
    if args.smoke:
        scenarios = scenarios[:1]
    elif len(scenarios) != 8 or args.max_windows != len(path_v1.ANCHORS):
        raise ValueError("formal A/B requires the audited 8 scenarios and all 21 anchors")
    if args.base_genome_id != "NR020" or args.composition_genome_id is not None:
        raise ValueError("actor-contract A/B freezes exactly one NR020 genome")
    if any(int(row["seed"]) in residual.SEALED_SEEDS for row in scenarios):
        raise ValueError("actor-contract A/B cannot open sealed seeds")
    args.train_opponent = list(dict.fromkeys(
        str(row["opponent"]) for row in scenarios if row["split"] == "train_proxy"
    ))
    args.validation_opponent = list(dict.fromkeys(
        str(row["opponent"]) for row in scenarios if row["split"] == "heldout_proxy"
    ))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (bundle, baseline_id, _, tapes, genomes, _, donors,
     inputs, source) = path_v1.load_experiment(args)
    provenance, provenance_inputs = _load_provenance(args.block_library)
    replay_cache: dict[tuple[str, int], dict[int, tuple[Any, Any]]] = {}
    rows: list[dict[str, Any]] = []
    trajectories = []
    for index, spec in enumerate(scenarios, 1):
        decisions, trajectory = _scenario(
            bundle, baseline_id, tapes, genomes["NR020"], donors,
            provenance, replay_cache, spec, args.max_windows,
        )
        rows.extend(decisions)
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "actor_matching_ab_progress", "index": index,
            "total": len(scenarios), "states": len(rows),
            "A_positive": sum(value["A"]["positive"] for value in rows),
            "B_positive": sum(value["B"]["positive"] for value in rows),
        }, sort_keys=True), flush=True)
    summary = summarize(rows)
    overall = summary["overall"]
    if overall["keep_equivalence_failures"] or overall["B_budget_cap_violations"]:
        status = "invalid_ab_contract"
        decision = "Fix KEEP equivalence or candidate-budget accounting before interpreting the A/B."
    elif overall["B_new_repairs"] or overall["B_positive_states"] > overall["A_positive_states"]:
        status = "actor_matching_expands_oracle_coverage"
        decision = (
            "Use identity union actor-contract candidates rather than replacing A: "
            "next test the union inside execute-one/replan before training a larger selector."
        )
    elif overall["B_beats_A"] > overall["B_loses_A"]:
        status = "actor_matching_improves_oracle_value"
        decision = "Actor matching improves best-arm value but not coverage; retain it as candidate normalization and validate on more replay states."
    elif overall["B_loses_A"]:
        status = "actor_matching_not_sufficient"
        decision = "Do not replace identity candidates. Use A union B or learn assignment jointly with candidate value."
    else:
        status = "actor_matching_oracle_neutral"
        decision = "Do not add actor matching to the online agent yet; the matched vocabulary has no terminal-oracle advantage on this slice."
    decisions_path = output / "paired_states.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    summary_path = output / "summary.json"
    _write_jsonl(decisions_path, rows)
    _write_jsonl(trajectories_path, trajectories)
    residual._atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "oracle_not_trained_agent": True,
        "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020", "route_ids": list(genomes["NR020"]),
            "sha256": path_v1._sha256(list(genomes["NR020"])),
        },
        "matching_contract": audit["matching_contract"],
        "candidate_contract": {
            "A": "current same-index i->i",
            "B": "farmer fixed; rectangular Hungarian hands; unmatched runtime actors KEEP base",
            "donor_shortlist": "identical diversity-aware K<=3 shortlist",
            "budget": "B capped by A per-donor/per-kind SINGLE/PAIR/ALL raw budget",
            "budget_shortfall": "reported per donor and kind when matching cannot form an eligible variant; never padded or hidden",
            "decision_offset": 1,
            "market_filter": "native H4-local market_diff == 0",
            "trajectory": "frozen NR020; every oracle candidate discarded",
        },
        "scenarios": scenarios,
        "summary": summary,
        "all_trajectories_complete": all(row["completed"] for row in trajectories),
        "replay_sources_read": len(replay_cache),
        "sealed_test_executed": False,
        "source": source,
        "inputs": {
            **inputs,
            "actor_contract_audit": {
                "path": str(audit_path), "sha256": residual._sha256_file(audit_path),
            },
            "donor_provenance": provenance_inputs,
        },
        "implementation": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": residual._sha256_file(Path(__file__).resolve()),
            },
            "native_pyd": {
                "path": str(Path(path_v1.native_extension.__file__).resolve()),
                "sha256": residual._sha256_file(Path(path_v1.native_extension.__file__).resolve()),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            path.name: residual._artifact(path, count)
            for path, count in (
                (decisions_path, len(rows)), (trajectories_path, len(trajectories)),
                (summary_path, None),
            )
        },
    }
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "actor_matching_ab_complete", "status": status,
        "output": str(output), "summary": overall,
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--block-library", type=Path, default=path_v1.DEFAULT_BLOCK_LIBRARY)
    result.add_argument("--heldout-team", action="append", default=None)
    result.add_argument("--actor-contract-audit", type=Path, default=DEFAULT_AUDIT)
    result.add_argument("--base-genome-id", default="NR020")
    result.add_argument("--composition-genome-id", default=None)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.max_windows < 1 or args.max_windows > len(path_v1.ANCHORS):
        raise ValueError("invalid anchor window count")
    if args.smoke:
        args.max_windows = min(args.max_windows, 2)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
