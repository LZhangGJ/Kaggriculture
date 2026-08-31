#!/usr/bin/env python3
"""Fixed-budget oracle A/B for guaranteeing an eligible farmer SINGLE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
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

import run_actor_contract_matching_oracle_ab as actor_ab
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual
import scan_multi_farmer_path_oracle_v1 as oracle_scan

SCHEMA = "farmer-guaranteed-candidate-ab-v1"
STRICT_LIBRARY = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\adaptive_tail_oracle_v0"
    r"\block_library_strict_unit_path_v2_prototypes_20260829"
)
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\multi_farmer_path_residual_v1"
    r"\farmer_guaranteed_candidate_ab_v1_strict_20260829"
)


def farmer_eligibility(
    base: Sequence[Sequence[Sequence[Any]]],
    donor_trace: Sequence[Sequence[Sequence[Any]]],
) -> tuple[bool, int]:
    changed = 0
    moved = False
    for offset in range(path_v1.PLAN_HORIZON):
        before = path_v1._unit(base[offset][0])
        after = path_v1._unit(donor_trace[offset][0])
        if before == after:
            continue
        changed += 1
        moved |= (
            str(before[0]) in path_v1.MOVE_OPS
            or str(after[0]) in path_v1.MOVE_OPS
        )
    return changed >= 2 and moved, changed


def guarantee_farmer_single(
    variants: Sequence[tuple[Any, ...]],
    farmer_variant: tuple[Any, ...],
    farmer_eligible: bool,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    """Replace the lowest-priority SINGLE without changing kind counts."""

    values = list(variants)
    singles = [index for index, row in enumerate(values) if row[1] == "SINGLE"]
    farmer_present = any(
        row[1] == "SINGLE" and tuple(row[2]) == ((0, 0),)
        for row in values
    )
    trigger = bool(farmer_eligible and not farmer_present and singles)
    replaced_code = None
    if trigger:
        index = singles[-1]
        replaced_code = str(values[index][0])
        values[index] = farmer_variant
    return values, {
        "farmer_eligible": bool(farmer_eligible),
        "farmer_single_present_A": bool(farmer_present),
        "replacement_triggered": trigger,
        "replacement_capacity_missing": bool(
            farmer_eligible and not farmer_present and not singles
        ),
        "replaced_code": replaced_code,
        "inserted_code": str(farmer_variant[0]) if trigger else None,
    }


def _raw_candidate(row: Sequence[Any]) -> path_v1.PathCandidate:
    code, kind, assignments, donor_id, cluster, support, rank, novelty, units = row
    payload = path_v1._canonical_bytes(units)
    return path_v1.PathCandidate(
        code=str(code), kind=str(kind), units=tuple(units),
        assignments=tuple(assignments), donor_id=donor_id,
        donor_cluster=cluster, donor_support=int(support),
        donor_rank=int(rank), trace_novelty=int(novelty),
        aliases=(str(code),), raw_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _keep(base: tuple[tuple[tuple[Any, ...], ...], ...]) -> path_v1.PathCandidate:
    return path_v1.PathCandidate(
        code="KEEP", kind="KEEP", units=base, assignments=tuple(),
        donor_id=None, donor_cluster=None, donor_support=0, donor_rank=-1,
        trace_novelty=0, aliases=("KEEP",), raw_sha256=path_v1._sha256(base),
    )


def candidate_arms(
    tape: Sequence[Mapping[str, Any]],
    selected_donors: Sequence[path_v1.DonorBlock],
    anchor: int,
    step: int,
    actor_count: int,
) -> tuple[list[path_v1.PathCandidate], list[path_v1.PathCandidate], dict[str, Any]]:
    base = path_v1._unit_trace(tape, step, actor_count)
    a_values = [_keep(base)]
    c_values = [_keep(base)]
    donor_rows = []
    for donor_rank, donor in enumerate(selected_donors):
        trace = path_v1._block_unit_trace(
            donor.actions, step - anchor, actor_count, path_v1.PLAN_HORIZON,
        )
        variants_a = path_v1._donor_variants(
            base, donor, actor_count, path_v1.PLAN_HORIZON,
            step - anchor, donor_rank, True,
        )
        eligible, novelty = farmer_eligibility(base, trace)
        farmer_row = (
            f"GUARANTEE_FARMER_A0_{donor.block_id}", "SINGLE", ((0, 0),),
            donor.block_id, donor.cluster_id, donor.support, donor_rank,
            novelty, path_v1._donor_patch(base, trace, (0,)),
        )
        variants_c, audit = guarantee_farmer_single(
            variants_a, farmer_row, eligible,
        )
        counts_a = Counter(str(row[1]) for row in variants_a)
        counts_c = Counter(str(row[1]) for row in variants_c)
        if counts_a != counts_c:
            raise AssertionError("farmer guarantee changed per-donor kind budget")
        a_values.extend(_raw_candidate(row) for row in variants_a)
        c_values.extend(_raw_candidate(row) for row in variants_c)
        donor_rows.append({
            "donor_id": donor.block_id, **audit,
            "raw_kind_budget_A": dict(sorted(counts_a.items())),
            "raw_kind_budget_C": dict(sorted(counts_c.items())),
        })
    candidates_a = actor_ab._deduplicate(a_values)
    generated_c = actor_ab._deduplicate(c_values)
    a_kind = Counter(value.kind for value in candidates_a)
    c_pre_kind = Counter(value.kind for value in generated_c)
    candidates_c = actor_ab.cap_by_kind(generated_c, a_kind)
    c_kind = Counter(value.kind for value in candidates_c)
    survived_aliases = {
        alias for candidate in candidates_c for alias in candidate.aliases
    }
    for row in donor_rows:
        row["guaranteed_survived_post_cap"] = bool(
            row["replacement_triggered"]
            and row["inserted_code"] in survived_aliases
        )
        row["replacement_priority"] = (
            "last raw SINGLE from _donor_variants "
            "(lowest novelty, then highest actor index)"
        )
    kinds = sorted(set(a_kind) | set(c_pre_kind) | set(c_kind))
    return candidates_a, candidates_c, {
        "donors": donor_rows,
        "selected_donor_count": len(selected_donors),
        "replacement_triggers": sum(row["replacement_triggered"] for row in donor_rows),
        "replacement_survived_post_cap": sum(
            row["guaranteed_survived_post_cap"] for row in donor_rows
        ),
        "eligible_farmer_donors": sum(row["farmer_eligible"] for row in donor_rows),
        "capacity_missing": sum(row["replacement_capacity_missing"] for row in donor_rows),
        "A_raw_before_dedup": len(a_values), "C_raw_before_dedup": len(c_values),
        "A_after_dedup_by_kind": dict(sorted(a_kind.items())),
        "C_after_dedup_before_cap_by_kind": dict(sorted(c_pre_kind.items())),
        "pre_cap_surplus_by_kind": {
            kind: c_pre_kind[kind] - a_kind[kind] for kind in kinds
            if c_pre_kind[kind] > a_kind[kind]
        },
        "C_after_cap_by_kind": dict(sorted(c_kind.items())),
        "post_cap_shortfall_by_kind": {
            kind: a_kind[kind] - c_kind[kind] for kind in kinds
            if a_kind[kind] > c_kind[kind]
        },
    }


def canonical_union(
    candidates_a: Sequence[path_v1.PathCandidate],
    candidates_c: Sequence[path_v1.PathCandidate],
) -> tuple[list[path_v1.PathCandidate], list[int], list[int]]:
    union = []
    index: dict[bytes, int] = {}

    def add(values: Sequence[path_v1.PathCandidate]) -> list[int]:
        result = []
        for candidate in values:
            key = path_v1._canonical_bytes(candidate.units)
            if key not in index:
                index[key] = len(union)
                union.append(candidate)
            result.append(index[key])
        return result

    indices_a = add(candidates_a)
    indices_c = add(candidates_c)
    if not indices_a or indices_a[0] != 0 or indices_c[0] != 0:
        raise AssertionError("A and C must share union KEEP")
    return union, indices_a, indices_c


def _arm_from_union(
    candidates: Sequence[path_v1.PathCandidate],
    indices: Sequence[int],
    union_rewards: np.ndarray,
    union_market_diff: np.ndarray,
    seat: int,
) -> dict[str, Any]:
    rewards = union_rewards[np.asarray(indices, np.int32)]
    market_diff = union_market_diff[np.asarray(indices, np.int32)]
    best = oracle_scan.select_path_pure_index(rewards, market_diff, seat)
    keep_rank = adaptive.reward_rank(rewards[0], seat)
    best_rank = adaptive.reward_rank(rewards[best], seat)
    selected = candidates[best]
    return {
        "candidate_count": len(candidates),
        "candidate_kind_counts": dict(sorted(Counter(
            candidate.kind for candidate in candidates
        ).items())),
        "path_pure_candidate_count": int(np.count_nonzero(market_diff == 0)),
        "market_diff_candidate_count": int(np.count_nonzero(market_diff)),
        "market_diff_step_count": int(market_diff.sum()),
        "keep_rewards": rewards[0].tolist(),
        "keep_rank": [int(keep_rank[0]), float(keep_rank[1])],
        "best_rewards": rewards[best].tolist(),
        "best_rank": [int(best_rank[0]), float(best_rank[1])],
        "best_code": selected.code,
        "best_kind": selected.kind,
        "best_donor_id": selected.donor_id,
        "best_assignments": [list(value) for value in selected.assignments],
        "delta_outcome": int(best_rank[0] - keep_rank[0]),
        "best_delta_margin": float(best_rank[1] - keep_rank[1]),
        "positive": bool(best_rank > keep_rank),
        "loss_to_win_repair": bool(keep_rank[0] == 0 and best_rank[0] == 2),
    }


def _paired(a: Mapping[str, Any], c: Mapping[str, Any]) -> dict[str, Any]:
    rank_a = (int(a["best_rank"][0]), float(a["best_rank"][1]))
    rank_c = (int(c["best_rank"][0]), float(c["best_rank"][1]))
    return {
        "C_vs_A_rank": int((rank_c > rank_a) - (rank_c < rank_a)),
        "C_minus_A_best_margin": float(rank_c[1] - rank_a[1]),
        "gain_retained": bool(a["positive"] and c["positive"]),
        "full_gain_retained": bool(a["positive"] and rank_c >= rank_a),
        "new_positive": bool(not a["positive"] and c["positive"]),
        "lost_positive": bool(a["positive"] and not c["positive"]),
        "repair_retained": bool(a["loss_to_win_repair"] and c["loss_to_win_repair"]),
        "new_repair": bool(not a["loss_to_win_repair"] and c["loss_to_win_repair"]),
        "lost_repair": bool(a["loss_to_win_repair"] and not c["loss_to_win_repair"]),
    }


def _union_oracle_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the row-wise A union C oracle; this is not a selector."""

    a_candidates = sum(int(row["A"]["candidate_count"]) for row in rows)
    increments = [
        int(row["union_candidate_count"]) - int(row["A"]["candidate_count"])
        for row in rows
    ]
    if any(value < 0 for value in increments):
        raise AssertionError("A must be a subset of the canonical union")
    u_positive = sum(
        bool(row["A"]["positive"] or row["C"]["positive"]) for row in rows
    )
    u_repairs = sum(
        bool(row["A"]["loss_to_win_repair"] or row["C"]["loss_to_win_repair"])
        for row in rows
    )
    u_beats = sum(int(row["paired"]["C_vs_A_rank"]) > 0 for row in rows)
    gains = [
        float(row["paired"]["C_minus_A_best_margin"])
        if int(row["paired"]["C_vs_A_rank"]) > 0 else 0.0
        for row in rows
    ]
    positive_deltas = [
        float(max(
            (row["A"], row["C"]),
            key=lambda arm: (int(arm["best_rank"][0]), float(arm["best_rank"][1])),
        )["best_delta_margin"])
        for row in rows
        if bool(row["A"]["positive"] or row["C"]["positive"])
    ]
    incremental = sum(increments)
    return {
        "U_positive_states": u_positive,
        "U_positive_coverage": u_positive / len(rows) if rows else 0.0,
        "U_positive_coverage_delta_vs_A": (
            u_positive - sum(bool(row["A"]["positive"]) for row in rows)
        ) / len(rows) if rows else 0.0,
        "U_gain_retained": sum(bool(row["A"]["positive"]) for row in rows),
        "U_full_gain_retained": sum(bool(row["A"]["positive"]) for row in rows),
        "U_new_positive": sum(bool(row["paired"]["new_positive"]) for row in rows),
        "U_lost_positive": 0,
        "U_positive_delta_margin_sum": float(sum(positive_deltas)),
        "U_mean_positive_delta_margin": (
            float(np.mean(positive_deltas)) if positive_deltas else 0.0
        ),
        "U_loss_to_win_repairs": u_repairs,
        "U_repairs_retained": sum(
            bool(row["A"]["loss_to_win_repair"]) for row in rows
        ),
        "U_new_repairs": sum(bool(row["paired"]["new_repair"]) for row in rows),
        "U_repairs_lost": 0,
        "U_beats_A": u_beats,
        "U_ties_A": len(rows) - u_beats,
        "U_loses_A": 0,
        "sum_U_minus_A_best_margin": float(sum(gains)),
        "mean_U_minus_A_best_margin": float(np.mean(gains)) if gains else 0.0,
        "max_U_minus_A_best_margin": max(gains, default=0.0),
        "U_incremental_canonical_candidates_vs_A": incremental,
        "U_incremental_canonical_candidate_rate_vs_A": (
            incremental / a_candidates if a_candidates else 0.0
        ),
    }


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a_positive = sum(bool(row["A"]["positive"]) for row in rows)
    c_positive = sum(bool(row["C"]["positive"]) for row in rows)
    selected = sum(int(row["candidate_audit"]["selected_donor_count"]) for row in rows)
    eligible = sum(int(row["candidate_audit"]["eligible_farmer_donors"]) for row in rows)
    triggers = sum(int(row["candidate_audit"]["replacement_triggers"]) for row in rows)
    survived = sum(
        int(row["candidate_audit"]["replacement_survived_post_cap"]) for row in rows
    )
    return {
        **_union_oracle_metrics(rows),
        "states": len(rows),
        "A_logical_candidate_rollouts": sum(int(row["A"]["candidate_count"]) for row in rows),
        "C_logical_candidate_rollouts": sum(int(row["C"]["candidate_count"]) for row in rows),
        "native_union_candidate_rollouts": sum(int(row["union_candidate_count"]) for row in rows),
        "union_saved_rollouts": sum(int(row["union_saved_rollouts"]) for row in rows),
        "A_positive_states": a_positive,
        "A_positive_coverage": a_positive / len(rows) if rows else 0.0,
        "C_positive_states": c_positive,
        "C_positive_coverage": c_positive / len(rows) if rows else 0.0,
        "positive_coverage_delta": (c_positive - a_positive) / len(rows) if rows else 0.0,
        "A_gain_retained": sum(bool(row["paired"]["gain_retained"]) for row in rows),
        "A_full_gain_retained": sum(bool(row["paired"]["full_gain_retained"]) for row in rows),
        "A_lost_positive": sum(bool(row["paired"]["lost_positive"]) for row in rows),
        "C_new_positive": sum(bool(row["paired"]["new_positive"]) for row in rows),
        "A_loss_to_win_repairs": sum(bool(row["A"]["loss_to_win_repair"]) for row in rows),
        "C_loss_to_win_repairs": sum(bool(row["C"]["loss_to_win_repair"]) for row in rows),
        "repairs_retained": sum(bool(row["paired"]["repair_retained"]) for row in rows),
        "C_new_repairs": sum(bool(row["paired"]["new_repair"]) for row in rows),
        "A_repairs_lost": sum(bool(row["paired"]["lost_repair"]) for row in rows),
        "C_beats_A": sum(int(row["paired"]["C_vs_A_rank"]) > 0 for row in rows),
        "C_ties_A": sum(int(row["paired"]["C_vs_A_rank"]) == 0 for row in rows),
        "C_loses_A": sum(int(row["paired"]["C_vs_A_rank"]) < 0 for row in rows),
        "mean_C_minus_A_best_margin": float(np.mean([
            float(row["paired"]["C_minus_A_best_margin"]) for row in rows
        ])) if rows else 0.0,
        "max_C_minus_A_best_margin": max((
            float(row["paired"]["C_minus_A_best_margin"]) for row in rows
        ), default=0.0),
        "min_C_minus_A_best_margin": min((
            float(row["paired"]["C_minus_A_best_margin"]) for row in rows
        ), default=0.0),
        "selected_donor_slots": selected,
        "farmer_eligible_donor_slots": eligible,
        "replacement_trigger_donor_slots": triggers,
        "replacement_survived_post_cap": survived,
        "replacement_survival_rate": survived / triggers if triggers else 0.0,
        "replacement_trigger_states": sum(
            int(row["candidate_audit"]["replacement_triggers"]) > 0 for row in rows
        ),
        "replacement_rate_selected": triggers / selected if selected else 0.0,
        "replacement_rate_eligible": triggers / eligible if eligible else 0.0,
        "replacement_capacity_missing": sum(
            int(row["candidate_audit"]["capacity_missing"]) for row in rows
        ),
        "pre_cap_surplus_candidates": sum(sum(
            int(value) for value in row["candidate_audit"]["pre_cap_surplus_by_kind"].values()
        ) for row in rows),
        "post_cap_shortfall_candidates": sum(sum(
            int(value) for value in row["candidate_audit"]["post_cap_shortfall_by_kind"].values()
        ) for row in rows),
        "A_market_diff_candidates": sum(int(row["A"]["market_diff_candidate_count"]) for row in rows),
        "C_market_diff_candidates": sum(int(row["C"]["market_diff_candidate_count"]) for row in rows),
        "keep_equivalence_failures": sum(not bool(row["keep_equivalent"]) for row in rows),
        "A_parity_checks": sum(bool(row["A_parity_checked"]) for row in rows),
        "A_parity_failures": sum(
            bool(row["A_parity_checked"]) and not bool(row["A_parity_exact"])
            for row in rows
        ),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {field: defaultdict(list) for field in ("split", "anchor", "opponent")}
    for row in rows:
        for field in groups:
            groups[field][str(row[field])].append(row)
    result = {"overall": _group(rows)}
    for field, group in groups.items():
        values = sorted(
            group.items(),
            key=(lambda item: int(item[0])) if field == "anchor" else None,
        )
        result[f"by_{field}"] = {key: _group(rows_) for key, rows_ in values}
    result["sentinels"] = {
        "NT0051_anchor504": _group([
            row for row in rows
            if row["opponent"] == "NT0051" and int(row["anchor"]) == 504
        ]),
        "anchor216": _group([
            row for row in rows if int(row["anchor"]) == 216
        ]),
    }
    return result


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    spec: Mapping[str, Any],
    max_windows: int,
    check_a_parity: bool,
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
            actor_count = len(path_v1._positions(observation))
            selected, selection = path_v1.select_diverse_donors(
                tape, donors[anchor], anchor, actor_count,
                decision_step=step, top_k=path_v1.MAX_PAIRS,
            )
            candidates_a, candidates_c, audit = candidate_arms(
                tape, selected, anchor, step, actor_count,
            )
            parity_checked = bool(check_a_parity and len(rows) < 6)
            parity_exact = True
            if parity_checked:
                reference_a = path_v1.effective_candidates(
                    path_v1.donor_candidates(
                        tape, selected, anchor, actor_count, decision_step=step,
                    ),
                    path_v1.PLAN_HORIZON,
                )
                parity_exact = (
                    [value.raw_sha256 for value in candidates_a]
                    == [value.raw_sha256 for value in reference_a]
                )
                if not parity_exact:
                    raise AssertionError("replicated A arm differs from shared generator")
            union, indices_a, indices_c = canonical_union(candidates_a, candidates_c)
            rewards, market_diff = oracle_scan._rollout(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, union,
            )
            arm_a = _arm_from_union(candidates_a, indices_a, rewards, market_diff, seat)
            arm_c = _arm_from_union(candidates_c, indices_c, rewards, market_diff, seat)
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                path_v1.STOPS[segment:],
            ), np.float64)[0]
            keep_equivalent = bool(
                np.array_equal(np.asarray(arm_a["keep_rewards"]), reference)
                and np.array_equal(np.asarray(arm_c["keep_rewards"]), reference)
            )
            rows.append({
                "schema": SCHEMA, "split": str(spec["split"]),
                "genome_id": "NR020", "opponent": opponent, "seed": seed,
                "seat": seat, "step": int(step), "anchor": anchor,
                "route_id": route_id, "runtime_actor_count": actor_count,
                "selected_donor_ids": [donor.block_id for donor in selected],
                "selection": selection, "candidate_audit": audit,
                "union_candidate_count": len(union),
                "union_saved_rollouts": len(candidates_a) + len(candidates_c) - len(union),
                "A": arm_a, "C": arm_c, "paired": _paired(arm_a, arm_c),
                "A_parity_checked": parity_checked,
                "A_parity_exact": parity_exact,
                "keep_equivalent": keep_equivalent, "committed": "KEEP",
            })
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            )
            for player in (0, 1)
        ])
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("farmer-guarantee frozen trajectory did not complete")
    return rows, {
        "split": str(spec["split"]), "opponent": opponent, "seed": seed,
        "seat": seat, "rewards": list(map(float, env.rewards)),
        "decisions": len(rows), "completed": True,
        "A_parity_checks": sum(row["A_parity_checked"] for row in rows),
        "A_parity_exact": all(
            row["A_parity_exact"] for row in rows if row["A_parity_checked"]
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    residual._atomic_text(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _python_runtime() -> dict[str, Any]:
    origins = {
        name: getattr(sys.modules.get(name), "__file__", None)
        for name in ("joblib", "sklearn")
    }
    return {
        "executable": str(Path(sys.executable).resolve()),
        "version": sys.version,
        "module_origins": origins,
        "import_stubs_detected": any(value is None for value in origins.values()),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    row = report["summary"]["overall"]
    anchor216 = report["summary"]["sentinels"]["anchor216"]
    nt504 = report["summary"]["sentinels"]["NT0051_anchor504"]
    return (
        "# FARMER-GUARANTEED-CANDIDATE-AB-v1\n\n"
        f"Status: {report['status']}\n\n"
        "This is a terminal-oracle candidate-pruning A/B, not a trained agent.\n\n"
        f"A/C positive states: {row['A_positive_states']}/{row['C_positive_states']}; "
        f"C beats/ties/loses A: {row['C_beats_A']}/{row['C_ties_A']}/{row['C_loses_A']}.\n\n"
        f"U=A union C oracle positive states: {row['U_positive_states']} "
        f"({row['U_positive_coverage']:.6f}); U beats A: {row['U_beats_A']}. "
        "U is an oracle vocabulary, not a selector.\n\n"
        f"U incremental canonical candidates vs A: "
        f"{row['U_incremental_canonical_candidates_vs_A']} "
        f"({row['U_incremental_canonical_candidate_rate_vs_A']:.6f}).\n\n"
        f"Replacement triggers: {row['replacement_trigger_donor_slots']}/{row['selected_donor_slots']} donor slots. "
        f"Native union rollouts: {row['native_union_candidate_rollouts']}; saved: {row['union_saved_rollouts']}.\n\n"
        f"NT0051/anchor504 C beats/ties/loses: {nt504['C_beats_A']}/{nt504['C_ties_A']}/{nt504['C_loses_A']}. "
        f"anchor216: {anchor216['C_beats_A']}/{anchor216['C_ties_A']}/{anchor216['C_loses_A']}.\n\n"
        f"Decision: {report['decision']}\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    train_seeds = tuple(range(args.train_seed_start, args.train_seed_start + args.seed_count))
    heldout_seeds = tuple(range(
        args.validation_seed_start, args.validation_seed_start + args.seed_count,
    ))
    scenarios = [
        {"split": split, "opponent": opponent, "seed": seed, "seat": seat}
        for split, opponents, seeds in (
            ("train_proxy", path_v1.TRAIN_OPPONENTS, train_seeds),
            ("heldout_proxy", path_v1.HELDOUT_OPPONENTS, heldout_seeds),
        )
        for opponent in opponents for seed in seeds for seat in (0, 1)
    ]
    if args.smoke:
        scenarios = scenarios[:1]
    elif len(scenarios) != 48 or args.seed_count != 2 or args.max_windows != len(path_v1.ANCHORS):
        raise ValueError("formal A/B requires 12 proxies x 2 seeds x 2 seats x 21 anchors")
    if (set(train_seeds) | set(heldout_seeds)) & residual.SEALED_SEEDS:
        raise ValueError("farmer A/B cannot open sealed seeds")
    if args.base_genome_id != "NR020" or args.composition_genome_id is not None:
        raise ValueError("farmer A/B freezes exactly NR020")
    args.train_opponent = list(path_v1.TRAIN_OPPONENTS)
    args.validation_opponent = list(path_v1.HELDOUT_OPPONENTS)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle, baseline_id, _, tapes, genomes, _, donors, inputs, source = (
        path_v1.load_experiment(args)
    )
    rows: list[dict[str, Any]] = []
    trajectories = []
    for index, spec in enumerate(scenarios, 1):
        decisions, trajectory = _scenario(
            bundle, baseline_id, tapes, genomes["NR020"], donors,
            spec, args.max_windows, index == 1,
        )
        rows.extend(decisions)
        trajectories.append(trajectory)
        if index % 4 == 0 or index == len(scenarios):
            print(json.dumps({
                "event": "farmer_guarantee_ab_progress", "index": index,
                "total": len(scenarios), "states": len(rows),
                "A_positive": sum(row["A"]["positive"] for row in rows),
                "C_positive": sum(row["C"]["positive"] for row in rows),
            }, sort_keys=True), flush=True)
    summary = summarize(rows)
    overall = summary["overall"]
    anchor216 = summary["sentinels"]["anchor216"]
    if (
        overall["keep_equivalence_failures"]
        or overall["post_cap_shortfall_candidates"]
        or overall["A_parity_failures"]
        or (not args.smoke and overall["A_parity_checks"] < 6)
    ):
        status = "invalid_fairness_contract"
        decision = "Fix KEEP equality or post-canonical budget parity."
    elif overall["A_lost_positive"] or overall["C_loses_A"]:
        status = "farmer_guarantee_mixed"
        decision = (
            "Reject fixed C replacement: it loses A-positive/value states. "
            "Retain U=A union C only as an oracle vocabulary for a future selector."
        )
    elif overall["C_new_repairs"] or (
        overall["C_positive_states"] > overall["A_positive_states"]
        and overall["C_loses_A"] == 0
    ):
        status = "farmer_guarantee_expands_coverage"
        decision = "Keep the fixed-budget guarantee and test execute-one/replan."
    elif overall["C_beats_A"] > overall["C_loses_A"] and anchor216["C_loses_A"] == 0:
        status = "farmer_guarantee_improves_value"
        decision = "Keep the fixed-budget guarantee; value improves without anchor216 regression."
    elif overall["C_beats_A"]:
        status = "farmer_guarantee_mixed"
        decision = "Do not hard-code replacement; retain U as oracle vocabulary."
    else:
        status = "farmer_guarantee_neutral"
        decision = "Do not change pruning; the guarantee adds no oracle value."
    split_manifest = {
        "schema": "farmer-guaranteed-candidate-ab-split-v1",
        "train_opponents": list(path_v1.TRAIN_OPPONENTS),
        "heldout_opponents": list(path_v1.HELDOUT_OPPONENTS),
        "train_seeds": list(train_seeds), "heldout_seeds": list(heldout_seeds),
        "seats": [0, 1],
        "anchors": list(map(int, path_v1.ANCHORS[:args.max_windows])),
        "offsets": [1], "trajectory": "frozen NR020; candidates discarded",
        "sealed_test_executed": False,
    }
    split_manifest["sha256"] = path_v1._sha256(split_manifest)
    decisions_path = output / "paired_states.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    summary_path = output / "summary.json"
    split_path = output / "split_manifest.json"
    _write_jsonl(decisions_path, rows)
    _write_jsonl(trajectories_path, trajectories)
    residual._atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(split_path, json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": SCHEMA, "status": status, "decision": decision,
        "oracle_not_trained_agent": True, "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020", "route_ids": list(genomes["NR020"]),
            "sha256": path_v1._sha256(list(genomes["NR020"])),
        },
        "candidate_contract": {
            "A": "current identity variants",
            "C": "eligible missing farmer0 replaces lowest-priority donor SINGLE",
            "U": (
                "canonical A union C vocabulary with row-wise terminal-oracle max; "
                "derived at zero additional rollout and not a selector or trained agent"
            ),
            "replacement_priority": (
                "last raw SINGLE emitted by _donor_variants; actor order is "
                "descending novelty then ascending actor, so this is lowest "
                "novelty then highest actor index"
            ),
            "raw_budget": "identical per donor and kind; never increased",
            "canonical_budget": "C capped to A per kind; differences audited",
            "survival": "every trigger audited after global dedup and cap",
            "A_parity": "first scenario first six real anchors compare ordered raw_sha256 with shared generator",
            "rollout": "A union C canonical traces evaluated once",
            "market_filter": "native H4-local market_diff == 0",
        },
        "split": split_manifest, "summary": summary,
        "A_parity": {
            "checks": overall["A_parity_checks"],
            "exact": overall["A_parity_failures"] == 0,
        },
        "all_trajectories_complete": all(row["completed"] for row in trajectories),
        "sealed_test_executed": False, "source": source, "inputs": inputs,
        "implementation": {
            "python_runtime": _python_runtime(),
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": residual._sha256_file(Path(__file__).resolve()),
            },
            "native_pyd": {
                "path": str(Path(path_v1.native_extension.__file__).resolve()),
                "sha256": residual._sha256_file(Path(path_v1.native_extension.__file__).resolve()),
            },
            "shared_path_v1": {
                "path": str(Path(path_v1.__file__).resolve()),
                "sha256": residual._sha256_file(Path(path_v1.__file__).resolve()),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            path.name: residual._artifact(path, count)
            for path, count in (
                (decisions_path, len(rows)), (trajectories_path, len(trajectories)),
                (summary_path, None), (split_path, None),
            )
        },
    }
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "farmer_guarantee_ab_complete", "status": status,
        "output": str(output), "summary": overall,
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--block-library", type=Path, default=STRICT_LIBRARY)
    result.add_argument("--heldout-team", action="append", default=None)
    result.add_argument("--base-genome-id", default="NR020")
    result.add_argument("--composition-genome-id", default=None)
    result.add_argument("--train-seed-start", type=int, default=path_v1.TRAIN_SEEDS[0])
    result.add_argument("--validation-seed-start", type=int, default=path_v1.VALIDATION_SEEDS[0])
    result.add_argument("--seed-count", type=int, default=2)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.seed_count < 1:
        raise ValueError("seed count must be positive")
    if args.max_windows < 1 or args.max_windows > len(path_v1.ANCHORS):
        raise ValueError("invalid anchor window count")
    if args.smoke:
        args.seed_count = 1
        args.max_windows = min(args.max_windows, 2)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
