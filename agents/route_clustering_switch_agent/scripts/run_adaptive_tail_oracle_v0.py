#!/usr/bin/env python3
"""Run the real-snapshot, receding-horizon ADAPTIVE-TAIL-ORACLE-v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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

import run_block_mvp_continuation_multitail_v1 as continuation
import fast_kaggriculture._fast_kaggriculture as native_extension
from fast_kaggriculture import Config, FastEnv, NativeAgentState
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_switch_features import (
    RouteSwitchHistory,
    route_switch_feature_names,
    route_switch_vector,
)


HORIZON = 719
ANCHORS = tuple(range(216, 697, 24))
MARKET_START, MARKET_STOP = 86, 104
HISTORY_START, HISTORY_STOP = 115, 129
STATE_INDICES = np.asarray(
    [*range(MARKET_START), *range(MARKET_STOP, HISTORY_START)], dtype=np.int64
)
STATE_NAMES = tuple(
    name for index, name in enumerate(route_switch_feature_names())
    if index in set(map(int, STATE_INDICES))
)
DEFAULT_OLD_PREPARED = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\block_mvp_continuation_multitail_v1\prepared"
)
DEFAULT_NEW_PREPARED = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\adaptive_tail_oracle_v0"
    r"\latest_replay_batch_510184144_514050427\prepared"
)
DEFAULT_BLOCK_LIBRARY = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\adaptive_tail_oracle_v0\block_library"
)
DEFAULT_QUERIES = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\block_mvp_continuation_multitail_v1_championship_final\run\queries.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\adaptive_tail_oracle_v0"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def reward_rank(rewards: Sequence[float], seat: int) -> tuple[int, float]:
    margin = float(rewards[seat]) - float(rewards[1 - seat])
    return (2 if margin > 0 else 1 if margin == 0 else 0, margin)


def choose_candidate(
    route_ids: Sequence[str], rewards: np.ndarray, seat: int, current: str,
) -> tuple[str, int, tuple[int, float]]:
    """Choose strict improvement only; exact ties keep the current tail."""

    if rewards.shape != (len(route_ids), 2) or current not in route_ids:
        raise ValueError("candidate rewards disagree with route IDs/current route")
    current_index = list(route_ids).index(current)
    best_index = current_index
    best_rank = reward_rank(rewards[current_index], seat)
    for index, route_id in enumerate(route_ids):
        rank = reward_rank(rewards[index], seat)
        if rank > best_rank or (
            rank == best_rank
            and route_id != current
            and route_ids[best_index] != current
            and route_id < route_ids[best_index]
        ):
            best_index, best_rank = index, rank
    return str(route_ids[best_index]), best_index, best_rank


def snapshot_contract(
    observation: Mapping[str, Any], tape: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State/plan contract without market environment or fake history fields."""

    observation = dict(observation)
    history = RouteSwitchHistory()
    full = route_switch_vector(observation, history, tape)
    state = full[STATE_INDICES].astype(np.float32, copy=True)
    plan = continuation.fixed_commitments(observation, tape)
    layout, unlocked = continuation.own_layout(observation)
    contract = np.concatenate((state, plan)).astype(np.float32)
    if (
        state.shape != (97,) or plan.shape != (18,) or contract.shape != (115,)
        or not np.isfinite(contract).all()
    ):
        raise ValueError("invalid market/history-free snapshot contract")
    return state, plan, layout, unlocked


def _load_banks(
    roots: Sequence[Path],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    bank: dict[str, list[dict[str, Any]]] = {}
    signatures: list[dict[str, Any]] = []
    for root in roots:
        manifest, part, _ = continuation.load_prepared_bank(root)
        for route_id, tape in part.items():
            # MTV1 route IDs hash the step-216 tail. Prefixes may legitimately
            # differ across replay occurrences and are never executed here.
            if route_id in bank and bank[route_id][216:] != tape[216:]:
                raise ValueError(f"route ID collision with different tape: {route_id}")
            bank.setdefault(route_id, tape)
        manifest_path = root.resolve() / "candidate_manifest.json"
        signatures.append({
            "root": str(root.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "pool_segment_sha256": str(manifest["pool_segment_sha256"]),
            "unique_tail_count": int(manifest["unique_tail_count"]),
        })
    if not bank:
        raise ValueError("adaptive oracle requires at least one replay tail")
    return bank, signatures


def _load_active_routes(path: Path, bank: Mapping[str, Any]) -> tuple[dict[int, list[str]], dict[str, Any]]:
    manifest_path = path.resolve() / "block_library_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "adaptive-tail-block-library-v0":
        raise ValueError("invalid adaptive block library schema")
    if tuple(map(int, manifest.get("anchors", []))) != ANCHORS:
        raise ValueError("adaptive block library anchors do not match the oracle")
    result: dict[int, list[str]] = {}
    active = dict(manifest.get("active_representatives") or {})
    for anchor in ANCHORS:
        rows = list(active.get(str(anchor), []) or [])
        route_ids = list(dict.fromkeys(str(row["source_route_id"]) for row in rows))
        if len(rows) != 8 or len(route_ids) != 8:
            raise ValueError(f"step {anchor} must have exactly 8 distinct active block routes")
        if any(route_id not in bank for route_id in route_ids):
            raise ValueError(f"invalid active block routes at step {anchor}")
        result[anchor] = route_ids
    return result, {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "block_count": int(manifest["block_count"]),
        "active_count": sum(len(values) for values in result.values()),
    }


def _load_queries(
    path: Path, opening: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row for row in rows
        if str(row.get("opening")) == opening and int(row.get("checkpoint", -1)) == 216
    ]
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in selected:
        key = (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
        if key in unique:
            raise ValueError(f"duplicate frozen query state: {key}")
        unique[key] = row
    if not unique:
        raise ValueError("no checkpoint-216 B0 queries found")
    result = [unique[key] for key in sorted(unique)]
    return result, {
        "path": str(path.resolve()), "sha256": _sha256_file(path),
        "states": len(result),
        "hard_states": sum(int(row["baseline_outcome"]) == 0 for row in result),
    }


def _pairs(candidate_indices: Sequence[int], opponent: int, seat: int) -> np.ndarray:
    values = np.empty((len(candidate_indices), 2), dtype=np.int64)
    if seat == 0:
        values[:, 0] = candidate_indices
        values[:, 1] = opponent
    else:
        values[:, 0] = opponent
        values[:, 1] = candidate_indices
    return values


def _baseline_rewards(
    bundle: NativeTeammateBundle, opening: int, opponent: int, seed: int, seat: int,
) -> tuple[float, float]:
    result = bundle.executor.play(
        opening if seat == 0 else opponent,
        opponent if seat == 0 else opening,
        int(seed),
    )
    return tuple(map(float, result["rewards"]))


def run_state(
    bundle: NativeTeammateBundle,
    opening_id: str,
    opponent_id: str,
    seed: int,
    seat: int,
    candidate_by_anchor: Mapping[int, Sequence[str]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    opening = bundle.index(opening_id)
    opponent = bundle.index(opponent_id)
    env = FastEnv(Config(), int(seed))
    states = [NativeAgentState(), NativeAgentState()]
    initial_pair = (opening, opponent) if seat == 0 else (opponent, opening)
    bundle.executor.advance_segment(env, states[0], states[1], *initial_pair, 216)
    if int(env.step_count) != 216:
        raise RuntimeError("native prefix did not stop at step 216")

    current_id = opening_id
    decisions: list[dict[str, Any]] = []
    vectors: list[dict[str, Any]] = []
    first_selected_rank: tuple[int, float] | None = None
    rollout_steps = 0
    for anchor in ANCHORS:
        if int(env.step_count) != anchor:
            raise RuntimeError(f"adaptive branch missed boundary {anchor}")
        candidates = list(dict.fromkeys([current_id, *candidate_by_anchor[anchor]]))
        indices = [bundle.index(route_id) for route_id in candidates]
        rewards = np.asarray(
            bundle.executor.rollout_from_batch(
                env, states[0], states[1], _pairs(indices, opponent, seat)
            ),
            dtype=np.float64,
        )
        if rewards.shape != (len(candidates), 2) or not np.isfinite(rewards).all():
            raise RuntimeError("native snapshot rollout returned invalid rewards")
        chosen_id, chosen_index, chosen_rank = choose_candidate(
            candidates, rewards, seat, current_id
        )
        current_index = candidates.index(current_id)
        current_rank = reward_rank(rewards[current_index], seat)
        if chosen_rank < current_rank:
            raise AssertionError("adaptive decision regressed against KEEP")
        if first_selected_rank is None:
            first_selected_rank = chosen_rank
        observation = dict(env.observation(seat))
        observation["step"] = anchor
        observation["player"] = seat
        state, plan, layout, unlocked = snapshot_contract(
            observation, tapes[chosen_id]
        )
        top = sorted(
            range(len(candidates)),
            key=lambda index: (reward_rank(rewards[index], seat), candidates[index]),
            reverse=True,
        )[:5]
        decisions.append({
            "mode": mode,
            "step": anchor,
            "current_route_id": current_id,
            "selected_route_id": chosen_id,
            "switched": chosen_id != current_id,
            "candidate_count": len(candidates),
            "current_outcome": current_rank[0],
            "current_margin": current_rank[1],
            "selected_outcome": chosen_rank[0],
            "selected_margin": chosen_rank[1],
            "top5": [
                {
                    "route_id": candidates[index],
                    "rewards": rewards[index].astype(float).tolist(),
                    "outcome": reward_rank(rewards[index], seat)[0],
                    "margin": reward_rank(rewards[index], seat)[1],
                }
                for index in top
            ],
        })
        vectors.append({
            "step": anchor, "seat": seat, "mode": mode,
            "selected_route_id": chosen_id,
            "state": state, "plan": plan,
            "contract": np.concatenate((state, plan)),
            "layout": layout, "unlocked": unlocked,
        })
        stop = min(anchor + 24, HORIZON)
        chosen = indices[chosen_index]
        active_pair = (chosen, opponent) if seat == 0 else (opponent, chosen)
        bundle.executor.advance_segment(
            env, states[0], states[1], *active_pair, stop
        )
        rollout_steps += len(candidates) * (HORIZON - anchor)
        current_id = chosen_id

    if not env.done or int(env.step_count) != HORIZON:
        raise RuntimeError("adaptive branch did not finish exactly at step 719")
    final_rewards = tuple(map(float, env.rewards))
    final_rank = reward_rank(final_rewards, seat)
    if first_selected_rank is None or final_rank < first_selected_rank:
        raise AssertionError("receding-horizon monotonicity was violated")
    if final_rank != (
        int(decisions[-1]["selected_outcome"]),
        float(decisions[-1]["selected_margin"]),
    ):
        raise AssertionError("last snapshot rollout disagrees with committed execution")
    return ({
        "mode": mode,
        "opening": opening_id,
        "opponent": opponent_id,
        "seed": int(seed),
        "seat": int(seat),
        "rewards": list(final_rewards),
        "outcome": final_rank[0],
        "margin": final_rank[1],
        "step216_static_choice": str(decisions[0]["selected_route_id"]),
        "step216_static_outcome": int(decisions[0]["selected_outcome"]),
        "step216_static_margin": float(decisions[0]["selected_margin"]),
        "adaptive_margin_gain_vs_step216_static": (
            final_rank[1] - float(decisions[0]["selected_margin"])
        ),
        "switches": sum(bool(row["switched"]) for row in decisions),
        "post216_switches": sum(
            bool(row["switched"]) and int(row["step"]) > 216 for row in decisions
        ),
        "unique_selected_routes": sorted({
            str(row["selected_route_id"]) for row in decisions
        }),
        "rollout_candidate_steps": rollout_steps,
    }, decisions, vectors)


def _save_vectors(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    np.savez_compressed(
        temporary,
        state=np.stack([row["state"] for row in rows]).astype(np.float32),
        plan=np.stack([row["plan"] for row in rows]).astype(np.float32),
        contract=np.stack([row["contract"] for row in rows]).astype(np.float32),
        layout=np.stack([row["layout"] for row in rows]).astype(np.uint8),
        unlocked=np.stack([row["unlocked"] for row in rows]).astype(np.uint8),
        step=np.asarray([row["step"] for row in rows], np.int16),
        seat=np.asarray([row["seat"] for row in rows], np.int8),
        mode=np.asarray([row["mode"] for row in rows]),
        selected_route_id=np.asarray([row["selected_route_id"] for row in rows]),
        state_key=np.asarray([row["state_key"] for row in rows]),
        state_feature_names=np.asarray(STATE_NAMES),
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prepared_values = args.prepared_root or [DEFAULT_OLD_PREPARED, DEFAULT_NEW_PREPARED]
    prepared_roots = [Path(value).resolve() for value in prepared_values]
    bank, bank_signatures = _load_banks(prepared_roots)
    active, block_signature = _load_active_routes(args.block_library, bank)
    (
        baseline_id, baseline_tape, _, _, v1_args, v1_manifest,
    ) = continuation.load_frozen_v1_openings(args.v1_root)
    queries, query_signature = _load_queries(args.queries, baseline_id)
    if args.max_states:
        queries = queries[:args.max_states]
    hard_keys = {
        (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
        for row in queries if int(row["baseline_outcome"]) == 0
    }
    opponents = sorted({str(row["opponent"]) for row in queries})
    additional = {baseline_id: baseline_tape, **bank}
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*opponents, *additional)),
    )
    for method in ("advance_segment", "rollout_from_batch"):
        if not hasattr(bundle.executor, method):
            raise RuntimeError(f"native extension is missing {method}; rebuild it")

    state_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    # The frozen baseline remains a legal fallback at every receding-horizon
    # boundary, even after the oracle has temporarily switched to a donor tail.
    all_routes = [baseline_id, *sorted(bank)]
    for query in queries:
        opponent_id = str(query["opponent"])
        seed, seat = int(query["seed"]), int(query["seat"])
        baseline_rewards = _baseline_rewards(
            bundle, bundle.index(baseline_id), bundle.index(opponent_id), seed, seat
        )
        baseline_rank = reward_rank(baseline_rewards, seat)
        expected = (int(query["baseline_outcome"]), float(query["baseline_margin"]))
        if baseline_rank != expected:
            raise RuntimeError(
                f"frozen baseline query drift for {opponent_id}/{seed}/{seat}: "
                f"{baseline_rank} != {expected}"
            )
        modes = {"cluster8": active}
        if (opponent_id, seed, seat) in hard_keys and not args.skip_full_bank:
            modes["full_bank"] = {anchor: all_routes for anchor in ANCHORS}
        for mode, candidates in modes.items():
            result, decisions, vectors = run_state(
                bundle, baseline_id, opponent_id, seed, seat,
                candidates, additional, mode,
            )
            result.update({
                "baseline_rewards": list(baseline_rewards),
                "baseline_outcome": baseline_rank[0],
                "baseline_margin": baseline_rank[1],
                "baseline_loss": baseline_rank[0] == 0,
            })
            state_key = f"{opponent_id}|{seed}|{seat}|{mode}"
            for row in decisions:
                row["state_key"] = state_key
            for row in vectors:
                row["state_key"] = state_key
            state_rows.append(result)
            decision_rows.extend(decisions)
            vector_rows.extend(vectors)

    cluster = [row for row in state_rows if row["mode"] == "cluster8"]
    full = [row for row in state_rows if row["mode"] == "full_bank"]
    hard_cluster = [row for row in cluster if row["baseline_loss"]]
    easy_cluster = [row for row in cluster if not row["baseline_loss"]]
    repair = lambda rows: sum(
        bool(row["baseline_loss"]) and int(row["outcome"]) == 2 for row in rows
    )
    nonloss_repair = lambda rows: sum(
        bool(row["baseline_loss"]) and int(row["outcome"]) > 0 for row in rows
    )
    static_repair = lambda rows: sum(
        bool(row["baseline_loss"]) and int(row["step216_static_outcome"]) == 2
        for row in rows
    )
    adaptive_only_repair = lambda rows: sum(
        bool(row["baseline_loss"])
        and int(row["step216_static_outcome"]) == 0
        and int(row["outcome"]) == 2
        for row in rows
    )
    nondegrading = lambda row: (int(row["outcome"]), float(row["margin"])) >= (
        int(row["baseline_outcome"]), float(row["baseline_margin"])
    )
    selected_nonbaseline = {
        route_id
        for row in hard_cluster for route_id in row["unique_selected_routes"]
        if route_id != baseline_id
    }
    state_key = lambda row: (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
    cluster_repair_keys = {
        state_key(row) for row in hard_cluster if int(row["outcome"]) == 2
    }
    full_repair_keys = {
        state_key(row) for row in full if int(row["outcome"]) == 2
    }
    retained_repair_keys = cluster_repair_keys & full_repair_keys
    missing_full_repair_keys = full_repair_keys - cluster_repair_keys
    repaired_opponents = {key[0] for key in cluster_repair_keys}
    cluster_repair_retention = len(retained_repair_keys) / max(1, len(full_repair_keys))
    cluster_raw_win_rate = sum(
        int(row["outcome"]) == 2 for row in cluster
    ) / max(1, len(cluster))
    baseline_raw_win_rate = sum(
        int(row["baseline_outcome"]) == 2 for row in cluster
    ) / max(1, len(cluster))
    gates = {
        "G0_all_terminal_outputs_finite": all(
            np.isfinite(row["rewards"]).all() for row in state_rows
        ),
        "G1_cluster8_never_regresses_frozen_baseline_rank": all(
            nondegrading(row) for row in cluster
        ),
        "G2_cluster8_turns_at_least_2_of_8_hard_states_into_wins": (
            len(hard_cluster) == 8 and repair(hard_cluster) >= 2
        ),
        "G4_full_bank_receding_oracle_never_regresses": all(
            nondegrading(row) for row in full
        ),
        "G5_at_least_1_repair_requires_a_post216_reroute": (
            adaptive_only_repair(hard_cluster) >= 1
            and any(
                int(row["step216_static_outcome"]) == 0
                and int(row["outcome"]) == 2
                and int(row["post216_switches"]) >= 1
                for row in hard_cluster
            )
        ),
        "G6_cluster8_retains_80pct_of_full_bank_hard_repairs": (
            bool(full_repair_keys) and cluster_repair_retention >= .80
        ),
        "G7_repairs_cover_at_least_2_distinct_opponents": (
            len(repaired_opponents) >= 2
        ),
        "G8_cluster8_raw_win_rate_improves_over_90pct_baseline": (
            baseline_raw_win_rate == .90 and cluster_raw_win_rate > baseline_raw_win_rate
        ),
    }
    report = {
        "schema": "adaptive-tail-oracle-v0",
        "status": (
            "passed_greedy_adaptive_feasibility"
            if all(gates.values()) else "falsified_greedy_adaptive_feasibility"
        ),
        "engine": "C++ NativeTeammateExecutor real snapshot/NativeAgentState copy",
        "oracle": "greedy clairvoyant constant-tail value-to-go; commit winner for 24 steps",
        "selector_trained": False,
        "market_environment_features_used": False,
        "history_features_used": False,
        "implementation": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "native_extension_path": str(Path(native_extension.__file__).resolve()),
            "native_extension_sha256": _sha256_file(
                Path(native_extension.__file__).resolve()
            ),
        },
        "anchors": list(ANCHORS),
        "prepared_inputs": bank_signatures,
        "unique_replay_tails": len(bank),
        "block_library": block_signature,
        "queries": query_signature,
        "evaluated_states": len(queries),
        "hard_states": len(hard_cluster),
        "cluster8": {
            "states": len(cluster),
            "baseline_raw_win_rate": baseline_raw_win_rate,
            "raw_win_rate": cluster_raw_win_rate,
            "mean_margin": float(np.mean([row["margin"] for row in cluster])),
            "hard_win_repairs": repair(hard_cluster),
            "hard_nonloss_repairs": nonloss_repair(hard_cluster),
            "step216_static_hard_win_repairs": static_repair(hard_cluster),
            "adaptive_only_hard_win_repairs": adaptive_only_repair(hard_cluster),
            "hard_repair_retention_vs_full_bank": cluster_repair_retention,
            "retained_full_repair_state_keys": sorted("|".join(map(str, key)) for key in retained_repair_keys),
            "missing_full_repair_state_keys": sorted("|".join(map(str, key)) for key in missing_full_repair_keys),
            "repaired_opponents": sorted(repaired_opponents),
            "easy_regressions": sum(not nondegrading(row) for row in easy_cluster),
            "unique_nonbaseline_hard_choices": sorted(selected_nonbaseline),
        },
        "full_bank_hard": {
            "states": len(full),
            "hard_win_repairs": repair(full),
            "hard_nonloss_repairs": nonloss_repair(full),
            "step216_static_hard_win_repairs": static_repair(full),
            "adaptive_only_hard_win_repairs": adaptive_only_repair(full),
            "mean_margin": float(np.mean([row["margin"] for row in full])) if full else None,
        },
        "rollout_candidate_steps": sum(int(row["rollout_candidate_steps"]) for row in state_rows),
        "full_game_equivalents": sum(
            int(row["rollout_candidate_steps"]) for row in state_rows
        ) / HORIZON,
        "gates": gates,
        "diagnostics": {
            "hard_states_use_at_least_2_nonbaseline_tails": (
                len(selected_nonbaseline) >= 2
            ),
        },
        "v1_actions_sha256": str(v1_manifest["actions_sha256"]),
        "artifacts": {
            "states": "states.jsonl", "decisions": "decisions.jsonl",
            "decision_vectors": "decision_vectors.npz",
        },
        "scope": [
            "This is a greedy receding-horizon feasibility test, not a global block-composition upper bound.",
            "Each decision is made from the real continued Simulator and NativeAgentState snapshot.",
            "Replay market inventory/prices and untracked history are excluded from the selector contract.",
            "A negative result falsifies constant-tail value-to-go greedy switching only; it does not falsify GA/beam search over block combinations.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = args.output_root.resolve() / "run"
    states_path = output / "states.jsonl"
    decisions_path = output / "decisions.jsonl"
    vectors_path = output / "decision_vectors.npz"
    _atomic_text(states_path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in state_rows
    ))
    _atomic_text(decisions_path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decision_rows
    ))
    _save_vectors(vectors_path, vector_rows)
    report["artifacts"] = {
        "states": {
            "file": states_path.name, "rows": len(state_rows),
            "sha256": _sha256_file(states_path),
        },
        "decisions": {
            "file": decisions_path.name, "rows": len(decision_rows),
            "sha256": _sha256_file(decisions_path),
        },
        "decision_vectors": {
            "file": vectors_path.name, "rows": len(vector_rows),
            "sha256": _sha256_file(vectors_path),
        },
    }
    _atomic_text(
        output / "FINAL_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--prepared-root", type=Path, action="append",
        default=None,
    )
    result.add_argument("--block-library", type=Path, default=DEFAULT_BLOCK_LIBRARY)
    result.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--max-states", type=int, default=0)
    result.add_argument("--skip-full-bank", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
