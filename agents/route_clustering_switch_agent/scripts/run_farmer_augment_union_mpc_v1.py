#!/usr/bin/env python3
"""Train-gated H1 MPC selector over A plus guaranteed-farmer candidates.

The experiment first scans a frozen NR020 train split.  It stops before feature
materialization when the farmer additions do not beat the best path-pure A arm
often enough.  If the signal gate passes, it trains an A backbone and a
farmer-vs-A residual gate using leave-one-opponent-out predictions only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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

import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_farmer_guaranteed_candidate_ab_v1 as farmer_ab
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual
import scan_multi_farmer_path_oracle_v1 as oracle_scan


SCHEMA = "farmer-augment-union-mpc-v1"
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1\farmer_augment_union_mpc_v1"
)
SIGNAL_FOLDS = 4
SIGNAL_MIN_DECISIONS = 8
SIGNAL_MIN_OPPONENTS = 2
SIGNAL_MIN_ANCHORS = 2
OOF_MIN_FIRES = 4


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


def seed_fold_map(seeds: Sequence[int], folds: int = SIGNAL_FOLDS) -> dict[int, int]:
    """Pre-register stable seed folds; LOPO remains the model-selection split."""

    if folds < 2:
        raise ValueError("seed diagnostic needs at least two folds")
    ordered = tuple(sorted(set(map(int, seeds))))
    return {seed: index % folds for index, seed in enumerate(ordered)}


def _margin(rewards: Sequence[float], seat: int) -> float:
    return float(rewards[seat]) - float(rewards[1 - seat])


def _strict_best(indices: Sequence[int], rewards: np.ndarray, seat: int) -> int:
    if not indices:
        raise ValueError("strict best requires at least one candidate")
    best = int(indices[0])
    rank = adaptive.reward_rank(rewards[best], seat)
    for raw in indices[1:]:
        index = int(raw)
        candidate_rank = adaptive.reward_rank(rewards[index], seat)
        if candidate_rank > rank:
            best, rank = index, candidate_rank
    return best


def _union_membership(
    candidates_a: Sequence[path_v1.PathCandidate],
    candidates_c: Sequence[path_v1.PathCandidate],
) -> tuple[list[path_v1.PathCandidate], list[int], list[int], list[int]]:
    union, indices_a, indices_c = farmer_ab.canonical_union(candidates_a, candidates_c)
    if len(indices_a) != len(set(indices_a)):
        raise AssertionError("canonical A unexpectedly contains duplicate traces")
    a_set = set(indices_a)
    farmer_indices = sorted(set(indices_c) - a_set)
    if [union[index].raw_sha256 for index in indices_a] != [
        candidate.raw_sha256 for candidate in candidates_a
    ]:
        raise AssertionError("canonical union changed an A candidate SHA or order")
    for index in farmer_indices:
        candidate = union[index]
        if candidate.kind != "SINGLE" or candidate.assignments != ((0, 0),):
            raise AssertionError("U minus A contains a non-farmer candidate")
        if not any(alias.startswith("GUARANTEE_FARMER_A0_") for alias in candidate.aliases):
            raise AssertionError("farmer addition lost its auditable alias")
    return union, list(indices_a), list(indices_c), farmer_indices


def h1_union_candidates(
    tape: Sequence[Mapping[str, Any]],
    donor_blocks: Sequence[path_v1.DonorBlock],
    anchor: int,
    step: int,
    actor_count: int,
) -> tuple[
    list[path_v1.PathCandidate], list[int], list[int], dict[str, Any],
]:
    """Return canonical H1 U, ordered A indices, and the strict farmer additions."""

    selected, selection = path_v1.select_diverse_donors(
        tape, donor_blocks, anchor, actor_count,
        top_k=path_v1.MAX_PAIRS, decision_step=step,
    )
    a_h4, c_h4, candidate_audit = farmer_ab.candidate_arms(
        tape, selected, anchor, step, actor_count,
    )
    candidates_a = path_v1.effective_candidates(a_h4, path_v1.MPC1_OVERRIDE_HORIZON)
    candidates_c = path_v1.effective_candidates(c_h4, path_v1.MPC1_OVERRIDE_HORIZON)
    union, indices_a, _, farmer_indices = _union_membership(candidates_a, candidates_c)

    reference = path_v1.effective_candidates(path_v1.donor_candidates(
        tape, selected, anchor, actor_count, decision_step=step,
    ), path_v1.MPC1_OVERRIDE_HORIZON)
    reference_sha = [candidate.raw_sha256 for candidate in reference]
    a_sha = [union[index].raw_sha256 for index in indices_a]
    if a_sha != reference_sha:
        raise AssertionError("A candidate SHA/order differs from the shared H1 generator")
    return union, indices_a, farmer_indices, {
        "selection": selection,
        "candidate_audit": candidate_audit,
        "selected_donor_ids": [donor.block_id for donor in selected],
        "a_parity_exact": True,
        "a_candidate_sha256": a_sha,
    }


def _candidate_record(
    index: int,
    candidate: path_v1.PathCandidate,
    rewards: np.ndarray,
    market_diff: np.ndarray,
    seat: int,
    membership: str,
) -> dict[str, Any]:
    return {
        "union_index": int(index), "membership": membership,
        "code": candidate.code, "kind": candidate.kind,
        "sha256": candidate.raw_sha256,
        "assignments": [list(value) for value in candidate.assignments],
        "aliases": list(candidate.aliases),
        "rewards": list(map(float, rewards[index])),
        "margin": _margin(rewards[index], seat),
        "market_diff_steps": int(market_diff[index]),
        "path_pure": bool(market_diff[index] == 0),
    }


def collect_label_scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genome: Sequence[str],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    opponent: str,
    opponent_index: int,
    seed: int,
    seat: int,
    max_windows: int,
    decision_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    env, states, _ = residual.prefix_with_history(bundle, baseline_id, opponent, seed, seat)
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    active = frozenset(
        anchor + offset
        for anchor in path_v1.ANCHORS[:max_windows]
        for offset in path_v1.WINDOW_OFFSETS
        if anchor + offset < path_v1.HORIZON
    )
    rows: list[dict[str, Any]] = []
    decision_id = int(decision_start)
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        if step in active:
            observation = dict(env.observation(seat))
            observation.update(step=step, player=seat)
            anchor = int(path_v1.ANCHORS[segment])
            route_id = str(genome[segment])
            tape = tapes[route_id]
            actor_count = max(1, len(path_v1._positions(observation)))
            union, a_indices, farmer_indices, audit = h1_union_candidates(
                tape, donors[anchor], anchor, step, actor_count,
            )
            rewards, market_diff = oracle_scan._rollout(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, union,
            )
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                path_v1.STOPS[segment:],
            ), np.float64)[0]
            if market_diff[0] != 0 or not np.array_equal(rewards[0], reference):
                raise RuntimeError("H1 U KEEP differs from the frozen schedule")
            pure_a = [index for index in a_indices if market_diff[index] == 0]
            if not pure_a or pure_a[0] != 0:
                raise RuntimeError("path-pure A lost KEEP")
            best_a = _strict_best(pure_a, rewards, seat)
            best_a_margin = _margin(rewards[best_a], seat)
            a_set, farmer_set = set(a_indices), set(farmer_indices)
            candidate_rows = [
                _candidate_record(
                    index, candidate, rewards, market_diff, seat,
                    "A" if index in a_set else "F" if index in farmer_set else "C_ALIAS",
                )
                for index, candidate in enumerate(union)
            ]
            farmer_rows = []
            for index in farmer_indices:
                value = dict(candidate_rows[index])
                value["novelty_uplift_vs_best_path_pure_A"] = (
                    float(value["margin"] - best_a_margin) if value["path_pure"] else None
                )
                farmer_rows.append(value)
            rows.append({
                "schema": SCHEMA, "split": "train_labels_only",
                "opponent": opponent, "opponent_index": int(opponent_index),
                "seed": int(seed), "seat": int(seat), "step": int(step),
                "anchor": anchor, "segment": segment, "route_id": route_id,
                "decision": decision_id, "runtime_actor_count": actor_count,
                "union_candidate_sha256": [candidate.raw_sha256 for candidate in union],
                "a_union_indices": list(map(int, a_indices)),
                "farmer_union_indices": list(map(int, farmer_indices)),
                "best_path_pure_a_union_index": int(best_a),
                "best_path_pure_a_margin": float(best_a_margin),
                "candidates": candidate_rows, "farmer_candidates": farmer_rows,
                "candidate_audit": audit,
                "native_union_rollout_calls": 1,
                "committed": "KEEP",
            })
            decision_id += 1
        pair = schedule[segment]
        env.step([
            bundle.executor.action_at(env, player, int(pair[player]), states[player])
            for player in (0, 1)
        ])
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("frozen U label trajectory did not complete")
    return rows, {
        "opponent": opponent, "seed": int(seed), "seat": int(seat),
        "rewards": list(map(float, env.rewards)), "decisions": len(rows),
        "completed": True,
    }, decision_id


def label_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    a = [candidate for candidate in candidates if candidate["membership"] == "A"]
    farmer = [candidate for candidate in candidates if candidate["membership"] == "F"]
    positive_decisions = sum(any(
        value["path_pure"]
        and float(value["novelty_uplift_vs_best_path_pure_A"]) > 0
        for value in row["farmer_candidates"]
    ) for row in rows)
    return {
        "decisions": len(rows),
        "native_union_rollout_calls": sum(int(row["native_union_rollout_calls"]) for row in rows),
        "union_all_candidates": len(candidates),
        "union_path_pure_candidates": sum(bool(value["path_pure"]) for value in candidates),
        "union_market_diff_candidates": sum(not bool(value["path_pure"]) for value in candidates),
        "union_market_diff_steps": sum(int(value["market_diff_steps"]) for value in candidates),
        "A_all_candidates": len(a),
        "A_path_pure_candidates": sum(bool(value["path_pure"]) for value in a),
        "F_all_candidates": len(farmer),
        "F_path_pure_candidates": sum(bool(value["path_pure"]) for value in farmer),
        "F_positive_novelty_candidates": sum(
            bool(value["path_pure"])
            and float(value["novelty_uplift_vs_best_path_pure_A"]) > 0
            for row in rows for value in row["farmer_candidates"]
        ),
        "F_positive_novelty_decisions": int(positive_decisions),
        "A_parity_failures": sum(not bool(row["candidate_audit"]["a_parity_exact"]) for row in rows),
    }


def signal_gate(
    rows: Sequence[Mapping[str, Any]],
    train_seeds: Sequence[int],
    *,
    minimum_decisions: int = SIGNAL_MIN_DECISIONS,
) -> dict[str, Any]:
    fold_by_seed = seed_fold_map(train_seeds, SIGNAL_FOLDS)
    positive = [row for row in rows if any(
        value["path_pure"]
        and float(value["novelty_uplift_vs_best_path_pure_A"]) > 0
        for value in row["farmer_candidates"]
    )]
    folds = sorted({fold_by_seed[int(row["seed"])] for row in positive})
    opponents = sorted({str(row["opponent"]) for row in positive})
    anchors = sorted({int(row["anchor"]) for row in positive})
    reasons = []
    if len(positive) < int(minimum_decisions):
        reasons.append("positive_novelty_decisions_below_minimum")
    if folds != list(range(SIGNAL_FOLDS)):
        reasons.append("positive_novelty_missing_seed_fold")
    if len(opponents) < SIGNAL_MIN_OPPONENTS:
        reasons.append("positive_novelty_opponent_coverage_below_minimum")
    if len(anchors) < SIGNAL_MIN_ANCHORS:
        reasons.append("positive_novelty_anchor_coverage_below_minimum")
    by_lopo = {
        opponent: sum(str(row["opponent"]) == opponent for row in positive)
        for opponent in sorted({str(row["opponent"]) for row in rows})
    }
    return {
        "passed": not reasons, "reasons": reasons,
        "minimum_positive_decisions": int(minimum_decisions),
        "positive_novelty_decisions": len(positive),
        "positive_seed_folds": folds, "required_seed_folds": list(range(SIGNAL_FOLDS)),
        "positive_opponents": opponents, "positive_anchors": anchors,
        "positive_by_lopo_opponent": by_lopo,
        "seed_fold_map": {str(seed): fold for seed, fold in fold_by_seed.items()},
        "contract": (
            "F strictly beats the best H1-local-market-identical A candidate; "
            "seed folds diagnose signal coverage only, not model calibration"
        ),
    }


def estimated_feature_bytes(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    a_rows = sum(len(row["a_union_indices"]) for row in rows)
    f_rows = sum(sum(bool(value["path_pure"]) for value in row["farmer_candidates"]) for row in rows)
    width = len(path_v1.FEATURE_NAMES)
    return {
        "A_rows": a_rows, "farmer_path_pure_rows": f_rows,
        "feature_width": width,
        "A_float32_bytes": a_rows * width * np.dtype(np.float32).itemsize,
        "farmer_base_float32_bytes": f_rows * width * np.dtype(np.float32).itemsize,
        "farmer_pair_float32_bytes": f_rows * width * 2 * np.dtype(np.float32).itemsize,
    }


def _decision_slices(decision: np.ndarray) -> list[np.ndarray]:
    if len(decision) == 0:
        return []
    if np.any(decision[1:] < decision[:-1]):
        raise ValueError("decision rows must stay contiguous")
    starts = np.flatnonzero(np.r_[True, decision[1:] != decision[:-1]])
    stops = np.r_[starts[1:], len(decision)]
    return [np.arange(start, stop, dtype=np.int64) for start, stop in zip(starts, stops, strict=True)]


def _thresholds(values: np.ndarray) -> list[float]:
    positive = np.asarray(values, np.float64)
    positive = positive[positive > 0]
    result = [0.0]
    if len(positive):
        result.extend(map(float, np.quantile(positive, [0.25, 0.5, 0.75, 0.9])))
        result.append(float(np.max(positive) + 1.0))
    return sorted(set(result))


def _selection_metrics(
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    positive: np.ndarray,
    calibration: Mapping[str, float],
    *,
    farmer_only: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    chosen_rows = []
    realized = []
    selected_opponents = []
    selected_seeds = []
    beta = float(calibration["beta"])
    threshold = float(calibration["threshold"])
    minimum = float(calibration["min_positive_fraction"])
    for indices in _decision_slices(arrays["decision"]):
        if farmer_only:
            chosen = -1
            score = mean[indices] - beta * std[indices]
            local = int(np.argmax(score))
            candidate = int(indices[local])
            if float(score[local]) > threshold and positive[candidate] >= minimum:
                chosen = candidate
            delta = 0.0 if chosen < 0 else float(arrays["deployment_uplift"][chosen])
        else:
            keep_rows = indices[arrays["edit"][indices] == path_v1.KIND_INDEX["KEEP"]]
            if len(keep_rows) != 1:
                raise RuntimeError("A OOF decision does not contain exactly one KEEP")
            chosen = int(keep_rows[0])
            edits = indices[arrays["edit"][indices] != path_v1.KIND_INDEX["KEEP"]]
            if len(edits):
                score = mean[edits] - beta * std[edits]
                local = int(np.argmax(score))
                candidate = int(edits[local])
                if float(score[local]) > threshold and positive[candidate] >= minimum:
                    chosen = candidate
            delta = float(arrays["delta_margin"][chosen])
        chosen_rows.append(chosen)
        realized.append(delta)
        if chosen >= 0 and (farmer_only or arrays["edit"][chosen] != path_v1.KIND_INDEX["KEEP"]):
            selected_opponents.append(int(arrays["opponent"][chosen]))
            selected_seeds.append(int(arrays["seed"][chosen]))
    realized_values = np.asarray(realized, np.float64)
    opponent_harm = {
        str(opponent): int(np.count_nonzero([
            value < 0 for value, row in zip(realized, chosen_rows, strict=True)
            if row >= 0 and int(arrays["opponent"][row]) == opponent
        ]))
        for opponent in sorted(set(map(int, arrays["opponent"])))
    }
    metrics = {
        "decisions": len(realized),
        "selected": int(np.count_nonzero(np.asarray(chosen_rows) >= 0)) if farmer_only else int(sum(
            row >= 0 and arrays["edit"][row] != path_v1.KIND_INDEX["KEEP"] for row in chosen_rows
        )),
        "sum_realized_delta": float(realized_values.sum()),
        "mean_realized_delta": float(realized_values.mean()) if len(realized_values) else 0.0,
        "harmful": int(np.count_nonzero(realized_values < 0)),
        "beneficial": int(np.count_nonzero(realized_values > 0)),
        "opponent_harm": opponent_harm,
        "selected_opponents": sorted(set(selected_opponents)),
        "selected_seeds": sorted(set(selected_seeds)),
        **{key: float(calibration[key]) for key in ("beta", "threshold", "min_positive_fraction")},
    }
    return metrics, np.asarray(chosen_rows, np.int64)


def zero_harm_calibration(
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    positive: np.ndarray,
    *,
    farmer_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    trials = []
    for beta in (0.0, 0.5, 1.0):
        scores = mean - beta * std
        for threshold in _thresholds(scores):
            for minimum in ((0.6, 0.7, 0.8, 0.9) if farmer_only else (0.5, 0.6, 0.7)):
                metrics, _ = _selection_metrics(
                    arrays, mean, std, positive,
                    {"beta": beta, "threshold": threshold, "min_positive_fraction": minimum},
                    farmer_only=farmer_only,
                )
                trials.append(metrics)
    allowed = [row for row in trials if row["harmful"] == 0 and not any(row["opponent_harm"].values())]
    if not allowed:
        raise RuntimeError("conservative threshold grid lost its KEEP-only fallback")
    chosen = max(allowed, key=lambda row: (
        row["sum_realized_delta"], row["beneficial"], -row["selected"],
        row["min_positive_fraction"], row["threshold"], row["beta"],
    ))
    chosen_metrics, chosen_rows = _selection_metrics(
        arrays, mean, std, positive, chosen, farmer_only=farmer_only,
    )
    return chosen_metrics, trials, chosen_rows


def _empty_feature_arrays(a_rows: int, farmer_rows: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    width = len(path_v1.FEATURE_NAMES)
    a = {
        "features": np.empty((a_rows, width), np.float32),
        "target_signed_log_margin": np.empty(a_rows, np.float32),
        "delta_margin": np.empty(a_rows, np.float64),
        "terminal_rewards": np.empty((a_rows, 2), np.float64),
        "seed": np.empty(a_rows, np.int64),
        "seat": np.empty(a_rows, np.int8),
        "opponent": np.empty(a_rows, np.int16),
        "step": np.empty(a_rows, np.int16),
        "edit": np.empty(a_rows, np.int8),
        "decision": np.empty(a_rows, np.int32),
        "selected_by_oracle": np.empty(a_rows, np.bool_),
        "candidate_sha256": np.empty(a_rows, "U64"),
        "union_index": np.empty(a_rows, np.int16),
        "path_pure": np.empty(a_rows, np.bool_),
    }
    farmer = {
        "features": np.empty((farmer_rows, width), np.float32),
        "novelty_uplift": np.empty(farmer_rows, np.float64),
        "delta_margin": np.empty(farmer_rows, np.float64),
        "seed": np.empty(farmer_rows, np.int64),
        "seat": np.empty(farmer_rows, np.int8),
        "opponent": np.empty(farmer_rows, np.int16),
        "step": np.empty(farmer_rows, np.int16),
        "anchor": np.empty(farmer_rows, np.int16),
        "decision": np.empty(farmer_rows, np.int32),
        "candidate_sha256": np.empty(farmer_rows, "U64"),
        "union_index": np.empty(farmer_rows, np.int16),
    }
    return a, farmer


def collect_feature_arrays(
    bundle: Any,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, Any],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    genome: Sequence[str],
    scenarios: Sequence[tuple[str, int, int]],
    opponents: Sequence[str],
    labels: Sequence[Mapping[str, Any]],
    max_windows: int,
    max_feature_bytes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Second pass: allocate once only after the label-only signal gate passes."""

    estimate = estimated_feature_bytes(labels)
    required = estimate["A_float32_bytes"] + estimate["farmer_base_float32_bytes"]
    if required > int(max_feature_bytes):
        raise MemoryError(
            f"dense feature preflight needs {required} bytes, above --max-feature-bytes"
        )
    a, farmer = _empty_feature_arrays(estimate["A_rows"], estimate["farmer_path_pure_rows"])
    by_key = {
        (str(row["opponent"]), int(row["seed"]), int(row["seat"]), int(row["step"])): row
        for row in labels
    }
    if len(by_key) != len(labels):
        raise RuntimeError("train decision keys are not unique")
    plan_tape = residual.scheduled_tape(baseline_tape, tapes, genome)
    a_at = farmer_at = parity_checks = 0
    active = frozenset(
        anchor + offset
        for anchor in path_v1.ANCHORS[:max_windows]
        for offset in path_v1.WINDOW_OFFSETS
        if anchor + offset < path_v1.HORIZON
    )
    for opponent, seed, seat in scenarios:
        env, states, history = residual.prefix_with_history(
            bundle, baseline_id, opponent, seed, seat,
        )
        schedule = residual._full_schedule(bundle, genome, opponent, seat)
        for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
            observation = dict(env.observation(seat))
            observation.update(step=step, player=seat)
            history.update(observation)
            segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
            if step in active:
                label = by_key[(opponent, int(seed), int(seat), int(step))]
                anchor = int(path_v1.ANCHORS[segment])
                route_id = str(genome[segment])
                tape = tapes[route_id]
                actor_count = max(1, len(path_v1._positions(observation)))
                union, a_indices, farmer_indices, _ = h1_union_candidates(
                    tape, donors[anchor], anchor, step, actor_count,
                )
                if [candidate.raw_sha256 for candidate in union] != list(
                    label["union_candidate_sha256"]
                ):
                    raise RuntimeError("feature pass regenerated a different U candidate panel")
                source = path_v1._source_positions(carriers.get(route_id), step)
                features = path_v1.candidate_feature_rows(
                    observation, history, plan_tape, tape, source, union, seat, step,
                )
                records = list(label["candidates"])
                keep_margin = float(records[0]["margin"])
                best_all_a = max(
                    a_indices,
                    key=lambda index: adaptive.reward_rank(records[index]["rewards"], seat),
                )
                for index in a_indices:
                    record = records[index]
                    delta = float(record["margin"] - keep_margin)
                    candidate = union[index]
                    a["features"][a_at] = features[index]
                    a["target_signed_log_margin"][a_at] = residual._signed_log(delta)
                    a["delta_margin"][a_at] = delta
                    a["terminal_rewards"][a_at] = record["rewards"]
                    a["seed"][a_at] = int(seed)
                    a["seat"][a_at] = int(seat)
                    a["opponent"][a_at] = int(opponents.index(opponent))
                    a["step"][a_at] = int(step)
                    a["edit"][a_at] = path_v1.KIND_INDEX[candidate.kind]
                    a["decision"][a_at] = int(label["decision"])
                    a["selected_by_oracle"][a_at] = index == best_all_a
                    a["candidate_sha256"][a_at] = candidate.raw_sha256
                    a["union_index"][a_at] = int(index)
                    a["path_pure"][a_at] = bool(record["path_pure"])
                    a_at += 1
                f_by_index = {
                    int(value["union_index"]): value for value in label["farmer_candidates"]
                }
                for index in farmer_indices:
                    record = f_by_index[index]
                    if not bool(record["path_pure"]):
                        continue
                    farmer["features"][farmer_at] = features[index]
                    farmer["novelty_uplift"][farmer_at] = float(
                        record["novelty_uplift_vs_best_path_pure_A"]
                    )
                    farmer["delta_margin"][farmer_at] = float(record["margin"] - keep_margin)
                    farmer["seed"][farmer_at] = int(seed)
                    farmer["seat"][farmer_at] = int(seat)
                    farmer["opponent"][farmer_at] = int(opponents.index(opponent))
                    farmer["step"][farmer_at] = int(step)
                    farmer["anchor"][farmer_at] = int(anchor)
                    farmer["decision"][farmer_at] = int(label["decision"])
                    farmer["candidate_sha256"][farmer_at] = union[index].raw_sha256
                    farmer["union_index"][farmer_at] = int(index)
                    farmer_at += 1
                parity_checks += 1
            pair = schedule[segment]
            env.step([
                bundle.executor.action_at(env, player, int(pair[player]), states[player])
                for player in (0, 1)
            ])
        if not env.done or int(env.step_count) != path_v1.HORIZON:
            raise RuntimeError("feature replay trajectory did not complete")
    if a_at != len(a["features"]) or farmer_at != len(farmer["features"]):
        raise RuntimeError("preallocated feature panel row count changed on replay")
    return a, farmer, {
        "estimated": estimate, "allocated_feature_bytes": required,
        "parity_checks": parity_checks, "parity_failures": 0,
        "construction": "signal-gated second pass with exact preallocation",
    }


def _decision_weights(decision: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(decision, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _lopo_predictions(
    arrays: Mapping[str, np.ndarray],
    trees: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x = arrays["features"]
    y = arrays["target_signed_log_margin"]
    opponents = np.asarray(arrays["opponent"], np.int16)
    unique = tuple(sorted(set(map(int, opponents))))
    if len(unique) < 2:
        raise ValueError("LOPO ranker requires at least two opponents")
    mean = np.full(len(y), np.nan, np.float64)
    std = np.full(len(y), np.nan, np.float64)
    positive = np.full(len(y), np.nan, np.float64)
    folds = []
    for fold, opponent in enumerate(unique):
        valid = np.flatnonzero(opponents == opponent)
        train = np.flatnonzero(opponents != opponent)
        model = residual.ExtraTreesRegressor(**residual._model_params(
            trees, random_seed + fold * 1009,
        ))
        model.fit(
            x[train], y[train],
            sample_weight=_decision_weights(np.asarray(arrays["decision"])[train]),
        )
        predictions = residual._tree_predictions(model, x[valid])
        mean[valid] = predictions.mean(axis=1)
        std[valid] = predictions.std(axis=1)
        positive[valid] = (predictions > 0).mean(axis=1)
        folds.append({
            "left_out_opponent_index": opponent,
            "train_rows": len(train), "valid_rows": len(valid),
        })
    if not all(np.isfinite(value).all() for value in (mean, std, positive)):
        raise RuntimeError("LOPO prediction panel is incomplete")
    return mean, std, positive, folds


def fit_a_backbone(
    arrays: Mapping[str, np.ndarray],
    trees: int,
    cv_trees: int,
    random_seed: int,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    mean, std, positive, folds = _lopo_predictions(arrays, cv_trees, random_seed)
    chosen, trials, choices = zero_harm_calibration(
        arrays, mean, std, positive, farmer_only=False,
    )
    model = residual.ExtraTreesRegressor(**residual._model_params(trees, random_seed))
    model.fit(
        arrays["features"], arrays["target_signed_log_margin"],
        sample_weight=_decision_weights(arrays["decision"]),
    )
    calibration = {
        "primary_grouping": "leave-one-train-opponent-out",
        "folds": folds, "chosen": chosen, "trials": trials,
        "zero_harm_each_lopo_fold": not any(chosen["opponent_harm"].values()),
        "model_params": residual._model_params(trees, random_seed),
    }
    return model, calibration, {
        "mean": mean, "std": std, "positive_fraction": positive,
    }, choices


def farmer_pair_arrays(
    a: Mapping[str, np.ndarray],
    farmer: Mapping[str, np.ndarray],
    a_choices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pair F with the deployable A LOPO choice; preserve novelty separately."""

    slices = _decision_slices(a["decision"])
    if len(slices) != len(a_choices):
        raise ValueError("A OOF choices do not align with its decisions")
    choice_by_decision = {
        int(a["decision"][indices[0]]): int(choice)
        for indices, choice in zip(slices, a_choices, strict=True)
    }
    width = len(path_v1.FEATURE_NAMES)
    pair = np.empty((len(farmer["features"]), 2 * width), np.float32)
    deployment = np.empty(len(pair), np.float64)
    for index in range(len(pair)):
        decision = int(farmer["decision"][index])
        a_index = choice_by_decision[decision]
        f_values = farmer["features"][index]
        a_values = a["features"][a_index]
        pair[index, :width] = f_values
        pair[index, width:] = f_values - a_values
        deployment[index] = float(
            farmer["delta_margin"][index] - a["delta_margin"][a_index]
        )
    return {
        "features": pair,
        "target_signed_log_margin": np.asarray(
            [residual._signed_log(value) for value in deployment], np.float32,
        ),
        "deployment_uplift": deployment,
        "novelty_uplift": np.asarray(farmer["novelty_uplift"], np.float64),
        **{
            key: np.asarray(farmer[key])
            for key in (
                "seed", "seat", "opponent", "step", "anchor", "decision",
                "candidate_sha256", "union_index",
            )
        },
    }


def fit_farmer_uplift_gate(
    arrays: Mapping[str, np.ndarray],
    trees: int,
    cv_trees: int,
    random_seed: int,
    train_seeds: Sequence[int],
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    mean, std, positive, folds = _lopo_predictions(arrays, cv_trees, random_seed)
    chosen, trials, choices = zero_harm_calibration(
        arrays, mean, std, positive, farmer_only=True,
    )
    fold_by_seed = seed_fold_map(train_seeds)
    seed_fold_harm = {str(fold): 0 for fold in range(SIGNAL_FOLDS)}
    for indices, choice in zip(_decision_slices(arrays["decision"]), choices, strict=True):
        if choice < 0:
            continue
        fold = fold_by_seed[int(arrays["seed"][indices[0]])]
        seed_fold_harm[str(fold)] += int(arrays["deployment_uplift"][choice] < 0)
    model = residual.ExtraTreesRegressor(**residual._model_params(trees, random_seed))
    model.fit(
        arrays["features"], arrays["target_signed_log_margin"],
        sample_weight=_decision_weights(arrays["decision"]),
    )
    calibration = {
        "primary_grouping": "leave-one-train-opponent-out",
        "target": "F minus A-backbone LOPO-OOF choice",
        "folds": folds, "chosen": chosen, "trials": trials,
        "zero_harm_each_lopo_fold": not any(chosen["opponent_harm"].values()),
        "secondary_seed_fold_harm": seed_fold_harm,
        "model_params": residual._model_params(trees, random_seed),
    }
    return model, calibration, {
        "mean": mean, "std": std, "positive_fraction": positive,
    }, choices


def choose_farmer_override(
    model: Any,
    calibration: Mapping[str, Any],
    farmer_pair_features: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if len(farmer_pair_features) == 0:
        empty = np.empty(0, np.float64)
        return -1, empty, empty, empty
    predictions = residual._tree_predictions(model, farmer_pair_features)
    mean = predictions.mean(axis=1)
    std = predictions.std(axis=1)
    positive = (predictions > 0).mean(axis=1)
    chosen = calibration["chosen"] if "chosen" in calibration else calibration
    score = mean - float(chosen["beta"]) * std
    local = int(np.argmax(score))
    if not (
        float(score[local]) > float(chosen["threshold"])
        and positive[local] >= float(chosen["min_positive_fraction"])
    ):
        local = -1
    return local, mean, std, positive


def learned_scenario(
    bundle: Any,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, Any],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    opponent: str,
    genome: Sequence[str],
    seed: int,
    seat: int,
    a_model: Any,
    a_calibration: Mapping[str, Any],
    farmer_model: Any | None,
    farmer_calibration: Mapping[str, Any] | None,
    max_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run deployable H1 MPC; diagnostic rollouts never affect its choice."""

    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(baseline_tape, tapes, genome)
    active = frozenset(
        anchor + offset
        for anchor in path_v1.ANCHORS[:max_windows]
        for offset in path_v1.WINDOW_OFFSETS
        if anchor + offset < path_v1.HORIZON
    )
    decisions = []
    edits = farmer_fires = fallback_checks = selected_market_diff = 0
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        selected: path_v1.PathCandidate | None = None
        if step in active:
            anchor = int(path_v1.ANCHORS[segment])
            route_id = str(genome[segment])
            tape = tapes[route_id]
            actor_count = max(1, len(path_v1._positions(observation)))
            union, a_indices, farmer_indices, audit = h1_union_candidates(
                tape, donors[anchor], anchor, step, actor_count,
            )
            source = path_v1._source_positions(carriers.get(route_id), step)
            features = path_v1.candidate_feature_rows(
                observation, history, plan_tape, tape, source, union, seat, step,
            )
            a_features = features[np.asarray(a_indices, np.int32)]
            a_local, a_mean, a_std, a_positive = residual.model_choice(
                a_model, a_features, a_calibration["chosen"],
            )
            a_union_index = int(a_indices[a_local])
            selected_index = a_union_index
            f_local = -1
            f_mean = f_std = f_positive = np.empty(0, np.float64)
            if farmer_model is not None and farmer_indices:
                f_features = features[np.asarray(farmer_indices, np.int32)]
                pair_features = np.concatenate((
                    f_features, f_features - features[a_union_index],
                ), axis=1).astype(np.float32)
                f_local, f_mean, f_std, f_positive = choose_farmer_override(
                    farmer_model, farmer_calibration or {}, pair_features,
                )
                if f_local >= 0:
                    selected_index = int(farmer_indices[f_local])
                    farmer_fires += 1
                else:
                    fallback_checks += 1
                    if selected_index != a_union_index:
                        raise AssertionError("farmer gate fallback changed A")
            selected = union[selected_index]
            # The choice is frozen above.  This two-arm rollout audits only the
            # selected action's immediate H1 market effect; reward is ignored.
            _, selected_market = oracle_scan._rollout(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, (union[0], selected),
            )
            selected_market_diff += int(selected_market[1])
            edits += int(not selected.keep)
            decisions.append({
                "opponent": opponent, "seed": int(seed), "seat": int(seat),
                "step": int(step), "anchor": anchor, "route_id": route_id,
                "a_candidate_sha256": [union[index].raw_sha256 for index in a_indices],
                "farmer_candidate_sha256": [union[index].raw_sha256 for index in farmer_indices],
                "a_selected_union_index": a_union_index,
                "a_selected_sha256": union[a_union_index].raw_sha256,
                "farmer_gate_fired": bool(f_local >= 0),
                "selected_union_index": selected_index,
                "selected_sha256": selected.raw_sha256,
                "selected_kind": selected.kind,
                "selected_market_diff_steps": int(selected_market[1]),
                "fallback_exact_A": bool(f_local >= 0 or selected_index == a_union_index),
                "a_prediction_mean": a_mean.tolist(),
                "a_prediction_std": a_std.tolist(),
                "a_positive_tree_fraction": a_positive.tolist(),
                "farmer_prediction_mean": f_mean.tolist(),
                "farmer_prediction_std": f_std.tolist(),
                "farmer_positive_tree_fraction": f_positive.tolist(),
                "candidate_audit": audit,
                "choice_frozen_before_market_audit": True,
            })
        pair = schedule[segment]
        if selected is None or selected.keep:
            env.step([
                bundle.executor.action_at(env, player, int(pair[player]), states[player])
                for player in (0, 1)
            ])
        else:
            path_v1._commit_units(bundle, env, states, pair, seat, selected, 0)
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("learned farmer-U trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    outcome, margin = adaptive.reward_rank(rewards, seat)
    return ({
        "arm": "A_plus_farmer" if farmer_model is not None else "A_backbone",
        "opponent": opponent, "seed": int(seed), "seat": int(seat),
        "rewards": list(rewards), "outcome": outcome, "margin": margin,
        "edits": edits, "farmer_fires": farmer_fires,
        "fallback_exact_A_checks": fallback_checks,
        "selected_market_diff_steps": selected_market_diff,
        "completed": True,
    }, decisions)


def paired_validation_summary(
    a_rows: Sequence[Mapping[str, Any]],
    augmented_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    key = lambda row: (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
    a = {key(row): row for row in a_rows}
    augmented = {key(row): row for row in augmented_rows}
    if a.keys() != augmented.keys():
        raise RuntimeError("paired validation scenario keys differ")
    pairs = []
    for scenario in sorted(a):
        left, right = a[scenario], augmented[scenario]
        left_rank = (int(left["outcome"]), float(left["margin"]))
        right_rank = (int(right["outcome"]), float(right["margin"]))
        pairs.append({
            "opponent": scenario[0], "seed": scenario[1], "seat": scenario[2],
            "rank_compare": int((right_rank > left_rank) - (right_rank < left_rank)),
            "outcome_delta": right_rank[0] - left_rank[0],
            "margin_delta": right_rank[1] - left_rank[1],
            "farmer_fires": int(right["farmer_fires"]),
        })
    by_opponent = {
        opponent: {
            "scenarios": len(values),
            "sum_margin_delta": float(sum(row["margin_delta"] for row in values)),
            "outcome_regressions": sum(row["outcome_delta"] < 0 for row in values),
            "strict_improvements": sum(row["rank_compare"] > 0 for row in values),
        }
        for opponent, values in (
            (opponent, [row for row in pairs if row["opponent"] == opponent])
            for opponent in sorted({row["opponent"] for row in pairs})
        )
    }
    return {
        "scenarios": len(pairs),
        "outcome_regressions": sum(row["outcome_delta"] < 0 for row in pairs),
        "strict_improvements": sum(row["rank_compare"] > 0 for row in pairs),
        "sum_margin_delta": float(sum(row["margin_delta"] for row in pairs)),
        "farmer_fires": sum(row["farmer_fires"] for row in pairs),
        "by_opponent": by_opponent,
        "pairs": pairs,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    labels = report["train_label_summary"]
    signal = report["signal_gate"]
    lines = [
        "# FARMER-AUGMENT-UNION-MPC-v1",
        "",
        f"Status: {report['status']}",
        "",
        "This is an H1 proxy experiment, not submitted-agent pool evidence.",
        "",
        f"Train decisions: {labels['decisions']}; one-U-rollout calls: "
        f"{labels['native_union_rollout_calls']}.",
        "",
        f"U all/path-pure/market-diff candidates: {labels['union_all_candidates']}/"
        f"{labels['union_path_pure_candidates']}/{labels['union_market_diff_candidates']}.",
        "",
        f"Novel farmer-positive decisions: {signal['positive_novelty_decisions']}; "
        f"signal gate: {signal['passed']}.",
        "",
        f"Feature/model training executed: {report['training_executed']}.",
        f"Validation executed: {report['validation_executed']}.",
    ]
    if report.get("validation_summary"):
        validation = report["validation_summary"]
        lines.extend((
            "", f"Validation farmer fires: {validation['farmer_fires']}; "
            f"strict improvements/regressions: {validation['strict_improvements']}/"
            f"{validation['outcome_regressions']}; margin sum: "
            f"{validation['sum_margin_delta']:.3f}.",
        ))
    return "\n".join(lines) + "\n"


def _finalize(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(markdown_path, _report_markdown(report))
    print(json.dumps({
        "event": "farmer_augment_union_mpc_complete",
        "status": report["status"], "output": str(output),
        "signal_passed": report["signal_gate"]["passed"],
        "training_executed": report["training_executed"],
        "validation_executed": report["validation_executed"],
    }, sort_keys=True), flush=True)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    train_opponents = tuple(args.train_opponent or path_v1.TRAIN_OPPONENTS)
    validation_opponents = tuple(args.validation_opponent or path_v1.HELDOUT_OPPONENTS)
    train_seeds = tuple(range(
        args.train_seed_start, args.train_seed_start + args.train_seed_count,
    ))
    validation_seeds = tuple(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_seed_count,
    ))
    if set(train_opponents) & set(validation_opponents):
        raise ValueError("train and validation opponents must be disjoint")
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("train and validation seeds must be disjoint")
    if (set(train_seeds) | set(validation_seeds)) & residual.SEALED_SEEDS:
        raise ValueError("farmer U experiment cannot open sealed seeds")
    if args.base_genome_id != "NR020" or args.composition_genome_id is not None:
        raise ValueError("FARMER-AUGMENT-UNION-MPC-v1 freezes exactly NR020")
    if not args.smoke and (
        train_opponents != path_v1.TRAIN_OPPONENTS
        or validation_opponents != path_v1.HELDOUT_OPPONENTS
        or train_seeds != path_v1.TRAIN_SEEDS
        or validation_seeds != path_v1.VALIDATION_SEEDS[:4]
        or args.max_windows != len(path_v1.ANCHORS)
        or args.signal_min_decisions != SIGNAL_MIN_DECISIONS
    ):
        raise ValueError(
            "formal run requires fixed 6/6 opponents, 8/4 seeds, 21 windows and signal>=8"
        )
    output.mkdir(parents=True, exist_ok=False)
    split = {
        "schema": "farmer-augment-union-mpc-split-v1",
        "train_opponents": list(train_opponents),
        "validation_opponents": list(validation_opponents),
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "seats": [0, 1],
        "anchors": list(map(int, path_v1.ANCHORS[:args.max_windows])),
        "decision_offsets": list(path_v1.WINDOW_OFFSETS),
        "controller": "H1 score / H1 execute / replan every step",
        "primary_model_selection": "leave one complete train opponent out",
        "secondary_seed_diagnostic": seed_fold_map(train_seeds),
        "validation_oracle_opened_after_freeze": True,
        "sealed_test_executed": False,
    }
    split["sha256"] = path_v1._sha256(split)
    split_path = output / "split_manifest.json"
    residual._atomic_text(split_path, json.dumps(split, ensure_ascii=False, indent=2) + "\n")

    args.train_opponent = list(train_opponents)
    args.validation_opponent = list(validation_opponents)
    (bundle, baseline_id, baseline_tape, tapes, genomes, carriers, donors,
     inputs, source) = path_v1.load_experiment(args)
    genome = genomes["NR020"]
    train_scenarios = [
        (opponent, seed, seat)
        for seed in train_seeds for opponent in train_opponents for seat in (0, 1)
    ]
    label_rows: list[dict[str, Any]] = []
    trajectories = []
    decision_id = 0
    for index, (opponent, seed, seat) in enumerate(train_scenarios, 1):
        rows, trajectory, decision_id = collect_label_scenario(
            bundle, baseline_id, tapes, genome, donors,
            opponent, train_opponents.index(opponent), seed, seat,
            args.max_windows, decision_id,
        )
        label_rows.extend(rows)
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "farmer_u_train_labels_complete",
            "index": index, "total": len(train_scenarios),
            "opponent": opponent, "seed": seed, "seat": seat,
            "decisions": len(rows),
        }, sort_keys=True), flush=True)
    labels_path = output / "train_union_labels.jsonl"
    trajectories_path = output / "train_frozen_trajectories.jsonl"
    _write_jsonl(labels_path, label_rows)
    _write_jsonl(trajectories_path, trajectories)
    labels_summary = label_summary(label_rows)
    signal = signal_gate(
        label_rows, train_seeds, minimum_decisions=args.signal_min_decisions,
    )
    preflight = estimated_feature_bytes(label_rows)
    common_report: dict[str, Any] = {
        "schema": SCHEMA, "status": "running",
        "claim": "train-only candidate-conditioned H1 A plus farmer residual gate",
        "not_claimed": "not a submitted-agent pool result or strict lineage OOD proof",
        "splits": split, "baseline_route_id": baseline_id,
        "frozen_genome": {
            "genome_id": "NR020", "route_ids": list(genome),
            "sha256": path_v1._sha256(list(genome)),
        },
        "candidate_contract": {
            "A": "ordered SHA-identical current effective H1 candidates",
            "F": "canonical H1 traces in C but not A; farmer0 SINGLE only",
            "U": "canonical A union F, evaluated once then sliced",
            "novelty_uplift": "F minus best path-pure A oracle",
            "deployment_uplift": "F minus A-backbone LOPO-OOF choice",
            "market": "per-candidate H1 market_diff reported; not used to choose online",
        },
        "train_label_summary": labels_summary,
        "signal_gate": signal,
        "feature_preflight": preflight,
        "training_executed": False, "validation_executed": False,
        "all_train_trajectories_complete": all(row["completed"] for row in trajectories),
        "source": source, "inputs": inputs,
        "implementation": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": residual._sha256_file(Path(__file__).resolve()),
            },
            "shared_A_generator": {
                "path": str(Path(path_v1.__file__).resolve()),
                "sha256": residual._sha256_file(Path(path_v1.__file__).resolve()),
            },
            "farmer_generator": {
                "path": str(Path(farmer_ab.__file__).resolve()),
                "sha256": residual._sha256_file(Path(farmer_ab.__file__).resolve()),
            },
        },
        "artifacts": {
            labels_path.name: residual._artifact(labels_path, len(label_rows)),
            trajectories_path.name: residual._artifact(trajectories_path, len(trajectories)),
            split_path.name: residual._artifact(split_path),
        },
        "sealed_test_executed": False,
    }
    if labels_summary["A_parity_failures"] or (
        labels_summary["native_union_rollout_calls"] != labels_summary["decisions"]
    ):
        common_report["status"] = "invalid_candidate_or_rollout_contract"
        common_report["elapsed_seconds"] = time.perf_counter() - started
        return _finalize(output, common_report)
    if not signal["passed"]:
        common_report["status"] = "farmer_candidate_signal_insufficient"
        common_report["decision"] = (
            "Stop before 2844D feature allocation/model fitting; current farmer additions "
            "do not prove enough path-pure novelty beyond A."
        )
        common_report["elapsed_seconds"] = time.perf_counter() - started
        return _finalize(output, common_report)

    a_arrays, farmer_base, feature_audit = collect_feature_arrays(
        bundle, baseline_id, baseline_tape, tapes, carriers, donors, genome,
        train_scenarios, train_opponents, label_rows, args.max_windows,
        args.max_feature_bytes,
    )
    common_report["training_executed"] = True
    common_report["feature_audit"] = feature_audit
    a_model, a_calibration, a_oof, a_choices = fit_a_backbone(
        a_arrays, args.trees, args.cv_trees, args.random_seed,
    )
    expected_lopo = set(range(len(train_opponents)))
    farmer_lopo = set(map(int, farmer_base["opponent"]))
    if farmer_lopo != expected_lopo:
        common_report["status"] = "farmer_candidate_lopo_coverage_insufficient"
        common_report["decision"] = (
            "Farmer rows do not cover all six train opponents; do not calibrate a "
            "purported six-fold LOPO safety gate."
        )
        common_report["farmer_lopo_present"] = sorted(farmer_lopo)
        common_report["farmer_lopo_required"] = sorted(expected_lopo)
        common_report["elapsed_seconds"] = time.perf_counter() - started
        return _finalize(output, common_report)
    farmer_arrays = farmer_pair_arrays(a_arrays, farmer_base, a_choices)
    farmer_model, farmer_calibration, farmer_oof, farmer_choices = (
        fit_farmer_uplift_gate(
            farmer_arrays, args.trees, args.cv_trees,
            args.random_seed + 100_003, train_seeds,
        )
    )

    a_panel_path = output / "training_panel_A_backbone.npz"
    farmer_panel_path = output / "training_panel_farmer_uplift.npz"
    a_oof_path = output / "oof_A_backbone_lopo.npz"
    farmer_oof_path = output / "oof_farmer_uplift_lopo.npz"
    a_model_path = output / "A_backbone.joblib"
    farmer_model_path = output / "farmer_uplift_gate.joblib"
    _atomic_npz(a_panel_path, a_arrays)
    _atomic_npz(farmer_panel_path, farmer_arrays)
    _atomic_npz(a_oof_path, a_oof)
    _atomic_npz(farmer_oof_path, farmer_oof)
    _atomic_joblib(a_model_path, a_model)
    _atomic_joblib(farmer_model_path, farmer_model)
    a_manifest = {
        "schema": "farmer-u-A-backbone-v1",
        "model": "ExtraTreesRegressor candidate-conditioned H1 value",
        "feature_count": len(path_v1.FEATURE_NAMES),
        "training_rows": len(a_arrays["features"]),
        "training_decisions": len(np.unique(a_arrays["decision"])),
        "calibration": a_calibration,
        "model_sha256": residual._sha256_file(a_model_path),
        "panel_sha256": residual._sha256_file(a_panel_path),
        "oof_sha256": residual._sha256_file(a_oof_path),
    }
    farmer_manifest = {
        "schema": "farmer-u-uplift-gate-v1",
        "model": "ExtraTreesRegressor paired F versus A LOPO-OOF choice",
        "feature_count": 2 * len(path_v1.FEATURE_NAMES),
        "feature_contract": "[x_F, x_F - x_A_backbone_choice]",
        "training_rows": len(farmer_arrays["features"]),
        "training_decisions": len(np.unique(farmer_arrays["decision"])),
        "novelty_positive_rows": int(np.count_nonzero(farmer_arrays["novelty_uplift"] > 0)),
        "deployment_positive_rows": int(np.count_nonzero(farmer_arrays["deployment_uplift"] > 0)),
        "calibration": farmer_calibration,
        "model_sha256": residual._sha256_file(farmer_model_path),
        "panel_sha256": residual._sha256_file(farmer_panel_path),
        "oof_sha256": residual._sha256_file(farmer_oof_path),
    }
    a_manifest_path = output / "model_manifest_A_backbone.json"
    farmer_manifest_path = output / "model_manifest_farmer_uplift.json"
    residual._atomic_text(
        a_manifest_path, json.dumps(a_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        farmer_manifest_path,
        json.dumps(farmer_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    common_report["models"] = {
        "A_backbone": a_manifest, "farmer_uplift_gate": farmer_manifest,
    }
    for path, rows in (
        (a_panel_path, len(a_arrays["features"])),
        (farmer_panel_path, len(farmer_arrays["features"])),
        (a_oof_path, len(a_arrays["features"])),
        (farmer_oof_path, len(farmer_arrays["features"])),
        (a_model_path, None), (farmer_model_path, None),
        (a_manifest_path, None), (farmer_manifest_path, None),
    ):
        common_report["artifacts"][path.name] = residual._artifact(path, rows)

    farmer_oof_choice = farmer_calibration["chosen"]
    oof_gate = {
        "zero_harm_each_lopo_fold": bool(
            farmer_calibration["zero_harm_each_lopo_fold"]
        ),
        "zero_harm_each_seed_diagnostic_fold": not any(
            farmer_calibration["secondary_seed_fold_harm"].values()
        ),
        "minimum_fires": OOF_MIN_FIRES,
        "fires": int(farmer_oof_choice["selected"]),
        "selected_opponents": list(farmer_oof_choice["selected_opponents"]),
        "selected_seeds": list(farmer_oof_choice["selected_seeds"]),
    }
    oof_gate["passed"] = bool(
        oof_gate["zero_harm_each_lopo_fold"]
        and oof_gate["zero_harm_each_seed_diagnostic_fold"]
        and oof_gate["fires"] >= OOF_MIN_FIRES
        and len(oof_gate["selected_opponents"]) >= 2
        and len(oof_gate["selected_seeds"]) >= 2
    )
    common_report["oof_deployment_gate"] = oof_gate
    if not oof_gate["passed"]:
        common_report["status"] = "farmer_uplift_selector_oof_insufficient"
        common_report["decision"] = (
            "Novel candidate signal exists, but the LOPO deployment gate cannot fire "
            "at least four times across opponents/seeds with zero harm."
        )
        common_report["elapsed_seconds"] = time.perf_counter() - started
        return _finalize(output, common_report)

    frozen_paths = (
        a_model_path, farmer_model_path, a_manifest_path, farmer_manifest_path,
        a_panel_path, farmer_panel_path, a_oof_path, farmer_oof_path,
        split_path, labels_path,
    )
    frozen_identity = {
        str(path): (residual._sha256_file(path), path.stat().st_mtime_ns)
        for path in frozen_paths
    }
    validation_scenarios = [
        (opponent, seed, seat)
        for seed in validation_seeds
        for opponent in validation_opponents
        for seat in (0, 1)
    ]
    validation_a = []
    validation_augmented = []
    decisions_a = []
    decisions_augmented = []
    for index, (opponent, seed, seat) in enumerate(validation_scenarios, 1):
        a_row, a_decisions = learned_scenario(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors,
            opponent, genome, seed, seat, a_model, a_calibration,
            None, None, args.max_windows,
        )
        augmented_row, augmented_decisions = learned_scenario(
            bundle, baseline_id, baseline_tape, tapes, carriers, donors,
            opponent, genome, seed, seat, a_model, a_calibration,
            farmer_model, farmer_calibration, args.max_windows,
        )
        validation_a.append(a_row)
        validation_augmented.append(augmented_row)
        decisions_a.extend(a_decisions)
        decisions_augmented.extend(augmented_decisions)
        print(json.dumps({
            "event": "farmer_u_validation_complete",
            "index": index, "total": len(validation_scenarios),
            "opponent": opponent, "seed": seed, "seat": seat,
            "farmer_fires": augmented_row["farmer_fires"],
        }, sort_keys=True), flush=True)
    after_identity = {
        str(path): (residual._sha256_file(path), path.stat().st_mtime_ns)
        for path in frozen_paths
    }
    if after_identity != frozen_identity:
        raise RuntimeError("frozen train/model artifacts changed during validation")
    validation_summary = paired_validation_summary(validation_a, validation_augmented)
    fired = [row for row in decisions_augmented if row["farmer_gate_fired"]]
    fallback_failures = sum(
        not bool(row["fallback_exact_A"]) for row in decisions_augmented
        if not row["farmer_gate_fired"]
    )
    per_opponent_nonnegative = all(
        row["sum_margin_delta"] >= 0
        for row in validation_summary["by_opponent"].values()
    )
    validation_gate = {
        "models_frozen_before_validation": after_identity == frozen_identity,
        "all_trajectories_complete": all(
            bool(row["completed"]) for row in (*validation_a, *validation_augmented)
        ),
        "fallback_exact_A_failures": fallback_failures,
        "farmer_fires": len(fired),
        "fire_opponents": sorted({str(row["opponent"]) for row in fired}),
        "fire_seeds": sorted({int(row["seed"]) for row in fired}),
        "selected_farmer_market_diff_steps": sum(
            int(row["selected_market_diff_steps"]) for row in fired
        ),
        "outcome_regressions": validation_summary["outcome_regressions"],
        "strict_improvements": validation_summary["strict_improvements"],
        "sum_margin_delta": validation_summary["sum_margin_delta"],
        "each_opponent_nonnegative_margin_sum": per_opponent_nonnegative,
    }
    validation_gate["passed"] = bool(
        validation_gate["models_frozen_before_validation"]
        and validation_gate["all_trajectories_complete"]
        and validation_gate["fallback_exact_A_failures"] == 0
        and validation_gate["farmer_fires"] >= 4
        and len(validation_gate["fire_opponents"]) >= 2
        and len(validation_gate["fire_seeds"]) >= 2
        and validation_gate["selected_farmer_market_diff_steps"] == 0
        and validation_gate["outcome_regressions"] == 0
        and validation_gate["strict_improvements"] > 0
        and validation_gate["sum_margin_delta"] > 0
        and validation_gate["each_opponent_nonnegative_margin_sum"]
    )
    common_report["validation_executed"] = True
    common_report["validation_summary"] = validation_summary
    common_report["validation_gate"] = validation_gate
    common_report["status"] = (
        "farmer_augment_union_mpc_feasible_proxy"
        if validation_gate["passed"] else "farmer_augment_union_mpc_validation_failed"
    )
    common_report["decision"] = (
        "Candidate-conditioned farmer residual is safe enough for a larger proxy test."
        if validation_gate["passed"] else
        "Retain A; the farmer residual failed a pre-registered heldout safety/value gate."
    )
    validation_files = {
        "validation_A.jsonl": validation_a,
        "validation_A_plus_farmer.jsonl": validation_augmented,
        "validation_A_decisions.jsonl": decisions_a,
        "validation_A_plus_farmer_decisions.jsonl": decisions_augmented,
    }
    for name, rows in validation_files.items():
        path = output / name
        _write_jsonl(path, rows)
        common_report["artifacts"][name] = residual._artifact(path, len(rows))
    common_report["elapsed_seconds"] = time.perf_counter() - started
    return _finalize(output, common_report)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument(
        "--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--block-library", type=Path, default=farmer_ab.STRICT_LIBRARY)
    result.add_argument("--heldout-team", action="append", default=None)
    result.add_argument("--base-genome-id", default="NR020")
    result.add_argument("--composition-genome-id", default=None)
    result.add_argument("--train-opponent", action="append", default=None)
    result.add_argument("--validation-opponent", action="append", default=None)
    result.add_argument("--train-seed-start", type=int, default=path_v1.TRAIN_SEEDS[0])
    result.add_argument("--train-seed-count", type=int, default=len(path_v1.TRAIN_SEEDS))
    result.add_argument(
        "--validation-seed-start", type=int, default=path_v1.VALIDATION_SEEDS[0],
    )
    result.add_argument("--validation-seed-count", type=int, default=4)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--signal-min-decisions", type=int, default=SIGNAL_MIN_DECISIONS)
    result.add_argument("--trees", type=int, default=96)
    result.add_argument("--cv-trees", type=int, default=24)
    result.add_argument("--random-seed", type=int, default=2026082903)
    result.add_argument("--max-feature-bytes", type=int, default=4_000_000_000)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        args.train_opponent = [
            (args.train_opponent or list(path_v1.TRAIN_OPPONENTS))[0],
        ]
        args.validation_opponent = [
            (args.validation_opponent or list(path_v1.HELDOUT_OPPONENTS))[0],
        ]
        args.train_seed_count = 1
        args.validation_seed_count = 1
        args.max_windows = min(args.max_windows, 2)
        args.trees = min(args.trees, 8)
        args.cv_trees = min(args.cv_trees, 4)
    if (
        args.train_seed_count < 1 or args.validation_seed_count < 1
        or args.max_windows < 1 or args.max_windows > len(path_v1.ANCHORS)
        or args.signal_min_decisions < 1
        or args.trees < 8 or args.cv_trees < 4
        or args.max_feature_bytes < 1
    ):
        raise ValueError("invalid FARMER-AUGMENT-UNION-MPC-v1 budget")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

