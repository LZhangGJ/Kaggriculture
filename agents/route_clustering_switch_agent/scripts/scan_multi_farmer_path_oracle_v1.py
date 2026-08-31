#!/usr/bin/env python3
"""Scan H4 replay-donor path headroom without committing any candidate."""

from __future__ import annotations

import argparse
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

import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation
import run_multi_farmer_path_residual_v1 as path_v1
import run_route_residual_adapter_v0 as residual


SCHEMA = "multi-farmer-path-oracle-coverage-v1"
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1\oracle_coverage_scan_20260829a"
)


def select_path_pure_index(
    rewards: np.ndarray, market_diff: np.ndarray, seat: int,
) -> int:
    """Return the best H4-local-market-identical arm; KEEP wins ties."""

    rewards = np.asarray(rewards, np.float64)
    market_diff = np.asarray(market_diff, np.int32)
    if rewards.ndim != 2 or rewards.shape[1] != 2:
        raise ValueError("rewards must have shape [candidate, 2]")
    if market_diff.shape != (len(rewards),) or len(rewards) < 1:
        raise ValueError("market_diff must align with a non-empty reward batch")
    if market_diff[0] != 0:
        raise ValueError("KEEP must be H4-local-market-identical")
    best = 0
    rank = adaptive.reward_rank(rewards[0], seat)
    for index in np.flatnonzero(market_diff == 0):
        candidate_rank = adaptive.reward_rank(rewards[index], seat)
        if candidate_rank > rank:
            best, rank = int(index), candidate_rank
    return best


def _pure_pair_beats_components(
    candidates: Sequence[path_v1.PathCandidate],
    rewards: np.ndarray,
    market_diff: np.ndarray,
    seat: int,
) -> list[str]:
    ranks = [adaptive.reward_rank(rewards[index], seat) for index in range(len(candidates))]
    keep_rank = ranks[0]
    singles = {
        (candidate.donor_id, candidate.assignments[0][0]): ranks[index]
        for index, candidate in enumerate(candidates)
        if (
            market_diff[index] == 0
            and candidate.kind == "SINGLE"
            and len(candidate.assignments) == 1
        )
    }
    result = []
    for index, candidate in enumerate(candidates):
        if market_diff[index] != 0 or candidate.kind != "PAIR":
            continue
        parents = [
            singles.get((candidate.donor_id, actor))
            for actor, _ in candidate.assignments
        ]
        if all(parent is not None for parent in parents) and ranks[index] > max(
            keep_rank, *(parent for parent in parents if parent is not None),
        ):
            result.append(candidate.code)
    return result


def _rollout(
    bundle: Any,
    env: Any,
    states: Sequence[Any],
    schedule: np.ndarray,
    stops: np.ndarray,
    seat: int,
    candidates: Sequence[path_v1.PathCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    units, counts = path_v1.pack_unit_plans(candidates)
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
    if market_diff.shape != (len(candidates),) or np.any(market_diff < 0):
        raise RuntimeError("native path rollout returned invalid market diagnostics")
    return rewards, market_diff


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    headroom = [row for row in rows if row["strict_headroom"]]
    deltas = [float(row["best_delta_margin"]) for row in rows]
    scenarios = {
        (row["split"], row["genome_id"], row["opponent"], row["seed"], row["seat"])
        for row in rows
    }
    repaired_scenarios = {
        (row["split"], row["genome_id"], row["opponent"], row["seed"], row["seat"])
        for row in rows if row["loss_to_win_repair"]
    }
    return {
        "scenarios": len(scenarios),
        "decisions": len(rows),
        "candidate_rollouts": sum(int(row["candidate_count"]) for row in rows),
        "path_pure_candidates": sum(int(row["path_pure_candidate_count"]) for row in rows),
        "market_diff_candidates": sum(int(row["market_diff_candidate_count"]) for row in rows),
        "market_diff_steps": sum(int(row["market_diff_step_count"]) for row in rows),
        "keep_equivalence_failures": sum(not bool(row["keep_equivalent"]) for row in rows),
        "headroom_decisions": len(headroom),
        "headroom_rate": len(headroom) / len(rows) if rows else 0.0,
        "outcome_improvement_decisions": sum(
            int(row["delta_outcome"]) > 0 for row in rows
        ),
        "loss_to_win_repair_decisions": sum(
            bool(row["loss_to_win_repair"]) for row in rows
        ),
        "loss_to_win_repair_scenarios": len(repaired_scenarios),
        "mean_best_delta_margin": float(np.mean(deltas)) if deltas else 0.0,
        "max_best_delta_margin": max(deltas, default=0.0),
        "max_self_reward_delta": max(
            (float(row["delta_self_reward"]) for row in rows), default=0.0,
        ),
        "pair_beats_components_events": sum(
            len(row["pair_beats_components_codes"]) for row in rows
        ),
        "pair_beats_components_decisions": sum(
            bool(row["pair_beats_components_codes"]) for row in rows
        ),
    }


def summarize_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "split": defaultdict(list),
        "anchor": defaultdict(list),
        "offset": defaultdict(list),
        "opponent": defaultdict(list),
        "seat": defaultdict(list),
    }
    for row in rows:
        for field in groups:
            groups[field][str(row[field])].append(row)
    headroom = [row for row in rows if row["strict_headroom"]]
    return {
        "overall": _group_summary(rows),
        **{
            f"by_{field}": {
                key: _group_summary(values)
                for key, values in sorted(group.items())
            }
            for field, group in groups.items()
        },
        "headroom_best_kind": dict(sorted(Counter(
            str(row["best_kind"]) for row in headroom
        ).items())),
        "headroom_best_donor": dict(sorted(Counter(
            str(row["best_donor_id"]) for row in headroom
        ).items())),
        "anchors_with_headroom": sorted({int(row["anchor"]) for row in headroom}),
        "opponents_with_headroom": sorted({str(row["opponent"]) for row in headroom}),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    residual._atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _implementation_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": residual._sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genomes: Mapping[str, Sequence[str]],
    carriers: Mapping[str, Any],
    donors: Mapping[int, Sequence[path_v1.DonorBlock]],
    genome_id: str,
    opponent: str,
    seed: int,
    seat: int,
    split: str,
    max_windows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    genome = genomes[genome_id]
    env, states, _ = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    active_steps = frozenset(
        anchor + offset
        for anchor in path_v1.ANCHORS[:max_windows]
        for offset in path_v1.WINDOW_OFFSETS
        if anchor + offset < path_v1.HORIZON
    )
    decisions: list[dict[str, Any]] = []
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        if step in active_steps:
            segment, route_id, _, _, _, candidates = path_v1._route_context(
                genome, tapes, carriers, observation, donors, step,
            )
            rewards, market_diff = _rollout(
                bundle, env, states, schedule[segment:], path_v1.STOPS[segment:],
                seat, candidates,
            )
            reference = np.asarray(bundle.executor.rollout_schedule_batch(
                env, states[0], states[1], schedule[segment:][None, :, :],
                path_v1.STOPS[segment:],
            ), np.float64)[0]
            best = select_path_pure_index(rewards, market_diff, seat)
            keep_rank = adaptive.reward_rank(rewards[0], seat)
            best_rank = adaptive.reward_rank(rewards[best], seat)
            keep_self = float(rewards[0, seat])
            best_self = float(rewards[best, seat])
            selected = candidates[best]
            pair_beats = _pure_pair_beats_components(
                candidates, rewards, market_diff, seat,
            )
            nonzero = [
                {
                    "index": int(index), "code": candidates[index].code,
                    "steps": int(market_diff[index]),
                }
                for index in np.flatnonzero(market_diff)
            ]
            decisions.append({
                "schema": SCHEMA,
                "split": split,
                "genome_id": genome_id,
                "opponent": opponent,
                "seed": int(seed),
                "seat": int(seat),
                "step": int(step),
                "anchor": int(path_v1.ANCHORS[segment]),
                "offset": int(step - path_v1.ANCHORS[segment]),
                "segment": int(segment),
                "route_id": route_id,
                "candidate_count": len(candidates),
                "candidate_kind_counts": dict(sorted(Counter(
                    candidate.kind for candidate in candidates
                ).items())),
                "path_pure_candidate_count": int(np.count_nonzero(market_diff == 0)),
                "market_diff_candidate_count": len(nonzero),
                "market_diff_step_count": int(market_diff.sum()),
                "market_diff_nonzero": nonzero,
                "keep_equivalent": bool(np.array_equal(rewards[0], reference)),
                "keep_rewards": rewards[0].tolist(),
                "reference_rewards": reference.tolist(),
                "keep_outcome": int(keep_rank[0]),
                "keep_margin": float(keep_rank[1]),
                "keep_self_reward": keep_self,
                "best_index": int(best),
                "best_code": selected.code,
                "best_kind": selected.kind,
                "best_donor_id": selected.donor_id,
                "best_donor_cluster": selected.donor_cluster,
                "best_assignments": [list(value) for value in selected.assignments],
                "best_rewards": rewards[best].tolist(),
                "best_outcome": int(best_rank[0]),
                "best_margin": float(best_rank[1]),
                "best_self_reward": best_self,
                "delta_outcome": int(best_rank[0] - keep_rank[0]),
                "best_delta_margin": float(best_rank[1] - keep_rank[1]),
                "delta_self_reward": best_self - keep_self,
                "loss_to_win_repair": bool(keep_rank[0] == 0 and best_rank[0] == 2),
                "strict_headroom": bool(best_rank > keep_rank),
                "pair_beats_components_codes": pair_beats,
                "committed": "KEEP",
            })
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(
                env, player, int(route_pair[player]), states[player],
            )
            for player in (0, 1)
        ])
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("oracle coverage baseline trajectory did not complete")
    rewards = tuple(map(float, env.rewards))
    outcome, margin = adaptive.reward_rank(rewards, seat)
    return decisions, {
        "schema": SCHEMA,
        "split": split,
        "genome_id": genome_id,
        "opponent": opponent,
        "seed": int(seed),
        "seat": int(seat),
        "rewards": list(rewards),
        "outcome": int(outcome),
        "margin": float(margin),
        "decisions": len(decisions),
        "completed": True,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    overall = report["summary"]["overall"]
    split_lines = [
        f"| {name} | {row['decisions']} | {row['candidate_rollouts']} | "
        f"{row['headroom_decisions']} ({100 * row['headroom_rate']:.2f}%) | "
        f"{row['loss_to_win_repair_scenarios']}/{row['scenarios']} | "
        f"{row['max_best_delta_margin']:+.1f} | "
        f"{row['pair_beats_components_events']} | "
        f"{row['market_diff_candidates']} |"
        for name, row in report["summary"]["by_split"].items()
    ]
    return (
        "# MULTI-FARMER-PATH-ORACLE-COVERAGE-v1\n\n"
        f"Status: `{report['status']}`\n\n"
        "Candidates were evaluated from each frozen baseline state and never "
        "committed. The path-pure filter compares market actions with a scratch KEEP "
        "action during the four override steps only.\n\n"
        "| split | decisions | candidate rollouts | strict headroom | loss->win scenarios | max delta | pair beats component singles | H4 market-diff candidates |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(split_lines)
        + "\n\n"
        f"Overall: {overall['headroom_decisions']}/{overall['decisions']} decisions "
        f"({100 * overall['headroom_rate']:.2f}%) have strict path-pure headroom; "
        f"maximum terminal-margin delta is {overall['max_best_delta_margin']:+.1f}.\n\n"
        "## Decision\n\n"
        f"{report['decision']}\n\n"
        "## Evidence boundary\n\n"
        "The opponents are fixed-route native proxies, not submitted agents. The "
        "21-gene genome is frozen and hashed in FINAL_REPORT.json. Heldout team "
        "medoids are filtered, but the prebuilt cluster geometry may still have "
        "been influenced by heldout replay members; a strict lineage test must rebuild "
        "the clustering from allowed blocks only. The native market diagnostic covers "
        "the H4 override window, not later state-induced market-overlay changes, so this "
        "run does not prove a fully market-frozen continuation.\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    train_opponents = tuple(args.train_opponent or path_v1.TRAIN_OPPONENTS)
    heldout_opponents = tuple(args.validation_opponent or path_v1.HELDOUT_OPPONENTS)
    train_seeds = tuple(range(args.train_seed_start, args.train_seed_start + args.train_seed_count))
    heldout_seeds = tuple(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_seed_count,
    ))
    if set(train_opponents) & set(heldout_opponents):
        raise ValueError("train and heldout opponents must be disjoint")
    if set(train_seeds) & set(heldout_seeds):
        raise ValueError("train and heldout seeds must be disjoint")
    if (set(train_seeds) | set(heldout_seeds)) & residual.SEALED_SEEDS:
        raise ValueError("oracle coverage scan cannot open sealed route-search seeds")
    if not args.smoke and (
        len(train_opponents) < 6 or len(heldout_opponents) < 6
        or len(train_seeds) < 2 or len(heldout_seeds) < 2
        or args.max_windows != len(path_v1.ANCHORS)
    ):
        raise ValueError("formal coverage scan requires 6+6 opponents, two seeds, and 21 windows")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (bundle, baseline_id, _, tapes, genomes, carriers, donors,
     inputs, source) = path_v1.load_experiment(args)
    genome_ids = path_v1._selected_genome_ids(args)
    scenarios = [
        (split, genome_id, opponent, seed, seat)
        for split, opponents, seeds in (
            ("train_proxy", train_opponents, train_seeds),
            ("heldout_proxy", heldout_opponents, heldout_seeds),
        )
        for genome_id in genome_ids
        for opponent in opponents
        for seed in seeds
        for seat in (0, 1)
    ]
    decisions: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, (split, genome_id, opponent, seed, seat) in enumerate(scenarios, 1):
        rows, trajectory = _scenario(
            bundle, baseline_id, tapes, genomes, carriers, donors,
            genome_id, opponent, seed, seat, split, args.max_windows,
        )
        decisions.extend(rows)
        trajectories.append(trajectory)
        if index % 12 == 0 or index == len(scenarios):
            print(json.dumps({
                "event": "oracle_coverage_progress",
                "index": index,
                "total": len(scenarios),
                "split": split,
                "opponent": opponent,
                "seed": seed,
                "seat": seat,
                "headroom_so_far": sum(row["strict_headroom"] for row in decisions),
            }, sort_keys=True), flush=True)

    summary = summarize_decisions(decisions)
    overall = summary["overall"]
    anchors = len(summary["anchors_with_headroom"])
    if overall["keep_equivalence_failures"]:
        status = "invalid_keep_mismatch"
        decision = "KEEP does not reproduce the frozen schedule; fix the native audit before interpreting headroom."
    elif overall["headroom_rate"] < 0.01 or anchors < 3:
        status = "h4_vocabulary_sparse"
        decision = (
            "Do not enlarge the selector yet. Test H=8 and state-conditioned donor retrieval; "
            "the present H4/top-3 vocabulary exposes too little oracle signal."
        )
    elif overall["pair_beats_components_events"] == 0:
        status = "single_actor_headroom_only"
        decision = (
            "Path residuals have headroom, but no pair strictly beats KEEP and both "
            "corresponding component singles. Train the smaller candidate value model "
            "before adding pair-specific architecture."
        )
    else:
        status = "multi_farmer_h4_headroom_found"
        decision = (
            "The H4 vocabulary is worth learning. Use state-conditioned donor retrieval plus "
            "a candidate value model, keeping MPC execute-one/replan as the online policy."
        )
    if overall["market_diff_candidates"]:
        decision += " H4-local market-different candidates were excluded from every result."

    split_manifest = {
        "schema": "multi-farmer-path-oracle-coverage-split-v1",
        "train_opponents": list(train_opponents),
        "heldout_opponents": list(heldout_opponents),
        "train_seeds": list(train_seeds),
        "heldout_seeds": list(heldout_seeds),
        "seats": [0, 1],
        "genome_ids": list(genome_ids),
        "anchors": list(map(int, path_v1.ANCHORS[:args.max_windows])),
        "offsets": list(path_v1.WINDOW_OFFSETS),
        "trajectory_policy": "frozen baseline; every oracle candidate is discarded",
        "sealed_test_executed": False,
    }
    split_manifest["sha256"] = path_v1._sha256(split_manifest)
    decisions_path = output / "oracle_scan_decisions.jsonl"
    trajectories_path = output / "frozen_trajectories.jsonl"
    summary_path = output / "oracle_scan_summary.json"
    split_path = output / "split_manifest.json"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(trajectories_path, trajectories)
    residual._atomic_text(split_path, json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "baseline_route_id": baseline_id,
        "frozen_genomes": {
            genome_id: {
                "route_ids": list(genomes[genome_id]),
                "sha256": path_v1._sha256(list(genomes[genome_id])),
            }
            for genome_id in genome_ids
        },
        "source": source,
        "inputs": inputs,
        "implementation": {
            "scanner": _implementation_identity(Path(__file__)),
            "path_runner": _implementation_identity(Path(path_v1.__file__)),
            "native_pyd": _implementation_identity(
                Path(path_v1.native_extension.__file__)
            ),
        },
        "split": split_manifest,
        "summary": summary,
        "trajectory_count": len(trajectories),
        "all_trajectories_complete": all(row["completed"] for row in trajectories),
        "sealed_test_executed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            path.name: residual._artifact(path, rows)
            for path, rows in (
                (decisions_path, len(decisions)),
                (trajectories_path, len(trajectories)),
                (summary_path, None),
                (split_path, None),
            )
        },
    }
    report_path = output / "FINAL_REPORT.json"
    markdown_path = output / "FINAL_REPORT.md"
    residual._atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    residual._atomic_text(markdown_path, _markdown(report))
    print(json.dumps({
        "event": "oracle_coverage_complete",
        "status": status,
        "output": str(output),
        "summary": overall,
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
    result.add_argument("--base-genome-id", default="NR020")
    result.add_argument("--composition-genome-id", default=None)
    result.add_argument("--train-opponent", action="append", default=None)
    result.add_argument("--validation-opponent", action="append", default=None)
    result.add_argument("--train-seed-start", type=int, default=path_v1.TRAIN_SEEDS[0])
    result.add_argument("--train-seed-count", type=int, default=2)
    result.add_argument("--validation-seed-start", type=int, default=path_v1.VALIDATION_SEEDS[0])
    result.add_argument("--validation-seed-count", type=int, default=2)
    result.add_argument("--max-windows", type=int, default=len(path_v1.ANCHORS))
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        args.train_opponent = [(args.train_opponent or list(path_v1.TRAIN_OPPONENTS))[0]]
        args.validation_opponent = [
            (args.validation_opponent or list(path_v1.HELDOUT_OPPONENTS))[0]
        ]
        args.train_seed_count = args.validation_seed_count = 1
        args.max_windows = min(args.max_windows, 2)
    if args.train_seed_count < 1 or args.validation_seed_count < 1:
        raise ValueError("seed counts must be positive")
    if args.max_windows < 1 or args.max_windows > len(path_v1.ANCHORS):
        raise ValueError("invalid anchor window count")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
