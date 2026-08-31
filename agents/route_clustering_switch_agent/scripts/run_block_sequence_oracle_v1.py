#!/usr/bin/env python3
"""Search complete 21-block route sequences on real native snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fast_kaggriculture as fast_package
import fast_kaggriculture._fast_kaggriculture as native_extension
from fast_kaggriculture import Config, FastEnv, NativeAgentState
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src import teammate_expanded_routes
import run_adaptive_tail_oracle_v0 as adaptive
import run_block_mvp_continuation_multitail_v1 as continuation


SCHEMA = "block-sequence-oracle-v1"
HORIZON = adaptive.HORIZON
ANCHORS = adaptive.ANCHORS
STOPS = np.asarray([*ANCHORS[1:], HORIZON], dtype=np.int64)
TARGET_OPPONENTS = ("NR020", "NR026")
VALIDATION_SEEDS = tuple(range(2026085802, 2026085810))
# Frozen after the first smoke exposed the former 5810..5825 range.  These
# seeds are reachable only through the holdout-only --frozen-run control path.
TEST_SEEDS = tuple(range(2026086100, 2026086116))
CONTROL_SEED = 2026085800
DEFAULT_OUTPUT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\block_sequence_oracle_v1"
)
DEFAULT_V0_RUN = adaptive.DEFAULT_OUTPUT / "run"


def _sha256_file(path: Path) -> str:
    return adaptive._sha256_file(path)


def _state_key(opponent: str, seed: int, seat: int) -> str:
    return f"{opponent}|{seed}|{seat}"


def _switches(genome: Sequence[str], baseline_id: str) -> int:
    previous = baseline_id
    count = 0
    for route_id in genome:
        count += route_id != previous
        previous = route_id
    return count


def single_mutants(
    baseline_id: str, route_ids: Sequence[str], length: int = len(ANCHORS),
) -> list[tuple[str, ...]]:
    baseline = [baseline_id] * length
    result = []
    for gene in range(length):
        for route_id in route_ids:
            value = baseline.copy()
            value[gene] = route_id
            result.append(tuple(value))
    return result


def native_schedules(
    genomes: Sequence[Sequence[int]], opponent: int, seat: int,
) -> np.ndarray:
    values = np.asarray(genomes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(ANCHORS):
        raise ValueError("genomes must have shape [K,21]")
    result = np.empty((len(values), len(ANCHORS), 2), dtype=np.int64)
    result[:, :, seat] = values
    result[:, :, 1 - seat] = int(opponent)
    return result


@dataclass(frozen=True)
class BlockIndex:
    manifest_path: Path
    manifest_sha256: str
    vectors_path: Path
    vectors_sha256: str
    actions_path: Path
    actions_sha256: str
    anchors: np.ndarray
    block_ids: np.ndarray
    contracts: np.ndarray
    layouts: np.ndarray
    masks: np.ndarray
    routes_by_block: Mapping[str, tuple[str, ...]]
    active_by_anchor: Mapping[int, tuple[str, ...]]
    block_by_anchor_route: Mapping[tuple[int, str], str]


def load_block_index(root: Path, route_ids: set[str]) -> BlockIndex:
    root = root.resolve()
    manifest_path = root / "block_library_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "adaptive-tail-block-library-v0":
        raise ValueError("invalid block library schema")
    vectors_path = root / str(manifest["vectors_file"])
    actions_path = root / str(manifest["actions_file"])
    if _sha256_file(vectors_path) != str(manifest["vectors_sha256"]):
        raise ValueError("block vector digest mismatch")
    if _sha256_file(actions_path) != str(manifest["actions_sha256"]):
        raise ValueError("block action digest mismatch")
    with np.load(vectors_path, allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    required = {
        "anchors", "block_ids", "entry_contracts", "layouts",
        "unlocked_masks",
    }
    if not required.issubset(arrays):
        raise ValueError("block vector archive is missing required arrays")
    count = int(manifest["block_count"])
    shapes = {
        "anchors": (count,), "block_ids": (count,),
        "entry_contracts": (count, 147), "layouts": (count, 100),
        "unlocked_masks": (count, 4),
    }
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("block vector archive shape mismatch")
    rows = {str(row["block_id"]): dict(row) for row in manifest["blocks"]}
    if set(map(str, arrays["block_ids"])) != set(rows):
        raise ValueError("block manifest and vector IDs disagree")
    routes_by_block: dict[str, tuple[str, ...]] = {}
    block_by_anchor_route: dict[tuple[int, str], str] = {}
    for block_id, row in rows.items():
        sources = tuple(map(str, row.get("source_route_ids") or ()))
        if not sources or any(route_id not in route_ids for route_id in sources):
            raise ValueError(f"invalid source routes for block {block_id}")
        routes_by_block[block_id] = sources
        anchor = int(row["anchor"])
        for route_id in sources:
            key = (anchor, route_id)
            if key in block_by_anchor_route:
                raise ValueError(f"route maps to multiple blocks at {key}")
            block_by_anchor_route[key] = block_id
    if any(
        (anchor, route_id) not in block_by_anchor_route
        for anchor in ANCHORS for route_id in route_ids
    ):
        raise ValueError("not every replay route has a block at every anchor")
    active_by_anchor = {
        anchor: tuple(
            str(row["source_route_id"])
            for row in manifest["active_representatives"][str(anchor)]
        )
        for anchor in ANCHORS
    }
    return BlockIndex(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        vectors_path=vectors_path,
        vectors_sha256=_sha256_file(vectors_path),
        actions_path=actions_path,
        actions_sha256=_sha256_file(actions_path),
        anchors=np.asarray(arrays["anchors"], np.int16),
        block_ids=np.asarray(arrays["block_ids"]),
        contracts=np.asarray(arrays["entry_contracts"], np.float32),
        layouts=np.asarray(arrays["layouts"], np.uint8),
        masks=np.asarray(arrays["unlocked_masks"], np.uint8),
        routes_by_block=routes_by_block,
        active_by_anchor=active_by_anchor,
        block_by_anchor_route=block_by_anchor_route,
    )


def nearest_routes(
    blocks: BlockIndex,
    anchor: int,
    state: np.ndarray,
    layout: np.ndarray,
    mask: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(blocks.anchors == anchor)
    values = blocks.contracts[indices][:, adaptive.STATE_INDICES].astype(np.float64)
    query = np.asarray(state, np.float64)
    median = np.median(values, axis=0)
    scale = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    numeric = np.mean(((values - query) / scale) ** 2, axis=1)
    categorical = np.mean(
        np.concatenate((blocks.layouts[indices], blocks.masks[indices]), axis=1)
        != np.concatenate((layout, mask)),
        axis=1,
    )
    distance = .75 * numeric + .25 * categorical
    ordered = sorted(
        range(len(indices)),
        key=lambda local: (
            float(distance[local]), str(blocks.block_ids[indices[local]]),
        ),
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for local in ordered:
        block_id = str(blocks.block_ids[indices[local]])
        for route_id in blocks.routes_by_block[block_id]:
            if route_id in seen:
                continue
            seen.add(route_id)
            result.append({
                "route_id": route_id,
                "block_id": block_id,
                "distance": float(distance[local]),
            })
            if len(result) == limit:
                return result
    return result


def prefix_snapshot(
    bundle: NativeTeammateBundle,
    baseline: int,
    opponent: int,
    seed: int,
    seat: int,
) -> tuple[FastEnv, list[NativeAgentState]]:
    env = FastEnv(Config(), int(seed))
    states = [NativeAgentState(), NativeAgentState()]
    pair = (baseline, opponent) if seat == 0 else (opponent, baseline)
    bundle.executor.advance_segment(env, states[0], states[1], *pair, 216)
    if int(env.step_count) != 216:
        raise RuntimeError("prefix snapshot did not stop at step 216")
    return env, states


def reference_states(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    baseline_tape: Sequence[Mapping[str, Any]],
    opponent_id: str,
    seed: int,
    seat: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    baseline = bundle.index(baseline_id)
    opponent = bundle.index(opponent_id)
    env, states = prefix_snapshot(bundle, baseline, opponent, seed, seat)
    pair = (baseline, opponent) if seat == 0 else (opponent, baseline)
    result = {}
    for anchor, stop in zip(ANCHORS, STOPS, strict=True):
        observation = dict(env.observation(seat))
        observation.update(step=anchor, player=seat)
        state, _, layout, mask = adaptive.snapshot_contract(
            observation, baseline_tape,
        )
        result[anchor] = (state, layout, mask)
        bundle.executor.advance_segment(
            env, states[0], states[1], *pair, int(stop),
        )
    return result


class ScheduleEvaluator:
    def __init__(
        self,
        bundle: NativeTeammateBundle,
        baseline_id: str,
        opponent_id: str,
        seed: int,
        seat: int,
    ) -> None:
        self.bundle = bundle
        self.baseline_id = baseline_id
        self.opponent_id = opponent_id
        self.seed = int(seed)
        self.seat = int(seat)
        self.baseline = bundle.index(baseline_id)
        self.opponent = bundle.index(opponent_id)
        self.env, self.states = prefix_snapshot(
            bundle, self.baseline, self.opponent, seed, seat,
        )
        self.cache: dict[tuple[str, ...], tuple[float, float]] = {}
        self.candidate_steps = 0

    def evaluate(
        self, genomes: Iterable[Sequence[str]], batch_size: int = 4096,
    ) -> dict[tuple[str, ...], tuple[float, float]]:
        keys = list(dict.fromkeys(tuple(genome) for genome in genomes))
        if any(len(key) != len(ANCHORS) for key in keys):
            raise ValueError("every genome must have 21 route genes")
        unseen = [key for key in keys if key not in self.cache]
        for start in range(0, len(unseen), batch_size):
            chunk = unseen[start:start + batch_size]
            route_indices = [
                [self.bundle.index(route_id) for route_id in genome]
                for genome in chunk
            ]
            schedules = native_schedules(
                route_indices, self.opponent, self.seat,
            )
            rewards = np.asarray(
                self.bundle.executor.rollout_schedule_batch(
                    self.env, self.states[0], self.states[1], schedules, STOPS,
                ),
                np.float64,
            )
            if rewards.shape != (len(chunk), 2) or not np.isfinite(rewards).all():
                raise RuntimeError("native sequence rollout returned invalid rewards")
            self.cache.update({
                genome: tuple(map(float, reward))
                for genome, reward in zip(chunk, rewards, strict=True)
            })
            self.candidate_steps += len(chunk) * (HORIZON - 216)
        return {key: self.cache[key] for key in keys}

    def rank(self, genome: Sequence[str]) -> tuple[int, float]:
        return adaptive.reward_rank(self.cache[tuple(genome)], self.seat)

    def replay(self, genome: Sequence[str]) -> tuple[float, float]:
        env, states = prefix_snapshot(
            self.bundle, self.baseline, self.opponent, self.seed, self.seat,
        )
        for route_id, stop in zip(genome, STOPS, strict=True):
            route = self.bundle.index(route_id)
            pair = (route, self.opponent) if self.seat == 0 else (self.opponent, route)
            self.bundle.executor.advance_segment(
                env, states[0], states[1], *pair, int(stop),
            )
        if not env.done or int(env.step_count) != HORIZON:
            raise RuntimeError("best sequence replay did not terminate")
        return tuple(map(float, env.rewards))


def choose_pools(
    baseline_id: str,
    route_ids: Sequence[str],
    isolated_rewards: Mapping[tuple[str, ...], tuple[float, float]],
    seat: int,
    active: Mapping[int, Sequence[str]],
    nearest: Mapping[int, Sequence[Mapping[str, Any]]],
    top_single: int,
    cap: int,
) -> tuple[list[tuple[str, ...]], dict[int, list[str]]]:
    singles = single_mutants(baseline_id, route_ids)
    search_seeds: list[tuple[str, ...]] = []
    pools: dict[int, list[str]] = {}
    width = len(route_ids)
    for gene, anchor in enumerate(ANCHORS):
        rows = singles[gene * width:(gene + 1) * width]
        ranked = sorted(
            rows,
            key=lambda genome: (
                adaptive.reward_rank(isolated_rewards[genome], seat), genome[gene],
            ),
            reverse=True,
        )
        search_seeds.extend(ranked[:top_single])
        ordered = [
            baseline_id,
            *(genome[gene] for genome in ranked[:top_single]),
            *active[anchor],
            *(str(row["route_id"]) for row in nearest[anchor]),
        ]
        pools[anchor] = list(dict.fromkeys(ordered))[:cap]
    return search_seeds, pools


def _random_genome(
    pools: Mapping[int, Sequence[str]], rng: np.random.Generator,
) -> tuple[str, ...]:
    return tuple(
        str(rng.choice(pools[anchor])) for anchor in ANCHORS
    )


def breed(
    elites: Sequence[tuple[str, ...]],
    pools: Mapping[int, Sequence[str]],
    all_routes: Sequence[str],
    population: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    result = list(dict.fromkeys(elites))
    seen = set(result)
    attempts = 0
    while len(result) < population and attempts < population * 100:
        attempts += 1
        draw = float(rng.random())
        if draw < .45:
            child = list(elites[int(rng.integers(len(elites)))])
            for gene in rng.choice(len(ANCHORS), int(rng.integers(1, 4)), replace=False):
                child[int(gene)] = str(rng.choice(pools[ANCHORS[int(gene)]]))
        elif draw < .70:
            child = list(elites[int(rng.integers(len(elites)))])
            start = int(rng.integers(len(ANCHORS)))
            stop = min(len(ANCHORS), start + int(rng.integers(1, 5)))
            donor = str(rng.choice(all_routes))
            child[start:stop] = [donor] * (stop - start)
        elif draw < .90:
            left = elites[int(rng.integers(len(elites)))]
            right = elites[int(rng.integers(len(elites)))]
            cut0, cut1 = sorted(map(int, rng.integers(0, len(ANCHORS) + 1, 2)))
            child = [*left[:cut0], *right[cut0:cut1], *left[cut1:]]
        else:
            child = list(_random_genome(pools, rng))
        value = tuple(child)
        if value not in seen:
            seen.add(value)
            result.append(value)
    if len(result) < population:
        raise RuntimeError("genetic search could not create a unique population")
    return result


def evolve(
    evaluator: ScheduleEvaluator,
    baseline_id: str,
    route_ids: Sequence[str],
    pools: Mapping[int, Sequence[str]],
    single_seeds: Sequence[tuple[str, ...]],
    warm_start: tuple[str, ...] | None,
    population: int,
    generations: int,
    restarts: int,
    elite_count: int,
    patience: int,
    random_seed: int,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    baseline = (baseline_id,) * len(ANCHORS)
    constants = [(route_id,) * len(ANCHORS) for route_id in route_ids]
    evaluator.evaluate(constants)
    constant_seeds = sorted(
        constants,
        key=lambda genome: (evaluator.rank(genome), genome),
        reverse=True,
    )[:64]
    fixed = [baseline]
    if warm_start is not None:
        fixed.append(warm_start)
    fixed.extend((*constant_seeds, *single_seeds))
    logs: list[dict[str, Any]] = []
    evaluator.evaluate(fixed)
    best = max(
        dict.fromkeys(fixed),
        key=lambda genome: (evaluator.rank(genome), genome),
    )
    for restart in range(restarts):
        rng = np.random.default_rng(random_seed + restart * 104729)
        candidates = list(dict.fromkeys(fixed))
        while len(candidates) < population:
            candidates.append(_random_genome(pools, rng))
            candidates = list(dict.fromkeys(candidates))
        # The fixed seed set can be larger than the requested offspring
        # population.  Rank every fixed seed in generation zero; truncating
        # here would silently discard later anchors before they can become an
        # elite parent.
        stagnant = 0
        prior = evaluator.rank(best)
        for generation in range(generations + 1):
            evaluator.evaluate(candidates)
            ranked = sorted(
                candidates,
                key=lambda genome: (
                    evaluator.rank(genome),
                    -_switches(genome, baseline_id), genome,
                ),
                reverse=True,
            )
            if evaluator.rank(ranked[0]) > evaluator.rank(best):
                best, stagnant = ranked[0], 0
            else:
                stagnant += 1
            rank = evaluator.rank(best)
            logs.append({
                "restart": restart,
                "generation": generation,
                "outcome": rank[0],
                "margin": rank[1],
                "switches": _switches(best, baseline_id),
                "generation_candidates": len(candidates),
                "unique_evaluated": len(evaluator.cache),
            })
            if generation == generations or (stagnant >= patience and rank[0] == 2):
                break
            elites = ranked[:min(elite_count, len(ranked))]
            candidates = breed(elites, pools, route_ids, population, rng)
        if evaluator.rank(best) <= prior and evaluator.rank(best)[0] == 2:
            break
    return best, logs


def _load_v0_genomes(
    path: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any] | None]:
    decisions = path.resolve() / "decisions.jsonl"
    if not decisions.is_file():
        return {}, None
    rows = [
        json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("mode")) != "full_bank":
            continue
        grouped.setdefault(str(row["state_key"]), []).append(row)
    genomes = {
        "|".join(key.split("|")[:3]): tuple(
            str(row["selected_route_id"])
            for row in sorted(values, key=lambda row: int(row["step"]))
        )
        for key, values in grouped.items()
    }
    return genomes, {
        "path": str(decisions),
        "sha256": _sha256_file(decisions),
        "rows": len(rows),
    }


def _evaluate_fixed(
    bundle: NativeTeammateBundle,
    baseline_id: str,
    opponent_id: str,
    genome: tuple[str, ...],
    seed: int,
    seat: int,
) -> dict[str, Any]:
    baseline_rewards = adaptive._baseline_rewards(
        bundle, bundle.index(baseline_id), bundle.index(opponent_id), seed, seat,
    )
    evaluator = ScheduleEvaluator(
        bundle, baseline_id, opponent_id, seed, seat,
    )
    rewards = evaluator.evaluate([genome])[genome]
    baseline_rank = adaptive.reward_rank(baseline_rewards, seat)
    rank = adaptive.reward_rank(rewards, seat)
    return {
        "opponent": opponent_id,
        "seed": int(seed),
        "seat": int(seat),
        "baseline_rewards": list(baseline_rewards),
        "baseline_outcome": baseline_rank[0],
        "baseline_margin": baseline_rank[1],
        "rewards": list(rewards),
        "outcome": rank[0],
        "margin": rank[1],
        "margin_gain": rank[1] - baseline_rank[1],
        "outcome_regression": rank[0] < baseline_rank[0],
        "rank_regression": rank < baseline_rank,
        "nondegrading": rank >= baseline_rank,
        "candidate_steps": evaluator.candidate_steps,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "states": len(rows),
        "baseline_raw_win_rate": float(np.mean([
            int(row["baseline_outcome"]) == 2 for row in rows
        ])) if rows else None,
        "raw_win_rate": float(np.mean([
            int(row["outcome"]) == 2 for row in rows
        ])) if rows else None,
        "mean_margin": float(np.mean([row["margin"] for row in rows])) if rows else None,
        "mean_margin_gain": float(np.mean([
            row["margin_gain"] for row in rows
        ])) if rows else None,
        "outcome_regressions": sum(
            bool(row["outcome_regression"]) for row in rows
        ),
        "rank_regressions": sum(bool(row["rank_regression"]) for row in rows),
        "per_opponent_raw_win_rate": {
            opponent: float(np.mean([
                int(row["outcome"]) == 2
                for row in rows if str(row["opponent"]) == opponent
            ]))
            for opponent in sorted({str(row["opponent"]) for row in rows})
        },
    }


def composition_analysis(
    evaluator: ScheduleEvaluator,
    genome: tuple[str, ...],
    baseline_id: str,
    constant_rewards: Mapping[tuple[str, ...], tuple[float, float]],
) -> dict[str, Any]:
    rank = evaluator.rank(genome)
    unique_routes = sorted(set(genome))
    constituent_constant_wins = sum(
        adaptive.reward_rank(
            constant_rewards[(route_id,) * len(ANCHORS)], evaluator.seat,
        )[0] == 2
        for route_id in unique_routes if route_id != baseline_id
    )
    ablations = []
    for gene, route_id in enumerate(genome):
        if route_id == baseline_id:
            continue
        value = list(genome)
        value[gene] = baseline_id
        ablations.append((gene, tuple(value)))
    ablation_rewards = evaluator.evaluate(value for _, value in ablations)
    win_critical = [
        {
            "anchor": ANCHORS[gene],
            "route_id": genome[gene],
            "ablated_outcome": adaptive.reward_rank(
                ablation_rewards[value], evaluator.seat,
            )[0],
            "ablated_margin": adaptive.reward_rank(
                ablation_rewards[value], evaluator.seat,
            )[1],
        }
        for gene, value in ablations
        if rank[0] == 2
        and adaptive.reward_rank(ablation_rewards[value], evaluator.seat)[0] < 2
    ]
    margin_contributions = [
        {
            "anchor": ANCHORS[gene],
            "route_id": genome[gene],
            "ablated_outcome": adaptive.reward_rank(
                ablation_rewards[value], evaluator.seat,
            )[0],
            "ablated_margin": adaptive.reward_rank(
                ablation_rewards[value], evaluator.seat,
            )[1],
            "margin_drop": rank[1] - adaptive.reward_rank(
                ablation_rewards[value], evaluator.seat,
            )[1],
        }
        for gene, value in ablations
        if rank[1] - adaptive.reward_rank(
            ablation_rewards[value], evaluator.seat,
        )[1] >= 500
    ]
    return {
        "outcome": rank[0],
        "margin": rank[1],
        "switches": _switches(genome, baseline_id),
        "unique_routes": unique_routes,
        "genome_route_ids": list(genome),
        "constituent_constant_wins": constituent_constant_wins,
        "win_critical_ablations": win_critical,
        "margin_contributions": margin_contributions,
        "true_composition": (
            rank[0] == 2
            and _switches(genome, baseline_id) >= 2
            and constituent_constant_wins == 0
            and bool(win_critical)
        ),
    }


def _could_prove_composition(
    genome: tuple[str, ...],
    evaluator: ScheduleEvaluator,
    baseline_id: str,
    constant_rewards: Mapping[tuple[str, ...], tuple[float, float]],
) -> bool:
    return (
        _switches(genome, baseline_id) >= 2
        and all(
            adaptive.reward_rank(
                constant_rewards[(route_id,) * len(ANCHORS)], evaluator.seat,
            )[0] < 2
            for route_id in set(genome) if route_id != baseline_id
        )
    )


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"file": path.name, "rows": rows, "sha256": _sha256_file(path)}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _search_artifact_path(root: Path, raw: str) -> Path:
    root = root.resolve()
    path = (root / raw).resolve()
    if path == root or root not in path.parents:
        raise ValueError("search artifact path escapes the frozen run")
    return path


def _implementation_identity() -> dict[str, Any]:
    bundle_module = sys.modules[NativeTeammateBundle.__module__]
    modules = {
        "runner": Path(__file__).resolve(),
        "adaptive_oracle": Path(adaptive.__file__).resolve(),
        "continuation_oracle": Path(continuation.__file__).resolve(),
        "routed_intent": Path(continuation.routed_v2.__file__).resolve(),
        "block_mvp_v1": Path(continuation.routed_v2.v1.__file__).resolve(),
        "native_bundle": Path(bundle_module.__file__).resolve(),
        "expanded_routes": Path(teammate_expanded_routes.__file__).resolve(),
        "fast_package": Path(fast_package.__file__).resolve(),
        "native_extension": Path(native_extension.__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in modules.items()
    }


def _input_identity(
    bank_signatures: Sequence[Mapping[str, Any]],
    route_ids: Sequence[str],
    blocks: BlockIndex,
    query_signature: Mapping[str, Any],
    v1_args: argparse.Namespace,
    v1_manifest: Mapping[str, Any],
    v0_signature: Mapping[str, Any] | None,
) -> dict[str, Any]:
    v1_files = {
        name: {
            "path": str(Path(value).resolve()),
            "sha256": _sha256_file(Path(value).resolve()),
        }
        for name, value in (
            ("source", v1_args.source),
            ("base_actions", v1_args.base_actions),
            ("base_metadata", v1_args.base_metadata),
        )
    }
    return {
        "prepared": list(bank_signatures),
        "route_ids_sha256": _json_sha256(list(route_ids)),
        "block_manifest_sha256": blocks.manifest_sha256,
        "block_vectors_sha256": blocks.vectors_sha256,
        "block_actions_sha256": blocks.actions_sha256,
        "queries": dict(query_signature),
        "v1_files": v1_files,
        "v1_actions_sha256": str(v1_manifest["actions_sha256"]),
        "v0_decisions": None if v0_signature is None else dict(v0_signature),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = args.prepared_root or [
        adaptive.DEFAULT_OLD_PREPARED, adaptive.DEFAULT_NEW_PREPARED,
    ]
    bank, bank_signatures = adaptive._load_banks(
        [Path(value).resolve() for value in prepared]
    )
    route_ids = sorted(bank)
    blocks = load_block_index(args.block_library, set(route_ids))
    baseline_id, baseline_tape, _, _, v1_args, v1_manifest = (
        continuation.load_frozen_v1_openings(args.v1_root)
    )
    queries, query_signature = adaptive._load_queries(args.queries, baseline_id)
    query_map = {
        (str(row["opponent"]), int(row["seed"]), int(row["seat"])): row
        for row in queries
    }
    targets = [
        row for row in queries
        if str(row["opponent"]) in args.opponent
        and int(row["seat"]) == 0
        and int(row["baseline_outcome"]) == 0
    ]
    if len(targets) != len(args.opponent):
        raise ValueError("each target opponent must have one frozen seat-0 loss")
    targets.sort(key=lambda row: str(row["opponent"]))
    additional = {baseline_id: baseline_tape, **bank}
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*args.opponent, *additional)),
    )
    for method in ("advance_segment", "rollout_schedule_batch"):
        if not hasattr(bundle.executor, method):
            raise RuntimeError(f"native extension is missing {method}; rebuild it")
    v0_genomes, v0_signature = _load_v0_genomes(args.v0_run)

    discovery_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    mirror_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    best_by_opponent: dict[str, tuple[str, ...]] = {}
    total_candidate_steps = 0

    for target_index, query in enumerate(targets):
        opponent_id = str(query["opponent"])
        seed, seat = int(query["seed"]), int(query["seat"])
        evaluator = ScheduleEvaluator(
            bundle, baseline_id, opponent_id, seed, seat,
        )
        baseline = (baseline_id,) * len(ANCHORS)
        baseline_rewards = evaluator.evaluate([baseline])[baseline]
        expected = (int(query["baseline_outcome"]), float(query["baseline_margin"]))
        if adaptive.reward_rank(baseline_rewards, seat) != expected:
            raise RuntimeError("frozen baseline drifted before sequence search")

        references = reference_states(
            bundle, baseline_id, baseline_tape, opponent_id, seed, seat,
        )
        nearest = {
            anchor: nearest_routes(
                blocks, anchor, *references[anchor], args.nearest,
            )
            for anchor in ANCHORS
        }
        singles = single_mutants(baseline_id, route_ids)
        isolated = evaluator.evaluate(singles)
        single_seeds, pools = choose_pools(
            baseline_id, route_ids, isolated, seat,
            blocks.active_by_anchor, nearest,
            args.top_single, args.pool_cap,
        )
        print(json.dumps({
            "event": "isolated_discovery_complete",
            "opponent": opponent_id,
            "schedules": len(singles),
            "best_margin": max(
                adaptive.reward_rank(rewards, seat)[1]
                for rewards in isolated.values()
            ),
        }, sort_keys=True), flush=True)
        for anchor, pool in pools.items():
            pool_rows.append({
                "opponent": opponent_id,
                "seed": seed,
                "seat": seat,
                "anchor": anchor,
                "candidate_count": len(pool),
                "route_ids": pool,
                "block_ids": [
                    None if route_id == baseline_id
                    else blocks.block_by_anchor_route[(anchor, route_id)]
                    for route_id in pool
                ],
                "nearest": nearest[anchor],
            })

        warm = v0_genomes.get(_state_key(opponent_id, seed, seat))
        if warm is not None and any(route_id not in additional for route_id in warm):
            raise ValueError("v0 warm-start route is missing from the union bank")
        search_seed = int(args.random_seed + target_index * 1000003)
        best, logs = evolve(
            evaluator, baseline_id, route_ids, pools, single_seeds, warm,
            args.population, args.generations, args.restarts,
            args.elites, args.patience, search_seed,
        )
        print(json.dumps({
            "event": "sequence_search_complete",
            "opponent": opponent_id,
            "rank": evaluator.rank(best),
            "unique_schedules": len(evaluator.cache),
            "switches": _switches(best, baseline_id),
        }, sort_keys=True), flush=True)
        for row in logs:
            generation_rows.append({
                "opponent": opponent_id, "seed": seed, "seat": seat, **row,
            })
        rewards = evaluator.cache[best]
        replayed = evaluator.replay(best)
        if replayed != rewards:
            raise RuntimeError("best schedule batch reward disagrees with replay")
        constants = [(route_id,) * len(ANCHORS) for route_id in route_ids]
        constant_rewards = evaluator.evaluate(constants)
        best_constant = max(
            constants,
            key=lambda genome: adaptive.reward_rank(constant_rewards[genome], seat),
        )
        best_constant_rank = adaptive.reward_rank(
            constant_rewards[best_constant], seat,
        )
        best_rank = adaptive.reward_rank(rewards, seat)
        all_winners = sorted(
            (
                genome for genome in evaluator.cache
                if evaluator.rank(genome)[0] == 2
            ),
            key=lambda genome: (
                evaluator.rank(genome), -_switches(genome, baseline_id), genome,
            ),
            reverse=True,
        )
        analyzed = all_winners[:16] or [best]
        composition_archive = [
            composition_analysis(
                evaluator, genome, baseline_id, constant_rewards,
            )
            for genome in analyzed
        ]
        best_analysis = next((
            row for row in composition_archive
            if tuple(row["genome_route_ids"]) == best
        ), None)
        if best_analysis is None:
            best_analysis = composition_analysis(
                evaluator, best, baseline_id, constant_rewards,
            )
        composition_evidence = next(
            (row for row in composition_archive if bool(row["true_composition"])),
            None,
        )
        if composition_evidence is None:
            analyzed_genomes = {
                tuple(row["genome_route_ids"]) for row in composition_archive
            }
            for genome in all_winners:
                if genome in analyzed_genomes or not _could_prove_composition(
                    genome, evaluator, baseline_id, constant_rewards,
                ):
                    continue
                analysis = composition_analysis(
                    evaluator, genome, baseline_id, constant_rewards,
                )
                if bool(analysis["true_composition"]):
                    composition_evidence = analysis
                    break
        if warm is not None:
            warm_rank = evaluator.rank(warm)
            warm_sha = hashlib.sha256(json.dumps(
                list(warm), ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
        else:
            warm_rank, warm_sha = None, None
        best_source = (
            "v0_warm_start" if warm is not None and best == warm
            else "baseline" if best == baseline
            else "constant_route" if len(set(best)) == 1
            else "isolated_block_seed" if best in set(single_seeds)
            else "ga_offspring"
        )
        best_by_opponent[opponent_id] = best
        discovery_rows.append({
            "opponent": opponent_id,
            "seed": seed,
            "seat": seat,
            "baseline_rewards": list(baseline_rewards),
            "baseline_outcome": expected[0],
            "baseline_margin": expected[1],
            "best_constant_route": best_constant[0],
            "best_constant_outcome": best_constant_rank[0],
            "best_constant_margin": best_constant_rank[1],
            "rewards": list(rewards),
            "outcome": best_rank[0],
            "margin": best_rank[1],
            "margin_gain": best_rank[1] - expected[1],
            "gain_vs_best_constant": best_rank[1] - best_constant_rank[1],
            "switches": _switches(best, baseline_id),
            "unique_routes": best_analysis["unique_routes"],
            "genome_route_ids": list(best),
            "genome_block_ids": [
                None if route_id == baseline_id
                else blocks.block_by_anchor_route[(anchor, route_id)]
                for anchor, route_id in zip(ANCHORS, best, strict=True)
            ],
            "constituent_constant_wins": best_analysis["constituent_constant_wins"],
            "win_critical_ablations": best_analysis["win_critical_ablations"],
            "margin_contributions": best_analysis["margin_contributions"],
            "true_composition": best_analysis["true_composition"],
            "composition_archive": composition_archive,
            "composition_evidence": composition_evidence,
            "composition_proven": composition_evidence is not None,
            "best_source": best_source,
            "warm_start_sha256": warm_sha,
            "warm_start_outcome": None if warm_rank is None else warm_rank[0],
            "warm_start_margin": None if warm_rank is None else warm_rank[1],
            "ga_gain_vs_warm": None if warm_rank is None else best_rank[1] - warm_rank[1],
            "isolated_schedules": len(singles),
            "unique_schedules_evaluated": len(evaluator.cache),
            "candidate_steps": evaluator.candidate_steps,
        })
        total_candidate_steps += evaluator.candidate_steps

        mirror = query_map.get((opponent_id, seed, 1))
        if mirror is None:
            raise ValueError("frozen mirror-seat query is missing")
        mirror_row = _evaluate_fixed(
            bundle, baseline_id, opponent_id, best, seed, 1,
        )
        mirror_expected = (
            int(mirror["baseline_outcome"]), float(mirror["baseline_margin"]),
        )
        if (
            mirror_expected[0] != 0
            or (
                int(mirror_row["baseline_outcome"]),
                float(mirror_row["baseline_margin"]),
            ) != mirror_expected
        ):
            raise RuntimeError("frozen mirror-seat baseline is not the expected loss")
        mirror_row["frozen_baseline_verified"] = True
        mirror_rows.append(mirror_row)
        total_candidate_steps += int(mirror_row["candidate_steps"])
        for mirror_seat in (0, 1):
            control_expected_row = query_map.get((
                opponent_id, CONTROL_SEED, mirror_seat,
            ))
            if control_expected_row is None:
                raise ValueError("frozen easy-control query is missing")
            row = _evaluate_fixed(
                bundle, baseline_id, opponent_id, best,
                CONTROL_SEED, mirror_seat,
            )
            control_expected = (
                int(control_expected_row["baseline_outcome"]),
                float(control_expected_row["baseline_margin"]),
            )
            if control_expected[0] != 2 or (
                int(row["baseline_outcome"]),
                float(row["baseline_margin"]),
            ) != control_expected:
                raise RuntimeError(
                    "frozen easy-control baseline drifted before evaluation"
                )
            row["frozen_baseline_verified"] = True
            control_rows.append(row)
            total_candidate_steps += int(row["candidate_steps"])

    for opponent_id, genome in sorted(best_by_opponent.items()):
        for seed in VALIDATION_SEEDS:
            for seat in (0, 1):
                row = _evaluate_fixed(
                    bundle, baseline_id, opponent_id, genome, seed, seat,
                )
                row["split"] = "validation"
                validation_rows.append(row)
                total_candidate_steps += int(row["candidate_steps"])

    discovery_wins = sum(int(row["outcome"]) == 2 for row in discovery_rows)
    mirror_wins = sum(int(row["outcome"]) == 2 for row in mirror_rows)
    validation = _summary(validation_rows)
    test = _summary(test_rows)
    exact_scope = set(args.opponent) == set(TARGET_OPPONENTS)
    gates = {
        "G_SCOPE_exactly_NR020_and_NR026": exact_scope,
        "G0_best_schedule_replay_matches_batch": True,
        "G1_search_never_loses_to_best_constant_route": all(
            (int(row["outcome"]), float(row["margin"])) >= (
                int(row["best_constant_outcome"]),
                float(row["best_constant_margin"]),
            )
            for row in discovery_rows
        ),
        "G2_both_targeted_seat0_losses_become_wins": (
            exact_scope and discovery_wins == len(discovery_rows) == 2
        ),
        "G3_same_sequences_win_the_mirror_seats": (
            exact_scope and mirror_wins == len(mirror_rows) == 2
        ),
        "G4_at_least_one_win_requires_true_block_composition": any(
            bool(row["composition_proven"]) for row in discovery_rows
        ),
        "G5_seed5800_controls_have_no_outcome_regression": all(
            not bool(row["outcome_regression"]) for row in control_rows
        ),
        "G6_validation_aggregate90_each_opponent80": (
            float(validation["raw_win_rate"] or 0) >= .90
            and all(
                float(value) >= .80
                for value in validation["per_opponent_raw_win_rate"].values()
            )
        ),
        "G7_test_aggregate90_each_opponent80": (
            False
        ),
    }
    existential = all(gates[name] for name in (
        "G_SCOPE_exactly_NR020_and_NR026",
        "G0_best_schedule_replay_matches_batch",
        "G1_search_never_loses_to_best_constant_route",
        "G2_both_targeted_seat0_losses_become_wins",
        "G3_same_sequences_win_the_mirror_seats",
        "G4_at_least_one_win_requires_true_block_composition",
    ))
    holdout_eligible = exact_scope and all(
        gates[name] for name in gates if name != "G7_test_aggregate90_each_opponent80"
    )
    robust = False
    status = (
        "partial_scope_smoke" if not exact_scope
        else
        "passed_search_gates_pending_holdout" if holdout_eligible
        else "passed_existential_but_failed_validation" if existential
        else "partial_sequence_improvement" if any(
            float(row["margin_gain"]) > 0 for row in discovery_rows
        )
        else "no_sequence_improvement_within_budget"
    )

    output = args.output_root.resolve() / "run"
    output.mkdir(parents=True, exist_ok=True)
    paths_rows = (
        (output / "discovery.jsonl", discovery_rows),
        (output / "candidate_pools.jsonl", pool_rows),
        (output / "generations.jsonl", generation_rows),
        (output / "mirror.jsonl", mirror_rows),
        (output / "controls.jsonl", control_rows),
        (output / "validation.jsonl", validation_rows),
        (output / "test.jsonl", test_rows),
    )
    for path, rows in paths_rows:
        adaptive._atomic_text(path, "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ))
    split_manifest = {
        "validation_seeds": list(VALIDATION_SEEDS),
        "holdout_seeds": "sealed behind --frozen-run",
        "test_executed": False,
        "grouping": "seed keeps both opponents and both seats together",
    }
    split_manifest["sha256"] = hashlib.sha256(json.dumps(
        split_manifest, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    implementation_identity = _implementation_identity()
    inputs_identity = _input_identity(
        bank_signatures, route_ids, blocks, query_signature,
        v1_args, v1_manifest, v0_signature,
    )
    frozen_path = output / "frozen_genomes.json"
    frozen_payload = {
        "schema": "block-sequence-oracle-v1-frozen-genomes",
        "mode": "search_only",
        "baseline_route_id": baseline_id,
        "horizon": HORIZON,
        "anchors": list(ANCHORS),
        "stops": list(map(int, STOPS)),
        "opponents": sorted(best_by_opponent),
        "holdout_eligible": holdout_eligible,
        "search_gates": gates,
        "validation": validation,
        "split_manifest": split_manifest,
        "implementation": implementation_identity,
        "inputs": inputs_identity,
        "genomes": {
            opponent: {
                "route_ids": list(genome),
                "block_ids": [
                    None if route_id == baseline_id
                    else blocks.block_by_anchor_route[(anchor, route_id)]
                    for anchor, route_id in zip(ANCHORS, genome, strict=True)
                ],
                "genome_sha256": _json_sha256(list(genome)),
            }
            for opponent, genome in sorted(best_by_opponent.items())
        },
    }
    adaptive._atomic_text(
        frozen_path,
        json.dumps(frozen_payload, ensure_ascii=False, indent=2) + "\n",
    )
    report = {
        "schema": SCHEMA,
        "mode": "search_only",
        "status": status,
        "holdout_eligible": holdout_eligible,
        "selector_trained": False,
        "search": (
            "all 21x227 isolated-block counterfactuals, then full 21-gene "
            "terminal-fitness evolutionary search"
        ),
        "runtime_semantics": (
            "each gene selects a source full tape for one 24-step window; "
            "Simulator and NativeAgentState remain continuous"
        ),
        "implementation": implementation_identity,
        "inputs": {
            "prepared": bank_signatures,
            "unique_replay_tails": len(bank),
            "block_manifest": {
                "path": str(blocks.manifest_path),
                "sha256": blocks.manifest_sha256,
                "blocks": len(blocks.block_ids),
            },
            "block_vectors": {
                "path": str(blocks.vectors_path),
                "sha256": blocks.vectors_sha256,
            },
            "block_actions": {
                "path": str(blocks.actions_path),
                "sha256": blocks.actions_sha256,
            },
            "queries": query_signature,
            "v1_actions_sha256": str(v1_manifest["actions_sha256"]),
            "v0_run": str(args.v0_run.resolve()),
            "v0_decisions": v0_signature,
            "splits": split_manifest,
        },
        "budget": {
            "population": args.population,
            "population_semantics": (
                "offspring population; generation zero ranks every fixed seed"
            ),
            "generation_zero_candidates_per_target": [
                int(row["generation_candidates"])
                for row in generation_rows
                if int(row["restart"]) == 0 and int(row["generation"]) == 0
            ],
            "generations": args.generations,
            "offspring_generations": args.generations,
            "restarts": args.restarts,
            "elites": args.elites,
            "patience": args.patience,
            "top_single_per_anchor": args.top_single,
            "nearest_per_anchor": args.nearest,
            "candidate_pool_cap": args.pool_cap,
            "random_seed": args.random_seed,
        },
        "targeted_discovery": {
            "states": len(discovery_rows),
            "wins": discovery_wins,
            "mean_baseline_margin": float(np.mean([
                row["baseline_margin"] for row in discovery_rows
            ])),
            "mean_best_constant_margin": float(np.mean([
                row["best_constant_margin"] for row in discovery_rows
            ])),
            "mean_sequence_margin": float(np.mean([
                row["margin"] for row in discovery_rows
            ])),
            "mean_margin_gain": float(np.mean([
                row["margin_gain"] for row in discovery_rows
            ])),
            "composition_proven_states": sum(
                bool(row["composition_proven"]) for row in discovery_rows
            ),
        },
        "mirror": _summary(mirror_rows),
        "controls": _summary(control_rows),
        "validation": validation,
        "test": test,
        "gates": gates,
        "rollout_candidate_steps": total_candidate_steps,
        "full_game_equivalents": total_candidate_steps / HORIZON,
        "artifacts": {
            **{
                path.stem: _artifact(path, len(rows))
                for path, rows in paths_rows
            },
            "frozen_genomes": {
                "file": frozen_path.name,
                "sha256": _sha256_file(frozen_path),
                "opponents": len(best_by_opponent),
            },
        },
        "scope": [
            "The discovery search sees terminal rewards for one frozen hard market seed.",
            "Validation uses one frozen per-opponent sequence without terminal best-of selection.",
            "Holdout runs later in --frozen-run mode, which cannot search.",
            "A budget failure does not falsify action-level mutation or a larger sequence search.",
            "Passing this two-opponent experiment is not the final 20-opponent agent gate.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    adaptive._atomic_text(
        output / "FINAL_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def run_holdout(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one immutable search result without entering the search path."""
    started = time.perf_counter()
    search_root = args.frozen_run.resolve()
    report_path = search_root / "FINAL_REPORT.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing search report: {report_path}")
    search_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        search_report.get("schema") != SCHEMA
        or search_report.get("mode") != "search_only"
        or bool(search_report.get("inputs", {}).get("splits", {}).get(
            "test_executed", True,
        ))
    ):
        raise ValueError("holdout requires an untested search-only report")
    frozen_meta = search_report.get("artifacts", {}).get("frozen_genomes")
    if not isinstance(frozen_meta, dict):
        raise ValueError("search report does not pin frozen genomes")
    frozen_path = _search_artifact_path(
        search_root, str(frozen_meta.get("file", "")),
    )
    if (
        not frozen_path.is_file()
        or _sha256_file(frozen_path) != str(frozen_meta.get("sha256"))
    ):
        raise ValueError("frozen genome artifact digest mismatch")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("schema") != "block-sequence-oracle-v1-frozen-genomes"
        or frozen.get("mode") != "search_only"
        or not bool(frozen.get("holdout_eligible"))
    ):
        raise ValueError("search result is not eligible for holdout")
    if sorted(map(str, frozen.get("opponents", ()))) != sorted(TARGET_OPPONENTS):
        raise ValueError("holdout requires exactly NR020 and NR026")
    if set(args.opponent) != set(TARGET_OPPONENTS):
        raise ValueError("holdout opponent scope cannot be changed")

    # Verify every search artifact before the first rollout.  This also makes
    # the frozen report, discovery evidence, and validation evidence one unit.
    search_snapshot: dict[Path, tuple[str, int]] = {}
    for meta in search_report.get("artifacts", {}).values():
        if not isinstance(meta, dict) or "file" not in meta or "sha256" not in meta:
            raise ValueError("invalid search artifact manifest")
        path = _search_artifact_path(search_root, str(meta["file"]))
        if path in search_snapshot:
            raise ValueError("duplicate search artifact path")
        if not path.is_file() or _sha256_file(path) != str(meta["sha256"]):
            raise ValueError(f"search artifact digest mismatch: {path.name}")
        search_snapshot[path] = (_sha256_file(path), path.stat().st_mtime_ns)
    if report_path in search_snapshot:
        raise ValueError("search report cannot also be a mutable artifact")
    search_snapshot[report_path] = (
        _sha256_file(report_path), report_path.stat().st_mtime_ns,
    )

    prepared = args.prepared_root or [
        adaptive.DEFAULT_OLD_PREPARED, adaptive.DEFAULT_NEW_PREPARED,
    ]
    bank, bank_signatures = adaptive._load_banks(
        [Path(value).resolve() for value in prepared]
    )
    route_ids = sorted(bank)
    blocks = load_block_index(args.block_library, set(route_ids))
    baseline_id, baseline_tape, _, _, v1_args, v1_manifest = (
        continuation.load_frozen_v1_openings(args.v1_root)
    )
    _, query_signature = adaptive._load_queries(args.queries, baseline_id)
    _, v0_signature = _load_v0_genomes(args.v0_run)
    current_identity = _input_identity(
        bank_signatures, route_ids, blocks, query_signature,
        v1_args, v1_manifest, v0_signature,
    )
    implementation_identity = _implementation_identity()
    if frozen.get("inputs") != current_identity:
        raise ValueError("holdout inputs differ from the frozen search inputs")
    if frozen.get("implementation") != implementation_identity:
        raise ValueError("implementation changed after search")
    if (
        str(frozen.get("baseline_route_id")) != baseline_id
        or tuple(map(int, frozen.get("anchors", ()))) != ANCHORS
        or tuple(map(int, frozen.get("stops", ()))) != tuple(map(int, STOPS))
        or int(frozen.get("horizon", -1)) != HORIZON
    ):
        raise ValueError("frozen runtime contract does not match this runner")
    split_manifest = dict(frozen.get("split_manifest") or {})
    claimed_split_sha = str(split_manifest.pop("sha256", ""))
    if _json_sha256(split_manifest) != claimed_split_sha:
        raise ValueError("frozen split manifest digest mismatch")
    if search_report["inputs"]["splits"] != frozen["split_manifest"]:
        raise ValueError("search report and frozen split manifest disagree")

    frozen_genomes = dict(frozen.get("genomes") or {})
    if set(frozen_genomes) != set(TARGET_OPPONENTS):
        raise ValueError("frozen genome set has missing or extra opponents")
    genomes: dict[str, tuple[str, ...]] = {}
    allowed_routes = set(route_ids) | {baseline_id}
    for opponent in TARGET_OPPONENTS:
        record = dict(frozen_genomes[opponent])
        genome = tuple(map(str, record.get("route_ids") or ()))
        block_ids = list(record.get("block_ids") or ())
        if len(genome) != len(ANCHORS) or len(block_ids) != len(ANCHORS):
            raise ValueError(f"invalid frozen genome length for {opponent}")
        if any(route_id not in allowed_routes for route_id in genome):
            raise ValueError(f"unknown route in frozen genome for {opponent}")
        expected_blocks = [
            None if route_id == baseline_id
            else blocks.block_by_anchor_route[(anchor, route_id)]
            for anchor, route_id in zip(ANCHORS, genome, strict=True)
        ]
        if block_ids != expected_blocks:
            raise ValueError(f"frozen route/block mapping drifted for {opponent}")
        if str(record.get("genome_sha256")) != _json_sha256(list(genome)):
            raise ValueError(f"frozen genome digest mismatch for {opponent}")
        genomes[opponent] = genome

    output = (
        args.holdout_output.resolve()
        if args.holdout_output is not None
        else search_root.parent / "final_test"
    )
    if output == search_root or search_root in output.parents:
        raise ValueError("holdout output must be outside the immutable search run")
    additional = {baseline_id: baseline_tape, **bank}
    bundle = NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*TARGET_OPPONENTS, *additional)),
    )
    if not hasattr(bundle.executor, "rollout_schedule_batch"):
        raise RuntimeError("native extension is missing rollout_schedule_batch")
    output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    total_candidate_steps = 0
    for opponent in TARGET_OPPONENTS:
        genome = genomes[opponent]
        genome_sha256 = _json_sha256(list(genome))
        for seed in TEST_SEEDS:
            for seat in (0, 1):
                row = _evaluate_fixed(
                    bundle, baseline_id, opponent, genome, seed, seat,
                )
                row.update({
                    "split": "test",
                    "genome_sha256": genome_sha256,
                })
                rows.append(row)
                total_candidate_steps += int(row["candidate_steps"])
    keys = {
        (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
        for row in rows
    }
    expected_rows = len(TEST_SEEDS) * len(TARGET_OPPONENTS) * 2
    if len(rows) != len(keys) or len(rows) != expected_rows:
        raise RuntimeError("holdout did not produce the exact frozen test matrix")
    after_snapshot = {
        path: (_sha256_file(path), path.stat().st_mtime_ns)
        for path in search_snapshot
    }
    if after_snapshot != search_snapshot:
        raise RuntimeError("search artifacts changed during holdout")

    test_path = output / "test.jsonl"
    adaptive._atomic_text(test_path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    test = _summary(rows)
    gates = dict(frozen["search_gates"])
    gates["G7_test_aggregate90_each_opponent80"] = (
        float(test["raw_win_rate"] or 0) >= .90
        and set(test["per_opponent_raw_win_rate"]) == set(TARGET_OPPONENTS)
        and all(
            float(value) >= .80
            for value in test["per_opponent_raw_win_rate"].values()
        )
    )
    robust = all(bool(value) for value in gates.values())
    report = {
        "schema": SCHEMA,
        "mode": "holdout_only",
        "status": (
            "passed_robust_sequence_feasibility"
            if robust else "failed_frozen_holdout"
        ),
        "search_was_executed": False,
        "source_search": {
            "report_path": str(report_path),
            "report_sha256": _sha256_file(report_path),
            "frozen_genomes_path": str(frozen_path),
            "frozen_genomes_sha256": _sha256_file(frozen_path),
        },
        "implementation": implementation_identity,
        "test_split": {
            "seeds": list(TEST_SEEDS),
            "opponents": list(TARGET_OPPONENTS),
            "seats": [0, 1],
            "rows": len(rows),
        },
        "test": test,
        "gates": gates,
        "rollout_candidate_steps": total_candidate_steps,
        "artifacts": {"test": _artifact(test_path, len(rows))},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path_out = output / "FINAL_TEST_REPORT.json"
    adaptive._atomic_text(
        report_path_out,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--block-library", type=Path, default=adaptive.DEFAULT_BLOCK_LIBRARY)
    result.add_argument("--queries", type=Path, default=adaptive.DEFAULT_QUERIES)
    result.add_argument("--v1-root", type=Path, default=continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--v0-run", type=Path, default=DEFAULT_V0_RUN)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--frozen-run", type=Path, default=None,
        help="holdout-only mode: load a completed search run and skip search",
    )
    result.add_argument("--holdout-output", type=Path, default=None)
    result.add_argument("--opponent", action="append", default=None)
    result.add_argument("--population", type=int, default=1024)
    result.add_argument("--generations", type=int, default=25)
    result.add_argument("--restarts", type=int, default=3)
    result.add_argument("--elites", type=int, default=64)
    result.add_argument("--patience", type=int, default=6)
    result.add_argument("--top-single", type=int, default=16)
    result.add_argument("--nearest", type=int, default=8)
    result.add_argument("--pool-cap", type=int, default=40)
    result.add_argument("--random-seed", type=int, default=2026082801)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.opponent is None:
        args.opponent = list(TARGET_OPPONENTS)
    args.opponent = list(dict.fromkeys(map(str, args.opponent)))
    if (
        args.population < 256 or args.generations < 1 or args.restarts < 1
        or args.elites < 2 or args.elites >= args.population
        or args.patience < 1 or args.top_single < 1 or args.nearest < 1
        or args.pool_cap < 2
    ):
        raise ValueError("invalid BLOCK-SEQUENCE-ORACLE-v1 search budget")
    if args.frozen_run is not None:
        run_holdout(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
