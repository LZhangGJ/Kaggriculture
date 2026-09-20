"""Distill and evaluate shallow opponent-state route decision trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np

from meta_agent.src.equilibrium_selector import EquilibriumOpeningSelector
from meta_agent.src.recurrent_meta import market_town_vector, public_farm_vector
from meta_agent.src.replay_trie_agent import ReplayTrieAgent
from meta_agent.src.simple_route_tree import (
    CommittedOpponentRouteSelector,
    DirectOpponentTreeSelector,
    SimpleOpponentTreeSelector,
    opponent_route_features,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "route-runtime-v8"
DESCRIPTORS = ROOT / "branch-descriptors-v8.pkl"
CHECKPOINT = ROOT / "recurrent-route-best.pt"
MIXTURE = ROOT / "opening-mixture-v8.json"
EXPERIMENT = ROOT / "experiments/simple-route-trees"
STRONG_OPPONENTS = (
    "public_b85", "schedule_93311715", "frontier_soil",
    "adaptive_farming", "kaito_v35",
)
CONFIG = {
    "episodeSteps": 720, "turnsPerDay": 24, "shedCapacity": 100,
    "boardSize": 10, "startingMoney": 3000, "maxMarketOrders": 10,
    "farmHandCostMult": 1,
}
_POLICIES: dict[tuple[str, str], tuple[ReplayTrieAgent, Any]] = {}
_OPPONENTS: dict[tuple[str, str], Any] = {}


def _worker_init() -> None:
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _salt(seed: int, seat: int) -> int:
    raw = f"simple-route-tree-v1:{seed}:{seat}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _policy(label: str, tree_path: str, salt: int) -> tuple[ReplayTrieAgent, Any]:
    key = (label, tree_path)
    cached = _POLICIES.get(key)
    if cached is None:
        common = dict(
            tree_q_model_path=None,
            recurrent_model_path=str(CHECKPOINT),
            residual_weight=1.0,
            enable_trading=False,
            epsilon=0.0,
            top_k=16,
            prior_weight=0.0,
            seed=0,
            mixture_path=str(MIXTURE),
            mixture_salt=salt,
        )
        if label == "teacher":
            selector = EquilibriumOpeningSelector(
                str(RUNTIME), str(DESCRIPTORS), **common
            )
        else:
            with Path(tree_path).open("rb") as handle:
                model_kind = pickle.load(handle).get("model_kind", "rank_distillation")
            Selector = (
                DirectOpponentTreeSelector if model_kind == "direct_family"
                else CommittedOpponentRouteSelector if model_kind == "committed_route"
                else SimpleOpponentTreeSelector
            )
            selector = Selector(
                str(RUNTIME), str(DESCRIPTORS),
                tree_model_path=tree_path, stochastic_softmax=True, **common,
            )
        policy = ReplayTrieAgent(
            RUNTIME, selector_override=selector,
            manage_sells=True, lead_sells=True, lead_turns=5,
            lead_batch=20, lead_max_distance=8, repair_weeds=True,
            weed_replay_steps=8,
        )
        cached = policy, selector
        _POLICIES[key] = cached
    policy, selector = cached
    selector.fixed_mixture_salt = int(salt)
    selector.mixture_salt = int(salt)
    return policy, selector


def _opponent(name: str, engine: str):
    key = (name, engine)
    if key not in _OPPONENTS:
        if engine == "fast":
            from kaggrl.community_agents import community_agent_source
            from meta_agent.src.benchmark_community_round_robin import LoadedAgent
            _OPPONENTS[key] = LoadedAgent(
                community_agent_source(name),
                f"simple_tree_{name}_{os.getpid()}",
            )
        else:
            from kaggrl import CommunityAgent
            _OPPONENTS[key] = CommunityAgent(name)
    return _OPPONENTS[key]


def _run_episode(
    label: str, tree_path: str, opponent_name: str, seed: int, seat: int,
    engine: str, collect: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    policy, selector = _policy(label, tree_path, _salt(seed, seat))
    opponent = _opponent(opponent_name, engine)
    agents = [policy, opponent] if seat == 0 else [opponent, policy]
    if engine == "fast":
        from fast_kaggriculture import Config, FastEnv
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        while not env.done:
            step = int(env.step_count)
            for observation in observations:
                observation["step"] = step
            actions = [
                agents[index](observations[index], CONFIG) for index in range(2)
            ]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        statuses = ["DONE", "DONE"]
    else:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"seed": seed}, debug=True)
        env.run(agents)
        rewards = [float(state.reward) for state in env.state]
        statuses = [str(state.status) for state in env.state]
    own, other = rewards[seat], rewards[1 - seat]
    result: dict[str, Any] = {
        "label": label, "opponent": opponent_name, "seed": seed, "seat": seat,
        "own_reward": own, "opponent_reward": other, "margin": own - other,
        "result": "win" if own > other else "loss" if own < other else "draw",
        "statuses": statuses, "elapsed_seconds": time.perf_counter() - started,
    }
    if collect:
        trace = selector.trace_arrays(top_k=16)
        checkpoints = np.asarray(trace["checkpoints"], dtype=np.int16)
        opponent_public = np.asarray(trace["opponent_public"], dtype=np.float32)
        market_town = np.asarray(trace["market_town"], dtype=np.float32)
        result["checkpoints"] = checkpoints
        result["selected"] = np.asarray(trace["selected"], dtype=np.int8)
        result["candidate_counts"] = np.asarray(
            trace["candidate_counts"], dtype=np.int8
        )
        result["opponent_counts"] = np.stack([
            opponent_route_features(
                opponent_public[checkpoint], market_town[checkpoint], "opponent_counts"
            )
            for checkpoint in checkpoints
        ])
        result["opponent_context"] = np.stack([
            opponent_route_features(
                opponent_public[checkpoint], market_town[checkpoint], "opponent_context"
            )
            for checkpoint in checkpoints
        ])
    return result


def _collect_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    opponent, seed, seat = task
    try:
        return _run_episode("teacher", "", opponent, seed, seat, "fast", True)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "task": task}


def _evaluation_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    label, tree_path, opponent, seed, seat, engine = task
    try:
        return _run_episode(label, tree_path, opponent, seed, seat, engine, False)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "task": task}


def _parallel(tasks, worker, workers: int) -> list[dict[str, Any]]:
    count = min(workers or (os.cpu_count() or 1), len(tasks))
    results = []
    with mp.get_context("spawn").Pool(count, initializer=_worker_init) as pool:
        for row in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(row)
            if len(results) % 100 == 0 or len(results) == len(tasks):
                print(f"[{len(results):05d}/{len(tasks):05d}]", flush=True)
    errors = [row for row in results if "error" in row]
    if errors:
        raise RuntimeError(f"{len(errors)} worker failures; first={errors[0]}")
    return results


def collect_dataset(output: Path, seeds: Sequence[int], workers: int) -> None:
    tasks = [
        (opponent, seed, seat)
        for opponent in STRONG_OPPONENTS for seed in seeds for seat in (0, 1)
    ]
    rows = _parallel(tasks, _collect_worker, workers)
    rows.sort(key=lambda row: (row["opponent"], row["seed"], row["seat"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        checkpoints=np.stack([row["checkpoints"] for row in rows]),
        selected=np.stack([row["selected"] for row in rows]),
        candidate_counts=np.stack([row["candidate_counts"] for row in rows]),
        opponent_counts=np.stack([row["opponent_counts"] for row in rows]),
        opponent_context=np.stack([row["opponent_context"] for row in rows]),
        opponents=np.asarray([row["opponent"] for row in rows]),
        seeds=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        seats=np.asarray([row["seat"] for row in rows], dtype=np.int8),
        margins=np.asarray([row["margin"] for row in rows], dtype=np.float32),
    )
    print(json.dumps({"output": str(output), "games": len(rows)}, indent=2))


def train_trees(
    train_path: Path,
    valid_path: Path,
    output: Path,
    feature_set: str,
    min_valid_accuracy: float = 0.0,
    min_valid_gain: float = 0.0,
) -> None:
    from sklearn.tree import DecisionTreeClassifier
    train = np.load(train_path)
    valid = np.load(valid_path)
    checkpoints = sorted(set(np.asarray(train["checkpoints"]).reshape(-1).tolist()))
    trees: dict[int, Any] = {}
    metrics: dict[int, Any] = {}
    total_correct = total_rows = total_zero = 0
    for checkpoint in checkpoints:
        train_mask = (train["checkpoints"] == checkpoint) & (train["candidate_counts"] > 1)
        valid_mask = (valid["checkpoints"] == checkpoint) & (valid["candidate_counts"] > 1)
        x_train = np.asarray(train[feature_set][train_mask], dtype=np.float32)
        y_train = np.asarray(train["selected"][train_mask], dtype=np.int64)
        x_valid = np.asarray(valid[feature_set][valid_mask], dtype=np.float32)
        y_valid = np.asarray(valid["selected"][valid_mask], dtype=np.int64)
        if len(x_train) < 20 or not len(x_valid):
            continue
        best = None
        min_leaf = max(10, len(x_train) // 100)
        for depth in (2, 3, 4, 5):
            tree = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=min_leaf, random_state=20260824
            ).fit(x_train, y_train)
            accuracy = float(np.mean(tree.predict(x_valid) == y_valid))
            candidate = (accuracy, -depth, tree)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        accuracy, negative_depth, tree = best
        prediction = tree.predict(x_valid)
        rank0_accuracy = float(np.mean(y_valid == 0))
        accepted = (
            accuracy >= min_valid_accuracy
            and accuracy - rank0_accuracy >= min_valid_gain
        )
        if accepted:
            trees[int(checkpoint)] = tree
        metrics[int(checkpoint)] = {
            "train_rows": len(x_train), "valid_rows": len(x_valid),
            "depth": -negative_depth, "leaves": int(tree.get_n_leaves()),
            "accuracy": accuracy,
            "rank0_accuracy": rank0_accuracy,
            "teacher_nonzero_rate": float(np.mean(y_valid != 0)),
            "accepted": accepted,
        }
        total_correct += int(np.sum(prediction == y_valid))
        total_zero += int(np.sum(y_valid == 0))
        total_rows += len(y_valid)
    payload = {
        "schema_version": 1, "feature_set": feature_set,
        "checkpoint": str(CHECKPOINT), "trees": trees,
        "selection": {
            "min_valid_accuracy": min_valid_accuracy,
            "min_valid_gain": min_valid_gain,
        },
        "metrics": metrics,
        "validation": {
            "actionable_rows": total_rows,
            "accuracy": total_correct / max(1, total_rows),
            "rank0_accuracy": total_zero / max(1, total_rows),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    print(json.dumps({
        "output": str(output), "feature_set": feature_set,
        "trees": len(trees), **payload["validation"],
    }, indent=2))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    result = {
        "games": len(rows), "wins": wins, "draws": draws,
        "losses": len(rows) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
    }
    result["versus"] = {
        opponent: _summary_without_versus(
            [row for row in rows if row["opponent"] == opponent]
        ) for opponent in STRONG_OPPONENTS
    }
    return result


def _summary_without_versus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    return {
        "games": len(rows), "wins": wins, "draws": draws,
        "losses": len(rows) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
    }


def evaluate(
    output: Path, seeds: Sequence[int], workers: int, engine: str,
    counts_tree: Path, context_tree: Path,
) -> None:
    policies = {
        "teacher": "", "opponent_counts": str(counts_tree.resolve()),
        "opponent_context": str(context_tree.resolve()),
    }
    tasks = [
        (label, tree, opponent, seed, seat, engine)
        for label, tree in policies.items()
        for opponent in STRONG_OPPONENTS for seed in seeds for seat in (0, 1)
    ]
    started = time.perf_counter()
    rows = _parallel(tasks, _evaluation_worker, workers)
    rows.sort(key=lambda row: (row["label"], row["opponent"], row["seed"], row["seat"]))
    report = {
        "schema_version": 1, "engine": engine,
        "kaggle_environments_version": version("kaggle-environments"),
        "checkpoint": str(CHECKPOINT), "seeds": list(seeds),
        "seats": [0, 1], "opponents": list(STRONG_OPPONENTS),
        "summary": {
            label: _summary([row for row in rows if row["label"] == label])
            for label in policies
        },
        "wall_seconds": time.perf_counter() - started,
        "games": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


def _range(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if not separator:
        return (int(value),)
    return tuple(range(int(start), int(stop)))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--seeds", type=_range, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--workers", type=int, default=0)
    train = subparsers.add_parser("train")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--valid", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--feature-set", choices=("opponent_counts", "opponent_context"), required=True
    )
    train.add_argument("--min-valid-accuracy", type=float, default=0.0)
    train.add_argument("--min-valid-gain", type=float, default=0.0)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--seeds", type=_range, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--engine", choices=("fast", "official"), default="fast")
    evaluation.add_argument(
        "--counts-tree", type=Path,
        default=EXPERIMENT / "opponent-counts.pkl",
    )
    evaluation.add_argument(
        "--context-tree", type=Path,
        default=EXPERIMENT / "opponent-context.pkl",
    )
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect_dataset(args.output, args.seeds, args.workers)
    elif args.command == "train":
        train_trees(
            args.train, args.valid, args.output, args.feature_set,
            args.min_valid_accuracy, args.min_valid_gain,
        )
    else:
        evaluate(
            args.output, args.seeds, args.workers, args.engine,
            args.counts_tree, args.context_tree,
        )


if __name__ == "__main__":
    main()
