#!/usr/bin/env python3
"""Paired FastEnv screening of global-family medoid route sub-agents."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np

from meta_agent.src.recurrent_meta import encode_meta_observation
from meta_agent.src.replay_trie_agent import ReplayTrieAgent
from meta_agent.src.selector import RouteDecision


ROOT = Path(__file__).resolve().parents[1]
RUNTIME: Path | None = None
CHECKPOINTS: tuple[int, ...] = ()
CONFIG = {
    "episodeSteps": 720, "turnsPerDay": 24, "shedCapacity": 100,
    "boardSize": 10, "startingMoney": 3000, "maxMarketOrders": 10,
    "farmHandCostMult": 1,
}
_POLICIES: dict[tuple[int, str], ReplayTrieAgent] = {}


class FixedRouteSelector:
    def __init__(self, runtime: Path, route_id: str) -> None:
        with (runtime / "manifest.pkl").open("rb") as handle:
            manifest = pickle.load(handle)
        self.route_id = str(route_id)
        self.current_route_id = self.route_id
        self.checkpoints = [int(value) for value in manifest["checkpoints"]]
        offset, _ = manifest["route_offsets"][self.route_id]
        with (runtime / "actions.bin").open("rb") as handle:
            handle.seek(int(offset))
            self.actions = pickle.load(handle)

    def reset(self) -> None:
        self.current_route_id = self.route_id

    def select(self, fingerprint: dict[str, Any], checkpoint: int) -> RouteDecision:
        next_checkpoint = next(
            (value for value in self.checkpoints if value > checkpoint), checkpoint
        )
        return RouteDecision(
            checkpoint=int(checkpoint), next_checkpoint=int(next_checkpoint),
            route_id=self.route_id, family_hash=f"fixed:{self.route_id}",
            changed_route=False, score=0.0, distance={}, support=1,
            median_reward=0.0,
        )

    def action(self, step: int) -> dict[str, Any]:
        if 0 <= step < len(self.actions):
            return self.actions[step]
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _policy(lane: int, route_id: str) -> ReplayTrieAgent:
    assert RUNTIME is not None
    key = (int(lane), str(route_id))
    if key not in _POLICIES:
        selector = FixedRouteSelector(RUNTIME, route_id)
        _POLICIES[key] = ReplayTrieAgent(
            RUNTIME, selector_override=selector,
            manage_sells=True, lead_sells=True, lead_turns=5,
            lead_batch=20, lead_max_distance=8,
            repair_weeds=True, weed_replay_steps=8,
        )
    return _POLICIES[key]


def _worker_init(runtime: str, checkpoints: tuple[int, ...]) -> None:
    global RUNTIME, CHECKPOINTS
    RUNTIME = Path(runtime)
    CHECKPOINTS = tuple(int(value) for value in checkpoints)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"


def _state_vector(observation: dict[str, Any]) -> np.ndarray:
    encoded = encode_meta_observation(observation)
    return np.concatenate([
        encoded.self_public, encoded.opponent_public,
        encoded.private_plan, encoded.market_town,
    ]).astype(np.float16)


def _play(task: tuple[int, int, int, str, str, int, int]) -> dict[str, Any]:
    task_id, own_index, opponent_index, own_route, opponent_route, seed, seat = task
    try:
        from fast_kaggriculture import Config, FastEnv

        own_policy = _policy(0, own_route)
        opponent_policy = _policy(1, opponent_route)
        agents = [own_policy, opponent_policy] if seat == 0 else [opponent_policy, own_policy]
        env = FastEnv(Config(), int(seed))
        observations = list(env.reset(int(seed)))
        states = []
        checkpoint_set = set(CHECKPOINTS)
        while not env.done:
            step = int(env.step_count)
            for player, observation in enumerate(observations):
                observation["player"] = player
                observation["step"] = step
            if step in checkpoint_set:
                states.append(_state_vector(observations[seat]))
            actions = [agents[player](observations[player], CONFIG) for player in range(2)]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        own_reward = rewards[seat]
        other_reward = rewards[1 - seat]
        margin = own_reward - other_reward
        if len(states) != len(CHECKPOINTS):
            raise RuntimeError(f"captured {len(states)} states, expected {len(CHECKPOINTS)}")
        return {
            "task_id": task_id, "own_index": own_index,
            "opponent_index": opponent_index, "seed": seed, "seat": seat,
            "reward": own_reward, "opponent_reward": other_reward,
            "margin": margin,
            "score": float(margin > 0) + 0.5 * float(margin == 0),
            "states": np.stack(states),
        }
    except Exception as error:
        return {
            "task_id": task_id, "error": f"{type(error).__name__}: {error}",
            "task": task[1:],
        }


def _range(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else (int(value),)


def _completed(directory: Path) -> set[int]:
    result: set[int] = set()
    for path in sorted(directory.glob("shard-*.npz")):
        with np.load(path) as shard:
            result.update(int(value) for value in shard["task_ids"])
    return result


def _write_shard(directory: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    rows.sort(key=lambda row: int(row["task_id"]))
    path = directory / f"shard-{shard_index:05d}.npz"
    np.savez_compressed(
        path,
        task_ids=np.asarray([row["task_id"] for row in rows], dtype=np.int32),
        own_indices=np.asarray([row["own_index"] for row in rows], dtype=np.int16),
        opponent_indices=np.asarray([row["opponent_index"] for row in rows], dtype=np.int16),
        seeds=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        seats=np.asarray([row["seat"] for row in rows], dtype=np.int8),
        rewards=np.asarray([row["reward"] for row in rows], dtype=np.float32),
        opponent_rewards=np.asarray([row["opponent_reward"] for row in rows], dtype=np.float32),
        margins=np.asarray([row["margin"] for row in rows], dtype=np.float32),
        scores=np.asarray([row["score"] for row in rows], dtype=np.float32),
        states=np.stack([row["states"] for row in rows]),
    )
    return path


def _load_results(directory: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(directory.glob("shard-*.npz")):
        with np.load(path) as shard:
            for key in shard.files:
                values[key].append(shard[key])
    return {key: np.concatenate(parts) for key, parts in values.items()}


def _summarize(
    data: dict[str, np.ndarray], own: list[dict[str, Any]], opponents: list[dict[str, Any]]
) -> dict[str, Any]:
    versus = {}
    for own_index, own_row in enumerate(own):
        local = {}
        own_mask = data["own_indices"] == own_index
        for opponent_index, opponent_row in enumerate(opponents):
            mask = own_mask & (data["opponent_indices"] == opponent_index)
            if not mask.any():
                continue
            local[str(opponent_row["family"])] = {
                "games": int(mask.sum()),
                "score_rate": float(data["scores"][mask].mean()),
                "mean_margin": float(data["margins"][mask].mean()),
            }
        mask = own_mask
        versus[str(own_row["route_id"])] = {
            "family": str(own_row["family"]), "games": int(mask.sum()),
            "score_rate": float(data["scores"][mask].mean()),
            "mean_margin": float(data["margins"][mask].mean()),
            "versus": local,
        }
    return versus


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / "route-runtime-v8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_range, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=(24, 48, 72, 96, 120, 144, 168))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--own-limit", type=int, default=0)
    parser.add_argument("--opponent-limit", type=int, default=0)
    args = parser.parse_args(argv)
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    own = list(payload["candidates"])
    opponents = list(payload["opponent_representatives"])
    if args.own_limit > 0:
        own = own[: args.own_limit]
    if args.opponent_limit > 0:
        opponents = opponents[: args.opponent_limit]
    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1, "runtime": str(args.runtime),
        "candidate_source": str(args.candidates), "own": own,
        "opponents": opponents, "seeds": list(args.seeds),
        "seats": [0, 1], "checkpoints": list(args.checkpoints),
    }
    selection_path = args.output / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        if previous != selection:
            raise RuntimeError("output directory contains a different search selection")
    else:
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    tasks = []
    task_id = 0
    for own_index, own_row in enumerate(own):
        for opponent_index, opponent_row in enumerate(opponents):
            for seed in args.seeds:
                for seat in (0, 1):
                    tasks.append((
                        task_id, own_index, opponent_index,
                        str(own_row["route_id"]), str(opponent_row["route_id"]),
                        int(seed), seat,
                    ))
                    task_id += 1
    completed = _completed(args.output)
    pending = [task for task in tasks if task[0] not in completed]
    workers = min(args.workers or (os.cpu_count() or 1), max(1, len(pending)))
    print(json.dumps({
        "tasks": len(tasks), "completed": len(completed), "pending": len(pending),
        "workers": workers, "own": len(own), "opponents": len(opponents),
    }), flush=True)
    shard_index = len(list(args.output.glob("shard-*.npz")))
    buffer = []
    errors = []
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(
            workers, initializer=_worker_init,
            initargs=(str(args.runtime), tuple(args.checkpoints)),
        ) as pool:
            for index, row in enumerate(pool.imap_unordered(_play, pending, chunksize=1), 1):
                if "error" in row:
                    errors.append(row)
                else:
                    buffer.append(row)
                if len(buffer) >= args.shard_size:
                    path = _write_shard(args.output, shard_index, buffer)
                    shard_index += 1
                    buffer = []
                    print(f"[{len(completed)+index}/{len(tasks)}] wrote {path.name}", flush=True)
        if buffer:
            _write_shard(args.output, shard_index, buffer)
    if errors:
        (args.output / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{len(errors)} games failed; first={errors[0]}")
    data = _load_results(args.output)
    report = {
        "schema_version": 1, "engine": "FastEnv default official parameters",
        "paired": {"seeds": list(args.seeds), "seats": [0, 1]},
        "games": int(len(data["task_ids"])), "workers": workers,
        "checkpoints": list(args.checkpoints),
        "summary": _summarize(data, own, opponents),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "games": report["games"],
        "summary": str(args.output / "summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
