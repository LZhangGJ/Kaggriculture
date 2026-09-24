#!/usr/bin/env python3
"""Smoke the existing C++ route executor as a route-agnostic RL opponent pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from meta_agent.src.teammate_expanded_routes import load_action_tapes


ROOT = Path(__file__).resolve().parents[3]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def build_pool(actions_path: Path, metadata_path: Path, max_source_failures: int):
    # Import after OMP_NUM_THREADS is set so --threads controls libgomp too.
    from fast_kaggriculture import NativeTeammateExecutor

    actions = load_action_tapes(actions_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in metadata["opponent_routes"]
        if entry["route_id"] in actions and (
            max_source_failures < 0
            or int(entry.get("source_execution_hard_failures") or 0)
            <= max_source_failures
        )
    ]
    if len(entries) < 2:
        raise ValueError("the selected replay pool needs at least two routes")
    pass_tape = [PASS] * 719
    executor = NativeTeammateExecutor(
        [actions[entry["route_id"]] for entry in entries],
        pass_tape, pass_tape, [pass_tape] * 5, [pass_tape] * 5,
    )
    return executor, entries


def tasks(count: int, routes: int, seed0: int, mode: str) -> np.ndarray:
    result = np.full((count, 7), -1, dtype=np.int64)
    for pair in range(count // 2):
        seed = seed0 + pair
        if mode == "fixed":
            left, right = pair % routes, (pair + 1) % routes
        else:
            left = splitmix64(seed ^ 0xA0761D6478BD642F) % routes
            right = splitmix64(seed ^ 0xE7037ED1A0B428DB) % routes
            if right == left:
                right = (right + 1) % routes
        result[2 * pair, :3] = left, right, seed
        result[2 * pair + 1, :3] = right, left, seed
    return result


def step_api_self_check(executor, route: int, seed: int) -> tuple[float, float]:
    from fast_kaggriculture import Config, FastEnv, NativeReplayOpponent

    opponent = NativeReplayOpponent(executor, route)
    probe = FastEnv(Config(), seed)
    try:
        opponent.action(probe, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("step API accepted an env/step mismatch")

    def run_once() -> tuple[tuple[float, float], str]:
        env = FastEnv(Config(), seed)
        opponent.reset(route)
        digest = hashlib.sha256()
        while not env.done:
            action = opponent.action(env, 1, env.step_count)
            digest.update(json.dumps(action, sort_keys=True).encode())
            env.step([PASS, action])
        return tuple(env.rewards), digest.hexdigest()

    first, second = run_once(), run_once()
    assert first == second
    return first[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=ROOT / "agent/route_actions.json.zlib")
    parser.add_argument("--metadata", type=Path, default=ROOT / "agent/route_library.json")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2620000000)
    parser.add_argument("--mode", choices=("fixed", "hash"), default="hash")
    parser.add_argument("--max-source-failures", type=int, default=-1,
                        help="-1 keeps all representative routes; 0 keeps clean-source routes")
    parser.add_argument("--repair-mask", type=int, default=0,
                        help="existing NativeRepairOptions bit mask; 0 is the frozen route shell")
    parser.add_argument("--threads", type=int, default=min(192, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.threads < 1:
        parser.error("--games must be positive and even; --threads must be >=1")
    if not 0 <= args.repair_mask <= 127:
        parser.error("--repair-mask must be in [0,127]")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    setup_started = time.perf_counter()
    executor, entries = build_pool(args.actions, args.metadata, args.max_source_failures)
    step_rewards = step_api_self_check(executor, 0, args.seed)
    batch = tasks(args.games, len(entries), args.seed, args.mode)
    assert np.array_equal(batch, tasks(args.games, len(entries), args.seed, args.mode))
    assert np.all(batch[:, 3:] == -1)
    assert np.array_equal(batch[0::2, 0], batch[1::2, 1])
    assert np.array_equal(batch[0::2, 1], batch[1::2, 0])
    assert np.array_equal(batch[0::2, 2], batch[1::2, 2])
    setup_seconds = time.perf_counter() - setup_started

    started = time.perf_counter()
    rewards = np.asarray(executor.play_batch(batch, True, args.repair_mask), dtype=np.float64)
    simulation_seconds = time.perf_counter() - started
    audit_count = min(args.games, 256)
    audit_rewards, audit = executor.play_audit_batch(
        batch[:audit_count], True, args.repair_mask
    )
    audit_rewards, audit = np.asarray(audit_rewards), np.asarray(audit)

    # One compact check catches selection drift, shared per-episode state, ABI
    # mistakes, and non-finite terminal simulation without adding a test stack.
    repeat = np.asarray(executor.play_batch(
        batch[: min(16, args.games)], True, args.repair_mask
    ))
    assert np.array_equal(rewards[: len(repeat)], repeat)
    assert np.array_equal(rewards[:audit_count], audit_rewards)
    assert rewards.shape == (args.games, 2) and np.isfinite(rewards).all()
    assert ((audit[:, :, 2] == -1) | ((audit[:, :, 2] >= 0) & (audit[:, :, 2] < 719))).all()

    used = np.unique(batch[:, :2])
    failures = audit[:, :, :2].sum(axis=2)
    margins = rewards[:, 0] - rewards[:, 1]
    report = {
        "schema": "native-replay-pool-smoke-v1",
        "executor": "fastkag::NativeTeammateExecutor",
        "neutral_special_economy": True,
        "repair_mask": args.repair_mask,
        "selection": args.mode,
        "selection_scope": "episode reset only; no opponent identity reaches policy",
        "pool_routes": len(entries),
        "routes_exercised": int(len(used)),
        "games": args.games,
        "paired_seeds": args.games // 2,
        "both_seats": True,
        "steps_per_game": 719,
        "route_choices_per_player_game": 1,
        "route_switches_per_player_game": 0,
        "step_api_deterministic": True,
        "step_api_sync_guard": True,
        "step_api_route": 0,
        "step_api_route_id": entries[0]["route_id"],
        "step_api_rewards": step_rewards,
        "switch_columns_all_minus_one": bool(np.all(batch[:, 3:] == -1)),
        "audit_games": audit_count,
        "threads": args.threads,
        "setup_seconds": setup_seconds,
        "simulation_seconds": simulation_seconds,
        "games_per_second": args.games / simulation_seconds,
        "terminal_games": int(np.isfinite(rewards).all(axis=1).sum()),
        "terminal_cash_mean": rewards.mean(axis=0).tolist(),
        "terminal_cash_min": rewards.min(axis=0).tolist(),
        "terminal_cash_max": rewards.max(axis=0).tolist(),
        "player0_wins": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "player0_losses": int((margins < 0).sum()),
        "absolute_cash_margin_mean": float(np.abs(margins).mean()),
        "zero_macro_failure_player_games": int((failures == 0).sum()),
        "macro_failure_player_games": int((failures > 0).sum()),
        "mean_macro_unit_failures": float(audit[:, :, 0].mean()),
        "mean_macro_market_failures": float(audit[:, :, 1].mean()),
        "first_macro_failure_min": int(audit[:, :, 2][audit[:, :, 2] >= 0].min())
        if (audit[:, :, 2] >= 0).any() else -1,
        "actions_sha256": hashlib.sha256(args.actions.read_bytes()).hexdigest(),
        "families": [entry["family"] for entry in entries],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
