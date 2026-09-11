#!/usr/bin/env python3
"""Replay raw actions in the native simulator and audit macro-plan failures.

The output rows are aligned exactly with the input macro feature cache.  Python
only decodes replay JSON and schedules workers; all 719 game turns and failure
accounting run in the C++ simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INTENT_ANCHORS = (168, 288, 432, 576, 719)
INTENT_CATEGORY = {
    "WHEAT": 1, "CARROT": 2, "TOMATO": 3, "STRAWBERRY": 4,
    "MELON": 5, "GOOSE": 6, "COW": 7, "SHEEP": 8,
    "COOP": 9, "PASTURE": 10,
}


def _episode_id(path: Path) -> int:
    match = re.search(r"episode-(\d+)-replay", path.name)
    if not match:
        raise ValueError(f"not a replay path: {path}")
    return int(match.group(1))


def _make_config(fk, raw: dict):
    config = fk.Config()
    source = raw.get("configuration") or {}
    mapping = {
        "episodeSteps": "episode_steps",
        "boardSize": "board_size",
        "startingMoney": "starting_money",
        "maxMarketOrdersPerTurn": "max_market_orders",
        "turnsPerDay": "turns_per_day",
        "shedCapacity": "shed_capacity",
        "weedSpawnChance": "weed_spawn_chance",
        "townShopUnlockInterval": "town_shop_unlock_interval",
        "townShopSellInterval": "town_shop_sell_interval",
        "townCenterSellInterval": "town_center_sell_interval",
        "farmHandCostMult": "farm_hand_cost_mult",
    }
    for source_name, target_name in mapping.items():
        if source_name in source:
            setattr(config, target_name, source[source_name])
    return config


def _planned_layouts(steps: list, episode_steps: int) -> np.ndarray:
    """Cumulative intended production locations, independent of action success."""
    result = np.zeros((2, len(INTENT_ANCHORS), 100), dtype=np.int8)
    plans = np.zeros((2, 100), dtype=np.int8)
    anchor_index = {step: idx for idx, step in enumerate(INTENT_ANCHORS)}
    last = min(len(steps), episode_steps)
    for step in range(1, last):
        for player in range(2):
            previous = steps[step - 1][player].get("observation") or {}
            farms = previous.get("farms") or []
            farm = farms[player] if player < len(farms) else {}
            positions = [farm.get("farmer"), *(farm.get("hands") or [])]
            action = steps[step][player].get("action") or {}
            unit_actions = [action.get("farmer"), *(action.get("hands") or [])]
            for actor, raw_action in enumerate(unit_actions):
                order = list(raw_action or [])
                if not order or actor >= len(positions) or positions[actor] is None:
                    continue
                op = str(order[0])
                item = str(order[1]) if len(order) > 1 else ""
                category = 0
                if op == "PLANT" and item in INTENT_CATEGORY:
                    category = INTENT_CATEGORY[item]
                elif op == "BUILD_COOP":
                    category = INTENT_CATEGORY["COOP"]
                elif op == "BUILD_PASTURE":
                    category = INTENT_CATEGORY["PASTURE"]
                elif op == "PLACE" and item in ("GOOSE", "COW", "SHEEP"):
                    category = INTENT_CATEGORY[item]
                if category:
                    x, y = (int(value) for value in positions[actor])
                    if 0 <= x < 10 and 0 <= y < 10:
                        plans[player, y * 10 + x] = category
        if step in anchor_index:
            result[:, anchor_index[step], :] = plans
    return result


def _audit_one(path_text: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import orjson
    import fast_kaggriculture as fk

    path = Path(path_text)
    raw = orjson.loads(path.read_bytes())
    steps = raw["steps"]
    episode_steps = int((raw.get("configuration") or {}).get("episodeSteps", 720))
    last = min(len(steps), episode_steps)
    tapes = [
        [(steps[step][player].get("action") or {}) for step in range(1, last)]
        for player in range(2)
    ]
    seed = int((raw.get("info") or {}).get("seed", 0))
    metrics, simulated_rewards = fk.audit_raw_tapes(
        tapes[0], tapes[1], seed, _make_config(fk, raw)
    )
    planned_layouts = _planned_layouts(steps, episode_steps)
    stored_rewards = raw.get("rewards")
    if not stored_rewards or len(stored_rewards) != 2:
        stored_rewards = [steps[-1][player].get("reward") for player in range(2)]
    return (
        int((raw.get("info") or {}).get("EpisodeId") or _episode_id(path)),
        np.asarray(metrics, dtype=np.float64),
        np.asarray(simulated_rewards, dtype=np.float64),
        np.asarray(stored_rewards, dtype=np.float64),
        planned_layouts,
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"),
            np.quantile(values, (0, .1, .25, .5, .75, .9, .95, .99, 1)),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 1))
    args = parser.parse_args()

    import fast_kaggriculture as fk

    with np.load(args.features, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    keys = np.asarray(
        [(int(row["episode_id"]), int(row["player_index"])) for row in rows],
        dtype=np.int64,
    )
    wanted_episodes = set(keys[:, 0].tolist())
    replay_paths: dict[int, Path] = {}
    for replay_root in args.replay_root:
        for path in replay_root.glob("episode-*-replay.json"):
            episode = _episode_id(path)
            if episode in wanted_episodes:
                replay_paths.setdefault(episode, path)
    missing = sorted(wanted_episodes - replay_paths.keys())
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} replay files, first: {missing[:10]}")

    metric_names = list(fk.raw_tape_audit_metric_names())
    by_episode: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    failures: list[dict] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_audit_one, str(path)): episode
            for episode, path in replay_paths.items()
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                returned_episode, metrics, simulated, stored, planned_layouts = future.result()
                if returned_episode != episode:
                    raise ValueError(f"episode id mismatch: {episode} != {returned_episode}")
                by_episode[episode] = (metrics, simulated, stored, planned_layouts)
            except Exception as exc:  # keep a useful failure manifest
                failures.append({"episode_id": episode, "error": repr(exc)})
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"audited {completed}/{len(futures)} replays; errors={len(failures)}",
                    flush=True,
                )
    if failures:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"failures": failures}, indent=2) + "\n")
        raise RuntimeError(f"native audit failed for {len(failures)} replays")

    metrics = np.stack([by_episode[e][0][p] for e, p in keys])
    simulated_rewards = np.asarray([by_episode[e][1][p] for e, p in keys])
    stored_rewards = np.asarray([by_episode[e][2][p] for e, p in keys])
    opponent_stored_rewards = np.asarray([by_episode[e][2][1 - p] for e, p in keys])
    planned_layouts = np.stack([by_episode[e][3][p] for e, p in keys])
    reward_abs_error = np.abs(simulated_rewards - stored_rewards)
    index = {name: idx for idx, name in enumerate(metric_names)}
    unit_failures = metrics[:, index["unit_attempts"]] - metrics[:, index["unit_valid"]]
    market_failures = metrics[:, index["market_requested"]] - metrics[:, index["market_filled"]]
    hard_failures = unit_failures + market_failures
    attempts = metrics[:, index["unit_attempts"]] + metrics[:, index["market_requested"]]
    completion_rate = np.divide(
        attempts - hard_failures, attempts,
        out=np.ones_like(attempts), where=attempts > 0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        keys=keys,
        metric_names=np.asarray(metric_names),
        metrics=metrics,
        simulated_rewards=simulated_rewards,
        stored_rewards=stored_rewards,
        opponent_stored_rewards=opponent_stored_rewards,
        reward_abs_error=reward_abs_error,
        unit_failures=unit_failures,
        market_failures=market_failures,
        hard_failures=hard_failures,
        completion_rate=completion_rate,
        planned_layouts=planned_layouts,
        planned_layout_anchors=np.asarray(INTENT_ANCHORS, dtype=np.int16),
    )
    summary = {
        "schema_version": 1,
        "features": str(args.features),
        "replay_roots": [str(path) for path in args.replay_root],
        "replay_sides": len(rows),
        "unique_episodes": len(replay_paths),
        "workers": args.workers,
        "metric_names": metric_names,
        "exact_reward_matches": int(np.count_nonzero(reward_abs_error == 0)),
        "reward_mismatches": int(np.count_nonzero(reward_abs_error != 0)),
        "max_reward_abs_error": float(reward_abs_error.max(initial=0)),
        "zero_hard_failure_sides": int(np.count_nonzero(hard_failures == 0)),
        "zero_unit_failure_sides": int(np.count_nonzero(unit_failures == 0)),
        "zero_market_failure_sides": int(np.count_nonzero(market_failures == 0)),
        "hard_failure_quantiles": _quantiles(hard_failures),
        "completion_rate_quantiles": _quantiles(completion_rate),
        "metric_totals": {
            name: float(metrics[:, idx].sum()) for idx, name in enumerate(metric_names)
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
