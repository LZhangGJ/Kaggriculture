"""Generate structured BC shards by querying a strong local teacher agent."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_lab.fast_env import FastKaggricultureEnv, resolve_agent
from kaggriculture_lab.gpu_policy import encode_observation
from kaggriculture_lab.policy_v2 import (
    MARKET_TOKEN_INDEX,
    MAX_MARKET_ORDERS,
    MAX_QUANTITY,
    UNIT_INDEX,
    structured_action_targets,
)


def _safe_name(path: str) -> str:
    candidate = Path(path)
    return candidate.parent.name if candidate.suffix == ".py" else path.replace(":", "_")


def _worker(task: tuple[str, str, int, int, str]) -> dict[str, Any]:
    teacher_spec, opponent_spec, seed, teacher_seat, output_name = task
    teacher = resolve_agent(teacher_spec)
    opponent = resolve_agent(opponent_spec)
    env = FastKaggricultureEnv()
    env.reset(seed)

    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "unit_context": [],
        "unit_active": [],
        "unit_targets": [],
        "unit_quantity_targets": [],
        "unit_quantity_active": [],
        "market_targets": [],
        "market_quantity_targets": [],
        "market_quantity_active": [],
        "market_order_active": [],
    }
    stats: Counter[str] = Counter()

    while not env.done:
        observations = env.observations()
        teacher_observation = observations[teacher_seat]
        opponent_observation = observations[1 - teacher_seat]
        teacher_action = teacher(teacher_observation, env.configuration)
        opponent_action = opponent(opponent_observation, env.configuration)

        feature, context, _ = encode_observation(teacher_observation)
        targets = structured_action_targets([teacher_observation], [teacher_action])
        arrays["features"].append(feature.astype(np.float16))
        arrays["unit_context"].append(context.astype(np.float16))
        for key in targets:
            arrays[key].append(targets[key][0])

        raw_units = [teacher_action.get("farmer", ["PASS"]), *list(teacher_action.get("hands", []) or [])]
        for raw in raw_units:
            op = str(raw[0]) if raw else "PASS"
            item = raw[1] if len(raw) > 1 else None
            stats["unit_actions"] += 1
            stats["unit_unrepresentable"] += (op, item) not in UNIT_INDEX
            if len(raw) > 2:
                stats["quantities_clipped"] += int(raw[2]) > MAX_QUANTITY
        market = list(teacher_action.get("market", []) or [])
        stats["market_orders"] += len(market)
        stats["market_orders_dropped"] += max(0, len(market) - MAX_MARKET_ORDERS)
        for raw in market[:MAX_MARKET_ORDERS]:
            op = str(raw[0]) if raw else "NONE"
            item = raw[1] if len(raw) > 1 else None
            stats["market_unrepresentable"] += (
                op, None if op in ("HIRE", "BUY_LAND") else item
            ) not in MARKET_TOKEN_INDEX
            if len(raw) > 2:
                stats["quantities_clipped"] += int(raw[2]) > MAX_QUANTITY

        actions = (
            [teacher_action, opponent_action]
            if teacher_seat == 0
            else [opponent_action, teacher_action]
        )
        env.step(actions)

    state = env._require_state()
    teacher_reward = float(state[teacher_seat].reward)
    opponent_reward = float(state[1 - teacher_seat].reward)
    outcome = float((teacher_reward > opponent_reward) - (teacher_reward < opponent_reward))
    output = Path(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    stacked = {key: np.stack(values) for key, values in arrays.items()}
    stacked["value_targets"] = np.full(len(stacked["features"]), outcome, dtype=np.float16)
    np.savez(output, **stacked)
    stats["examples"] = len(stacked["features"])
    stats["wins"] = teacher_reward > opponent_reward
    stats["ties"] = teacher_reward == opponent_reward
    stats["losses"] = teacher_reward < opponent_reward
    return {
        "output": str(output),
        "teacher": teacher_spec,
        "opponent": opponent_spec,
        "seed": seed,
        "teacher_seat": teacher_seat,
        "teacher_reward": teacher_reward,
        "opponent_reward": opponent_reward,
        "stats": dict(stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--opponent", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Kaggriculture\data\processed\teacher_bc_v2"))
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("at least five seeds are required for train/validation/test splits")

    teacher_name = _safe_name(args.teacher)
    tasks: list[tuple[str, str, int, int, str]] = []
    assignments: dict[str, str] = {}
    for seed_offset in range(args.seeds):
        split_fraction = seed_offset / args.seeds
        split = "train" if split_fraction < 0.8 else "validation" if split_fraction < 0.9 else "test"
        seed = args.seed_start + seed_offset
        for opponent_spec in args.opponent:
            opponent_name = _safe_name(opponent_spec)
            for teacher_seat in (0, 1):
                filename = f"{teacher_name}__vs__{opponent_name}__seed{seed}__seat{teacher_seat}.npz"
                output = args.output_dir / split / filename
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f"output already exists; pass --overwrite: {output}")
                assignments[filename] = split
                tasks.append((args.teacher, opponent_spec, seed, teacher_seat, str(output)))

    total: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    if args.workers <= 1:
        results = map(_worker, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_worker, tasks, chunksize=1)
    try:
        for index, result in enumerate(results, start=1):
            episodes.append(result)
            total.update(result["stats"])
            print(
                f"[{index}/{len(tasks)}] seed={result['seed']} seat={result['teacher_seat']} "
                f"teacher={result['teacher_reward']:.0f} opponent={result['opponent_reward']:.0f} "
                f"output={result['output']}"
            )
    finally:
        if args.workers > 1:
            executor.shutdown()

    manifest = {
        "schema_version": 2,
        "teacher": args.teacher,
        "opponents": args.opponent,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "assignments": assignments,
        "stats": dict(total),
        "episodes": episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(dict(total), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
