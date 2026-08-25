"""Screen several fixed-capacity trace banks in one JAX process.

The rollout functions are created once per seat.  Every job must use an
identically padded trace-bank shape, batch and route chunk, so Rank families
reuse the same compiled executables while retaining separate event seeds and
auditable receipts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from run_hasegawa_route_screen_vs_fc24b import (
    Events,
    ROOT,
    SHOP_NAMES,
    build_events_v1,
    load_bank,
    load_hasegawa_trace_bank_v3,
    load_high_potential_runtime_tables_v1,
    load_tables,
    make_rollout,
    reset,
)


def screen_job(
    spec: dict,
    *,
    rollouts: dict[int, object],
    batch: int,
    route_chunk: int,
    route_start: str,
    compilation_cache: Path,
) -> dict:
    bank_path = ROOT / spec["bank"]
    output = ROOT / spec["output"]
    trace_bank = load_hasegawa_trace_bank_v3(bank_path)
    capacity = int(trace_bank.source_reward.shape[0])
    route_ids = np.asarray(
        spec.get("route_ids", list(range(int(spec["route_count"])))), dtype=np.int32
    )
    if route_ids.size == 0 or np.any(route_ids < 0) or np.any(route_ids >= capacity):
        raise ValueError(f"invalid route IDs for {spec['name']}: {route_ids.tolist()}")
    if np.unique(route_ids).size != route_ids.size:
        raise ValueError(f"duplicate route IDs for {spec['name']}")
    route_count = int(route_ids.size)

    seeds = np.arange(int(spec["seed_start"]), int(spec["seed_start"]) + batch, dtype=np.int32)
    expanded_seeds = np.repeat(seeds, route_chunk)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(expanded_seeds))

    cash = np.zeros((2, batch, route_count), np.int32)
    fc_cash = np.zeros_like(cash)
    invalid = np.zeros_like(cash)
    resync = np.zeros_like(cash)
    hard = np.zeros_like(cash)
    first_shop = np.full((2, batch), -1, np.int8)
    observed_shops = np.full((2, batch, route_count, 8), -1, np.int8)
    done_all = True
    timings: list[dict] = []

    for seat in (0, 1):
        rollout = rollouts[seat]
        for start in range(0, route_count, route_chunk):
            valid = np.arange(start, min(start + route_chunk, route_count), dtype=np.int32)
            padded = np.pad(valid, (0, route_chunk - len(valid)), mode="edge")
            forced = np.tile(route_ids[padded], batch)
            tick = time.perf_counter()
            result = rollout(initial, events, jnp.asarray(forced, dtype=jnp.int16), trace_bank)
            jax.block_until_ready(result[0])
            elapsed = time.perf_counter() - tick
            money, done, carry, observed = jax.device_get(result)
            shape = (batch, route_chunk)
            n = len(valid)
            cash[seat][:, valid] = money[:, seat].reshape(shape)[:, :n]
            fc_cash[seat][:, valid] = money[:, 1 - seat].reshape(shape)[:, :n]
            invalid[seat][:, valid] = carry.invalid_intent_total.reshape(shape)[:, :n]
            resync[seat][:, valid] = carry.resync_total.reshape(shape)[:, :n]
            hard[seat][:, valid] = carry.hard_counter_total.reshape(shape)[:, :n]
            observed_shops[seat][:, valid, :] = observed.reshape(batch, route_chunk, 8)[:, :n, :]
            first_shop[seat] = observed_shops[seat, :, valid[0], 0]
            done_all = done_all and bool(np.all(done))
            timings.append(
                {"seat": seat, "start": int(valid[0]), "end": int(valid[-1]), "seconds": elapsed}
            )
            print(
                json.dumps(
                    {
                        "job": spec["name"],
                        "seat": seat,
                        "routes": [int(valid[0]), int(valid[-1])],
                        "seconds": elapsed,
                        "done": bool(np.all(done)),
                    }
                ),
                flush=True,
            )

    margin = cash - fc_cash
    source_episode = np.asarray(trace_bank.source_episode_id)[route_ids]
    source_reward = np.asarray(trace_bank.source_reward)[route_ids]
    source_first_shop = np.asarray(trace_bank.source_shop_sequence)[route_ids, 0]
    groups = []
    for shop_id, shop_name in enumerate(SHOP_NAMES):
        event_mask = first_shop[0] == shop_id
        eligible = source_first_shop == shop_id
        if not np.any(event_mask) or not np.any(eligible):
            continue
        group_margin = margin[:, event_mask, :].reshape(-1, route_count)
        group_invalid = invalid[:, event_mask, :].reshape(-1, route_count)
        wins = np.mean(group_margin > 0, axis=0)
        means = np.mean(group_margin, axis=0)
        invalid_means = np.mean(group_invalid, axis=0)
        columns = np.flatnonzero(eligible)
        order = np.lexsort(
            (source_reward[columns], -invalid_means[columns], means[columns], wins[columns])
        )[::-1]
        top = []
        for column in columns[order[: min(10, columns.size)]]:
            top.append(
                {
                    "route_id": int(route_ids[column]),
                    "source_episode_id": int(source_episode[column]),
                    "source_reward": int(source_reward[column]),
                    "games": int(group_margin.shape[0]),
                    "win_rate": float(wins[column]),
                    "mean_margin": float(means[column]),
                    "invalid_mean": float(invalid_means[column]),
                }
            )
        groups.append(
            {
                "first_shop_id": shop_id,
                "first_shop": shop_name,
                "seed_count": int(np.sum(event_mask)),
                "eligible_routes": int(np.sum(eligible)),
                "top_routes": top,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = output.with_suffix(".npz")
    np.savez_compressed(
        matrix_path,
        seeds=seeds,
        first_shop=first_shop,
        cash=cash,
        opponent_cash=fc_cash,
        margin=margin,
        invalid=invalid,
        resync=resync,
        hard=hard,
        observed_shops=observed_shops,
        source_episode_id=source_episode,
        source_reward=source_reward,
        source_first_shop=source_first_shop,
        route_ids=route_ids,
    )
    payload = {
        "schema": "kaggriculture.front40_fusion.hasegawa-route-screen-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "job": spec["name"],
        "trace_bank": str(bank_path),
        "trace_bank_capacity": capacity,
        "seed_start": int(spec["seed_start"]),
        "seed_count": batch,
        "route_count": route_count,
        "route_ids": route_ids.tolist(),
        "route_chunk": route_chunk,
        "route_start": route_start,
        "compilation_cache": str(compilation_cache.resolve()),
        "games": int(2 * batch * route_count),
        "all_done": done_all,
        "hard_counter_total": int(np.sum(hard)),
        "groups": groups,
        "timings": timings,
        "matrix": str(matrix_path),
        "status": "PASS" if done_all and int(np.sum(hard)) == 0 else "FAIL",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"job": spec["name"], "status": payload["status"], "games": payload["games"]}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--route-chunk", type=int, default=8)
    parser.add_argument(
        "--route-start", choices=("first_shop", "step0", "step24", "step72", "step144"),
        default="first_shop",
    )
    parser.add_argument("--compilation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tables = load_tables()
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    activation_step = {
        "first_shop": None,
        "step0": 0,
        "step24": 24,
        "step72": 72,
        "step144": 144,
    }[args.route_start]
    rollouts = {
        seat: make_rollout(
            tables, latest_bank, old_bank, runtime, seat, activation_step=activation_step
        )
        for seat in (0, 1)
    }

    rows = []
    expected_capacity = None
    for spec in manifest["jobs"]:
        with np.load(ROOT / spec["bank"], allow_pickle=False) as bank:
            capacity = int(bank["source_reward"].shape[0])
        if expected_capacity is None:
            expected_capacity = capacity
        elif capacity != expected_capacity:
            raise ValueError(f"trace-bank capacity differs: {capacity} != {expected_capacity}")
        rows.append(
            screen_job(
                spec,
                rollouts=rollouts,
                batch=args.batch,
                route_chunk=args.route_chunk,
                route_start=args.route_start,
                compilation_cache=args.compilation_cache,
            )
        )

    result = {
        "schema": "kaggriculture.front40_fusion.multi-padded-route-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "bank_capacity": expected_capacity,
        "jobs": [
            {"name": row["job"], "output": str(manifest["jobs"][i]["output"]), "games": row["games"]}
            for i, row in enumerate(rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
