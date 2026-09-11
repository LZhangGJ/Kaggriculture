"""Screen every Hasegawa complete route against FC24B after first-shop reveal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_PREFIX_LOCK  # noqa: E402
from kaggriculture_jax.constants import SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def make_rollout(
    tables,
    latest_bank,
    old_bank,
    runtime,
    trace_player,
    activation_step: int | None = None,
):
    fc_player = 1 - trace_player

    @jax.jit
    def rollout(initial, events, forced_route, trace_bank):
        batch = initial.step.shape[0]

        def body(value, _):
            states, trace_carry, fc_carry = value
            activate = (
                states.town_count > 0
                if activation_step is None
                else states.step.astype(jnp.int32) >= activation_step
            )
            route = jnp.where(activate, forced_route, trace_carry.branch_id).astype(jnp.int16)
            fc_action, fc_carry = fc24_terminal_crop_salvage_player_action_v1(
                states, tables, latest_bank, old_bank, runtime, fc_carry, fc_player
            )
            states, trace_carry, _, _ = hasegawa_step_with_external_v3(
                states,
                trace_carry,
                trace_bank,
                fc_action,
                trace_player,
                events,
                tables,
                ROUTER_PREFIX_LOCK,
                None,
                forced_route=route,
                forced_lock=activate,
            )
            return (states, trace_carry, fc_carry), None

        trace_carry = initialize_hasegawa_carry_v3(batch, trace_bank.bootstrap_route_id)
        fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
        (terminal, trace_carry, _), _ = jax.lax.scan(
            body, (initial, trace_carry, fc_carry), None, length=719
        )
        return terminal.money, terminal.done, trace_carry, terminal.town_shops

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--route-chunk", type=int, default=8)
    parser.add_argument(
        "--route-ids",
        default="",
        help="Optional comma-separated global trace-bank route IDs to screen.",
    )
    parser.add_argument("--seed-start", type=int, default=162001)
    parser.add_argument(
        "--route-start",
        choices=("first_shop", "step0", "step24", "step72", "step144"),
        default="first_shop",
    )
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    available_routes = int(trace_bank.source_reward.shape[0])
    selected_route_ids = np.asarray(
        [int(value) for value in args.route_ids.split(",") if value]
        if args.route_ids
        else list(range(available_routes)),
        dtype=np.int32,
    )
    if (
        selected_route_ids.size == 0
        or np.any(selected_route_ids < 0)
        or np.any(selected_route_ids >= available_routes)
    ):
        raise ValueError(f"invalid route IDs: {selected_route_ids.tolist()}")
    if np.unique(selected_route_ids).size != selected_route_ids.size:
        raise ValueError("route IDs must be unique")
    route_count = int(selected_route_ids.size)
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    expanded_seeds = np.repeat(seeds, args.route_chunk)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(expanded_seeds))

    cash = np.zeros((2, args.batch, route_count), np.int32)
    fc_cash = np.zeros_like(cash)
    invalid = np.zeros_like(cash)
    resync = np.zeros_like(cash)
    hard = np.zeros_like(cash)
    first_shop = np.full((2, args.batch), -1, np.int8)
    observed_shops = np.full((2, args.batch, route_count, 8), -1, np.int8)
    done_all = True
    timings = []
    for seat in (0, 1):
        activation_step = {
            "first_shop": None,
            "step0": 0,
            "step24": 24,
            "step72": 72,
            "step144": 144,
        }[args.route_start]
        rollout = make_rollout(
            tables,
            latest_bank,
            old_bank,
            runtime,
            seat,
            activation_step=activation_step,
        )
        for start in range(0, route_count, args.route_chunk):
            valid = np.arange(start, min(start + args.route_chunk, route_count), dtype=np.int32)
            padded = np.pad(valid, (0, args.route_chunk - len(valid)), mode="edge")
            forced = np.tile(selected_route_ids[padded], args.batch)
            tick = time.perf_counter()
            result = rollout(initial, events, jnp.asarray(forced, dtype=jnp.int16), trace_bank)
            jax.block_until_ready(result[0])
            elapsed = time.perf_counter() - tick
            money, done, carry, observed_shop_sequence = jax.device_get(result)
            shape = (args.batch, args.route_chunk)
            n = len(valid)
            cash[seat][:, valid] = money[:, seat].reshape(shape)[:, :n]
            fc_cash[seat][:, valid] = money[:, 1 - seat].reshape(shape)[:, :n]
            invalid[seat][:, valid] = carry.invalid_intent_total.reshape(shape)[:, :n]
            resync[seat][:, valid] = carry.resync_total.reshape(shape)[:, :n]
            hard[seat][:, valid] = carry.hard_counter_total.reshape(shape)[:, :n]
            shop_shape = (args.batch, args.route_chunk, 8)
            observed_shops[seat][:, valid, :] = observed_shop_sequence.reshape(shop_shape)[:, :n, :]
            first_shop[seat] = observed_shops[seat, :, valid[0], 0]
            done_all = done_all and bool(np.all(done))
            timings.append({"seat": seat, "start": int(valid[0]), "end": int(valid[-1]), "seconds": elapsed})
            print(json.dumps({"seat": seat, "routes": [int(valid[0]), int(valid[-1])], "seconds": elapsed, "done": bool(np.all(done))}), flush=True)

    margin = cash - fc_cash
    source_episode = np.asarray(trace_bank.source_episode_id)[selected_route_ids]
    source_reward = np.asarray(trace_bank.source_reward)[selected_route_ids]
    source_first_shop = np.asarray(trace_bank.source_shop_sequence)[selected_route_ids, 0]
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
        route_ids = np.flatnonzero(eligible)
        order = np.lexsort(
            (source_reward[route_ids], -invalid_means[route_ids], means[route_ids], wins[route_ids])
        )[::-1]
        top = []
        for route_id in route_ids[order[: min(10, route_ids.size)]]:
            top.append(
                {
                    "route_id": int(selected_route_ids[route_id]),
                    "source_episode_id": int(source_episode[route_id]),
                    "source_reward": int(source_reward[route_id]),
                    "games": int(group_margin.shape[0]),
                    "win_rate": float(wins[route_id]),
                    "mean_margin": float(means[route_id]),
                    "invalid_mean": float(invalid_means[route_id]),
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

    matrix_path = args.output.with_suffix(".npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        route_ids=selected_route_ids,
    )
    payload = {
        "schema": "kaggriculture.front40_fusion.hasegawa-route-screen-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "seed_count": args.batch,
        "route_count": route_count,
        "route_ids": selected_route_ids.tolist(),
        "route_chunk": args.route_chunk,
        "route_start": args.route_start,
        "compilation_cache": str(args.compilation_cache.resolve()),
        "games": int(2 * args.batch * route_count),
        "all_done": done_all,
        "hard_counter_total": int(np.sum(hard)),
        "groups": groups,
        "timings": timings,
        "matrix": str(matrix_path),
        "status": "PASS" if done_all and int(np.sum(hard)) == 0 else "FAIL",
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output), "games": payload["games"]}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
