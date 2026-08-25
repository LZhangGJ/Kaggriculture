"""Seat-swapped GPU Arena for Hasegawa V3 observable-prefix routing."""

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
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import (  # noqa: E402
    ROUTER_FIRST_SHOP_MAP,
    ROUTER_PREFIX_COMPATIBLE,
    ROUTER_PREFIX_LOCK,
    ROUTER_PREFIX_PUBLIC_STATE,
    ROUTER_PREFIX_REWARD,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_exact5_arena_v2 import TARGETS  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    initialize_public_g02_carry_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import public_v27_player_action_exact_v1  # noqa: E402


ROUTER_MODES = {
    "first_shop_map": ROUTER_FIRST_SHOP_MAP,
    "prefix_compatible": ROUTER_PREFIX_COMPATIBLE,
    "prefix_lock": ROUTER_PREFIX_LOCK,
    "prefix_reward": ROUTER_PREFIX_REWARD,
    "prefix_state": ROUTER_PREFIX_PUBLIC_STATE,
}


def make_rollout(
    hbank, exact_bank, runtime, tables, kind, mode, route_id, h_player,
    router_mode, route_map, compatibility_threshold, compatibility_slack,
    lock_shop_count,
):
    opponent = 1 - h_player

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), route_id, dtype=jnp.int32)

        def body(value, _):
            states, hcarry, opponent_carry = value
            if kind == "v20":
                action, opponent_carry = high_potential_v20_player_action_v1(
                    states, tables, exact_bank, runtime, opponent_carry, opponent, mode
                )
            elif kind == "v25":
                action, opponent_carry = public_v27_player_action_exact_v1(
                    states, runtime, exact_bank, route_ids, opponent_carry, opponent
                )
            else:
                action, opponent_carry = public_rc5_weed_player_action_v1(
                    states, exact_bank, route_ids, opponent_carry, opponent
                )
            states, hcarry, _, _ = hasegawa_step_with_external_v3(
                states, hcarry, hbank, action, h_player, events, tables,
                router_mode, route_map, compatibility_threshold, compatibility_slack,
                lock_shop_count,
            )
            return (states, hcarry, opponent_carry), None

        hcarry = initialize_hasegawa_carry_v3(batch, hbank.bootstrap_route_id)
        opponent_carry = (
            initialize_high_potential_v20_carry_v1(batch)
            if kind == "v20"
            else initialize_public_g02_carry_v1(batch)
        )
        (terminal, hcarry, _), _ = jax.lax.scan(
            body, (initial, hcarry, opponent_carry), None, length=719
        )
        return terminal.money, terminal.done, hcarry

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--exact-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--opponents", default="all5")
    parser.add_argument("--router", choices=sorted(ROUTER_MODES), default="prefix_state")
    parser.add_argument("--route-map", type=Path)
    parser.add_argument("--compatibility-threshold", type=int, default=15000)
    parser.add_argument("--compatibility-slack", type=int, default=5000)
    parser.add_argument("--lock-shop-count", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=154001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    requested = list(TARGETS) if args.opponents == "all5" else [
        value.strip() for value in args.opponents.split(",") if value.strip()
    ]
    hbank = load_hasegawa_trace_bank_v3(args.hasegawa_bank)
    if args.router == "first_shop_map":
        if args.route_map is None:
            raise ValueError("--route-map is required for first_shop_map")
        route_map_payload = json.loads(args.route_map.read_text(encoding="utf-8"))
        route_map = jnp.asarray(route_map_payload["route_ids"], dtype=jnp.int16)
        if route_map.shape != (8,):
            raise ValueError(f"route_ids must contain 8 entries, got {route_map.shape}")
    else:
        route_map = jnp.zeros((8,), dtype=jnp.int16)
    exact_bank = load_bank(args.exact_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    rows = []
    for name in requested:
        kind, mode, route_id = TARGETS[name]
        for h_player in (0, 1):
            rollout = make_rollout(
                hbank, exact_bank, runtime, tables, kind, mode, route_id,
                h_player, ROUTER_MODES[args.router], route_map,
                args.compatibility_threshold, args.compatibility_slack,
                args.lock_shop_count,
            )
            start = time.perf_counter()
            first = rollout(initial, events)
            jax.block_until_ready(first[0])
            first_seconds = time.perf_counter() - start
            start = time.perf_counter()
            money, done, carry = jax.device_get(rollout(initial, events))
            steady = time.perf_counter() - start
            opponent = 1 - h_player
            hcash, ocash = money[:, h_player], money[:, opponent]
            row = {
                "opponent": name,
                "hasegawa_seat": h_player,
                "games": args.batch,
                "win_rate": float(np.mean(hcash > ocash)),
                "hasegawa_cash_mean": float(np.mean(hcash)),
                "opponent_cash_mean": float(np.mean(ocash)),
                "mean_margin": float(np.mean(hcash - ocash)),
                "all_done": bool(np.all(done)),
                "route_switch_total": int(np.sum(carry.route_switch_total)),
                "hard_counter_total": int(np.sum(carry.hard_counter_total)),
                "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
                "resync_total": int(np.sum(carry.resync_total)),
                "compile_and_first_seconds": first_seconds,
                "steady_seconds": steady,
                "transitions_per_second": args.batch * 719 / steady,
                "cash": hcash.astype(int).tolist(),
                "opponent_cash": ocash.astype(int).tolist(),
                "final_route": carry.branch_id.astype(int).tolist(),
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"cash", "opponent_cash", "final_route"}}), flush=True)
    payload = {
        "schema": "kaggriculture.hasegawa_jax_v3.exact5_arena",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "router": args.router,
        "route_map": None if args.route_map is None else str(args.route_map),
        "compatibility_threshold": args.compatibility_threshold,
        "compatibility_slack": args.compatibility_slack,
        "lock_shop_count": args.lock_shop_count,
        "results": rows,
        "status": "PASS" if rows and all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
