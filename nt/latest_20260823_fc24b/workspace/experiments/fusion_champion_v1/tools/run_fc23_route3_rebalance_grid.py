#!/usr/bin/env python3
"""Vectorized FC22 first-YARN route-rebalance counterfactual screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc23_route3_rebalance_parameter_player_action_v1,
    initialize_fusion_champion_feed_value_carry_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


# name, enabled, switch step, target route
# route 0=10C4S milk-support; route 1=8C6S default; route 2=6C8S three-YARN.
VARIANTS = (
    ("source_fc22", False, 32767, 3),
    ("r3_to_r0_s72", True, 72, 0),
    ("r3_to_r0_s120", True, 120, 0),
    ("r3_to_r0_s168", True, 168, 0),
    ("r3_to_r0_s216", True, 216, 0),
    ("r3_to_r1_s72", True, 72, 1),
    ("r3_to_r1_s120", True, 120, 1),
    ("r3_to_r1_s168", True, 168, 1),
    ("r3_to_r1_s216", True, 216, 1),
    ("r3_to_r2_s72", True, 72, 2),
    ("r3_to_r2_s120", True, 120, 2),
    ("r3_to_r2_s168", True, 168, 2),
    ("r3_to_r2_s216", True, 216, 2),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--opponents", default="local_prt_v6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

    opponents = [value.strip() for value in args.opponents.split(",") if value.strip()]
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(seeds, len(VARIANTS))
    batch = int(expanded.size)
    repeat = lambda values, dtype=None: jnp.repeat(
        jnp.asarray(values, dtype=dtype), args.seeds
    )
    enabled = repeat([row[1] for row in VARIANTS], jnp.bool_)
    switch_step = repeat([row[2] for row in VARIANTS], jnp.int16)
    target_route = repeat([row[3] for row in VARIANTS], jnp.int8)

    old_receipt = json.loads(
        (
            ROOT
            / "experiments/expert_business_agent_v2/receipts/"
            "jax_full37_mixed_exact_proxy_bank_v1.json"
        ).read_text(encoding="utf-8")
    )
    resources = {
        "old_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "jax_full37_mixed_exact_proxy_bank_v1.npz"
        ),
        "latest_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "latest_public8_route_bank_v1.npz"
        ),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    ids = {name: index for index, name in enumerate(names)}
    if any(name not in ids for name in opponents):
        raise ValueError("unknown opponent")
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc23_route3_rebalance_parameter_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["old_bank"],
                resources["runtime"],
                carry,
                enabled,
                switch_step,
                target_route,
                player,
            )

        return policy

    policies = (make_candidate_policy(0), make_candidate_policy(1))
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    rows = []
    started = perf_counter()
    for opponent_name in opponents:
        money = []
        opponent_id = ids[opponent_name]
        for seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded))
            candidate_carry = initialize_fusion_champion_feed_value_carry_v1(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - seat, **resources
            )
            for _ in range(719):
                action, candidate_carry = policies[seat](states, candidate_carry)
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                states = (
                    simulator(states, action, opponent_action, events)
                    if seat == 0
                    else simulator(states, opponent_action, action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            if any(
                int(np.sum(np.asarray(getattr(terminal, name))))
                for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            ):
                raise AssertionError("simulator safety counter hit")
            money.append(
                np.asarray(terminal.money, dtype=np.int64).reshape(
                    len(VARIANTS), args.seeds, 2
                )
            )

        for index, variant in enumerate(VARIANTS):
            own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
            rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
            margin = own - rival
            row = {
                "opponent": opponent_name,
                "variant": variant[0],
                "enabled": bool(variant[1]),
                "switch_step": int(variant[2]),
                "target_route": int(variant[3]),
                "games": int(margin.size),
                "wins": int(np.sum(margin > 0)),
                "ties": int(np.sum(margin == 0)),
                "losses": int(np.sum(margin < 0)),
                "score_rate": float(
                    np.mean(margin > 0) + 0.5 * np.mean(margin == 0)
                ),
                "mean_candidate_cash": float(np.mean(own)),
                "mean_opponent_cash": float(np.mean(rival)),
                "mean_margin": float(np.mean(margin)),
                "per_game": [
                    {
                        "seed": int(seeds[i % args.seeds]),
                        "candidate_seat": 0 if i < args.seeds else 1,
                        "candidate_cash": int(own[i]),
                        "opponent_cash": int(rival[i]),
                        "margin": int(margin[i]),
                        "result": (
                            "win" if margin[i] > 0 else "tie" if margin[i] == 0 else "loss"
                        ),
                    }
                    for i in range(margin.size)
                ],
            }
            rows.append(row)
            print(
                json.dumps({key: value for key, value in row.items() if key != "per_game"}),
                flush=True,
            )

    payload = {
        "schema": "kaggriculture.fc23.route3_rebalance_grid.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "truth_boundary": (
            "counterfactual screen keyed only by the route selected from public town shops; "
            "no opponent identity, seed, future event, or private rival state"
        ),
        "seed_start": int(seeds[0]),
        "seed_count": int(seeds.size),
        "games_per_variant_opponent": int(seeds.size * 2),
        "variants": [
            {
                "name": row[0],
                "enabled": row[1],
                "switch_step": row[2],
                "target_route": row[3],
            }
            for row in VARIANTS
        ],
        "opponents": opponents,
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
