#!/usr/bin/env python3
"""Diagnose how the wheat-8 opening changes Rank14's step-192 route.

This is an offline audit only.  The opponent label and selected skeleton are
never exposed to the candidate policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc16_moon_h4_player_action_v1,
    fc19_moon_h4_wheat8_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.constants import ANIMALS, CROPS, TileKind  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    extract_base48,
    load_bank,
)
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def _counts(board: jax.Array, size: int) -> np.ndarray:
    host = np.asarray(jax.device_get(board))
    return np.stack(
        [np.sum(host == value, axis=(1, 2)) for value in range(size)], axis=1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=594001)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    tables = load_tables()
    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    old_receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(encoding="utf-8")
    )
    router = build_router_arrays(old_receipt)
    resources = {
        "old_bank": old_bank,
        "latest_bank": latest_bank,
        "runtime": runtime,
        "tables": tables,
        "router": router,
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    opponent_name = "gold_proxy_rank14_recursion"
    opponent_id = names.index(opponent_name)
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(tables)
    batch = int(seeds.size)

    rows = []
    for variant in ("fc16", "fc19_wheat8"):
        for player in (0, 1):
            rival = 1 - player

            @jax.jit
            def candidate_policy(states, carry):
                if variant == "fc16":
                    return fc16_moon_h4_player_action_v1(
                        states, tables, latest_bank, old_bank, runtime, carry, player
                    )
                return fc19_moon_h4_wheat8_player_action_v1(
                    states, tables, latest_bank, old_bank, runtime, carry, player
                )

            opponent_policy = rr.make_agent_policy(
                opponent_id, rival, **resources
            )
            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = initialize_fusion_champion_moon_market_carry_v1(batch)
            opponent_carry = rr.initialize_agent_carry(opponent_id, batch, router)
            decision = None
            selected_route = None
            for _ in range(719):
                if int(jax.device_get(states.step[0])) == 192:
                    decision = {
                        "features48": np.asarray(jax.device_get(extract_base48(states, rival))),
                        "money": np.asarray(jax.device_get(states.money)),
                        "active_units": np.asarray(jax.device_get(jnp.sum(states.unit_active, axis=2))),
                        "unlocked": np.asarray(
                            jax.device_get(
                                jnp.sum(states.tile_kind != int(TileKind.LOCKED), axis=(2, 3))
                            )
                        ),
                        "candidate_crops": _counts(states.tile_crop[:, player], len(CROPS)),
                        "candidate_animals": _counts(states.tile_animal[:, player], len(ANIMALS)),
                        "opponent_crops": _counts(states.tile_crop[:, rival], len(CROPS)),
                        "opponent_animals": _counts(states.tile_animal[:, rival], len(ANIMALS)),
                    }
                candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
                opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
                if int(jax.device_get(states.step[0])) == 192:
                    selected_route = np.asarray(jax.device_get(opponent_carry.skeleton_id))
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if player == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal_money = np.asarray(jax.device_get(states.money), dtype=np.int64)
            if decision is None or selected_route is None:
                raise AssertionError("step-192 diagnostic was not captured")
            own = terminal_money[:, player]
            opp = terminal_money[:, rival]
            margins = own - opp
            route_rows = []
            for route in np.unique(selected_route):
                mask = selected_route == route
                route_rows.append({
                    "route_id": int(route),
                    "games": int(np.sum(mask)),
                    "wins": int(np.sum(margins[mask] > 0)),
                    "score_rate": float(np.mean(margins[mask] > 0) + 0.5 * np.mean(margins[mask] == 0)),
                    "mean_margin": float(np.mean(margins[mask])),
                    "mean_candidate_step192_money": float(np.mean(decision["money"][mask, player])),
                    "mean_opponent_step192_money": float(np.mean(decision["money"][mask, rival])),
                })
            rows.append({
                "variant": variant,
                "candidate_seat": player,
                "games": batch,
                "wins": int(np.sum(margins > 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_margin": float(np.mean(margins)),
                "route_distribution": route_rows,
                "step192_means": {
                    "candidate_money": float(np.mean(decision["money"][:, player])),
                    "opponent_money": float(np.mean(decision["money"][:, rival])),
                    "candidate_active_units": float(np.mean(decision["active_units"][:, player])),
                    "candidate_unlocked": float(np.mean(decision["unlocked"][:, player])),
                    "candidate_crops": np.mean(decision["candidate_crops"], axis=0).tolist(),
                    "candidate_animals": np.mean(decision["candidate_animals"], axis=0).tolist(),
                    "opponent_crops": np.mean(decision["opponent_crops"], axis=0).tolist(),
                    "opponent_animals": np.mean(decision["opponent_animals"], axis=0).tolist(),
                },
                "per_game": [
                    {
                        "seed": int(seed),
                        "route_id": int(route),
                        "margin": int(margin),
                        "candidate_step192_money": int(money[player]),
                        "opponent_step192_money": int(money[rival]),
                        "features48": features.tolist(),
                    }
                    for seed, route, margin, money, features in zip(
                        seeds.tolist(),
                        selected_route.tolist(),
                        margins.tolist(),
                        decision["money"],
                        decision["features48"],
                        strict=True,
                    )
                ],
            })
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "per_game"}), flush=True)

    payload = {
        "schema": "kaggriculture.fc19-rank14-route-shift-diagnostic.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "visibility": "opponent route and label are offline diagnostics only",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "rows": rows,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
