#!/usr/bin/env python3
"""Replay the first rival request against FC16 in both seat orientations."""

from __future__ import annotations

import argparse
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
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc16_moon_h4_player_action_v1,
    initialize_fusion_champion_moon_market_carry_v1,
)
from kaggriculture_jax.codec import encode_actions  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def select_player(action: Action, player: int) -> Action:
    return Action(*(value[player : player + 1] for value in action))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--own-team", default="QQ Farming")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    tables = load_tables()
    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    simulator = jax.jit(rr.make_simulator_step(tables))
    rows = []
    for episode_path in args.episode:
        replay = json.loads(episode_path.read_text(encoding="utf-8"))
        names = list((replay.get("info") or {}).get("TeamNames") or [])
        if args.own_team not in names or len(names) != 2:
            raise ValueError(f"cannot identify own seat from {names}")
        recorded_own = names.index(args.own_team)
        recorded_rival = 1 - recorded_own
        rival_raw = replay["steps"][1][recorded_rival].get("action") or {}
        encoded = encode_actions([rival_raw, rival_raw])
        seed = int((replay.get("info") or {})["seed"])
        weed, shops = build_events_v1([seed])
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        for candidate_seat in (0, 1):
            state = jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))
            carry = initialize_fusion_champion_moon_market_carry_v1(1)
            candidate_action, _ = jax.jit(
                fc16_moon_h4_player_action_v1, static_argnums=(6,)
            )(state, tables, latest, old_bank, runtime, carry, candidate_seat)
            rival_action = select_player(encoded, candidate_seat)
            state = (
                simulator(state, candidate_action, rival_action, events)
                if candidate_seat == 0
                else simulator(state, rival_action, candidate_action, events)
            )
            jax.block_until_ready(state.money)
            host = jax.device_get(state)
            rival = 1 - candidate_seat
            rows.append(
                {
                    "episode_id": int((replay.get("info") or {})["EpisodeId"]),
                    "candidate_seat": candidate_seat,
                    "money_own": int(host.money[0, candidate_seat]),
                    "money_rival": int(host.money[0, rival]),
                    "active_units_own": int(np.sum(host.unit_active[0, candidate_seat])),
                    "active_units_rival": int(np.sum(host.unit_active[0, rival])),
                    "pastures_rival": int(
                        np.sum(np.asarray(host.tile_kind[0, rival]) == 2)
                    ),
                    "coops_rival": int(
                        np.sum(np.asarray(host.tile_kind[0, rival]) == 3)
                    ),
                    "rival_opening": rival_raw,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "kaggriculture.public-loss-opening-both-seats.v1",
                "status": "PASS",
                "backend": jax.default_backend(),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "rows": rows, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
