#!/usr/bin/env python3
"""Audit public step-1 fingerprints of every frozen old-bank opening."""

from __future__ import annotations

import json
import os
import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", choices=("old", "latest6"), default="old")
    parser.add_argument(
        "--episode",
        type=Path,
        help="Repeat this Replay rival's first request across the selected bank batch.",
    )
    parser.add_argument("--own-team", default="QQ Farming")
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
    opponent_bank = (
        old_bank
        if args.bank == "old"
        else load_bank(
            ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
        )
    )
    receipt_path = ROOT / (
        "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
        if args.bank == "old"
        else "experiments/expert_business_agent_v2/receipts/latest_public6_20260822_route_bank_v1.json"
    )
    receipt = json.loads(
        receipt_path.read_text(
            encoding="utf-8"
        )
    )
    skeletons = receipt.get("skeletons") or receipt.get("routes") or []
    names = {
        int(row.get("skeleton_id", row.get("route_id", index))):
        f'{row.get("opponent", row.get("name", "UNKNOWN"))}::{row.get("route", row.get("source_route", "UNKNOWN"))}'
        for index, row in enumerate(skeletons)
    }
    batch = int(opponent_bank.unit_op.shape[0])
    episode_id = None
    if args.episode is None:
        seeds = np.full((batch,), 20260823, dtype=np.int32)
        old_action = Action(*(value[:, 0] for value in opponent_bank))
    else:
        replay = json.loads(args.episode.read_text(encoding="utf-8"))
        team_names = list((replay.get("info") or {}).get("TeamNames") or [])
        if args.own_team not in team_names or len(team_names) != 2:
            raise ValueError(f"cannot identify own seat from {team_names}")
        recorded_rival = 1 - team_names.index(args.own_team)
        raw = replay["steps"][1][recorded_rival].get("action") or {}
        encoded = encode_actions([raw, raw])
        old_action = Action(
            *(jnp.repeat(value[0:1], batch, axis=0) for value in encoded)
        )
        seeds = np.full(
            (batch,), int((replay.get("info") or {})["seed"]), dtype=np.int32
        )
        episode_id = int((replay.get("info") or {})["EpisodeId"])
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = jax.jit(rr.make_simulator_step(tables))
    rows = []
    for candidate_seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(seeds))
        carry = initialize_fusion_champion_moon_market_carry_v1(batch)
        candidate_action, _ = jax.jit(fc16_moon_h4_player_action_v1, static_argnums=(6,))(
            states, tables, latest, old_bank, runtime, carry, candidate_seat
        )
        states = (
            simulator(states, candidate_action, old_action, events)
            if candidate_seat == 0
            else simulator(states, old_action, candidate_action, events)
        )
        jax.block_until_ready(states.money)
        state = jax.device_get(states)
        rival = 1 - candidate_seat
        for route in range(batch):
            rows.append(
                {
                    "candidate_seat": candidate_seat,
                    "route": route,
                    "name": names.get(route, "UNKNOWN"),
                    "money_own": int(state.money[route, candidate_seat]),
                    "money_rival": int(state.money[route, rival]),
                    "active_units_own": int(np.sum(state.unit_active[route, candidate_seat])),
                    "active_units_rival": int(np.sum(state.unit_active[route, rival])),
                    "pastures_rival": int(np.sum(np.asarray(state.tile_kind[route, rival]) == 2)),
                    "coops_rival": int(np.sum(np.asarray(state.tile_kind[route, rival]) == 3)),
                }
            )
    target = [
        row
        for row in rows
        if row["money_own"] == 105 and row["money_rival"] == 250
    ]
    suffix = args.bank if episode_id is None else f"episode{episode_id}_both_seats"
    output = ROOT / f"experiments/fusion_champion_v1/receipts/step1_{suffix}_opening_fingerprints_v1.json"
    output.write_text(
        json.dumps(
            {
                "schema": "kaggriculture.step1-oldbank-opening-fingerprints.v1",
                "status": "PASS",
                "backend": jax.default_backend(),
                "bank": args.bank,
                "episode_id": episode_id,
                "rows": rows,
                "target_money_105_250": target,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "matches": target, "output": str(output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
