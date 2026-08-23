#!/usr/bin/env python3
"""One-step FC15/Boatlee-V21 opening probe without a full-season rollout."""

from __future__ import annotations

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
    fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
)


def action_row(action, index: int) -> dict:
    host = jax.device_get(action)
    unit_count = int(np.asarray(host.unit_count)[index])
    market_count = int(np.asarray(host.market_count)[index])
    return {
        "unit_count": unit_count,
        "unit_op": np.asarray(host.unit_op)[index, :unit_count].astype(int).tolist(),
        "unit_item": np.asarray(host.unit_item)[index, :unit_count].astype(int).tolist(),
        "unit_amount": np.asarray(host.unit_amount)[index, :unit_count].astype(int).tolist(),
        "market_count": market_count,
        "market_op": np.asarray(host.market_op)[index, :market_count].astype(int).tolist(),
        "market_item": np.asarray(host.market_item)[index, :market_count].astype(int).tolist(),
        "market_amount": np.asarray(host.market_amount)[index, :market_count].astype(int).tolist(),
    }


def main() -> int:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    tables = load_tables()
    old_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest8_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    latest6_bank = load_bank(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    )
    runtime = load_high_potential_runtime_tables_v1(
        ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    seeds = np.arange(842001, 842009, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = jax.jit(rr.make_simulator_step(tables))

    for candidate_seat in (0, 1):
        rival = 1 - candidate_seat
        states = jax.vmap(reset)(jnp.asarray(seeds))
        candidate_carry = initialize_fusion_champion_carry_v3(len(seeds))
        opponent_carry = initialize_boatlee_v21_carry_v1(len(seeds))

        @jax.jit
        def candidate_policy(state, carry):
            return fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
                state, tables, latest8_bank, old_bank, runtime, carry, candidate_seat
            )

        @jax.jit
        def opponent_policy(state, carry):
            return boatlee_v21_player_action_v1(
                state, runtime, latest6_bank, carry, rival
            )

        candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
        opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
        states = (
            simulator(states, candidate_action, opponent_action, events)
            if candidate_seat == 0
            else simulator(states, opponent_action, candidate_action, events)
        )
        states = jax.device_get(states)
        print(
            {
                "candidate_seat": candidate_seat,
                "candidate_action0": action_row(candidate_action, 0),
                "opponent_action0": action_row(opponent_action, 0),
                "candidate_money_step1": np.asarray(states.money)[:, candidate_seat].astype(int).tolist(),
                "opponent_money_step1": np.asarray(states.money)[:, rival].astype(int).tolist(),
                "candidate_hires_step1": np.asarray(states.hires_today)[:, candidate_seat].astype(int).tolist(),
                "opponent_hires_step1": np.asarray(states.hires_today)[:, rival].astype(int).tolist(),
                "b21_route_before_step1_decision": np.asarray(jax.device_get(opponent_carry.route)).astype(int).tolist(),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
