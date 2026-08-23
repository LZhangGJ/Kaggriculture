#!/usr/bin/env python3
"""Paired GPU screen for a weed-repair barrier around planned HIRE steps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    k320_clone_aware_weed_mass_hire_guard_player_action_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)


VARIANTS = (
    ("source_bounded8", False, 0, 1, False),
    ("all_actor_h2_min1", True, 2, 1, False),
    ("farmer_h2_min1", True, 2, 1, True),
    ("farmer_h2_min3", True, 2, 3, True),
    ("farmer_h2_min5", True, 2, 5, True),
    ("farmer_h4_min5", True, 4, 5, True),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    old_receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(encoding="utf-8")
    )
    resources = {
        "old_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"),
        "latest_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"),
        "runtime": load_high_potential_runtime_tables_v1(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError(f"unknown opponent: {args.opponent}")
    opponent_id = names.index(args.opponent)

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(VARIANTS))
    enabled = jnp.repeat(jnp.asarray([row[1] for row in VARIANTS], dtype=jnp.bool_), args.seeds)
    horizons = jnp.repeat(jnp.asarray([row[2] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    minimum_hires = jnp.repeat(jnp.asarray([row[3] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    farmer_only = jnp.repeat(jnp.asarray([row[4] for row in VARIANTS], dtype=jnp.bool_), args.seeds)
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return k320_clone_aware_weed_mass_hire_guard_player_action_v1(
                states,
                resources["tables"],
                resources["runtime"],
                resources["latest_bank"],
                carry,
                enabled,
                horizons,
                minimum_hires,
                farmer_only,
                player,
            )

        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    money = []
    repairs = []
    batch = len(expanded)
    for seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded))
        candidate_carry = initialize_high_potential_v20_carry_v1(batch)
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        cumulative_repairs = jnp.zeros((batch,), dtype=jnp.int16)
        for _ in range(719):
            before = candidate_carry.weed_start
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            cumulative_repairs += jnp.sum(candidate_carry.weed_start != before, axis=1).astype(jnp.int16)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        if any(int(np.sum(np.asarray(getattr(terminal, name)))) for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")):
            raise AssertionError("simulator safety counter hit")
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2))
        repairs.append(np.asarray(jax.device_get(cumulative_repairs), dtype=np.int16).reshape(len(VARIANTS), args.seeds))

    rows = []
    margins_all = []
    for index, (name, use_barrier, horizon, minimum_hire_count, only_farmer) in enumerate(VARIANTS):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        repair_count = np.concatenate((repairs[0][index], repairs[1][index]))
        margins = own - rival
        margins_all.append(margins)
        rows.append(
            {
                "variant": name,
                "use_hire_barrier": bool(use_barrier),
                "hire_guard_horizon": int(horizon),
                "minimum_hires": int(minimum_hire_count),
                "farmer_only": bool(only_farmer),
                "games": int(len(margins)),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_candidate_cash": float(np.mean(own)),
                "mean_opponent_cash": float(np.mean(rival)),
                "mean_margin": float(np.mean(margins)),
                "mean_repairs": float(np.mean(repair_count)),
                "per_game": [
                    {
                        "seed": int(base_seeds[i % args.seeds]),
                        "candidate_seat": 0 if i < args.seeds else 1,
                        "candidate_cash": int(own[i]),
                        "opponent_cash": int(rival[i]),
                        "margin": int(margins[i]),
                        "repairs": int(repair_count[i]),
                    }
                    for i in range(len(margins))
                ],
            }
        )

    source = margins_all[0]
    paired = []
    for index, row in enumerate(rows[1:], start=1):
        candidate = margins_all[index]
        paired.append(
            {
                "variant": row["variant"],
                "improved_games": int(np.sum(candidate > source)),
                "unchanged_games": int(np.sum(candidate == source)),
                "regressed_games": int(np.sum(candidate < source)),
                "source_losses_rescued": int(np.sum((source < 0) & (candidate > 0))),
                "source_wins_harmed": int(np.sum((source > 0) & (candidate < 0))),
                "mean_margin_delta": float(np.mean(candidate - source)),
            }
        )

    payload = {
        "schema": "kaggriculture.fusion_champion.k320-weed-hire-barrier-screen.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "paired_delta": paired,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": [{k: v for k, v in row.items() if k != "per_game"} for row in rows], "paired_delta": paired, "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
