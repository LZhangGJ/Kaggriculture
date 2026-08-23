#!/usr/bin/env python3
"""Batched screen of no-yarn cow/sheep investment caps."""

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
for path in (ROOT / "experiments/fusion_champion_v1/src", ROOT / "experiments/expert_business_agent_v2/tools"):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import initialize_x562_route_rule_carry_v1, x562_stable_route_animal_cap_player_action_v1  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


VARIANTS = (
    ("baseline_10c4s", 10, 4),
    ("6c4s", 6, 4),
    ("7c4s", 7, 4),
    ("8c4s", 8, 4),
    ("9c4s", 9, 4),
    ("12c4s", 12, 4),
    ("8c2s", 8, 2),
    ("8c3s", 8, 3),
    ("10c2s", 10, 2),
    ("10c3s", 10, 3),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponent", default="gold_proxy_rank14_recursion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(old_bank_path),
        "latest_bank": load_bank(latest_bank_path),
        "runtime": load_high_potential_runtime_tables_v1(runtime_path),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    if args.opponent not in names:
        raise ValueError("unknown opponent")
    opponent_id = names.index(args.opponent)
    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded_seeds = np.tile(base_seeds, len(VARIANTS))
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    cow_cap = jnp.repeat(jnp.asarray([row[1] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    sheep_cap = jnp.repeat(jnp.asarray([row[2] for row in VARIANTS], dtype=jnp.int16), args.seeds)
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return x562_stable_route_animal_cap_player_action_v1(
                states, resources["runtime"], resources["latest_bank"], carry, cow_cap, sheep_cap, player
            )
        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    batch = len(expanded_seeds)
    orientation_money = []
    for candidate_seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
        candidate_carry = initialize_x562_route_rule_carry_v1(batch)
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(candidate_seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - candidate_seat, **resources)
        for _ in range(719):
            candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = simulator(states, candidate_action, opponent_action, events) if candidate_seat == 0 else simulator(states, opponent_action, candidate_action, events)
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        if int(np.sum(np.asarray(terminal.hand_cap_hits))) or int(np.sum(np.asarray(terminal.market_loop_cap_hits))) or int(np.sum(np.asarray(terminal.price_lut_oob))):
            raise AssertionError("simulator safety counter hit")
        orientation_money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(VARIANTS), args.seeds, 2))

    first, second = orientation_money
    rows, all_margins = [], []
    for index, (name, cows, sheep) in enumerate(VARIANTS):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival
        all_margins.append(margins)
        rows.append({
            "variant": name, "cow_cap": cows, "sheep_cap": sheep,
            "games": int(margins.size), "wins": int(np.sum(margins > 0)), "ties": int(np.sum(margins == 0)), "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)), "mean_opponent_cash": float(np.mean(rival)), "mean_margin": float(np.mean(margins)),
            "per_game": [{"seed": int(base_seeds[game % args.seeds]), "candidate_seat": 0 if game < args.seeds else 1, "margin": int(margin)} for game, margin in enumerate(margins.tolist())],
        })
    matrix = np.stack(all_margins, axis=0)
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.x562_animal_cap_screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "backend": jax.default_backend(), "device": str(jax.devices()[0]), "official_package_version": "1.32.7",
        "opponent": args.opponent, "seed_start": args.seed_start, "seed_count": args.seeds, "seat_protocol": "same seeds with seats swapped",
        "base_route_rule": "step168: any visible YARN -> route7, else route12; counters and preempt disabled; caps only on route12",
        "rows": rows,
        "oracle": {"wins": int(np.sum(oracle > 0)), "ties": int(np.sum(oracle == 0)), "losses": int(np.sum(oracle < 0)), "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)), "mean_margin": float(np.mean(oracle))},
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": [{k: v for k, v in row.items() if k != "per_game"} for row in rows], "oracle": payload["oracle"], "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
