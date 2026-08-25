"""GPU cross-play among the five official-parity local agents."""

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
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from run_exact5_arena_v2 import TARGETS  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.public_g02_gpu import initialize_public_g02_carry_v1, public_rc5_weed_player_action_v1  # noqa: E402
from strategic_v5.public_v25_gpu import public_v27_player_action_exact_v1  # noqa: E402


def initialize_carry(kind, batch):
    return initialize_high_potential_v20_carry_v1(batch) if kind == "v20" else initialize_public_g02_carry_v1(batch)


def action_for(states, tables, bank, runtime, carry, player, spec):
    kind, mode, route_id = spec
    route_ids = jnp.full((states.step.shape[0],), route_id, dtype=jnp.int32)
    if kind == "v20":
        return high_potential_v20_player_action_v1(states, tables, bank, runtime, carry, player, mode)
    if kind == "v25":
        return public_v27_player_action_exact_v1(states, runtime, bank, route_ids, carry, player)
    return public_rc5_weed_player_action_v1(states, bank, route_ids, carry, player)


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(bank, runtime, tables, candidate_spec, opponent_spec, candidate_seat):
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]

        def body(value, _):
            states, candidate_carry, opponent_carry = value
            candidate_action, candidate_carry = action_for(
                states, tables, bank, runtime, candidate_carry, candidate_seat, candidate_spec
            )
            opponent_action, opponent_carry = action_for(
                states, tables, bank, runtime, opponent_carry, opponent_seat, opponent_spec
            )
            joint = pair(candidate_action, opponent_action) if candidate_seat == 0 else pair(opponent_action, candidate_action)
            states = batched_step_sync(states, joint, events, tables)
            return (states, candidate_carry, opponent_carry), None

        candidate_carry = initialize_carry(candidate_spec[0], batch)
        opponent_carry = initialize_carry(opponent_spec[0], batch)
        (terminal, _, _), _ = jax.lax.scan(
            body, (initial, candidate_carry, opponent_carry), None, length=719
        )
        return terminal.money, terminal.done

    return rollout


def names(value):
    return list(TARGETS) if value == "all5" else [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidates", default="all5")
    parser.add_argument("--opponents", default="all5")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=154001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    bank = load_bank(args.bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    rows = []
    for candidate in names(args.candidates):
        for opponent in names(args.opponents):
            for seat in (0, 1):
                rollout = make_rollout(bank, runtime, tables, TARGETS[candidate], TARGETS[opponent], seat)
                start = time.perf_counter()
                result = rollout(initial, events)
                jax.block_until_ready(result[0])
                elapsed = time.perf_counter() - start
                money, done = jax.device_get(result)
                other = 1 - seat
                own_cash, opponent_cash = money[:, seat], money[:, other]
                row = {
                    "candidate": candidate,
                    "opponent": opponent,
                    "candidate_seat": seat,
                    "games": args.batch,
                    "win_rate": float(np.mean(own_cash > opponent_cash)),
                    "mean_margin": float(np.mean(own_cash - opponent_cash)),
                    "candidate_cash_mean": float(np.mean(own_cash)),
                    "opponent_cash_mean": float(np.mean(opponent_cash)),
                    "all_done": bool(np.all(done)),
                    "compile_and_run_seconds": elapsed,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    payload = {
        "schema": "kaggriculture.exact5_crossplay.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "games_per_seat": args.batch,
        "results": rows,
        "status": "PASS" if rows and all(row["all_done"] for row in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
