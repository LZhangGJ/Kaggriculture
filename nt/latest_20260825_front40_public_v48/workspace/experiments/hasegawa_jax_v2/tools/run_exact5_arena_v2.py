"""Small seat-swapped GPU Arena against the five frozen local public agents."""

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
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v2 import (  # noqa: E402
    hasegawa_step_with_external_v2,
    initialize_hasegawa_carry_v2,
    load_hasegawa_trace_bank_v2,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MODE_BOATLEE, MODE_RAY_K320, MODE_TETSUTANI,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.public_g02_gpu import initialize_public_g02_carry_v1, public_rc5_weed_player_action_v1  # noqa: E402
from strategic_v5.public_v25_gpu import public_v27_player_action_exact_v1  # noqa: E402


TARGETS = {
    "boatlee_v20": ("v20", MODE_BOATLEE, -1),
    "ray_k320": ("v20", MODE_RAY_K320, -1),
    "tetsutani": ("v20", MODE_TETSUTANI, -1),
    "kaito_v27": ("v25", -1, 8),
    "flex_v59": ("flex", -1, 9),
}


def make_rollout(hbank, exact_bank, runtime, tables, kind, mode, route_id, h_player):
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
            states, hcarry, _, _ = hasegawa_step_with_external_v2(
                states, hcarry, hbank, action, h_player, events, tables
            )
            return (states, hcarry, opponent_carry), None

        hcarry = initialize_hasegawa_carry_v2(batch)
        opponent_carry = initialize_high_potential_v20_carry_v1(batch) if kind == "v20" else initialize_public_g02_carry_v1(batch)
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
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=151001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    requested = list(TARGETS) if args.opponents == "all5" else [x.strip() for x in args.opponents.split(",") if x.strip()]
    hbank = load_hasegawa_trace_bank_v2(args.hasegawa_bank)
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
            rollout = make_rollout(hbank, exact_bank, runtime, tables, kind, mode, route_id, h_player)
            start = time.perf_counter()
            first = rollout(initial, events)
            jax.block_until_ready(first[0])
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
            money, done, carry = jax.device_get(rollout(initial, events))
            steady = time.perf_counter() - start
            opponent = 1 - h_player
            hcash, ocash = money[:, h_player], money[:, opponent]
            row = {
                "opponent": name, "hasegawa_seat": h_player, "games": args.batch,
                "win_rate": float(np.mean(hcash > ocash)),
                "hasegawa_cash_mean": float(np.mean(hcash)), "opponent_cash_mean": float(np.mean(ocash)),
                "mean_margin": float(np.mean(hcash - ocash)), "all_done": bool(np.all(done)),
                "hard_counter_total": int(np.sum(carry.hard_counter_total)),
                "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
                "resync_total": int(np.sum(carry.resync_total)),
                "compile_and_first_seconds": compile_seconds, "steady_seconds": steady,
                "transitions_per_second": args.batch * 719 / steady,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "schema": "kaggriculture.hasegawa_jax_v2.exact5_small_arena",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]), "seed_start": args.seed_start,
        "screening_only": True, "results": rows,
        "status": "PASS" if rows and all(r["all_done"] and r["hard_counter_total"] == 0 for r in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
