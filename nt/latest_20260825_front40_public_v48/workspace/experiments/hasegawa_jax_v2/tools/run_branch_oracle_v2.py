"""Counterfactual branch audit for Hasegawa V2.

For every seed this tool runs the current first-shop router plus all nine frozen
Hasegawa programs against the same opponent and event stream.  It measures
whether better branch selection alone can reach the requested win-rate ceiling.
"""

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


NUM_BRANCHES = 9
NUM_VARIANTS = 1 + NUM_BRANCHES  # current router, then forced branch 0..8


def make_rollout(hbank, exact_bank, runtime, tables, kind, mode, route_id, h_player):
    opponent = 1 - h_player

    @jax.jit
    def rollout(initial, events, forced_branch, force_enabled):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), route_id, dtype=jnp.int32)

        def body(value, _):
            states, hcarry, opponent_carry = value
            # Counterfactual branches may only diverge after the first shop is
            # actually public.  Locking them at reset would leak future random
            # information and overstate the deployable route-selection ceiling.
            activate_forced = force_enabled & (states.town_count > 0)
            hcarry = hcarry._replace(
                branch_id=jnp.where(activate_forced, forced_branch, hcarry.branch_id).astype(jnp.int8),
                branch_locked=hcarry.branch_locked | activate_forced,
                branch_lock_step=jnp.where(
                    activate_forced & (~hcarry.branch_locked), states.step, hcarry.branch_lock_step
                ).astype(jnp.int16),
            )
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


def summarize(name, seat, money, done, carry, seeds):
    n = len(seeds)
    money = money.reshape(n, NUM_VARIANTS, 2)
    done = done.reshape(n, NUM_VARIANTS)
    invalid = carry.invalid_intent_total.reshape(n, NUM_VARIANTS)
    resync = carry.resync_total.reshape(n, NUM_VARIANTS)
    hard = carry.hard_counter_total.reshape(n, NUM_VARIANTS)
    player, opponent = seat, 1 - seat
    hcash = money[:, :, player]
    ocash = money[:, :, opponent]
    margin = hcash - ocash
    best = np.argmax(margin[:, 1:], axis=1) + 1
    best_margin = np.take_along_axis(margin, best[:, None], axis=1)[:, 0]
    best_cash = np.take_along_axis(hcash, best[:, None], axis=1)[:, 0]
    current_margin = margin[:, 0]
    route_rows = []
    for variant in range(NUM_VARIANTS):
        route_rows.append({
            "variant": "current" if variant == 0 else f"branch_{variant - 1}",
            "win_rate": float(np.mean(margin[:, variant] > 0)),
            "cash_mean": float(np.mean(hcash[:, variant])),
            "mean_margin": float(np.mean(margin[:, variant])),
            "hard_total": int(np.sum(hard[:, variant])),
            "invalid_total": int(np.sum(invalid[:, variant])),
            "resync_total": int(np.sum(resync[:, variant])),
        })
    row = {
        "opponent": name,
        "hasegawa_seat": seat,
        "games": n,
        "all_done": bool(np.all(done)),
        "current_win_rate": float(np.mean(current_margin > 0)),
        "current_mean_margin": float(np.mean(current_margin)),
        "oracle_win_rate": float(np.mean(best_margin > 0)),
        "oracle_cash_mean": float(np.mean(best_cash)),
        "oracle_mean_margin": float(np.mean(best_margin)),
        "oracle_branch_counts": np.bincount(best - 1, minlength=NUM_BRANCHES).tolist(),
        "routes": route_rows,
    }
    detail = {
        "seeds": np.asarray(seeds, dtype=np.int32),
        "hcash": hcash.astype(np.int32),
        "ocash": ocash.astype(np.int32),
        "margin": margin.astype(np.int32),
        "invalid": invalid.astype(np.int32),
        "resync": resync.astype(np.int32),
        "hard": hard.astype(np.int32),
        "oracle_variant": best.astype(np.int8),
    }
    return row, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--exact-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--opponents", default="all5")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=153001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    requested = list(TARGETS) if args.opponents == "all5" else [
        x.strip() for x in args.opponents.split(",") if x.strip()
    ]
    hbank = load_hasegawa_trace_bank_v2(args.hasegawa_bank)
    exact_bank = load_bank(args.exact_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    base_seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    expanded_seeds = np.repeat(base_seeds, NUM_VARIANTS)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(expanded_seeds))
    forced_template = np.concatenate(([-1], np.arange(NUM_BRANCHES, dtype=np.int8)))
    forced_branch = jnp.asarray(np.tile(forced_template, args.batch))
    force_enabled = forced_branch >= 0
    forced_branch = jnp.maximum(forced_branch, 0)
    rows, details = [], {}
    for name in requested:
        kind, mode, route_id = TARGETS[name]
        for seat in (0, 1):
            rollout = make_rollout(hbank, exact_bank, runtime, tables, kind, mode, route_id, seat)
            start = time.perf_counter()
            result = rollout(initial, events, forced_branch, force_enabled)
            jax.block_until_ready(result[0])
            elapsed = time.perf_counter() - start
            money, done, carry = jax.device_get(result)
            row, detail = summarize(name, seat, money, done, carry, base_seeds)
            row["compile_and_run_seconds"] = elapsed
            rows.append(row)
            details[f"{name}_seat{seat}"] = detail
            print(json.dumps({k: v for k, v in row.items() if k != "routes"}), flush=True)
    payload = {
        "schema": "kaggriculture.hasegawa_jax_v2.branch_oracle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "games_per_seat": args.batch,
        "variants_per_seed": NUM_VARIANTS,
        "results": rows,
        "status": "PASS" if rows and all(r["all_done"] for r in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **{
        f"{key}__{field}": value
        for key, detail in details.items()
        for field, value in detail.items()
    })
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
