"""GPU double-seat gate for a Top-40 trace-policy candidate against FC24B."""

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
    ROOT / "experiments/front40_fusion_v1/src",
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_FIRST_SHOP_MAP  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def make_rollout(hbank, latest_bank, old_bank, runtime, tables, route_map, trace_player):
    fc_player = 1 - trace_player

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]

        def body(value, _):
            states, trace_carry, fc_carry = value
            fc_action, fc_carry = fc24_terminal_crop_salvage_player_action_v1(
                states,
                tables,
                latest_bank,
                old_bank,
                runtime,
                fc_carry,
                fc_player,
            )
            states, trace_carry, _, _ = hasegawa_step_with_external_v3(
                states,
                trace_carry,
                hbank,
                fc_action,
                trace_player,
                events,
                tables,
                ROUTER_FIRST_SHOP_MAP,
                route_map,
                15000,
                5000,
                2,
            )
            return (states, trace_carry, fc_carry), None

        trace_carry = initialize_hasegawa_carry_v3(batch, hbank.bootstrap_route_id)
        fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(batch)
        (terminal, trace_carry, _), _ = jax.lax.scan(
            body, (initial, trace_carry, fc_carry), None, length=719
        )
        return terminal.money, terminal.done, trace_carry

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--route-map", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=161001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    hbank = load_hasegawa_trace_bank_v3(args.trace_bank)
    mapping = json.loads(args.route_map.read_text(encoding="utf-8"))
    route_map = jnp.asarray(mapping["route_ids"], dtype=jnp.int16)
    if route_map.shape != (8,):
        raise ValueError(f"route map must be shape (8,), got {route_map.shape}")
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    rows = []
    for trace_player in (0, 1):
        rollout = make_rollout(
            hbank, latest_bank, old_bank, runtime, tables, route_map, trace_player
        )
        start = time.perf_counter()
        first = rollout(initial, events)
        jax.block_until_ready(first[0])
        compile_and_first = time.perf_counter() - start
        start = time.perf_counter()
        money, done, carry = jax.device_get(rollout(initial, events))
        steady = time.perf_counter() - start
        fc_player = 1 - trace_player
        candidate_cash = money[:, trace_player]
        fc_cash = money[:, fc_player]
        wins = candidate_cash > fc_cash
        row = {
            "candidate": args.candidate_name,
            "candidate_seat": trace_player,
            "games": args.batch,
            "wins": int(np.sum(wins)),
            "win_rate": float(np.mean(wins)),
            "candidate_cash_mean": float(np.mean(candidate_cash)),
            "fc24b_cash_mean": float(np.mean(fc_cash)),
            "mean_margin": float(np.mean(candidate_cash - fc_cash)),
            "all_done": bool(np.all(done)),
            "route_switch_total": int(np.sum(carry.route_switch_total)),
            "hard_counter_total": int(np.sum(carry.hard_counter_total)),
            "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
            "resync_total": int(np.sum(carry.resync_total)),
            "compile_and_first_seconds": compile_and_first,
            "steady_seconds": steady,
            "transitions_per_second": args.batch * 719 / steady,
            "candidate_cash": candidate_cash.astype(int).tolist(),
            "fc24b_cash": fc_cash.astype(int).tolist(),
            "final_route": carry.branch_id.astype(int).tolist(),
        }
        rows.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in {"candidate_cash", "fc24b_cash", "final_route"}},
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_games = sum(row["games"] for row in rows)
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-agent-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "candidate": args.candidate_name,
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "aggregate": {
            "games": total_games,
            "wins": sum(row["wins"] for row in rows),
            "win_rate": sum(row["wins"] for row in rows) / total_games,
            "mean_margin": sum(row["mean_margin"] * row["games"] for row in rows) / total_games,
            "invalid_intent_per_game": sum(row["invalid_intent_total"] for row in rows) / total_games,
        },
        "results": rows,
        "status": "PASS" if all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
