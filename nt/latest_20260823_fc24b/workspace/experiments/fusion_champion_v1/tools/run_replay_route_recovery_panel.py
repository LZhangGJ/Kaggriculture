#!/usr/bin/env python3
"""Strict-JAX panel for a Replay route with Hasegawa V2 deviation recovery."""

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
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from hasegawa_jax_v2 import (  # noqa: E402
    hasegawa_step_with_external_v2,
    initialize_hasegawa_carry_v2,
    load_hasegawa_trace_bank_v2,
)
from kaggriculture_jax.constants import PRODUCTS, SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


DEFAULT_OPPONENTS = (
    "rayk_k320_adaptive_rank1",
    "gold_proxy_rank12_ai_b2b67_saas",
    "local_prt_v6",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated explicit seeds; overrides --seed-start/--seeds.",
    )
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    old_receipt = json.loads((
        ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    ).read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"),
        "latest_bank": load_bank(ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    bank = load_hasegawa_trace_bank_v2(args.bank.resolve())
    names = [row["name"] for row in rr.ROSTER]
    requested = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not requested or any(name not in names for name in requested):
        raise ValueError("unknown or empty opponent list")
    if args.seed_list:
        seeds = np.asarray(
            [int(value.strip()) for value in args.seed_list.split(",") if value.strip()],
            dtype=np.int32,
        )
    elif args.seed_start is not None:
        seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    else:
        raise ValueError("provide --seed-list or --seed-start")
    if seeds.size < 1 or len(set(seeds.tolist())) != int(seeds.size):
        raise ValueError("seed list must be non-empty and unique")
    batch = int(seeds.size)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in requested:
        opponent_id = names.index(opponent_name)
        orientation = []
        carry_totals = []
        for candidate_seat in (0, 1):
            states = initial
            candidate_carry = initialize_hasegawa_carry_v2(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )

            @jax.jit
            def combined_step(current, route_carry, rival_carry):
                opponent_action, rival_carry = opponent_policy(current, rival_carry)
                current, route_carry, _, _ = hasegawa_step_with_external_v2(
                    current,
                    route_carry,
                    bank,
                    opponent_action,
                    candidate_seat,
                    events,
                    resources["tables"],
                )
                return current, route_carry, rival_carry

            for _ in range(719):
                states, candidate_carry, opponent_carry = combined_step(
                    states, candidate_carry, opponent_carry
                )
            jax.block_until_ready(states.money)
            terminal, final_carry = jax.device_get((states, candidate_carry))
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            own = np.asarray(terminal.money[:, candidate_seat], dtype=np.int64)
            rival = np.asarray(terminal.money[:, 1 - candidate_seat], dtype=np.int64)
            orientation.append(
                {
                    "own": own,
                    "rival": rival,
                    "town_count": np.asarray(terminal.town_count, dtype=np.int16),
                    "town_shops": np.asarray(terminal.town_shops, dtype=np.int16),
                    "market_price": np.asarray(terminal.market_price, dtype=np.int64),
                }
            )
            carry_totals.append(
                {
                    "invalid": int(np.sum(final_carry.invalid_intent_total)),
                    "resync": int(np.sum(final_carry.resync_total)),
                    "trim": int(np.sum(final_carry.market_trim_total)),
                    "hard": int(np.sum(final_carry.hard_counter_total)),
                }
            )

        own = np.concatenate((orientation[0]["own"], orientation[1]["own"]))
        rival = np.concatenate((orientation[0]["rival"], orientation[1]["rival"]))
        margins = own - rival
        row = {
            "opponent": opponent_name,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "seat0_score_rate": float(np.mean(orientation[0]["own"] > orientation[0]["rival"])),
            "seat1_score_rate": float(np.mean(orientation[1]["own"] > orientation[1]["rival"])),
            "recovery_totals_by_seat": carry_totals,
            "per_game": [
                {
                    "seed": int(seeds[index % batch]),
                    "candidate_seat": 0 if index < batch else 1,
                    "candidate_cash": int(own[index]),
                    "opponent_cash": int(rival[index]),
                    "margin": int(margins[index]),
                    "town_shops": [
                        SHOP_NAMES[int(value)]
                        for value in orientation[0 if index < batch else 1]["town_shops"][
                            index % batch,
                            : int(
                                orientation[0 if index < batch else 1]["town_count"][index % batch]
                            ),
                        ]
                    ],
                    "terminal_market_prices": {
                        name: int(value)
                        for name, value in zip(
                            PRODUCTS,
                            orientation[0 if index < batch else 1]["market_price"][index % batch],
                            strict=True,
                        )
                    },
                }
                for index in range(margins.size)
            ],
        }
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "per_game"}), flush=True)

    payload = {
        "schema": "kaggriculture.fusion_champion.replay-route-recovery-panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(
            all(side["hard"] == 0 for side in row["recovery_totals_by_seat"])
            for row in rows
        ) else "FAIL",
        "truth_boundary": "Replay route plus generic recovery probe; still not a deployable Agent.",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "bank": str(args.bank.resolve()),
        "seed_start": args.seed_start,
        "seed_count": batch,
        "seed_values": seeds.astype(int).tolist(),
        "seat_protocol": "same independent events with seats swapped",
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
