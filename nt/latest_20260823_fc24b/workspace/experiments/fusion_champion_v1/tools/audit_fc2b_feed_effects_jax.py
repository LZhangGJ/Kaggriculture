#!/usr/bin/env python3
"""Measure FC2B animal-service intents versus effects on independent JAX games."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import FLAG_CARED, FLAG_FED, UnitOp  # noqa: E402
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
WHEAT_ID = 0


def flag_is_set(flags, bit: int):
    return (flags & jnp.asarray(bit, dtype=jnp.uint8)) != 0


def service_effect_masks(states, action, player: int):
    batch = states.step.shape[0]
    unit_width = action.unit_op.shape[1]
    active = jnp.arange(unit_width)[None, :] < action.unit_count[:, None]
    pos = states.unit_pos[:, player]
    y = jnp.clip(pos[:, :, 0].astype(jnp.int32), 0, 9)
    x = jnp.clip(pos[:, :, 1].astype(jnp.int32), 0, 9)
    batch_index = jnp.arange(batch)[:, None]
    animal = states.tile_animal[batch_index, player, y, x]
    flags = states.tile_flags[batch_index, player, y, x]
    wheat = states.unit_inventory[:, player, :, WHEAT_ID].astype(jnp.int32)

    feed = active & (action.unit_op == int(UnitOp.FEED))
    feed_no_animal = feed & (animal < 0)
    feed_already = feed & (animal >= 0) & flag_is_set(flags, FLAG_FED)
    feed_no_wheat = feed & (animal >= 0) & ~feed_already & (wheat <= 0)
    feed_effective = feed & (animal >= 0) & ~feed_already & (wheat > 0)

    care = active & (action.unit_op == int(UnitOp.CARE))
    care_no_animal = care & (animal < 0)
    care_already = care & (animal >= 0) & flag_is_set(flags, FLAG_CARED)
    care_effective = care & (animal >= 0) & ~care_already
    return {
        "feed_intent": jnp.sum(feed, axis=1).astype(jnp.int32),
        "feed_effective": jnp.sum(feed_effective, axis=1).astype(jnp.int32),
        "feed_no_animal": jnp.sum(feed_no_animal, axis=1).astype(jnp.int32),
        "feed_already": jnp.sum(feed_already, axis=1).astype(jnp.int32),
        "feed_no_wheat": jnp.sum(feed_no_wheat, axis=1).astype(jnp.int32),
        "care_intent": jnp.sum(care, axis=1).astype(jnp.int32),
        "care_effective": jnp.sum(care_effective, axis=1).astype(jnp.int32),
        "care_no_animal": jnp.sum(care_no_animal, axis=1).astype(jnp.int32),
        "care_already": jnp.sum(care_already, axis=1).astype(jnp.int32),
    }


def summarize_metric(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.int64)
    return {
        "total": int(np.sum(values)),
        "mean_per_game": float(np.mean(values)),
        "median_per_game": float(np.median(values)),
        "p95_per_game": float(np.quantile(values, 0.95)),
        "games_positive": int(np.sum(values > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")

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
    names = [row["name"] for row in rr.ROSTER]
    requested = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not requested or any(name not in names for name in requested):
        raise ValueError("unknown or empty opponent list")
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(resources["tables"])
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in requested:
        opponent_id = names.index(opponent_name)
        orientation = []
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = initialize_fusion_champion_carry_v3(args.seeds)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, args.seeds, resources["router"]
            )

            @jax.jit
            def candidate_policy(current, carry):
                return fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
                    current,
                    resources["tables"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    resources["runtime"],
                    carry,
                    candidate_seat,
                )

            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )
            totals = {
                name: jnp.zeros((args.seeds,), dtype=jnp.int32)
                for name in (
                    "feed_intent", "feed_effective", "feed_no_animal",
                    "feed_already", "feed_no_wheat", "care_intent",
                    "care_effective", "care_no_animal", "care_already",
                )
            }
            daily_totals = {
                name: jnp.zeros((args.seeds, 30), dtype=jnp.int32)
                for name in totals
            }

            @jax.jit
            def update_service_totals(
                current, candidate_action, accumulated, daily, day_index
            ):
                step_metrics = service_effect_masks(
                    current, candidate_action, candidate_seat
                )
                accumulated = {
                    name: accumulated[name] + step_metrics[name]
                    for name in accumulated
                }
                daily = {
                    name: daily[name].at[:, day_index].add(step_metrics[name])
                    for name in daily
                }
                return accumulated, daily

            for step_index in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                totals, daily_totals = update_service_totals(
                    states,
                    candidate_action,
                    totals,
                    daily_totals,
                    jnp.asarray(step_index // 24, dtype=jnp.int32),
                )
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if candidate_seat == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            metrics = {name: np.asarray(jax.device_get(value)) for name, value in totals.items()}
            daily_metrics = {
                name: np.asarray(jax.device_get(value))
                for name, value in daily_totals.items()
            }
            own = np.asarray(terminal.money[:, candidate_seat], dtype=np.int64)
            rival = np.asarray(terminal.money[:, 1 - candidate_seat], dtype=np.int64)
            orientation.append(
                {
                    "own": own,
                    "rival": rival,
                    "metrics": metrics,
                    "daily_metrics": daily_metrics,
                }
            )

        own = np.concatenate([value["own"] for value in orientation])
        rival = np.concatenate([value["rival"] for value in orientation])
        margins = own - rival
        metrics = {
            name: np.concatenate([value["metrics"][name] for value in orientation])
            for name in orientation[0]["metrics"]
        }
        daily_metrics = {
            name: np.concatenate(
                [value["daily_metrics"][name] for value in orientation], axis=0
            )
            for name in orientation[0]["daily_metrics"]
        }
        invalid_feed = metrics["feed_intent"] - metrics["feed_effective"]
        invalid_feed_daily = (
            daily_metrics["feed_intent"] - daily_metrics["feed_effective"]
        )
        row = {
            "opponent": opponent_name,
            "games": int(margins.size),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_cash": float(np.mean(own)),
            "mean_margin": float(np.mean(margins)),
            "service": {name: summarize_metric(value) for name, value in metrics.items()},
            "invalid_feed": summarize_metric(invalid_feed),
            "invalid_feed_mean_wins": float(np.mean(invalid_feed[margins > 0])) if np.any(margins > 0) else None,
            "invalid_feed_mean_losses": float(np.mean(invalid_feed[margins < 0])) if np.any(margins < 0) else None,
            "daily": {
                "invalid_feed_mean": np.mean(invalid_feed_daily, axis=0).tolist(),
                "invalid_feed_mean_wins": (
                    np.mean(invalid_feed_daily[margins > 0], axis=0).tolist()
                    if np.any(margins > 0) else None
                ),
                "invalid_feed_mean_losses": (
                    np.mean(invalid_feed_daily[margins < 0], axis=0).tolist()
                    if np.any(margins < 0) else None
                ),
                "feed_no_animal_mean": np.mean(
                    daily_metrics["feed_no_animal"], axis=0
                ).tolist(),
                "feed_already_mean": np.mean(
                    daily_metrics["feed_already"], axis=0
                ).tolist(),
                "feed_no_wheat_mean": np.mean(
                    daily_metrics["feed_no_wheat"], axis=0
                ).tolist(),
            },
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-service-effect-audit.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Pre-action intent/effect audit on official-parity JAX. Correlation with losses "
            "does not prove that replacing an invalid action improves terminal score."
        ),
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same independent events with seats swapped",
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
