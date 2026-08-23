#!/usr/bin/env python3
"""Screen implementable K320 route overrides after a real adaptive prefix.

Unlike ``run_k320_forced_route_screen.py``, this runner does not force a route
from reset.  K320 owns the opening and observes the public state normally.  At
the requested public decision step, the selected internal route is locked for
the remainder of the game.  This makes each arm executable online and avoids
using terminal outcomes to alter the prefix.
"""

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
    k320_clone_aware_preempt_player_action_v1,
)
from run_old_suffix_discovery import extract_decision_features  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MODE_RAY_K320,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)


ROUTES = (
    ("milk_support", 0),
    ("default", 1),
    ("three_yarn", 2),
    ("first_yarn", 3),
    ("two_yarn", 4),
)


def _parse_steps(raw: str) -> list[int]:
    steps = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not steps or any(step < 0 or step > 718 for step in steps):
        raise ValueError("switch steps must be a non-empty subset of [0, 718]")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--switch-steps", required=True)
    parser.add_argument(
        "--routes",
        default="",
        help="Optional comma-separated subset of milk_support,default,three_yarn,first_yarn,two_yarn.",
    )
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument(
        "--candidate-stack",
        choices=("frozen_k320", "clone_aware_preempt"),
        default="frozen_k320",
        help="Production and market stack used after the route decision.",
    )
    parser.add_argument("--save-decision-features", action="store_true")
    parser.add_argument("--feature-step", type=int, default=-1)
    parser.add_argument(
        "--compiled-loop",
        action="store_true",
        help="Run each rollout segment inside one GPU fori_loop instead of 719 Python calls.",
    )
    parser.add_argument(
        "--seed-batches",
        type=int,
        default=1,
        help="Consecutive same-sized event batches to run in one process and one compiled graph.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    switch_step_values = _parse_steps(args.switch_steps)
    feature_step = args.feature_step if args.feature_step >= 0 else switch_step_values[0]
    if args.save_decision_features and feature_step not in switch_step_values:
        raise ValueError("feature step must be one of the screened switch steps")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seed_batches < 1:
        raise ValueError("seed batches must be positive")

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
    if args.opponent not in names:
        raise ValueError(f"unknown opponent: {args.opponent}")
    opponent_id = names.index(args.opponent)

    route_filter = {value.strip() for value in args.routes.split(",") if value.strip()}
    known_routes = {name for name, _ in ROUTES}
    if not route_filter.issubset(known_routes):
        raise ValueError(f"unknown --routes values: {sorted(route_filter - known_routes)}")
    selected_routes = tuple(
        (name, route_id)
        for name, route_id in ROUTES
        if not route_filter or name in route_filter
    )
    variants = [("adaptive_source", -1, -1)]
    if args.include_dynamic:
        variants.append(("dynamic_public_route", -2, -1))
    variants.extend(
        (f"override_{route_name}_step{step}", route_id, step)
        for step in switch_step_values
        for route_name, route_id in selected_routes
    )
    seed_segments = [
        np.arange(
            args.seed_start + segment * args.seeds,
            args.seed_start + (segment + 1) * args.seeds,
            dtype=np.int32,
        )
        for segment in range(args.seed_batches)
    ]
    base_seeds = np.concatenate(seed_segments)
    event_segments = []
    for segment_seeds in seed_segments:
        expanded = np.tile(segment_seeds, len(variants))
        weed, shops = build_events_v1(expanded.tolist())
        event_segments.append(Events(jnp.asarray(weed), jnp.asarray(shops)))
    override_enabled = jnp.repeat(
        jnp.asarray([route_id >= 0 for _, route_id, _ in variants]), args.seeds
    )
    dynamic_enabled = jnp.repeat(
        jnp.asarray([route_id == -2 for _, route_id, _ in variants]), args.seeds
    )
    route_ids = jnp.repeat(
        jnp.asarray([max(route_id, 0) for _, route_id, _ in variants], dtype=jnp.int8),
        args.seeds,
    )
    switch_steps = jnp.repeat(
        jnp.asarray([max(step, 0) for _, _, step in variants], dtype=jnp.int32),
        args.seeds,
    )
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            # Dynamic arm deliberately discards only K320's sticky route lock.
            # _select_route then recomputes the route from the currently public
            # town-shop state on every step; all other K320 carry is preserved.
            carry = carry._replace(
                ray_route_locked=jnp.where(
                    dynamic_enabled,
                    jnp.zeros_like(carry.ray_route_locked),
                    carry.ray_route_locked,
                )
            )
            # Reapply the requested ID from the public decision step onward.
            # This intentionally overrides an earlier source lock while leaving
            # every action before the decision step byte-identical to K320.
            do_override = override_enabled & (states.step.astype(jnp.int32) >= switch_steps)
            carry = carry._replace(
                ray_route_locked=carry.ray_route_locked | do_override,
                ray_route_id=jnp.where(do_override, route_ids, carry.ray_route_id).astype(jnp.int8),
            )
            if args.candidate_stack == "clone_aware_preempt":
                return k320_clone_aware_preempt_player_action_v1(
                    states,
                    resources["tables"],
                    resources["runtime"],
                    resources["latest_bank"],
                    carry,
                    player,
                )
            return high_potential_v20_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["runtime"],
                carry,
                player,
                MODE_RAY_K320,
            )

        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    money = []
    orientation_features = []
    batch = len(variants) * args.seeds
    for seat in (0, 1):
        candidate_policy = make_candidate_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        seat_money = []
        seat_features = []
        if args.compiled_loop:
            @jax.jit
            def run_segment(
                states, candidate_carry, opponent_carry, segment_events, step_count
            ):
                def body(_, loop_carry):
                    loop_states, loop_candidate, loop_opponent = loop_carry
                    candidate_action, loop_candidate = candidate_policy(
                        loop_states, loop_candidate
                    )
                    opponent_action, loop_opponent = opponent_policy(
                        loop_states, loop_opponent
                    )
                    loop_states = (
                        simulator(
                            loop_states,
                            candidate_action,
                            opponent_action,
                            segment_events,
                        )
                        if seat == 0
                        else simulator(
                            loop_states,
                            opponent_action,
                            candidate_action,
                            segment_events,
                        )
                    )
                    return loop_states, loop_candidate, loop_opponent

                return jax.lax.fori_loop(
                    0,
                    step_count,
                    body,
                    (states, candidate_carry, opponent_carry),
                )

        for segment_seeds, events in zip(
            seed_segments, event_segments, strict=True
        ):
            expanded = np.tile(segment_seeds, len(variants))
            states = jax.vmap(reset)(jnp.asarray(expanded))
            candidate_carry = initialize_high_potential_v20_carry_v1(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            decision_features = None
            if args.compiled_loop:
                if args.save_decision_features:
                    states, candidate_carry, opponent_carry = run_segment(
                        states,
                        candidate_carry,
                        opponent_carry,
                        events,
                        jnp.asarray(feature_step, dtype=jnp.int32),
                    )
                    prefix_state = jax.tree_util.tree_map(
                        lambda value: value[: args.seeds], states
                    )
                    decision_features = extract_decision_features(
                        prefix_state, seat
                    )
                    states, candidate_carry, opponent_carry = run_segment(
                        states,
                        candidate_carry,
                        opponent_carry,
                        events,
                        jnp.asarray(719 - feature_step, dtype=jnp.int32),
                    )
                else:
                    states, candidate_carry, opponent_carry = run_segment(
                        states,
                        candidate_carry,
                        opponent_carry,
                        events,
                        jnp.asarray(719, dtype=jnp.int32),
                    )
            else:
                for action_index in range(719):
                    if args.save_decision_features and action_index == feature_step:
                        prefix_state = jax.tree_util.tree_map(
                            lambda value: value[:args.seeds], states
                        )
                        decision_features = extract_decision_features(
                            prefix_state, seat
                        )
                    candidate_action, candidate_carry = candidate_policy(
                        states, candidate_carry
                    )
                    opponent_action, opponent_carry = opponent_policy(
                        states, opponent_carry
                    )
                    states = (
                        simulator(states, candidate_action, opponent_action, events)
                        if seat == 0
                        else simulator(states, opponent_action, candidate_action, events)
                    )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            if any(
                int(np.sum(np.asarray(getattr(terminal, name))))
                for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            ):
                raise AssertionError("simulator safety counter hit")
            seat_money.append(
                np.asarray(terminal.money, dtype=np.int64).reshape(
                    len(variants), args.seeds, 2
                )
            )
            if args.save_decision_features:
                if decision_features is None:
                    raise AssertionError("decision features were not captured")
                seat_features.extend(decision_features)
        money.append(np.concatenate(seat_money, axis=1))
        if args.save_decision_features:
            orientation_features.append(seat_features)

    rows = []
    margins_all = []
    for index, (name, route_id, switch_step) in enumerate(variants):
        own = np.concatenate((money[0][index, :, 0], money[1][index, :, 1]))
        rival = np.concatenate((money[0][index, :, 1], money[1][index, :, 0]))
        margins = own - rival
        margins_all.append(margins)
        rows.append({
            "variant": name,
            "route_id": route_id,
            "switch_step": switch_step,
            "games": int(len(margins)),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "per_game": [
                {
                    "seed": int(base_seeds[game % len(base_seeds)]),
                    "candidate_seat": 0 if game < len(base_seeds) else 1,
                    "margin": int(margin),
                }
                for game, margin in enumerate(margins.tolist())
            ],
        })

    stacked = np.stack(margins_all)
    oracle = np.max(stacked, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.k320-route-override-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": int(len(base_seeds)),
        "seed_batch_size": args.seeds,
        "seed_batches": args.seed_batches,
        "seat_protocol": "same seeds with seats swapped",
        "execution": "compiled_fori_loop" if args.compiled_loop else "python_step_loop",
        "decision_semantics": "adaptive K320 prefix; public-step route lock; no future state in policy",
        "candidate_stack": args.candidate_stack,
        "route_filter": sorted(route_filter),
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    if args.save_decision_features:
        payload["feature_step"] = feature_step
        payload["decision_features"] = [
            {
                "seed": int(base_seeds[index]),
                "candidate_seat": seat,
                **orientation_features[seat][index],
            }
            for seat in (0, 1)
            for index in range(len(base_seeds))
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = [{key: value for key, value in row.items() if key != "per_game"} for row in rows]
    print(json.dumps({
        "status": "PASS",
        "best": sorted(compact, key=lambda row: (row["score_rate"], row["mean_margin"]), reverse=True),
        "oracle": payload["oracle"],
        "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
