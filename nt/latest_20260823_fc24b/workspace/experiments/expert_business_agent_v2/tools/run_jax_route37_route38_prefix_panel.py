#!/usr/bin/env python3
"""GPU same-prefix panel for Rank16 Route37 (8C4S) vs Route38 (6C6S).

Both streams are identical through step 191.  Features captured at step 192
are therefore same-state counterfactual context.  JAX outcomes remain a GPU
screen and require official Python 1.32.7 validation before promotion.
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
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_TOOLS = ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools"
sys.path.insert(0, str(SOURCE_TOOLS))

from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    BASE_NAMES,
    Events,
    TREND_INDICES,
    batched_step_sync,
    build_events_v1,
    build_router_arrays,
    extract_base48,
    full_opponent_action,
    initialize_full_opponent_carry,
    initialize_trace_player_carry_v1,
    load_bank,
    load_boatlee_trace_v1,
    pair,
    reset,
    sha256,
    skeleton_player_action_v1,
)


CANDIDATE_IDS = (37, 38)
ANCHOR_STEPS = (120, 168)
DECISION_STEP = 192
FEATURE_NAMES = BASE_NAMES + [
    f"delta_from_step{step}_{BASE_NAMES[index]}"
    for step in ANCHOR_STEPS
    for index in TREND_INDICES
]


def make_rollout(bank, boatlee_trace, tables, router, candidate_player: int):
    opponent_player = 1 - candidate_player

    @jax.jit
    def rollout(initial, events, candidate_ids, opponent_ids):
        batch_size = initial.step.shape[0]
        zero48 = jnp.zeros((batch_size, 48), dtype=jnp.float32)

        def body(value, _):
            states, candidate_carry, opponent_carry, anchor120, anchor168, decision192 = value

            def capture(captured):
                old120, old168, old192 = captured
                actor_features = extract_base48(states, candidate_player)
                old120 = jnp.where((states.step == 120)[:, None], actor_features, old120)
                old168 = jnp.where((states.step == 168)[:, None], actor_features, old168)
                old192 = jnp.where((states.step == 192)[:, None], actor_features, old192)
                return old120, old168, old192

            capture_step = jnp.any(
                states.step[0]
                == jnp.asarray((*ANCHOR_STEPS, DECISION_STEP), dtype=states.step.dtype)
            )
            anchor120, anchor168, decision192 = jax.lax.cond(
                capture_step,
                capture,
                lambda captured: captured,
                (anchor120, anchor168, decision192),
            )
            candidate_action, candidate_carry = skeleton_player_action_v1(
                states, tables, bank, candidate_ids, candidate_carry, candidate_player
            )
            opponent_action, opponent_carry = full_opponent_action(
                states,
                tables,
                bank,
                boatlee_trace,
                opponent_ids,
                opponent_carry,
                opponent_player,
                router,
            )
            actions = (
                pair(candidate_action, opponent_action)
                if candidate_player == 0
                else pair(opponent_action, candidate_action)
            )
            states = batched_step_sync(states, actions, events, tables)
            return (
                states, candidate_carry, opponent_carry,
                anchor120, anchor168, decision192,
            ), None

        result, _ = jax.lax.scan(
            body,
            (
                initial,
                initialize_trace_player_carry_v1(batch_size),
                initialize_full_opponent_carry(batch_size, opponent_ids, router),
                zero48,
                zero48,
                zero48,
            ),
            xs=None,
            length=719,
        )
        states, _, _, anchor120, anchor168, decision192 = result
        trend = jnp.asarray(TREND_INDICES, dtype=jnp.int32)
        features66 = jnp.concatenate((
            decision192,
            decision192[:, trend] - anchor120[:, trend],
            decision192[:, trend] - anchor168[:, trend],
        ), axis=1)
        return states.money, states.done, features66

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument(
        "--candidate-ids",
        type=int,
        nargs=2,
        default=CANDIDATE_IDS,
        metavar=("ROUTE37_ID", "ROUTE38_ID"),
        help="Skeleton ids assigned to Route37 and Route38 in this bank.",
    )
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path = args.bank.resolve()
    receipt_path = args.bank_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    candidate_ids_pair = tuple(int(value) for value in args.candidate_ids)
    compiled_ids = {int(value) for value in receipt["candidate_ids"]}
    if not set(candidate_ids_pair) <= compiled_ids:
        raise ValueError(f"candidate ids missing from bank: {candidate_ids_pair}")
    bank = load_bank(bank_path)
    boatlee_trace = load_boatlee_trace_v1()
    router = build_router_arrays(receipt)
    from kaggriculture_jax.state import load_tables
    tables = load_tables()

    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int32)
    opponent_values = np.arange(len(receipt["opponents"]), dtype=np.int16)
    batch_seeds = np.tile(seeds, len(opponent_values))
    batch_opponents = np.repeat(opponent_values, len(seeds))
    weed_by_seed, shops_by_seed = build_events_v1(seeds.tolist())
    events = Events(
        jnp.asarray(np.tile(weed_by_seed, (len(opponent_values), 1, 1))),
        jnp.asarray(np.tile(shops_by_seed, (len(opponent_values), 1, 1))),
    )
    initial = jax.vmap(reset)(jnp.asarray(batch_seeds))
    opponent_ids = jnp.asarray(batch_opponents, dtype=jnp.int32)
    wins = np.zeros((2, 2, 1, len(opponent_values), len(seeds)), dtype=np.bool_)
    margins = np.zeros_like(wins, dtype=np.int32)
    feature_by_candidate = np.zeros(
        (2, 2, len(opponent_values), len(seeds), len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    rollouts = [
        make_rollout(bank, boatlee_trace, tables, router, seat)
        for seat in (0, 1)
    ]
    timings: list[dict[str, object]] = []
    all_done = True
    for candidate_index, skeleton_id in enumerate(candidate_ids_pair):
        candidate_ids = jnp.full(opponent_ids.shape, skeleton_id, dtype=jnp.int32)
        for seat, rollout in enumerate(rollouts):
            started = perf_counter()
            money, done, feature_values = rollout(initial, events, candidate_ids, opponent_ids)
            jax.block_until_ready(money)
            elapsed = perf_counter() - started
            money_np = np.asarray(money, dtype=np.int64)
            done_np = np.asarray(done, dtype=bool)
            feature_np = np.asarray(feature_values, dtype=np.float32).reshape(
                len(opponent_values), len(seeds), len(FEATURE_NAMES)
            )
            own = money_np[:, seat]
            other = money_np[:, 1 - seat]
            wins[candidate_index, seat, 0] = (own > other).reshape(len(opponent_values), len(seeds))
            margins[candidate_index, seat, 0] = (own - other).reshape(len(opponent_values), len(seeds))
            feature_by_candidate[candidate_index, seat] = feature_np
            all_done &= bool(np.all(done_np))
            timings.append({
                "candidate_id": skeleton_id,
                "seat": seat,
                "games": len(batch_seeds),
                "seconds": elapsed,
                "transitions_per_second": len(batch_seeds) * 719 / elapsed,
            })

    max_feature_error = float(np.max(np.abs(feature_by_candidate - feature_by_candidate[0:1])))
    features_output = args.features_output.resolve()
    outcomes_output = args.outcomes_output.resolve()
    features_output.parent.mkdir(parents=True, exist_ok=True)
    outcomes_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        features_output,
        features=feature_by_candidate[0],
        feature_names=np.asarray(FEATURE_NAMES),
        opponent_ids=opponent_values,
        seeds=seeds,
        candidate_id=np.asarray(candidate_ids_pair[0], dtype=np.int16),
        decision_step=np.asarray(DECISION_STEP, dtype=np.int16),
    )
    np.savez_compressed(
        outcomes_output,
        candidate_ids=np.asarray(candidate_ids_pair, dtype=np.int16),
        opponent_ids=opponent_values,
        seeds=seeds,
        wins=wins,
        margins=margins,
    )
    total_games = sum(int(row["games"]) for row in timings)
    total_seconds = sum(float(row["seconds"]) for row in timings)
    result = {
        "schema": "route37-route38-jax-prefix-panel-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_GPU_SCREEN" if all_done and max_feature_error == 0.0 else "FAIL",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "candidate_ids": list(candidate_ids_pair),
        "decision_step": DECISION_STEP,
        "seed_start": int(seeds[0]),
        "seed_count": len(seeds),
        "opponent_count": len(opponent_values),
        "context_count": int(2 * len(opponent_values) * len(seeds)),
        "game_count": int(wins.size),
        "all_done": all_done,
        "same_state_feature_max_abs_error": max_feature_error,
        "same_state_pass": max_feature_error == 0.0,
        "timings": timings,
        "compile_inclusive_transitions_per_second": total_games * 719 / total_seconds,
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "bank_receipt": str(receipt_path),
        "bank_receipt_sha256": sha256(receipt_path),
        "features_output": str(features_output),
        "features_sha256": sha256(features_output),
        "outcomes_output": str(outcomes_output),
        "outcomes_sha256": sha256(outcomes_output),
        "truth_boundary": (
            "JAX GPU same-prefix screen. Feature equality proves an internal "
            "same-state panel only; official Python 1.32.7 validation remains mandatory."
        ),
    }
    receipt_output = args.receipt_output.resolve()
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS_GPU_SCREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
