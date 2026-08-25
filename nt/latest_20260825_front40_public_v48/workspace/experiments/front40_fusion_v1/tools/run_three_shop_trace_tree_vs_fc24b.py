"""Run a causal three-shop Replay route tree against frozen FC24B on GPU.

The candidate route can change only after the corresponding town shop is
publicly visible.  An optional route screen provides a strict per-game parity
oracle for the spliced online execution.
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
from hasegawa_jax_v3.agent import ROUTER_PREFIX_LOCK  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def load_tree(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = np.asarray(payload["route_ids"], dtype=np.int16)
    second = np.asarray(payload["second_route_ids_by_first_shop"], dtype=np.int16)
    third = np.asarray(payload["third_route_ids_by_shop"], dtype=np.int16)
    if first.shape != (8,) or second.shape != (8, 8) or third.shape != (8, 8, 8):
        raise ValueError("invalid three-shop route-tree shapes")
    return first, second, third


def select_carry(mask: jax.Array, old, new):
    def choose(old_value, new_value):
        shaped = mask.reshape((mask.shape[0],) + (1,) * (old_value.ndim - 1))
        return jnp.where(shaped, new_value, old_value)

    return jax.tree_util.tree_map(choose, old, new)


def make_rollout(
    trace_player,
    tables,
    latest_bank,
    old_bank,
    runtime,
    first_map,
    second_map,
    third_map,
):
    fc_player = 1 - trace_player

    @jax.jit
    def rollout(
        states,
        trace_carry,
        fc_carry,
        candidate_fc_carry,
        events,
        trace_bank,
        weak_shop_mask,
    ):
        def body(value, _):
            current, trace_value, fc_value, candidate_fc_value = value
            count = jnp.clip(current.town_count.astype(jnp.int32), 0, 8)
            first_shop = jnp.clip(current.town_shops[:, 0], 0, 7)
            second_shop = jnp.clip(current.town_shops[:, 1], 0, 7)
            third_shop = jnp.clip(current.town_shops[:, 2], 0, 7)
            first_route = first_map[first_shop]
            second_route = second_map[first_shop, second_shop]
            third_route = third_map[first_shop, second_shop, third_shop]
            selected_route = jnp.where(
                count >= 3,
                third_route,
                jnp.where(count >= 2, second_route, first_route),
            ).astype(jnp.int16)
            forced_lock = count > 0
            # ``forced_lock`` controls only the lock flag inside the Replay
            # executor; supplying ``forced_route`` still changes the active
            # program immediately.  Preserve the bootstrap program until the
            # first shop is actually public, matching the route-screen
            # activation semantics exactly.
            forced_route = jnp.where(
                forced_lock, selected_route, trace_value.branch_id
            ).astype(jnp.int16)
            fc_action, fc_value = fc24_terminal_crop_salvage_player_action_v1(
                current, tables, latest_bank, old_bank, runtime, fc_value, fc_player
            )
            candidate_fc_action, proposed_candidate_fc_value = (
                fc24_terminal_crop_salvage_player_action_v1(
                    current,
                    tables,
                    latest_bank,
                    old_bank,
                    runtime,
                    candidate_fc_value,
                    trace_player,
                )
            )
            use_fallback = forced_lock & weak_shop_mask[first_shop]
            candidate_fc_value = select_carry(
                use_fallback, candidate_fc_value, proposed_candidate_fc_value
            )
            current, trace_value, _, _ = hasegawa_step_with_external_v3(
                current,
                trace_value,
                trace_bank,
                fc_action,
                trace_player,
                events,
                tables,
                ROUTER_PREFIX_LOCK,
                None,
                forced_route=forced_route,
                forced_lock=forced_lock,
                controlled_action_override=candidate_fc_action,
                controlled_action_override_mask=use_fallback,
            )
            return (current, trace_value, fc_value, candidate_fc_value), None

        return jax.lax.scan(
            body,
            (states, trace_carry, fc_carry, candidate_fc_carry),
            None,
            length=719,
        )[0]

    return rollout


def screen_expected(
    matrix_path: Path,
    first_map: np.ndarray,
    second_map: np.ndarray,
    third_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    screen = np.load(matrix_path, allow_pickle=False)
    margin = np.asarray(screen["margin"])
    observed = np.asarray(screen["observed_shops"])
    route_ids = np.asarray(screen["route_ids"], dtype=np.int32)
    lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}
    expected_cash = np.empty(margin.shape[:2], dtype=np.int64)
    expected_opp = np.empty(margin.shape[:2], dtype=np.int64)
    expected_route = np.empty(margin.shape[:2], dtype=np.int16)
    for seat in range(margin.shape[0]):
        for seed in range(margin.shape[1]):
            first_shop = int(observed[seat, seed, 0, 0])
            base_route = int(first_map[first_shop])
            base_column = lookup[base_route]
            second_shop = int(observed[seat, seed, base_column, 1])
            second_route = int(second_map[first_shop, second_shop])
            second_column = lookup[second_route]
            third_shop = int(observed[seat, seed, second_column, 2])
            route = int(third_map[first_shop, second_shop, third_shop])
            column = lookup[route]
            expected_cash[seat, seed] = int(screen["cash"][seat, seed, column])
            expected_opp[seat, seed] = int(screen["opponent_cash"][seat, seed, column])
            expected_route[seat, seed] = route
    return expected_cash, expected_opp, expected_route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--route-tree", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--screen-matrix", type=Path, default=None)
    parser.add_argument(
        "--fallback-shop-ids",
        default="",
        help="Comma-separated public first-shop IDs that switch to FC24B after reveal.",
    )
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    first_np, second_np, third_np = load_tree(args.route_tree)
    first_map = jnp.asarray(first_np)
    second_map = jnp.asarray(second_np)
    third_map = jnp.asarray(third_np)
    fallback_ids = tuple(
        sorted({int(value) for value in args.fallback_shop_ids.split(",") if value})
    )
    if any(value < 0 or value >= 8 for value in fallback_ids):
        raise ValueError(f"invalid fallback shop IDs: {fallback_ids}")
    weak_shop_mask = jnp.asarray(
        [shop in fallback_ids for shop in range(8)], dtype=jnp.bool_
    )
    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    route_count = int(trace_bank.unit_op.shape[0])
    if np.any(first_np >= route_count) or np.any(second_np >= route_count) or np.any(third_np >= route_count):
        raise ValueError("route tree exceeds trace bank")
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    expected = (
        screen_expected(args.screen_matrix, first_np, second_np, third_np)
        if args.screen_matrix is not None
        else None
    )

    rows = []
    outcomes: list[dict[str, int | bool]] = []
    for trace_player in (0, 1):
        rollout = make_rollout(
            trace_player,
            tables,
            latest_bank,
            old_bank,
            runtime,
            first_map,
            second_map,
            third_map,
        )
        state = initial
        carry = initialize_hasegawa_carry_v3(args.batch, trace_bank.bootstrap_route_id)
        fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(args.batch)
        candidate_fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(
            args.batch
        )
        started = time.perf_counter()
        state, carry, fc_carry, candidate_fc_carry = rollout(
            state,
            carry,
            fc_carry,
            candidate_fc_carry,
            events,
            trace_bank,
            weak_shop_mask,
        )
        jax.block_until_ready(state.money)
        elapsed = time.perf_counter() - started
        state, carry = jax.device_get((state, carry))
        opponent = 1 - trace_player
        cash = np.asarray(state.money[:, trace_player], dtype=np.int64)
        opp_cash = np.asarray(state.money[:, opponent], dtype=np.int64)
        route = np.asarray(carry.branch_id, dtype=np.int16)
        first_shop_terminal = np.asarray(state.town_shops[:, 0], dtype=np.int8)
        fallback = np.isin(first_shop_terminal, fallback_ids)
        parity = {"cash_mismatch": 0, "opponent_cash_mismatch": 0, "route_mismatch": 0}
        parity_examples: list[dict[str, int]] = []
        if expected is not None:
            mismatch = (~fallback) & (
                (cash != expected[0][trace_player])
                | (opp_cash != expected[1][trace_player])
                | (route != expected[2][trace_player])
            )
            parity = {
                "cash_mismatch": int(np.sum((~fallback) & (cash != expected[0][trace_player]))),
                "opponent_cash_mismatch": int(
                    np.sum((~fallback) & (opp_cash != expected[1][trace_player]))
                ),
                "route_mismatch": int(
                    np.sum((~fallback) & (route != expected[2][trace_player]))
                ),
            }
            for index in np.flatnonzero(mismatch)[:16]:
                parity_examples.append(
                    {
                        "index": int(index),
                        "seed": int(seeds[index]),
                        "actual_cash": int(cash[index]),
                        "expected_cash": int(expected[0][trace_player, index]),
                        "actual_opponent_cash": int(opp_cash[index]),
                        "expected_opponent_cash": int(expected[1][trace_player, index]),
                        "actual_route": int(route[index]),
                        "expected_route": int(expected[2][trace_player, index]),
                    }
                )
        row = {
            "candidate_seat": trace_player,
            "games": args.batch,
            "wins": int(np.sum(cash > opp_cash)),
            "win_rate": float(np.mean(cash > opp_cash)),
            "mean_margin": float(np.mean(cash - opp_cash)),
            "candidate_cash_mean": float(np.mean(cash)),
            "opponent_cash_mean": float(np.mean(opp_cash)),
            "all_done": bool(np.all(state.done)),
            "invalid_intent_total": int(np.sum(carry.invalid_intent_total)),
            "resync_total": int(np.sum(carry.resync_total)),
            "hard_counter_total": int(np.sum(carry.hard_counter_total)),
            "route_switch_total": int(np.sum(carry.route_switch_total)),
            "fallback_games": int(np.sum(fallback)),
            "fallback_wins": int(np.sum((cash > opp_cash) & fallback)),
            "fallback_win_rate": (
                float(np.mean((cash > opp_cash)[fallback])) if np.any(fallback) else None
            ),
            "elapsed_seconds": elapsed,
            "transitions_per_second": args.batch * 719 / elapsed,
            "parity": parity,
            "parity_examples": parity_examples,
        }
        rows.append(row)
        for index in range(args.batch):
            outcomes.append(
                {
                    "seat": int(trace_player),
                    "first_shop": int(first_shop_terminal[index]),
                    "fallback": bool(fallback[index]),
                    "win": bool(cash[index] > opp_cash[index]),
                    "margin": int(cash[index] - opp_cash[index]),
                }
            )
        print(json.dumps(row), flush=True)

    games = 2 * args.batch
    parity_total = sum(sum(row["parity"].values()) for row in rows)
    status = "PASS" if (
        all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows)
        and parity_total == 0
    ) else "FAIL"
    by_first_shop = []
    for first_shop in range(8):
        selected = [row for row in outcomes if row["first_shop"] == first_shop]
        if not selected:
            continue
        by_first_shop.append(
            {
                "first_shop": first_shop,
                "mode": "FC24B_FALLBACK" if first_shop in fallback_ids else "THREE_SHOP_TRACE",
                "games": len(selected),
                "wins": int(sum(bool(row["win"]) for row in selected)),
                "win_rate": float(sum(bool(row["win"]) for row in selected) / len(selected)),
                "mean_margin": float(np.mean([int(row["margin"]) for row in selected])),
            }
        )
    payload = {
        "schema": "kaggriculture.front40_fusion.three-shop-trace-tree-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "device": str(jax.devices()[0]),
        "trace_bank": str(args.trace_bank),
        "route_tree": str(args.route_tree),
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "screen_matrix": str(args.screen_matrix) if args.screen_matrix is not None else None,
        "fallback_shop_ids": list(fallback_ids),
        "runtime_boundary": "only town shops already visible in the current state",
        "aggregate": {
            "games": games,
            "wins": int(sum(row["wins"] for row in rows)),
            "win_rate": float(sum(row["wins"] for row in rows) / games),
            "mean_margin": float(sum(row["mean_margin"] for row in rows) / 2),
            "parity_mismatch_total": parity_total,
        },
        "results": rows,
        "by_first_shop": by_first_shop,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "aggregate": payload["aggregate"], "output": str(args.output)}))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
