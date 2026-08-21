#!/usr/bin/env python3
"""Evaluate a fusion or frozen candidate against the strict-parity roster."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
    fc1a_sheep_pressure_player_action_v1,
    fc1b_x562_pressure_player_action_v1,
    fc1s_prt_suffix_pressure_player_action_v1,
    fc2a_rank14_plus_anti_mirror_player_action_v1,
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v1,
    initialize_fusion_champion_carry_v2,
    initialize_fusion_champion_carry_v3,
    initialize_kaito_x562_carry_v1,
    initialize_x562_route_rule_carry_v1,
    kaito_x562_market_player_action_v1,
    k320_clone_aware_preempt_player_action_v1,
    k320_preempt_parameter_player_action_v1,
    x562_route_rule_player_action_v1,
)
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


CRITICAL_NAMES = (
    "gold_proxy_rank14_recursion",
    "public_g04_soil_rain",
    "flexonafft_v59_multi_route",
    "public_g01_boatlee_v16",
    "gold_proxy_rank12_ai_b2b67_saas",
    "public_g02_rc5_c166",
    "local_prt_v6",
    "deniz_v111_8c4s_latest",
    "kaito_v36_latest",
    "x562_latest",
    "gold_proxy_rank07_junichiro_morita",
    "gold_proxy_rank19_manu_nicholas_jacob",
    "kaito_v27_midgame_reset",
    "rayk_k320_adaptive_rank1",
    "boatlee_v20_multi_route",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=540101)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--candidate",
        choices=(
            "fc1a",
            "fc1b",
            "fc1d",
            "fc1s",
            "fc2a_fusion",
            "fc2b_fusion",
            "k320",
            "k320_preempt_best",
            "k320_preempt_broad",
            "k320_preempt_clone_aware",
            "kaito_x562",
        ),
        default="fc1a",
    )
    parser.add_argument(
        "--frozen-candidate",
        default="",
        help="Optional frozen roster name; overrides --candidate.",
    )
    parser.add_argument("--full-roster", action="store_true")
    parser.add_argument(
        "--track-weed-repairs",
        action="store_true",
        help="Record candidate weed-repair starts per game without changing policy semantics.",
    )
    parser.add_argument(
        "--opponents",
        default="",
        help="Optional comma-separated frozen roster names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/receipts/fc1a_panel_seed540101_n32x2_v1.json",
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(old_bank_path)
    latest_bank = load_bank(latest_bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    tables = load_tables()
    router = build_router_arrays(old_receipt)
    boatlee_trace = load_boatlee_trace_v1()
    resources = dict(
        old_bank=old_bank,
        latest_bank=latest_bank,
        runtime=runtime,
        tables=tables,
        router=router,
        boatlee_trace=boatlee_trace,
    )

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(tables)
    names = [row["name"] for row in rr.ROSTER]
    selected = (
        [name.strip() for name in args.opponents.split(",") if name.strip()]
        if args.opponents
        else (names if args.full_roster else list(CRITICAL_NAMES))
    )
    ids = {name: index for index, name in enumerate(names)}
    if len(selected) != len(set(selected)) or any(name not in ids for name in selected):
        raise AssertionError("invalid panel roster")
    if args.frozen_candidate and args.frozen_candidate not in ids:
        raise AssertionError(f"invalid frozen candidate: {args.frozen_candidate}")

    frozen_candidate_id = (
        ids[args.frozen_candidate] if args.frozen_candidate else None
    )

    def make_candidate_policy(player: int):
        if frozen_candidate_id is not None:
            return rr.make_agent_policy(
                frozen_candidate_id, player, **resources
            )

        @jax.jit
        def policy(states, carry):
            if args.candidate == "fc1a":
                return fc1a_sheep_pressure_player_action_v1(
                    states, tables, latest_bank, runtime, carry, player
                )
            if args.candidate == "fc1b":
                return fc1b_x562_pressure_player_action_v1(
                    states, tables, latest_bank, runtime, carry, player
                )
            if args.candidate == "fc1s":
                return fc1s_prt_suffix_pressure_player_action_v1(
                    states,
                    tables,
                    latest_bank,
                    old_bank,
                    runtime,
                    carry,
                    player,
                )
            if args.candidate == "fc2a_fusion":
                return fc2a_rank14_plus_anti_mirror_player_action_v1(
                    states,
                    tables,
                    latest_bank,
                    old_bank,
                    runtime,
                    carry,
                    player,
                )
            if args.candidate == "fc2b_fusion":
                return fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
                    states,
                    tables,
                    latest_bank,
                    old_bank,
                    runtime,
                    carry,
                    player,
                )
            if args.candidate == "kaito_x562":
                return kaito_x562_market_player_action_v1(
                    states, latest_bank, runtime, carry, player
                )
            if args.candidate == "fc1d":
                batch = states.step.shape[0]
                return x562_route_rule_player_action_v1(
                    states,
                    runtime,
                    latest_bank,
                    carry,
                    jnp.full((batch,), 2, dtype=jnp.int8),
                    jnp.zeros((batch,), dtype=jnp.bool_),
                    player,
                )
            if args.candidate == "k320_preempt_best":
                batch = states.step.shape[0]
                return k320_preempt_parameter_player_action_v1(
                    states,
                    tables,
                    runtime,
                    latest_bank,
                    carry,
                    jnp.zeros((batch,), dtype=jnp.bool_),
                    jnp.full((batch,), 4, dtype=jnp.int16),
                    jnp.full((batch,), 6, dtype=jnp.int16),
                    jnp.full((batch,), 32, dtype=jnp.int16),
                    jnp.full((batch,), 4, dtype=jnp.int16),
                    jnp.zeros((batch,), dtype=jnp.int16),
                    jnp.full((batch,), 120, dtype=jnp.int16),
                    jnp.full((batch,), 200, dtype=jnp.int16),
                    player,
                )
            if args.candidate == "k320_preempt_broad":
                batch = states.step.shape[0]
                return k320_preempt_parameter_player_action_v1(
                    states,
                    tables,
                    runtime,
                    latest_bank,
                    carry,
                    jnp.zeros((batch,), dtype=jnp.bool_),
                    jnp.full((batch,), 4, dtype=jnp.int16),
                    jnp.full((batch,), 100, dtype=jnp.int16),
                    jnp.full((batch,), 12, dtype=jnp.int16),
                    jnp.full((batch,), 4, dtype=jnp.int16),
                    jnp.zeros((batch,), dtype=jnp.int16),
                    jnp.full((batch,), 120, dtype=jnp.int16),
                    jnp.full((batch,), 216, dtype=jnp.int16),
                    player,
                )
            if args.candidate == "k320_preempt_clone_aware":
                return k320_clone_aware_preempt_player_action_v1(
                    states,
                    tables,
                    runtime,
                    latest_bank,
                    carry,
                    player,
                )
            return high_potential_v20_player_action_v1(
                states,
                tables,
                latest_bank,
                runtime,
                carry,
                player,
                MODE_RAY_K320,
            )

        return policy

    candidate_policies = (make_candidate_policy(0), make_candidate_policy(1))
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in selected:
        opponent_id = ids[opponent_name]
        orientation_money = []
        orientation_terminal = []
        orientation_weed_repairs = []
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(seeds))
            if frozen_candidate_id is not None:
                candidate_carry = rr.initialize_agent_carry(
                    frozen_candidate_id, args.seeds, router
                )
            elif args.candidate == "fc1a":
                candidate_carry = initialize_fusion_champion_carry_v1(args.seeds)
            elif args.candidate == "fc1b":
                candidate_carry = initialize_fusion_champion_carry_v2(args.seeds)
            elif args.candidate in ("fc1s", "fc2a_fusion", "fc2b_fusion"):
                candidate_carry = initialize_fusion_champion_carry_v3(args.seeds)
            elif args.candidate == "kaito_x562":
                candidate_carry = initialize_kaito_x562_carry_v1(args.seeds)
            elif args.candidate == "fc1d":
                candidate_carry = initialize_x562_route_rule_carry_v1(args.seeds)
            else:
                candidate_carry = initialize_high_potential_v20_carry_v1(args.seeds)
            opponent_carry = rr.initialize_agent_carry(opponent_id, args.seeds, router)
            candidate_policy = candidate_policies[candidate_seat]
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )
            candidate_weed_repairs = jnp.zeros((args.seeds,), dtype=jnp.int16)
            for _ in range(719):
                if args.track_weed_repairs:
                    if args.candidate in ("fc1s", "fc2a_fusion", "fc2b_fusion"):
                        weed_before = candidate_carry.k320.weed_start
                    elif args.candidate in (
                        "k320",
                        "k320_preempt_best",
                        "k320_preempt_broad",
                        "k320_preempt_clone_aware",
                    ):
                        weed_before = candidate_carry.weed_start
                    else:
                        weed_before = None
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                if args.track_weed_repairs and weed_before is not None:
                    if args.candidate in ("fc1s", "fc2a_fusion", "fc2b_fusion"):
                        weed_after = candidate_carry.k320.weed_start
                    else:
                        weed_after = candidate_carry.weed_start
                    candidate_weed_repairs = candidate_weed_repairs + jnp.sum(
                        weed_after != weed_before, axis=1
                    ).astype(jnp.int16)
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                if candidate_seat == 0:
                    states = simulator(
                        states, candidate_action, opponent_action, events
                    )
                else:
                    states = simulator(
                        states, opponent_action, candidate_action, events
                    )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError(f"{opponent_name}: not all games DONE")
            if (
                int(np.sum(np.asarray(terminal.hand_cap_hits)))
                or int(np.sum(np.asarray(terminal.market_loop_cap_hits)))
                or int(np.sum(np.asarray(terminal.price_lut_oob)))
            ):
                raise AssertionError(f"{opponent_name}: simulator safety counter hit")
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
            orientation_terminal.append(terminal)
            orientation_weed_repairs.append(
                np.asarray(jax.device_get(candidate_weed_repairs), dtype=np.int16)
            )

        first, second = orientation_money
        candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
        opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
        margins = candidate_cash - opponent_cash
        weed_repairs = np.concatenate(orientation_weed_repairs)
        first_terminal, second_terminal = orientation_terminal
        candidate_shed = np.concatenate(
            (
                np.sum(np.asarray(first_terminal.shed[:, 0]), axis=1),
                np.sum(np.asarray(second_terminal.shed[:, 1]), axis=1),
            )
        )
        candidate_unit_inventory = np.concatenate(
            (
                np.sum(np.asarray(first_terminal.unit_inventory[:, 0]), axis=(1, 2)),
                np.sum(np.asarray(second_terminal.unit_inventory[:, 1]), axis=(1, 2)),
            )
        )
        candidate_map_yield = np.concatenate(
            (
                np.sum(np.asarray(first_terminal.tile_yield[:, 0]), axis=(1, 2)),
                np.sum(np.asarray(second_terminal.tile_yield[:, 1]), axis=(1, 2)),
            )
        )
        per_game = []
        for index, seed in enumerate(seeds.tolist()):
            for candidate_seat, terminal_money in ((0, first), (1, second)):
                candidate_value = int(terminal_money[index, candidate_seat])
                opponent_value = int(terminal_money[index, 1 - candidate_seat])
                margin = candidate_value - opponent_value
                per_game.append(
                    {
                        "seed": int(seed),
                        "candidate_seat": candidate_seat,
                        "candidate_cash": candidate_value,
                        "opponent_cash": opponent_value,
                        "margin": margin,
                        "candidate_weed_repairs": int(
                            orientation_weed_repairs[candidate_seat][index]
                        ),
                        "result": "win" if margin > 0 else ("tie" if margin == 0 else "loss"),
                    }
                )
        row = {
            "opponent": opponent_name,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(candidate_cash)),
            "mean_opponent_cash": float(np.mean(opponent_cash)),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "candidate_seat0_wins": int(np.sum((first[:, 0] - first[:, 1]) > 0)),
            "candidate_seat1_wins": int(np.sum((second[:, 1] - second[:, 0]) > 0)),
            "weed_repair_diagnostic": {
                "enabled": bool(args.track_weed_repairs),
                "mean_all": float(np.mean(weed_repairs)),
                "mean_wins": float(np.mean(weed_repairs[margins > 0])) if np.any(margins > 0) else 0.0,
                "mean_losses": float(np.mean(weed_repairs[margins < 0])) if np.any(margins < 0) else 0.0,
                "hist_all": {
                    str(int(value)): int(count)
                    for value, count in zip(*np.unique(weed_repairs, return_counts=True), strict=True)
                },
                "hist_losses": {
                    str(int(value)): int(count)
                    for value, count in zip(*np.unique(weed_repairs[margins < 0], return_counts=True), strict=True)
                } if np.any(margins < 0) else {},
            },
            "terminal_assets": {
                "mean_shed_units_all": float(np.mean(candidate_shed)),
                "mean_shed_units_losses": float(np.mean(candidate_shed[margins < 0])) if np.any(margins < 0) else 0.0,
                "mean_unit_inventory_all": float(np.mean(candidate_unit_inventory)),
                "mean_unit_inventory_losses": float(np.mean(candidate_unit_inventory[margins < 0])) if np.any(margins < 0) else 0.0,
                "mean_map_yield_all": float(np.mean(candidate_map_yield)),
                "mean_map_yield_losses": float(np.mean(candidate_map_yield[margins < 0])) if np.any(margins < 0) else 0.0,
            },
            "per_game": per_game,
        }
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "per_game"}), flush=True)

    elapsed = perf_counter() - started
    payload = {
        "schema": "kaggriculture.fusion_champion.panel.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate": args.frozen_candidate or (
            "fc1a_sheep_pressure"
            if args.candidate == "fc1a"
            else (
                "fc1b_x562_sheep_pressure"
                if args.candidate == "fc1b"
                else (
                "fc1s_x562_prt_suffix_sheep_pressure"
                if args.candidate == "fc1s"
                else (
                "fc2b_rank14_plus_k320_clone_aware_preempt"
                    if args.candidate == "fc2b_fusion"
                    else "fc2a_rank14_plus_k320_anti_mirror"
                    if args.candidate == "fc2a_fusion"
                    else (
                    "fc1d_x562_any_yarn_route7_no_counters"
                    if args.candidate == "fc1d"
                    else (
                    "kaito_v36_plus_x562_market_layers"
                    if args.candidate == "kaito_x562"
                    else (
                        "fc2_k320_anti_mirror_best_so_far_v1"
                        if args.candidate == "k320_preempt_best"
                        else "rayk_k320_adaptive_rank1"
                    )
                    )
                    )
                    )
                )
            )
        ),
        "change": (
            f"frozen strict-parity roster candidate: {args.frozen_candidate}"
            if args.frozen_candidate
            else (
                "K320; sticky visible opening sheep pressure (>=4S, <=2C, cash<=1000) forces 6C8S"
                if args.candidate == "fc1a"
                else (
                    "K320 normally; sticky visible opening sheep pressure selects complete X562 capability"
                    if args.candidate == "fc1b"
                    else (
                        "K320 normally; sheep pressure selects stable X562 and the frozen step288 PRT suffix selector"
                        if args.candidate == "fc1s"
                        else (
                        "FC1S Rank14 branch plus frozen FC2 h4/d6/cap32/q4/start120 STRAWBERRY/MILK/WOOL anti-mirror"
                        if args.candidate in ("fc2a_fusion", "fc2b_fusion")
                        else (
                        "X562 prefix; step168 any visible YARN selects route7 else route12; counters disabled"
                        if args.candidate == "fc1d"
                        else (
                        "complete Kaito V36 production plus current-state X562 market counters, guards, and liquidation"
                        if args.candidate == "kaito_x562"
                        else (
                            "frozen FC2 best-so-far: h4, distance6, cap32, q4, start120, STRAWBERRY/MILK/WOOL"
                            if args.candidate == "k320_preempt_best"
                            else "frozen K320 control"
                        )
                        )
                        )
                        )
                    )
                )
            )
        ),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "games_per_opponent": args.seeds * 2,
        "seat_protocol": "same seeds with seats swapped",
        "full_roster": bool(args.full_roster),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "min_score_rate": min(row["score_rate"] for row in rows),
        "mean_score_rate": float(np.mean([row["score_rate"] for row in rows])),
        "candidate_source": (
            {
                "path": str(Path(rr.__file__).resolve()),
                "sha256": sha256(Path(rr.__file__).resolve()),
                "roster_name": args.frozen_candidate,
            }
            if args.frozen_candidate
            else {
                "path": str((ROOT / "experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py").resolve()),
                "sha256": sha256(ROOT / "experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py"),
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
