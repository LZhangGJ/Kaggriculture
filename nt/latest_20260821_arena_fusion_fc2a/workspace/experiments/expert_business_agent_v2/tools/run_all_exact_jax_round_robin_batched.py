#!/usr/bin/env python3
"""Batched-anchor accelerator for the strict 28-agent JAX round robin.

Eight opponents share one 400-lane simulator batch.  The anchor policy runs
once over all lanes; each opponent policy runs on its exact 50-lane slice.
This preserves live opponent-state semantics while substantially reducing
Python-to-XLA dispatches versus running eight pair matches independently.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
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
TOOLS = ROOT / "experiments" / "expert_business_agent_v2" / "tools"
sys.path.insert(0, str(TOOLS))

from run_all_exact_jax_round_robin import (  # noqa: E402
    ROSTER,
    fit_bradley_terry,
    initialize_agent_carry,
    make_agent_policy,
    make_report,
    make_simulator_step,
    sha256,
    write_checkpoint,
)
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    load_bank,
)
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def slice_tree(tree, begin: int, end: int):
    return jax.tree.map(lambda value: value[begin:end], tree)


def concatenate_actions(actions: list[Action]) -> Action:
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *actions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=530001)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--anchor-chunk", type=int, default=8)
    parser.add_argument("--anchor-min", type=int, default=0)
    parser.add_argument("--anchor-max", type=int, default=26)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from a v2 checkpoint that persisted integrity counters",
    )
    parser.add_argument(
        "--old-bank",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz",
    )
    parser.add_argument(
        "--old-receipt",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json",
    )
    parser.add_argument(
        "--latest-bank",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz",
    )
    parser.add_argument(
        "--event-bank-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/all_exact_round_robin_events_seed530001_n50_v1.npz",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/receipts/all_exact_jax_round_robin_n28_seed530001_n50x2_v1.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/reports/ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md",
    )
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1 or args.anchor_chunk < 1 or not (1 <= args.steps <= 719):
        raise ValueError("invalid seeds, anchor chunk, or steps")
    if not (0 <= args.anchor_min <= args.anchor_max <= 26):
        raise ValueError("anchor range must satisfy 0 <= min <= max <= 26")
    if len(ROSTER) != 28:
        raise AssertionError("frozen roster changed")

    cache = ROOT / ".jax_cache" / "all_exact_round_robin"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    old_receipt_path = args.old_receipt.resolve()
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(args.old_bank.resolve())
    latest_bank = load_bank(args.latest_bank.resolve())
    runtime = load_high_potential_runtime_tables_v1(args.runtime.resolve())
    router = build_router_arrays(old_receipt)
    tables = load_tables()
    boatlee_trace = load_boatlee_trace_v1()

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    event_path = args.event_bank_output.resolve()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(event_path, event_seeds=seeds, weed_spawn=weed, shop_choice=shops)

    all_pairs = list(itertools.combinations(range(len(ROSTER)), 2))
    all_pairs = [
        pair for pair in all_pairs if args.anchor_min <= pair[0] <= args.anchor_max
    ]
    if args.pair_limit:
        all_pairs = all_pairs[: args.pair_limit]
    policy_cache = {}
    simulator_cache = {}
    compile_records = []
    compile_seconds = 0.0
    arena_seconds = 0.0
    pair_rows = []
    all_done = True
    hand_cap_hits_total = 0
    market_loop_cap_hits_total = 0
    price_lut_oob_total = 0
    previous_wall_seconds = 0.0
    receipt_path = args.receipt_output.resolve()
    if args.resume and receipt_path.is_file():
        resume_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not resume_payload.get("resumable_integrity_checkpoint_v2", False):
            raise RuntimeError("refusing to resume from checkpoint without persisted integrity counters")
        if (
            resume_payload.get("steps") != args.steps
            or resume_payload.get("seed_count") != args.seeds
            or resume_payload.get("anchor_min") != args.anchor_min
            or resume_payload.get("anchor_max") != args.anchor_max
        ):
            raise RuntimeError("resume protocol does not match command line")
        pair_rows = list(resume_payload.get("pair_results", []))
        compile_records = list(resume_payload.get("compile_records", []))
        compile_seconds = float(resume_payload.get("compile_seconds", 0.0))
        arena_seconds = float(resume_payload.get("arena_seconds", 0.0))
        all_done = bool(resume_payload.get("all_done", True))
        hand_cap_hits_total = int(resume_payload.get("hand_cap_hits_total", 0))
        market_loop_cap_hits_total = int(resume_payload.get("market_loop_cap_hits_total", 0))
        price_lut_oob_total = int(resume_payload.get("price_lut_oob_total", 0))
        previous_wall_seconds = float(resume_payload.get("wall_seconds", 0.0))
    completed_pairs = {
        (int(row["agent_a_id"]), int(row["agent_b_id"])) for row in pair_rows
    }
    pending_pairs = [pair for pair in all_pairs if pair not in completed_pairs]
    pairs_by_anchor: dict[int, list[int]] = {}
    for left, right in pending_pairs:
        pairs_by_anchor.setdefault(left, []).append(right)
    work = []
    for anchor, opponents in pairs_by_anchor.items():
        for begin in range(0, len(opponents), args.anchor_chunk):
            work.append((anchor, opponents[begin : begin + args.anchor_chunk]))

    started_all = perf_counter()
    device = jax.devices()[0]
    memory_before = device.memory_stats() or {}

    def get_policy(agent_id: int, player: int, batch: int):
        key = (agent_id, player, batch)
        new = key not in policy_cache
        if new:
            policy_cache[key] = make_agent_policy(
                agent_id,
                player,
                old_bank,
                latest_bank,
                runtime,
                tables,
                router,
                boatlee_trace,
            )
        return policy_cache[key], new

    for work_index, (anchor, real_opponents) in enumerate(work):
        opponents = list(real_opponents)
        batch = len(opponents) * args.seeds
        lane_seeds = np.tile(seeds, len(opponents))
        events = Events(
            jnp.asarray(np.tile(weed, (len(opponents), 1, 1))),
            jnp.asarray(np.tile(shops, (len(opponents), 1, 1))),
        )
        simulator_new = batch not in simulator_cache
        if simulator_new:
            simulator_cache[batch] = make_simulator_step(tables)
        simulator_step = simulator_cache[batch]
        orientation_money = []
        work_started = perf_counter()
        compile_for_work = 0.0

        for orientation in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(lane_seeds))
            anchor_player = orientation
            partner_player = 1 - orientation
            anchor_policy, anchor_new = get_policy(anchor, anchor_player, batch)
            anchor_carry = initialize_agent_carry(anchor, batch, router)
            partner_policies, partner_new, partner_carries = [], [], []
            for opponent in opponents:
                policy, new = get_policy(opponent, partner_player, args.seeds)
                partner_policies.append(policy)
                partner_new.append(new)
                partner_carries.append(
                    initialize_agent_carry(opponent, args.seeds, router)
                )

            for step in range(args.steps):
                call_started = perf_counter()
                anchor_action, anchor_carry = anchor_policy(states, anchor_carry)
                if anchor_new and step == 0:
                    jax.block_until_ready(anchor_action.unit_op)
                    seconds = perf_counter() - call_started
                    compile_for_work += seconds
                    row = {
                        "agent_id": anchor,
                        "agent": ROSTER[anchor]["name"],
                        "seat": anchor_player,
                        "batch": batch,
                        "seconds": seconds,
                    }
                    compile_records.append(row)
                    print(json.dumps({"phase": "compiled_anchor", **row}), flush=True)

                partner_actions = []
                for slot, (opponent, policy) in enumerate(
                    zip(opponents, partner_policies, strict=True)
                ):
                    begin, end = slot * args.seeds, (slot + 1) * args.seeds
                    substate = slice_tree(states, begin, end)
                    call_started = perf_counter()
                    action, partner_carries[slot] = policy(
                        substate, partner_carries[slot]
                    )
                    if partner_new[slot] and step == 0:
                        jax.block_until_ready(action.unit_op)
                        seconds = perf_counter() - call_started
                        compile_for_work += seconds
                        row = {
                            "agent_id": opponent,
                            "agent": ROSTER[opponent]["name"],
                            "seat": partner_player,
                            "batch": args.seeds,
                            "seconds": seconds,
                        }
                        compile_records.append(row)
                        print(json.dumps({"phase": "compiled_partner", **row}), flush=True)
                    partner_actions.append(action)
                partner_action = concatenate_actions(partner_actions)

                if orientation == 0:
                    action0, action1 = anchor_action, partner_action
                else:
                    action0, action1 = partner_action, anchor_action
                call_started = perf_counter()
                states = simulator_step(states, action0, action1, events)
                if simulator_new and orientation == 0 and step == 0:
                    jax.block_until_ready(states.step)
                    seconds = perf_counter() - call_started
                    compile_for_work += seconds
                    row = {
                        "agent": "simulator_step",
                        "seat": -1,
                        "batch": batch,
                        "seconds": seconds,
                    }
                    compile_records.append(row)
                    print(json.dumps({"phase": "compiled_simulator", **row}), flush=True)

            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
            real_lanes = len(real_opponents) * args.seeds
            all_done = all_done and (
                bool(np.all(np.asarray(terminal.done)[:real_lanes]))
                if args.steps == 719
                else True
            )
            hand_cap_hits_total += int(
                np.sum(np.asarray(terminal.hand_cap_hits)[:real_lanes])
            )
            market_loop_cap_hits_total += int(
                np.sum(np.asarray(terminal.market_loop_cap_hits)[:real_lanes])
            )
            price_lut_oob_total += int(
                np.sum(np.asarray(terminal.price_lut_oob)[:real_lanes])
            )

        compile_seconds += compile_for_work
        work_seconds = perf_counter() - work_started
        arena_seconds += max(0.0, work_seconds - compile_for_work)
        first, second = orientation_money
        for slot, opponent in enumerate(real_opponents):
            begin, end = slot * args.seeds, (slot + 1) * args.seeds
            a_cash = np.concatenate((first[begin:end, 0], second[begin:end, 1]))
            b_cash = np.concatenate((first[begin:end, 1], second[begin:end, 0]))
            margins = a_cash - b_cash
            wins = int(np.sum(margins > 0))
            losses = int(np.sum(margins < 0))
            ties = int(np.sum(margins == 0))
            pair_rows.append(
                {
                    "agent_a_id": anchor,
                    "agent_a": ROSTER[anchor]["name"],
                    "agent_b_id": opponent,
                    "agent_b": ROSTER[opponent]["name"],
                    "games": args.seeds * 2,
                    "a_wins": wins,
                    "ties": ties,
                    "a_losses": losses,
                    "a_score_rate": (wins + 0.5 * ties) / (args.seeds * 2),
                    "a_mean_cash": float(np.mean(a_cash)),
                    "b_mean_cash": float(np.mean(b_cash)),
                    "a_mean_margin": float(np.mean(margins)),
                    "a_median_margin": float(np.median(margins)),
                    "a_seat0_wins": int(np.sum((first[begin:end, 0] - first[begin:end, 1]) > 0)),
                    "a_seat1_wins": int(np.sum((second[begin:end, 1] - second[begin:end, 0]) > 0)),
                }
            )

        checkpoint = {
            "schema": "kaggriculture.all_exact_jax_round_robin.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
            "resumable_integrity_checkpoint_v2": True,
            "roster_count": len(ROSTER),
            "pair_count_planned": len(all_pairs),
            "pair_count_complete": len(pair_rows),
            "games_per_pair": args.seeds * 2,
            "seed_count": args.seeds,
            "steps": args.steps,
            "anchor_min": args.anchor_min,
            "anchor_max": args.anchor_max,
            "execution": "batched_anchor_v1",
            "compile_seconds": compile_seconds,
            "compile_records": compile_records,
            "arena_seconds": arena_seconds,
            "wall_seconds": previous_wall_seconds + perf_counter() - started_all,
            "all_done": all_done,
            "hand_cap_hits_total": hand_cap_hits_total,
            "market_loop_cap_hits_total": market_loop_cap_hits_total,
            "price_lut_oob_total": price_lut_oob_total,
            "pair_results": pair_rows,
        }
        write_checkpoint(args.receipt_output.resolve(), checkpoint)
        print(
            json.dumps(
                {
                    "phase": "anchor_chunk",
                    "chunk": work_index + 1,
                    "chunks": len(work),
                    "anchor": ROSTER[anchor]["name"],
                    "new_pairs": len(real_opponents),
                    "pairs_complete": len(pair_rows),
                    "pairs_total": len(all_pairs),
                    "seconds": work_seconds,
                }
            ),
            flush=True,
        )

    elapsed = previous_wall_seconds + perf_counter() - started_all
    ratings = fit_bradley_terry(pair_rows, len(ROSTER))
    totals = [
        {"wins": 0, "ties": 0, "losses": 0, "cash": [], "margin": [], "pairs": []}
        for _ in ROSTER
    ]
    for row in pair_rows:
        i, j = row["agent_a_id"], row["agent_b_id"]
        totals[i]["wins"] += row["a_wins"]
        totals[i]["ties"] += row["ties"]
        totals[i]["losses"] += row["a_losses"]
        totals[j]["wins"] += row["a_losses"]
        totals[j]["ties"] += row["ties"]
        totals[j]["losses"] += row["a_wins"]
        totals[i]["cash"].append(row["a_mean_cash"])
        totals[j]["cash"].append(row["b_mean_cash"])
        totals[i]["margin"].append(row["a_mean_margin"])
        totals[j]["margin"].append(-row["a_mean_margin"])
        totals[i]["pairs"].append((row["agent_b"], row["a_score_rate"]))
        totals[j]["pairs"].append((row["agent_a"], 1.0 - row["a_score_rate"]))

    ranking = []
    for rank, agent_id in enumerate(np.argsort(-ratings).tolist(), 1):
        total = totals[agent_id]
        games = total["wins"] + total["ties"] + total["losses"]
        worst_name, worst_score = min(total["pairs"], key=lambda x: x[1]) if total["pairs"] else ("N/A", 0.0)
        ranking.append(
            {
                "rank": rank,
                "agent_id": agent_id,
                "agent": ROSTER[agent_id]["name"],
                "bt_elo": float(ratings[agent_id]),
                "games": games,
                "wins": total["wins"],
                "ties": total["ties"],
                "losses": total["losses"],
                "score_rate": (total["wins"] + 0.5 * total["ties"]) / max(games, 1),
                "mean_cash": float(np.mean(total["cash"])) if total["cash"] else 0.0,
                "mean_margin": float(np.mean(total["margin"])) if total["margin"] else 0.0,
                "worst_opponent": worst_name,
                "worst_score_rate": float(worst_score),
            }
        )

    full_run = (
        args.steps == 719
        and not args.pair_limit
        and args.seeds == 50
        and args.anchor_min == 0
        and args.anchor_max == 26
    )
    integrity_pass = (
        (all_done if args.steps == 719 else True)
        and hand_cap_hits_total == 0
        and market_loop_cap_hits_total == 0
        and price_lut_oob_total == 0
    )
    status = (
        "PASS"
        if full_run and integrity_pass
        else ("PARTIAL_PASS" if integrity_pass and args.steps == 719 else ("SMOKE_PASS" if integrity_pass else "FAIL"))
    )
    aliases = [
        {"alias": alias, "canonical": row["name"], "reason": "source/action semantics identical"}
        for row in ROSTER
        for alias in row.get("aliases", ())
    ]
    games_total = len(pair_rows) * args.seeds * 2
    memory_after = device.memory_stats() or {}
    provenance_paths = (
        args.old_bank.resolve(),
        old_receipt_path,
        args.latest_bank.resolve(),
        args.runtime.resolve(),
        ROOT / "experiments/expert_business_agent_v2/receipts/weak6_jax_stepwise_parity_v3.json",
        ROOT / "experiments/expert_business_agent_v2/receipts/high_potential_exact5_historical_regression_after_latest8_v1.json",
        ROOT / "experiments/expert_business_agent_v2/receipts/latest_public8_jax_acceptance_v1.json",
        ROOT / "experiments/expert_business_agent_v2/tools/run_all_exact_jax_round_robin.py",
        Path(__file__).resolve(),
    )
    payload = {
        "schema": "kaggriculture.all_exact_jax_round_robin.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": str(device),
        "execution": "batched_anchor_v1",
        "roster_count": len(ROSTER),
        "roster": [dict(row) for row in ROSTER],
        "aliases": aliases,
        "excluded": {
            "route_skeletons": 46,
            "reason": "not complete strict-parity agents",
            "experimental_bc_ppo": "excluded",
            "unaccepted_proxy_agents": "excluded",
        },
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "event_seeds": seeds.tolist(),
        "event_bank": str(event_path),
        "event_bank_sha256": sha256(event_path),
        "steps": args.steps,
        "seat_protocol": "50 seeds as A-seat0/B-seat1 plus same 50 seeds swapped",
        "games_per_pair": args.seeds * 2,
        "pair_count": len(pair_rows),
        "pair_count_planned": len(all_pairs),
        "games_total": games_total,
        "transitions_total": games_total * args.steps,
        "anchor_chunk": args.anchor_chunk,
        "anchor_min": args.anchor_min,
        "anchor_max": args.anchor_max,
        "compile_seconds": compile_seconds,
        "compile_records": compile_records,
        "arena_seconds": arena_seconds,
        "wall_seconds": elapsed,
        "transitions_per_second": games_total * args.steps / max(arena_seconds, 1e-9),
        "all_done": all_done,
        "hand_cap_hits_total": hand_cap_hits_total,
        "market_loop_cap_hits_total": market_loop_cap_hits_total,
        "price_lut_oob_total": price_lut_oob_total,
        "integrity_pass": integrity_pass,
        "device_bytes_in_use_before": int(memory_before.get("bytes_in_use", 0)),
        "device_bytes_in_use_after": int(memory_after.get("bytes_in_use", 0)),
        "device_peak_bytes_in_use": int(memory_after.get("peak_bytes_in_use", 0)),
        "device_allocator_limit_bytes": int(memory_after.get("bytes_limit", 0)),
        "provenance": [{"path": str(path), "sha256": sha256(path)} for path in provenance_paths],
        "pair_results": pair_rows,
        "ranking": ranking,
        "truth_boundary": (
            "Gold proxy entries are strict-parity local imitation agents, not original gold source. "
            "This fixed-event local JAX ranking is not a current Public leaderboard score."
        ),
    }
    write_checkpoint(args.receipt_output.resolve(), payload)
    report_path = args.report_output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "receipt": str(args.receipt_output.resolve()),
                "report": str(report_path),
                "pairs": len(pair_rows),
                "games": games_total,
                "transitions_per_second": payload["transitions_per_second"],
            }
        ),
        flush=True,
    )
    return 0 if integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
