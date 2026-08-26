#!/usr/bin/env python3
"""Batched current-runtime screen of trace backbones versus one old28 group.

All opponents in one modest controller group share one GPU batch.  Splitting
the old28 roster into its seven accepted groups avoids recompiling or waiting
for one monolithic policy graph.  Candidate bank and public-shop maps remain
dynamic so one compiled step serves every trace backbone in the group.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
    ROOT / "experiments/front40_fusion_v1/tools",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_TWO_SHOP_COMPATIBLE_MAP  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    load_bank,
)
from run_trace_candidates_vs_fc24b_stepwise import pad_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


OLD_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
OLD_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
LATEST_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def loaded_project_python_sources() -> set[Path]:
    sources: set[Path] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix == ".py" and path.is_file():
            sources.add(path)
    return sources


def load_candidate(spec: dict, route_capacity: int):
    bank_path = ROOT / spec["bank"]
    route_path = ROOT / spec["route_map"]
    bank = pad_bank(load_hasegawa_trace_bank_v3(bank_path), route_capacity)
    route_payload = json.loads(route_path.read_text(encoding="utf-8"))
    route_map = jnp.asarray(route_payload["route_ids"], dtype=jnp.int16)
    if route_map.shape not in ((8,), (2, 8)):
        raise ValueError(f"{spec['name']} route map shape {route_map.shape}")
    if spec.get("second_route_map"):
        second_path = ROOT / spec["second_route_map"]
        second_payload = json.loads(second_path.read_text(encoding="utf-8"))
        key = (
            "second_route_ids_by_first_shop"
            if "second_route_ids_by_first_shop" in second_payload
            else "second_route_ids"
        )
        second_map = jnp.asarray(second_payload[key], dtype=jnp.int16)
    else:
        second_path = None
        second_map = jnp.repeat(
            jnp.arange(route_capacity, dtype=jnp.int16)[:, None], 8, axis=1
        )
    if second_map.shape not in ((route_capacity, 8), (8, 8)):
        raise ValueError(f"{spec['name']} second map shape {second_map.shape}")
    second_map_is_compact = second_map.shape == (8, 8)
    if second_map_is_compact and route_capacity != 8:
        padded = jnp.repeat(
            jnp.arange(route_capacity, dtype=jnp.int16)[:, None], 8, axis=1
        )
        second_map = padded.at[:8, :].set(second_map)
    return (
        bank,
        route_map,
        second_map,
        jnp.asarray(second_map_is_compact, dtype=jnp.bool_),
        [bank_path, route_path, second_path],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-names", default="")
    parser.add_argument("--seed-start", type=int, default=1_360_001)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--group",
        choices=("old_a", "old_b", "old_c", "dynamic", "high", "legacy", "latest"),
        required=True,
    )
    parser.add_argument(
        "--opponent-ids",
        default="",
        help=(
            "Optional comma-separated global roster IDs from the selected group. "
            "A single closed-over ID lets XLA prune unused group policies."
        ),
    )
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument(
        "--rollout-mode", choices=("stepwise", "scan"), default="scan"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

    cache = ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache_old28"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    if args.candidate_names:
        requested = {value.strip() for value in args.candidate_names.split(",") if value.strip()}
        candidates = [row for row in candidates if row["name"] in requested]
        missing = requested - {row["name"] for row in candidates}
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
    if not candidates:
        raise ValueError("no candidates selected")

    old_receipt = json.loads(OLD_RECEIPT.read_text(encoding="utf-8"))
    old_bank = load_bank(OLD_BANK)
    latest_bank = load_bank(LATEST_BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)
    router = build_router_arrays(old_receipt)
    tables = load_tables()
    boatlee_trace = load_boatlee_trace_v1()
    old_id_lut = jnp.asarray(
        np.concatenate(
            (rr.OLD_IDS, np.zeros((len(rr.ROSTER) - len(rr.OLD_IDS),), dtype=np.int32))
        )
    )
    _advance, group_action = rr.make_group_advance(
        args.group,
        args.group,
        old_bank,
        latest_bank,
        runtime,
        tables,
        router,
        boatlee_trace,
        old_id_lut,
        return_group_action=True,
    )

    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    group_opponent_ids = np.asarray(
        [index for index in range(len(rr.ROSTER)) if rr.GROUP_BY_ID[index] == args.group],
        dtype=np.int32,
    )
    if args.opponent_ids:
        requested_opponent_ids = np.asarray(
            [int(value.strip()) for value in args.opponent_ids.split(",") if value.strip()],
            dtype=np.int32,
        )
        invalid = sorted(set(requested_opponent_ids.tolist()) - set(group_opponent_ids.tolist()))
        if invalid:
            raise ValueError(
                f"opponent IDs {invalid} do not belong to group {args.group}; "
                f"valid={group_opponent_ids.tolist()}"
            )
        selected_opponent_ids = requested_opponent_ids
    else:
        selected_opponent_ids = group_opponent_ids
    opponent_count = int(selected_opponent_ids.size)
    if not opponent_count:
        raise AssertionError(f"empty group: {args.group}")
    opponent_ids_np = np.repeat(selected_opponent_ids, args.seeds)
    lane_seeds = np.tile(base_seeds, opponent_count)
    opponent_ids = jnp.asarray(opponent_ids_np)
    weed, shops = build_events_v1(lane_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(lane_seeds))

    step_functions = {}
    scan_functions = {}
    for candidate_seat in (0, 1):
        rival = 1 - candidate_seat

        @jax.jit
        def one_step(
            states,
            trace_carry,
            opponent_carry,
            trace_bank,
            route_map,
            second_map,
            second_map_is_compact,
            *,
            _trace_seat=candidate_seat,
            _rival=rival,
        ):
            rival_action, opponent_carry = group_action(
                args.group, states, opponent_ids, opponent_carry, _rival
            )
            states, trace_carry, _, _ = hasegawa_step_with_external_v3(
                states,
                trace_carry,
                trace_bank,
                rival_action,
                _trace_seat,
                events,
                tables,
                ROUTER_TWO_SHOP_COMPATIBLE_MAP,
                route_map,
                15_000,
                5_000,
                2,
                second_shop_route_map=second_map,
                second_shop_map_is_compact=second_map_is_compact,
            )
            return states, trace_carry, opponent_carry

        step_functions[candidate_seat] = one_step

        @jax.jit
        def full_rollout(
            states,
            trace_carry,
            opponent_carry,
            trace_bank,
            route_map,
            second_map,
            second_map_is_compact,
            *,
            _step=one_step,
        ):
            def body(value, _):
                current_states, current_trace, current_opponent = value
                return _step(
                    current_states,
                    current_trace,
                    current_opponent,
                    trace_bank,
                    route_map,
                    second_map,
                    second_map_is_compact,
                ), None

            return jax.lax.scan(
                body, (states, trace_carry, opponent_carry), None, length=719
            )[0]

        scan_functions[candidate_seat] = full_rollout

    rows = []
    input_paths = {manifest_path, OLD_BANK, OLD_RECEIPT, LATEST_BANK, RUNTIME, Path(__file__).resolve(), Path(rr.__file__).resolve()}
    compile_seconds = {}
    started = perf_counter()
    for candidate_index, spec in enumerate(candidates):
        bank, route_map, second_map, second_map_is_compact, paths = load_candidate(
            spec, args.route_capacity
        )
        input_paths.update(path for path in paths if path is not None)
        seat_money = []
        for candidate_seat in (0, 1):
            states = initial
            trace_carry = initialize_hasegawa_carry_v3(states.step.shape[0], bank.bootstrap_route_id)
            opponent_carry = rr.initialize_group_carry(
                args.group, opponent_ids, old_id_lut, router
            )
            before = perf_counter()
            if args.rollout_mode == "scan":
                states, trace_carry, opponent_carry = scan_functions[candidate_seat](
                    states,
                    trace_carry,
                    opponent_carry,
                    bank,
                    route_map,
                    second_map,
                    second_map_is_compact,
                )
            else:
                for _ in range(719):
                    states, trace_carry, opponent_carry = step_functions[candidate_seat](
                        states,
                        trace_carry,
                        opponent_carry,
                        bank,
                        route_map,
                        second_map,
                        second_map_is_compact,
                    )
            jax.block_until_ready(states.money)
            elapsed = perf_counter() - before
            if candidate_index == 0:
                compile_seconds[str(candidate_seat)] = elapsed
            terminal, final_carry = jax.device_get((states, trace_carry))
            if not bool(np.all(terminal.done)):
                raise AssertionError(f"{spec['name']} seat{candidate_seat}: unfinished")
            hard = {
                "hand_cap_hits": int(np.sum(terminal.hand_cap_hits)),
                "market_loop_cap_hits": int(np.sum(terminal.market_loop_cap_hits)),
                "price_lut_oob": int(np.sum(terminal.price_lut_oob)),
                "trace_hard_counter": int(np.sum(final_carry.hard_counter_total)),
            }
            if any(hard.values()):
                raise AssertionError(f"{spec['name']} seat{candidate_seat}: {hard}")
            money = np.asarray(terminal.money, dtype=np.int64).reshape(opponent_count, args.seeds, 2)
            seat_money.append(money)
            print(
                json.dumps(
                    {
                        "candidate": spec["name"],
                        "seat": candidate_seat,
                        "elapsed_seconds": elapsed,
                        "transitions_per_second": int(lane_seeds.size * 719 / elapsed),
                    }
                ),
                flush=True,
            )

        first, second = seat_money
        for local_id, global_id in enumerate(selected_opponent_ids.tolist()):
            opponent_spec = rr.ROSTER[global_id]
            own = np.concatenate((first[local_id, :, 0], second[local_id, :, 1]))
            rival_cash = np.concatenate((first[local_id, :, 1], second[local_id, :, 0]))
            margins = own - rival_cash
            rows.append(
                {
                    "candidate": spec["name"],
                    "family": spec["family"],
                    "opponent_id": global_id,
                    "opponent": opponent_spec["name"],
                    "games": int(margins.size),
                    "wins": int(np.sum(margins > 0)),
                    "ties": int(np.sum(margins == 0)),
                    "losses": int(np.sum(margins < 0)),
                    "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                    "mean_margin": float(np.mean(margins)),
                    "candidate_cash": own.astype(int).tolist(),
                    "opponent_cash": rival_cash.astype(int).tolist(),
                }
            )

    aggregates = []
    for spec in candidates:
        subset = [row for row in rows if row["candidate"] == spec["name"]]
        aggregates.append(
            {
                "candidate": spec["name"],
                "opponents": len(subset),
                "games": sum(row["games"] for row in subset),
                "mean_score_rate": float(np.mean([row["score_rate"] for row in subset])),
                "min_score_rate": min(row["score_rate"] for row in subset),
                "worst_opponent": min(subset, key=lambda row: row["score_rate"])["opponent"],
                "opponents_at_or_above_90pct": sum(row["score_rate"] >= 0.90 for row in subset),
            }
        )

    input_paths.update(loaded_project_python_sources())
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-backbones-vs-old28-batched.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "group": args.group,
        "selected_opponent_ids": selected_opponent_ids.astype(int).tolist(),
        "games_per_opponent": args.seeds * 2,
        "rollout_mode": args.rollout_mode,
        "candidate_count": len(candidates),
        "opponent_count": opponent_count,
        "compile_plus_first_candidate_seconds_by_seat": compile_seconds,
        "runtime_fingerprint": {
            "python_version": sys.version,
            "jax_version": importlib.metadata.version("jax"),
            "jaxlib_version": importlib.metadata.version("jaxlib"),
            "events": {
                "weed_sha256": sha256_array(weed),
                "shops_sha256": sha256_array(shops),
            },
            "files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
                for path in sorted(input_paths, key=lambda value: str(value))
            },
        },
        "aggregates": aggregates,
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "aggregates": aggregates}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
