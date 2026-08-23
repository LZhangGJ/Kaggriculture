#!/usr/bin/env python3
"""Compare FC2B and frozen Rank14 trajectories on paired PRT rescue seeds."""

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
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/fusion_champion_v1/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from diagnose_pair_trajectories import extract_snapshot  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import CROPS, ANIMALS, PRODUCTS  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


SNAPSHOT_AFTER_ACTION = tuple(sorted(set((0, 1, 7, 15) + tuple(range(23, 719, 24)) + (718,))))


def roster_policy(agent_id: int, player: int, resources: dict):
    return rr.make_agent_policy(
        agent_id,
        player,
        old_bank=resources["old_bank"],
        latest_bank=resources["latest_bank"],
        runtime=resources["runtime"],
        tables=resources["tables"],
        router=resources["router"],
        boatlee_trace=resources["boatlee_trace"],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def initialize_candidate(kind: str, batch: int, router):
    if kind == "fc2b":
        return initialize_fusion_champion_carry_v3(batch)
    if kind == "rank14":
        rank14_id = next(
            index
            for index, row in enumerate(rr.ROSTER)
            if row["name"] == "gold_proxy_rank14_recursion"
        )
        return rr.initialize_agent_carry(rank14_id, batch, router)
    raise ValueError(kind)


def make_candidate_policy(kind: str, player: int, resources: dict):
    if kind == "rank14":
        return roster_policy(resources["rank14_id"], player, resources)

    @jax.jit
    def policy(states, carry):
        return fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
            states,
            resources["tables"],
            resources["latest_bank"],
            resources["old_bank"],
            resources["runtime"],
            carry,
            player,
        )

    return policy


def run_orientation(
    kind: str,
    candidate_seat: int,
    seeds: np.ndarray,
    events: Events,
    resources: dict,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, dict[str, np.ndarray]]]]:
    batch = len(seeds)
    opponent_id = resources["prt_id"]
    states = jax.vmap(reset)(jnp.asarray(seeds))
    candidate_carry = initialize_candidate(kind, batch, resources["router"])
    opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
    candidate_policy = make_candidate_policy(kind, candidate_seat, resources)
    opponent_policy = roster_policy(opponent_id, 1 - candidate_seat, resources)
    simulator = rr.make_simulator_step(resources["tables"])
    snapshots: list[tuple[int, dict[str, np.ndarray]]] = []

    for action_index in range(719):
        candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
        opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
        if candidate_seat == 0:
            states = simulator(states, candidate_action, opponent_action, events)
        else:
            states = simulator(states, opponent_action, candidate_action, events)
        if action_index in SNAPSHOT_AFTER_ACTION:
            snapshot = extract_snapshot(states, own=candidate_seat)
            if kind == "fc2b":
                snapshot["route_own"] = np.asarray(
                    jax.device_get(candidate_carry.k320.ray_route_id), dtype=np.int16
                )
                snapshot["route_locked_own"] = np.asarray(
                    jax.device_get(candidate_carry.k320.ray_route_locked), dtype=bool
                )
            else:
                snapshot["route_own"] = np.full((batch,), 63, dtype=np.int16)
                snapshot["route_locked_own"] = np.ones((batch,), dtype=bool)
            snapshot["route_rival"] = np.full((batch,), opponent_id, dtype=np.int16)
            snapshots.append((action_index, snapshot))

    jax.block_until_ready(states.money)
    terminal = jax.device_get(states)
    if not bool(np.all(np.asarray(terminal.done))):
        raise AssertionError(f"{kind} seat{candidate_seat}: not all games DONE")
    if any(
        int(np.sum(np.asarray(value)))
        for value in (terminal.hand_cap_hits, terminal.market_loop_cap_hits, terminal.price_lut_oob)
    ):
        raise AssertionError(f"{kind} seat{candidate_seat}: simulator safety counter hit")
    money = np.asarray(terminal.money, dtype=np.int64)
    return money[:, candidate_seat], money[:, 1 - candidate_seat], snapshots


def serialize_game(
    kind: str,
    seed: int,
    candidate_seat: int,
    candidate_cash: int,
    opponent_cash: int,
    snapshots: list[tuple[int, dict[str, np.ndarray]]],
    index: int,
) -> dict:
    days = []
    for action_index, snapshot in snapshots:
        town_count = int(snapshot["town_count"][index])
        days.append(
            {
                "after_action_index": int(action_index),
                "state_step": int(action_index + 1),
                "money_own": int(snapshot["money_own"][index]),
                "money_rival": int(snapshot["money_rival"][index]),
                "hands_own": int(snapshot["hands_own"][index]),
                "hands_rival": int(snapshot["hands_rival"][index]),
                "unlocked_own": int(snapshot["unlocked_own"][index]),
                "unlocked_rival": int(snapshot["unlocked_rival"][index]),
                "tile_kinds_own": snapshot["tile_kinds_own"][index].astype(int).tolist(),
                "tile_kinds_rival": snapshot["tile_kinds_rival"][index].astype(int).tolist(),
                "crops_own": snapshot["crops_own"][index].astype(int).tolist(),
                "crops_rival": snapshot["crops_rival"][index].astype(int).tolist(),
                "animals_own": snapshot["animals_own"][index].astype(int).tolist(),
                "animals_rival": snapshot["animals_rival"][index].astype(int).tolist(),
                "crop_yield_own": snapshot["crop_yield_own"][index].astype(int).tolist(),
                "crop_yield_rival": snapshot["crop_yield_rival"][index].astype(int).tolist(),
                "animal_yield_own": snapshot["animal_yield_own"][index].astype(int).tolist(),
                "animal_yield_rival": snapshot["animal_yield_rival"][index].astype(int).tolist(),
                "shed_own": snapshot["shed_own"][index].astype(int).tolist(),
                "shed_animals_own": snapshot["shed_animals_own"][index].astype(int).tolist(),
                "seeds_own": snapshot["seeds_own"][index].astype(int).tolist(),
                "market_price": snapshot["market_price"][index].astype(int).tolist(),
                "market_inventory": snapshot["market_inventory"][index].astype(int).tolist(),
                "town_shops": snapshot["town_shops"][index, :town_count].astype(int).tolist(),
                "route_own_diagnostic": int(snapshot["route_own"][index]),
                "route_locked_own_diagnostic": bool(snapshot["route_locked_own"][index]),
            }
        )
    margin = candidate_cash - opponent_cash
    return {
        "candidate": kind,
        "seed": int(seed),
        "candidate_seat": int(candidate_seat),
        "candidate_cash": int(candidate_cash),
        "opponent_cash": int(opponent_cash),
        "margin": int(margin),
        "result": "win" if margin > 0 else ("tie" if margin == 0 else "loss"),
        "daily": days,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/receipts/fc10q_fc2b_rank14_prt_rescue_manifest_seed580001_n512x2_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/receipts/fc10r_fc2b_rank14_prt_rescue_trajectories_v1.json",
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = np.asarray(manifest["rescue_unique_seeds"], dtype=np.int32)
    if len(seeds) < 1:
        raise AssertionError("rescue manifest contains no seeds")
    rescue_keys = {
        (int(row["seed"]), int(row["candidate_seat"]))
        for row in manifest["categories"]["fc2b_loss_rank14_win"]
    }

    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(old_bank_path),
        "latest_bank": load_bank(latest_bank_path),
        "runtime": load_high_potential_runtime_tables_v1(runtime_path),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
        "rank14_id": next(
            index
            for index, row in enumerate(rr.ROSTER)
            if row["name"] == "gold_proxy_rank14_recursion"
        ),
        "prt_id": next(
            index for index, row in enumerate(rr.ROSTER) if row["name"] == "local_prt_v6"
        ),
    }
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    started = perf_counter()
    rows = []
    summaries = []
    for kind in ("fc2b", "rank14"):
        all_margins = []
        rescue_margins = []
        for candidate_seat in (0, 1):
            own_cash, rival_cash, snapshots = run_orientation(
                kind, candidate_seat, seeds, events, resources
            )
            margins = own_cash - rival_cash
            all_margins.extend(margins.tolist())
            for index, seed in enumerate(seeds.tolist()):
                if (seed, candidate_seat) not in rescue_keys:
                    continue
                rescue_margins.append(int(margins[index]))
                rows.append(
                    serialize_game(
                        kind,
                        seed,
                        candidate_seat,
                        int(own_cash[index]),
                        int(rival_cash[index]),
                        snapshots,
                        index,
                    )
                )
        rescue_array = np.asarray(rescue_margins, dtype=np.int64)
        summaries.append(
            {
                "candidate": kind,
                "all_seed_seat_games": len(all_margins),
                "rescue_games": len(rescue_margins),
                "rescue_wins": int(np.sum(rescue_array > 0)),
                "rescue_score_rate": float(np.mean(rescue_array > 0) + 0.5 * np.mean(rescue_array == 0)),
                "rescue_mean_margin": float(np.mean(rescue_array)),
            }
        )
        print(json.dumps(summaries[-1]), flush=True)

    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b_rank14_prt_capability_trajectory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": "offline counterfactual capability audit; route identity is not an online feature",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "crop_order": list(CROPS),
        "animal_order": list(ANIMALS),
        "product_order": list(PRODUCTS),
        "manifest": {"path": str(args.manifest.relative_to(ROOT)), "sha256": sha256(args.manifest)},
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (old_bank_path, old_receipt_path, latest_bank_path, runtime_path)
        ],
        "seed_count": len(seeds),
        "seeds": seeds.astype(int).tolist(),
        "rescue_game_count": len(rescue_keys),
        "snapshot_after_action_indices": list(SNAPSHOT_AFTER_ACTION),
        "summaries": summaries,
        "per_game": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
