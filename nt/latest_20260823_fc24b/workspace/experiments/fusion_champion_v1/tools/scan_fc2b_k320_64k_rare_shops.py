#!/usr/bin/env python3
"""Scan the frozen 64K event bank for rare public shop prefixes.

The policy is unchanged FC2B versus frozen K320.  Only public terminal shop
state and terminal money are retained, so the receipt stays compact.
"""

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
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-bank-dir",
        type=Path,
        default=(
            ROOT
            / "experiments/strategic_v4/artifacts/shared_event_banks/training_events_64k_v1"
        ),
    )
    parser.add_argument("--shop", default="PET_CAFE", choices=tuple(SHOP_NAMES))
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument(
        "--filter-mode",
        choices=("exact-repeat", "kobe-goose-prebuild", "kobe-goose-eligible"),
        default="exact-repeat",
        help=(
            "exact-repeat preserves the original repeated-shop scan. "
            "kobe-goose-prebuild selects the causally available first-two-shop "
            "condition. kobe-goose-eligible additionally excludes third-position YARN."
        ),
    )
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.prefix_length < 1:
        raise ValueError("--prefix-length must be positive")

    shard_paths = sorted(args.event_bank_dir.glob("events_*.npz"))
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"no event shards under {args.event_bank_dir}")

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
    }
    names = [row["name"] for row in rr.ROSTER]
    opponent_name = "rayk_k320_adaptive_rank1"
    opponent_id = names.index(opponent_name)
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
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

    candidate_policies = (make_candidate_policy(0), make_candidate_policy(1))
    opponent_policies = tuple(
        rr.make_agent_policy(opponent_id, 1 - seat, **resources) for seat in (0, 1)
    )
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    shop_id = SHOP_NAMES.index(args.shop)
    rare_games: list[dict] = []
    total_games = 0
    scanned_seeds = 0
    started = perf_counter()
    shard_receipts = []

    for shard_index, shard_path in enumerate(shard_paths):
        with np.load(shard_path, allow_pickle=False) as data:
            seeds = np.asarray(data["event_seeds"], dtype=np.int32)
            weed = np.asarray(data["weed_spawn"], dtype=np.bool_)
            shops = np.asarray(data["shop_choice"], dtype=np.int8)
        batch = int(seeds.size)
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        shard_started = perf_counter()
        shard_rare = 0

        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = initialize_fusion_champion_carry_v3(batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, batch, resources["router"]
            )
            candidate_policy = candidate_policies[candidate_seat]
            opponent_policy = opponent_policies[candidate_seat]
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
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
                raise AssertionError("not all games DONE")
            if any(
                int(np.sum(np.asarray(value)))
                for value in (
                    terminal.hand_cap_hits,
                    terminal.market_loop_cap_hits,
                    terminal.price_lut_oob,
                )
            ):
                raise AssertionError("simulator safety counter hit")

            town_count = np.asarray(terminal.town_count, dtype=np.int16)
            town_shops = np.asarray(terminal.town_shops, dtype=np.int16)
            prefix = town_shops[:, : args.prefix_length]
            if args.filter_mode == "exact-repeat":
                mask = (town_count >= args.prefix_length) & np.all(
                    prefix == shop_id, axis=1
                )
            else:
                if args.prefix_length != 3:
                    raise ValueError(
                        "kobe-goose-eligible requires --prefix-length 3"
                    )
                yarn = SHOP_NAMES.index("YARN_STORE")
                milk_support = np.asarray(
                    [
                        SHOP_NAMES.index("PIZZA_SHOP"),
                        SHOP_NAMES.index("ICE_CREAM_SHOP"),
                        SHOP_NAMES.index("SMOOTHIE_SHOP"),
                    ],
                    dtype=np.int16,
                )
                first_two = prefix[:, :2]
                no_yarn_first_two = np.all(first_two != yarn, axis=1)
                no_milk_first_two = ~np.any(
                    first_two[..., None] == milk_support[None, None, :],
                    axis=(1, 2),
                )
                mask = (
                    (town_count >= 3)
                    & no_yarn_first_two
                    & no_milk_first_two
                )
                if args.filter_mode == "kobe-goose-eligible":
                    mask = mask & (prefix[:, 2] != yarn)
            money = np.asarray(terminal.money, dtype=np.int64)
            for index in np.flatnonzero(mask):
                own = int(money[index, candidate_seat])
                rival = int(money[index, 1 - candidate_seat])
                count = int(town_count[index])
                ids = town_shops[index, :count].astype(int).tolist()
                rare_games.append(
                    {
                        "seed": int(seeds[index]),
                        "candidate_seat": candidate_seat,
                        "candidate_cash": own,
                        "opponent_cash": rival,
                        "margin": own - rival,
                        "town_shops": [SHOP_NAMES[value] for value in ids],
                        "event_shard": shard_path.name,
                    }
                )
                shard_rare += 1

        scanned_seeds += batch
        total_games += batch * 2
        shard_receipts.append(
            {
                "path": str(shard_path.resolve()),
                "sha256": sha256(shard_path),
                "seeds": batch,
                "rare_games": shard_rare,
                "elapsed_seconds": perf_counter() - shard_started,
            }
        )
        print(
            json.dumps(
                {
                    "shard": shard_index + 1,
                    "shards": len(shard_paths),
                    "scanned_seeds": scanned_seeds,
                    "rare_games": len(rare_games),
                }
            ),
            flush=True,
        )

    rare_margins = np.asarray(
        [row["margin"] for row in rare_games], dtype=np.int64
    )
    payload = {
        "schema": "kaggriculture.fusion_champion.fc7-rare-shop-scan.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "candidate": "fc2b_rank14_plus_k320_clone_aware_preempt",
        "opponent": opponent_name,
        "public_filter": (
            {"mode": "exact-repeat", "shop": args.shop, "prefix_length": args.prefix_length}
            if args.filter_mode == "exact-repeat"
            else {
                "mode": args.filter_mode,
                "prefix_length": 3,
                "first_two_exclude": [
                    "YARN_STORE",
                    "PIZZA_SHOP",
                    "ICE_CREAM_SHOP",
                    "SMOOTHIE_SHOP",
                ],
                "third_excludes": (
                    ["YARN_STORE"]
                    if args.filter_mode == "kobe-goose-eligible"
                    else []
                ),
            }
        ),
        "scanned_seeds": scanned_seeds,
        "total_games": total_games,
        "rare_games_count": len(rare_games),
        "rare_unique_seeds": len({row["seed"] for row in rare_games}),
        "rare_score_rate": (
            float(
                np.mean(rare_margins > 0)
                + 0.5 * np.mean(rare_margins == 0)
            )
            if rare_margins.size
            else None
        ),
        "rare_mean_margin": (
            float(np.mean(rare_margins)) if rare_margins.size else None
        ),
        "elapsed_seconds": perf_counter() - started,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (
                old_bank_path,
                old_receipt_path,
                latest_bank_path,
                runtime_path,
            )
        ],
        "event_shards": shard_receipts,
        "rare_games": rare_games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rare_games": len(rare_games),
                "output": str(args.output.resolve()),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
