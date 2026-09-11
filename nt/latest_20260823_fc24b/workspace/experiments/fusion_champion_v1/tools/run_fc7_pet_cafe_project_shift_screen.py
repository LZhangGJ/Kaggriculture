#!/usr/bin/env python3
"""Vectorized screen for the FC7 public-demand project-shift rule."""

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
    fc7_pet_cafe_project_shift_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


DEFAULT_OPPONENTS = ("rayk_k320_adaptive_rank1",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def config_grid() -> list[dict[str, int | str]]:
    configs: list[dict[str, int | str]] = [
        {
            "name": "source_fc2b",
            "seed_switch_step": 999,
            "plant_switch_step": 999,
            "sale_switch_step": 999,
            "sale_mode": 0,
            "animal_mode": 0,
            "animal_switch_step": 999,
            "goose_cap": 0,
            "feed_mode": 0,
            "feed_switch_step": 999,
            "release_step": 999,
        }
    ]
    for seed_step in (312, 336):
        for plant_step in (336, 360):
            for sale_mode, sale_step in ((0, 999), (1, 432), (1, 456)):
                configs.append(
                    {
                        "name": f"crop_s{seed_step}_p{plant_step}_sale{sale_mode}_{sale_step}",
                        "seed_switch_step": seed_step,
                        "plant_switch_step": plant_step,
                        "sale_switch_step": sale_step,
                        "sale_mode": sale_mode,
                        "animal_mode": 0,
                        "animal_switch_step": 999,
                        "goose_cap": 0,
                        "feed_mode": 0,
                        "feed_switch_step": 999,
                        "release_step": 999,
                    }
                )
    for animal_mode, goose_cap in ((0, 0), (1, 2), (1, 4)):
        for feed_mode, feed_step, release_step in (
            (0, 999, 999),
            (1, 336, 999),
            (1, 360, 999),
            (2, 999, 528),
            (3, 336, 528),
            (3, 360, 528),
        ):
            configs.append(
                {
                    "name": (
                        f"combo_animal{animal_mode}_g{goose_cap}"
                        f"_feed{feed_mode}_{feed_step}_release{release_step}"
                    ),
                    "seed_switch_step": 336,
                    "plant_switch_step": 360,
                    "sale_switch_step": 432,
                    "sale_mode": 1,
                    "animal_mode": animal_mode,
                    "animal_switch_step": 216,
                    "goose_cap": goose_cap,
                    "feed_mode": feed_mode,
                    "feed_switch_step": feed_step,
                    "release_step": release_step,
                }
            )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument(
        "--config-indices",
        default="",
        help="Optional comma-separated indices from the full config grid.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    seeds = np.asarray(
        [int(value.strip()) for value in args.seed_list.split(",") if value.strip()],
        dtype=np.int32,
    )
    if seeds.size < 1 or len(set(seeds.tolist())) != int(seeds.size):
        raise ValueError("--seed-list must contain unique seeds")

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
    opponents = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not opponents or any(value not in names for value in opponents):
        raise ValueError("unknown or empty opponent list")

    all_configs = config_grid()
    if args.config_indices:
        indices = [
            int(value.strip())
            for value in args.config_indices.split(",")
            if value.strip()
        ]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("--config-indices must contain unique indices")
        if any(index < 0 or index >= len(all_configs) for index in indices):
            raise ValueError("--config-indices contains an out-of-range index")
        configs = [all_configs[index] for index in indices]
    else:
        indices = list(range(len(all_configs)))
        configs = all_configs
    config_count = len(configs)
    seed_count = int(seeds.size)
    expanded_seeds = np.tile(seeds, config_count)
    weed, shops = build_events_v1(expanded_seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    parameters = {
        key: jnp.asarray(
            np.repeat([int(config[key]) for config in configs], seed_count),
            dtype=jnp.int32,
        )
        for key in (
            "seed_switch_step",
            "plant_switch_step",
            "sale_switch_step",
            "sale_mode",
            "animal_mode",
            "animal_switch_step",
            "goose_cap",
            "feed_mode",
            "feed_switch_step",
            "release_step",
        )
    }
    simulator = rr.make_simulator_step(resources["tables"])
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in opponents:
        opponent_id = names.index(opponent_name)
        orientations = []
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
            candidate_carry = initialize_fusion_champion_carry_v3(expanded_seeds.size)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, expanded_seeds.size, resources["router"]
            )
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )

            @jax.jit
            def candidate_policy(
                current,
                carry,
                seed_switch_step,
                plant_switch_step,
                sale_switch_step,
                sale_mode,
                animal_mode,
                animal_switch_step,
                goose_cap,
                feed_mode,
                feed_switch_step,
                release_step,
            ):
                return fc7_pet_cafe_project_shift_player_action_v1(
                    current,
                    resources["tables"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    resources["runtime"],
                    carry,
                    seed_switch_step,
                    plant_switch_step,
                    sale_switch_step,
                    sale_mode,
                    animal_mode,
                    animal_switch_step,
                    goose_cap,
                    feed_mode,
                    feed_switch_step,
                    release_step,
                    candidate_seat,
                )

            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states,
                    candidate_carry,
                    parameters["seed_switch_step"],
                    parameters["plant_switch_step"],
                    parameters["sale_switch_step"],
                    parameters["sale_mode"],
                    parameters["animal_mode"],
                    parameters["animal_switch_step"],
                    parameters["goose_cap"],
                    parameters["feed_mode"],
                    parameters["feed_switch_step"],
                    parameters["release_step"],
                )
                opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
                if candidate_seat == 0:
                    states = simulator(states, candidate_action, opponent_action, events)
                else:
                    states = simulator(states, opponent_action, candidate_action, events)
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
            orientations.append(terminal)

        for config_index, config in enumerate(configs):
            start = config_index * seed_count
            stop = start + seed_count
            game_rows = []
            for candidate_seat, terminal in enumerate(orientations):
                money = np.asarray(terminal.money[start:stop], dtype=np.int64)
                town_count = np.asarray(terminal.town_count[start:stop], dtype=np.int16)
                town_shops = np.asarray(terminal.town_shops[start:stop], dtype=np.int16)
                for index, seed in enumerate(seeds.tolist()):
                    own = int(money[index, candidate_seat])
                    rival = int(money[index, 1 - candidate_seat])
                    count = int(town_count[index])
                    ids = town_shops[index, :count].astype(int).tolist()
                    game_rows.append(
                        {
                            "seed": int(seed),
                            "candidate_seat": candidate_seat,
                            "candidate_cash": own,
                            "opponent_cash": rival,
                            "margin": own - rival,
                            "town_shops": [SHOP_NAMES[value] for value in ids],
                        }
                    )
            margins = np.asarray([row["margin"] for row in game_rows], dtype=np.int64)
            own_cash = np.asarray([row["candidate_cash"] for row in game_rows], dtype=np.int64)
            triple = np.asarray(
                [row["town_shops"][:3] == ["PET_CAFE"] * 3 for row in game_rows],
                dtype=bool,
            )
            rows.append(
                {
                    "opponent": opponent_name,
                    "config": config,
                    "games": int(margins.size),
                    "wins": int(np.sum(margins > 0)),
                    "ties": int(np.sum(margins == 0)),
                    "losses": int(np.sum(margins < 0)),
                    "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                    "mean_candidate_cash": float(np.mean(own_cash)),
                    "mean_margin": float(np.mean(margins)),
                    "triple_pet_games": int(np.sum(triple)),
                    "triple_pet_score_rate": float(
                        np.mean(margins[triple] > 0) + 0.5 * np.mean(margins[triple] == 0)
                    ) if np.any(triple) else None,
                    "triple_pet_mean_cash": float(np.mean(own_cash[triple])) if np.any(triple) else None,
                    "triple_pet_mean_margin": float(np.mean(margins[triple])) if np.any(triple) else None,
                    "per_game": game_rows,
                }
            )
        print(
            json.dumps(
                {
                    "opponent": opponent_name,
                    "configs": config_count,
                    "games_per_config": seed_count * 2,
                }
            ),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.fusion_champion.fc7-pet-cafe-project-shift-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": "Public-shop rule ablation; no opponent identity or future shop input.",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "seed_values": seeds.astype(int).tolist(),
        "full_config_count": len(all_configs),
        "selected_config_indices": indices,
        "seat_protocol": "same seeds with seats swapped",
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (old_bank_path, old_receipt_path, latest_bank_path, runtime_path)
        ],
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
