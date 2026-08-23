#!/usr/bin/env python3
"""Screen every strict JAX Agent against FC2B on triple-PET event seeds.

This is a conditional capability screen, not a global ranking.  It reuses the
independently frozen 64K event scan and therefore never selects seeds from the
candidate outcomes produced here.
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
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MarketOp,
    PRODUCTS,
    load_high_potential_runtime_tables_v1,
)


CARROT_ID = PRODUCTS.index("CARROT")


def parse_ids(raw: str) -> list[int]:
    if not raw.strip():
        return list(range(len(rr.ROSTER)))
    result = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not result or any(value < 0 or value >= len(rr.ROSTER) for value in result):
        raise ValueError("--agent-ids contains an invalid roster id")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rare-scan-receipt", type=Path, required=True)
    parser.add_argument("--agent-ids", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    scan = json.loads(args.rare_scan_receipt.read_text(encoding="utf-8"))
    public_filter = scan.get("public_filter", {})
    if public_filter != {"shop": "PET_CAFE", "prefix_length": 3}:
        raise ValueError(f"receipt is not the frozen triple-PET scan: {public_filter}")
    seeds = np.asarray(
        sorted({int(row["seed"]) for row in scan.get("rare_games", [])}),
        dtype=np.int32,
    )
    if seeds.size < 100:
        raise ValueError(f"expected at least 100 independent triple-PET seeds, got {seeds.size}")
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(seeds.size)

    old_receipt = json.loads((
        ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    ).read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(
            ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
        ),
        "latest_bank": load_bank(
            ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
        ),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    simulator = rr.make_simulator_step(resources["tables"])
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    agent_ids = parse_ids(args.agent_ids)

    def make_fc2b_policy(player: int):
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

    fc2b_policies = (make_fc2b_policy(1), make_fc2b_policy(0))
    rows: list[dict] = []
    started = perf_counter()
    for ordinal, agent_id in enumerate(agent_ids, start=1):
        name = str(rr.ROSTER[agent_id]["name"])
        orientations = []
        agent_started = perf_counter()
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = rr.initialize_agent_carry(
                agent_id, batch, resources["router"]
            )
            fc2b_carry = initialize_fusion_champion_carry_v3(batch)
            candidate_policy = rr.make_agent_policy(
                agent_id, candidate_seat, **resources
            )
            fc2b_policy = fc2b_policies[candidate_seat]
            carrot_sell_intents = jnp.zeros((batch,), dtype=jnp.int32)
            peak_carrot_tiles = jnp.zeros((batch,), dtype=jnp.int16)
            for _ in range(719):
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                fc2b_action, fc2b_carry = fc2b_policy(states, fc2b_carry)
                active = (
                    jnp.arange(candidate_action.market_op.shape[1])[None, :]
                    < candidate_action.market_count[:, None]
                )
                carrot_sell_intents += jnp.sum(
                    jnp.where(
                        active
                        & (candidate_action.market_op == MarketOp.SELL)
                        & (candidate_action.market_item == CARROT_ID),
                        jnp.maximum(candidate_action.market_amount, 0),
                        0,
                    ),
                    axis=1,
                ).astype(jnp.int32)
                peak_carrot_tiles = jnp.maximum(
                    peak_carrot_tiles,
                    jnp.sum(
                        states.tile_crop[:, candidate_seat] == CARROT_ID,
                        axis=(1, 2),
                    ).astype(jnp.int16),
                )
                states = (
                    simulator(states, candidate_action, fc2b_action, events)
                    if candidate_seat == 0
                    else simulator(states, fc2b_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            safety = {
                field: int(np.sum(np.asarray(getattr(terminal, field))))
                for field in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            }
            if any(safety.values()):
                raise AssertionError(f"simulator safety counter hit: {safety}")
            money = np.asarray(terminal.money, dtype=np.int64)
            own = money[:, candidate_seat]
            rival = money[:, 1 - candidate_seat]
            animal_count = np.sum(
                np.asarray(terminal.tile_animal[:, candidate_seat]) >= 0,
                axis=(1, 2),
            )
            orientations.append(
                {
                    "own": own,
                    "rival": rival,
                    "carrot_sell_intents": np.asarray(jax.device_get(carrot_sell_intents)),
                    "peak_carrot_tiles": np.asarray(jax.device_get(peak_carrot_tiles)),
                    "terminal_animals": animal_count,
                }
            )

        own = np.concatenate([value["own"] for value in orientations])
        rival = np.concatenate([value["rival"] for value in orientations])
        margins = own - rival
        carrot_sells = np.concatenate(
            [value["carrot_sell_intents"] for value in orientations]
        )
        carrot_tiles = np.concatenate(
            [value["peak_carrot_tiles"] for value in orientations]
        )
        animals = np.concatenate(
            [value["terminal_animals"] for value in orientations]
        )
        row = {
            "agent_id": agent_id,
            "agent": name,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_cash": float(np.mean(own)),
            "mean_fc2b_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "seat0_score_rate": float(
                np.mean(orientations[0]["own"] > orientations[0]["rival"])
                + 0.5 * np.mean(orientations[0]["own"] == orientations[0]["rival"])
            ),
            "seat1_score_rate": float(
                np.mean(orientations[1]["own"] > orientations[1]["rival"])
                + 0.5 * np.mean(orientations[1]["own"] == orientations[1]["rival"])
            ),
            "mean_carrot_sell_intents": float(np.mean(carrot_sells)),
            "mean_peak_carrot_tiles": float(np.mean(carrot_tiles)),
            "mean_terminal_animals": float(np.mean(animals)),
            "elapsed_seconds": perf_counter() - agent_started,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{ordinal}/{len(agent_ids)}",
                    **row,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    rows.sort(key=lambda row: (row["score_rate"], row["mean_margin"]), reverse=True)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc8-triple-pet-exact-agent-panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Conditional triple-PET capability screen against frozen FC2B; "
            "not a global Arena ranking and not sufficient for deployment."
        ),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "rare_scan_receipt": str(args.rare_scan_receipt.resolve()),
        "public_filter": public_filter,
        "independent_seeds": batch,
        "games_per_agent": batch * 2,
        "seat_protocol": "same frozen seeds with seats swapped",
        "opponent": "fc2b_rank14_plus_clone_aware_preempt",
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
