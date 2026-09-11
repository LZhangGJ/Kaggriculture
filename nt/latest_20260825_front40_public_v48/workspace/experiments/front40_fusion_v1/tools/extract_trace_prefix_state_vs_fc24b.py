"""Extract legal runtime state at a trace router decision point versus FC24B."""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def make_rollout(
    trace_player,
    tables,
    latest_bank,
    old_bank,
    runtime,
    bootstrap,
    decision_step,
    first_shop_routes,
):
    fc_player = 1 - trace_player

    @jax.jit
    def rollout(states, trace_carry, fc_carry, events, trace_bank):
        batch = states.step.shape[0]

        def body(value, _):
            current, trace_value, fc_value = value
            if first_shop_routes is None:
                forced = jnp.full((batch,), bootstrap, dtype=jnp.int16)
                unlocked = jnp.zeros((batch,), dtype=jnp.bool_)
            else:
                unlocked = current.town_count > 0
                first_shop = jnp.clip(current.town_shops[:, 0], 0, 7)
                mapped = first_shop_routes[first_shop]
                forced = jnp.where(unlocked, mapped, trace_value.branch_id).astype(jnp.int16)
            fc_action, fc_value = fc24_terminal_crop_salvage_player_action_v1(
                current, tables, latest_bank, old_bank, runtime, fc_value, fc_player
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
                forced_route=forced,
                forced_lock=unlocked,
            )
            return (current, trace_value, fc_value), None

        return jax.lax.scan(
            body,
            (states, trace_carry, fc_carry),
            None,
            length=decision_step,
        )

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--decision-step", type=int, default=72)
    parser.add_argument(
        "--first-shop-map",
        type=Path,
        default=None,
        help="Optional eight-route public first-shop map used before extracting state.",
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "own_money", "opponent_money", "own_tile_kind", "opponent_tile_kind",
            "own_tile_crop", "opponent_tile_crop", "own_tile_animal",
            "opponent_tile_animal", "own_tile_yield", "opponent_tile_yield",
            "own_tile_neglect", "opponent_tile_neglect", "own_tile_flags",
            "opponent_tile_flags", "own_unit_pos", "opponent_unit_pos",
            "own_unit_active", "opponent_unit_active", "own_unit_inventory",
            "own_shed", "own_seeds", "own_hires_today", "opponent_hires_today",
            "own_unlocked_count", "opponent_unlocked_count", "market_inventory",
            "market_price", "town_shops", "town_count", "invalid_intent_total",
            "resync_total",
        )
    }
    elapsed_by_seat = []
    bootstrap = int(np.asarray(jax.device_get(trace_bank.bootstrap_route_id)))
    first_shop_routes = None
    if args.first_shop_map is not None:
        payload = json.loads(args.first_shop_map.read_text(encoding="utf-8"))
        values = np.asarray(payload["route_ids"], dtype=np.int16)
        if values.shape != (8,):
            raise ValueError("first-shop map must contain eight route IDs")
        first_shop_routes = jnp.asarray(values)
    for seat in (0, 1):
        rollout = make_rollout(
            seat,
            tables,
            latest_bank,
            old_bank,
            runtime,
            bootstrap,
            args.decision_step,
            first_shop_routes,
        )
        start = time.perf_counter()
        state = initial
        carry = initialize_hasegawa_carry_v3(args.batch, bootstrap)
        fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(args.batch)
        (state, carry, fc_carry), _ = rollout(
            state, carry, fc_carry, events, trace_bank
        )
        jax.block_until_ready(state.money)
        seat_elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {"seat": seat, "stage": "prefix_state", "seconds": seat_elapsed}
            ),
            flush=True,
        )
        state, carry = jax.device_get((state, carry))
        elapsed_by_seat.append(seat_elapsed)
        opponent = 1 - seat
        fields["own_money"].append(np.asarray(state.money[:, seat]))
        fields["opponent_money"].append(np.asarray(state.money[:, opponent]))
        for name in ("tile_kind", "tile_crop", "tile_animal", "tile_yield", "tile_neglect", "tile_flags"):
            value = np.asarray(getattr(state, name))
            fields[f"own_{name}"].append(value[:, seat])
            fields[f"opponent_{name}"].append(value[:, opponent])
        fields["own_unit_pos"].append(np.asarray(state.unit_pos[:, seat]))
        fields["opponent_unit_pos"].append(np.asarray(state.unit_pos[:, opponent]))
        fields["own_unit_active"].append(np.asarray(state.unit_active[:, seat]))
        fields["opponent_unit_active"].append(np.asarray(state.unit_active[:, opponent]))
        fields["own_unit_inventory"].append(np.asarray(state.unit_inventory[:, seat]))
        fields["own_shed"].append(np.asarray(state.shed[:, seat]))
        fields["own_seeds"].append(np.asarray(state.seeds[:, seat]))
        fields["own_hires_today"].append(np.asarray(state.hires_today[:, seat]))
        fields["opponent_hires_today"].append(np.asarray(state.hires_today[:, opponent]))
        fields["own_unlocked_count"].append(np.asarray(state.unlocked_count[:, seat]))
        fields["opponent_unlocked_count"].append(np.asarray(state.unlocked_count[:, opponent]))
        for name in ("market_inventory", "market_price", "town_shops", "town_count"):
            fields[name].append(np.asarray(getattr(state, name)))
        fields["invalid_intent_total"].append(np.asarray(carry.invalid_intent_total))
        fields["resync_total"].append(np.asarray(carry.resync_total))

    arrays = {name: np.stack(value, axis=0) for name, value in fields.items()}
    arrays.update(
        seeds=seeds,
        candidate_seat=np.broadcast_to(np.asarray((0, 1), np.int8)[:, None], (2, args.batch)),
        decision_step=np.asarray(args.decision_step, np.int16),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    receipt = {
        "schema": "kaggriculture.front40_fusion.trace-prefix-state-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "device": str(jax.devices()[0]),
        "trace_bank": str(args.trace_bank),
        "decision_step": args.decision_step,
        "first_shop_map": str(args.first_shop_map) if args.first_shop_map else None,
        "seed_start": args.seed_start,
        "seed_count": args.batch,
        "seat_protocol": "same seeds with candidate seats swapped",
        "runtime_feature_boundary": "public state plus own private inventory only",
        "elapsed_seconds_by_seat": elapsed_by_seat,
        "invalid_intent_total": int(np.sum(arrays["invalid_intent_total"])),
        "resync_total": int(np.sum(arrays["resync_total"])),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "elapsed": elapsed_by_seat}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
