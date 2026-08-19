"""Trace one M3.5 acceptance candidate around feed misses and escapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.constants import FLAG_FED  # noqa: E402
from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m35_candidates import (  # noqa: E402
    m35_branch_coverage_panel_v2,
)
from project_route_search_v2.m35_controller import m35_unit_masks_v2  # noqa: E402
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
    make_m35_farm_rollout_v2,
)
from project_route_search_v2.m35_search import sample_m35_farm_genomes_v2  # noqa: E402


def _lane(tree, index: int):
    return jax.tree.map(lambda value: value[index : index + 1], tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--random-panel", action="store_true")
    args = parser.parse_args()
    if args.random_panel:
        panel = sample_m35_farm_genomes_v2(256, seed=350719)
        names = tuple(f"RANDOM_{index}" for index in range(256))
    else:
        panel, names = m35_branch_coverage_panel_v2()
    genome = _lane(panel, args.candidate)
    source_seeds, bank = load_event_bank(
        PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    )
    event_index = args.candidate % len(source_seeds)
    seed = int(np.asarray(source_seeds)[event_index])
    events = Events(
        bank.weed_spawn[event_index : event_index + 1],
        bank.shop_choice[event_index : event_index + 1],
    )
    carry = initialize_m35_rollout_carry_v2(
        jnp.asarray((seed,), dtype=jnp.int32), genome
    )
    initial = jax.device_get(carry.environment_state)
    rollout = jax.jit(make_m35_farm_rollout_v2(trace="diagnostic"))
    final, trace = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    joint, next_states, effects, controllers, hard = jax.device_get(trace)

    def state_at(step: int):
        if step == 0:
            return jax.tree.map(lambda value: np.asarray(value)[0], initial)
        return jax.tree.map(lambda value: np.asarray(value)[step - 1, 0], next_states)

    rows = []
    for step in range(719):
        pre = state_at(step)
        post_animal = np.asarray(next_states.tile_animal)[step, 0, 0]
        pre_animal = np.asarray(pre.tile_animal)[0]
        escaped = int(np.sum(pre_animal >= 0) - np.sum(post_animal >= 0))
        hard_miss = int(np.asarray(hard[0])[step, 0])
        if escaped <= 0 and hard_miss <= 0:
            continue
        context = []
        for prior in range(max(0, step - 23), step + 1):
            prior_state = state_at(prior)
            unit_count = int(np.asarray(joint.unit_count)[prior, 0, 0])
            market_count = int(np.asarray(joint.market_count)[prior, 0, 0])
            task_status = np.asarray(controllers.unit_tasks.status)[prior, 0]
            task_type = np.asarray(controllers.unit_tasks.task_type)[prior, 0]
            task_phase = np.asarray(controllers.unit_tasks.phase)[prior, 0]
            task_xy = np.stack(
                (
                    np.asarray(controllers.unit_tasks.target_x)[prior, 0],
                    np.asarray(controllers.unit_tasks.target_y)[prior, 0],
                ),
                axis=-1,
            )
            context.append(
                {
                    "step": prior,
                    "money": int(np.asarray(prior_state.money)[0]),
                    "hires": int(np.asarray(prior_state.hires_today)[0]),
                    "wheat": int(np.asarray(prior_state.shed)[0, 0])
                    + int(np.sum(np.asarray(prior_state.unit_inventory)[0, :, 0])),
                    "unit_pos": np.asarray(prior_state.unit_pos)[0].astype(int).tolist(),
                    "unit_op": np.asarray(joint.unit_op)[
                        prior, 0, 0, :unit_count
                    ].astype(int).tolist(),
                    "market_op": np.asarray(joint.market_op)[
                        prior, 0, 0, :market_count
                    ].astype(int).tolist(),
                    "market_item": np.asarray(joint.market_item)[
                        prior, 0, 0, :market_count
                    ].astype(int).tolist(),
                    "tasks": [
                        {
                            "unit": int(slot),
                            "type": int(task_type[slot]),
                            "phase": int(task_phase[slot]),
                            "xy": task_xy[slot].astype(int).tolist(),
                        }
                        for slot in np.flatnonzero(task_status == 1).tolist()
                    ],
                }
            )
        animals = []
        flags = np.asarray(pre.tile_flags)[0]
        neglect = np.asarray(pre.tile_neglect)[0]
        yield_ = np.asarray(pre.tile_yield)[0]
        for y, x in np.argwhere(pre_animal >= 0).tolist():
            animals.append(
                {
                    "xy": [x, y],
                    "species": int(pre_animal[y, x]),
                    "neglect": int(neglect[y, x]),
                    "fed": bool(int(flags[y, x]) & int(FLAG_FED)),
                    "yield": int(yield_[y, x]),
                }
            )
        rows.append(
            {
                "step": step,
                "day": step // 24 + 1,
                "turn": step % 24 + 1,
                "escaped": max(escaped, 0),
                "hard_miss": hard_miss,
                "animals": animals,
                "context": context,
            }
        )
    last_day_masks = {}
    for step in range(696, 719):
        pre = state_at(step)
        pre = jax.tree.map(lambda value: jnp.asarray(value)[None], pre)
        planned = jax.tree.map(
            lambda value, current=step: jnp.asarray(value)[current, 0][None],
            controllers,
        )
        crop_mask, animal_mask = m35_unit_masks_v2(
            pre, planned, genome, 0
        )
        last_day_masks[step] = (
            int(np.sum(np.asarray(crop_mask))),
            int(np.sum(np.asarray(animal_mask))),
        )

    effect_failures = []
    for step in range(719):
        counts = {
            name: int(np.asarray(value)[step, 0])
            for name, value in effects._asdict().items()
            if name in (
                "effect_mismatch_count",
                "owner_inactive_count",
                "deadline_missed_count",
                "resource_unavailable_count",
            )
            and int(np.asarray(value)[step, 0]) != 0
        }
        if not counts:
            continue
        pre = state_at(step)
        effect_failures.append(
            {
                "step": step,
                "day": step // 24 + 1,
                "turn": step % 24 + 1,
                "counts": counts,
                "components": {
                    component: {
                        name: int(np.asarray(value)[step, 0])
                        for name, value in getattr(effects, component)._asdict().items()
                        if int(np.asarray(value)[step, 0]) != 0
                    }
                    for component in ("e2", "e3", "crop_inventory")
                },
                "money": int(np.asarray(pre.money)[0]),
                "shed": np.asarray(pre.shed)[0].astype(int).tolist(),
                "unit_inventory": np.asarray(pre.unit_inventory)[0].astype(int).tolist(),
                "unit_pos": np.asarray(pre.unit_pos)[0].astype(int).tolist(),
                "unit_op": np.asarray(joint.unit_op)[step, 0, 0].astype(int).tolist(),
                "unit_item": np.asarray(joint.unit_item)[step, 0, 0].astype(int).tolist(),
                "market_op": np.asarray(joint.market_op)[step, 0, 0].astype(int).tolist(),
                "market_item": np.asarray(joint.market_item)[step, 0, 0].astype(int).tolist(),
                "market_amount": np.asarray(joint.market_amount)[step, 0, 0].astype(int).tolist(),
                "task_type": np.asarray(controllers.unit_tasks.task_type)[step, 0].astype(int).tolist(),
                "task_phase": np.asarray(controllers.unit_tasks.phase)[step, 0].astype(int).tolist(),
                "task_status": np.asarray(controllers.unit_tasks.status)[step, 0].astype(int).tolist(),
                "task_deadline": np.asarray(controllers.unit_tasks.deadline_step)[step, 0].astype(int).tolist(),
                "task_target_x": np.asarray(controllers.unit_tasks.target_x)[step, 0].astype(int).tolist(),
                "task_target_y": np.asarray(controllers.unit_tasks.target_y)[step, 0].astype(int).tolist(),
                "market_task_type": np.asarray(controllers.market_tasks.task_type)[step, 0].astype(int).tolist(),
                "market_task_item": np.asarray(controllers.market_tasks.item_id)[step, 0].astype(int).tolist(),
                "market_task_quantity": np.asarray(controllers.market_tasks.quantity)[step, 0].astype(int).tolist(),
                "market_task_deadline": np.asarray(controllers.market_tasks.deadline_step)[step, 0].astype(int).tolist(),
                "market_task_status": np.asarray(controllers.market_tasks.status)[step, 0].astype(int).tolist(),
            }
        )

    result = {
        "candidate": args.candidate,
        "name": names[args.candidate],
        "seed": seed,
        "backend": jax.default_backend(),
        "crop_metrics": {
            name: int(np.asarray(value)[0])
            for name, value in jax.device_get(final.crop_metrics)._asdict().items()
            if int(np.asarray(value)[0]) != 0
        },
        "animal_metrics": {
            name: int(np.asarray(value)[0])
            for name, value in jax.device_get(final.animal_metrics)._asdict().items()
            if int(np.asarray(value)[0]) != 0
        },
        "failures": rows,
        "effect_failures": effect_failures,
        "terminal_tiles": [
            {
                "xy": [int(x), int(y)],
                "kind": int(np.asarray(final.environment_state.tile_kind)[0, 0, y, x]),
                "crop": int(np.asarray(final.environment_state.tile_crop)[0, 0, y, x]),
                "animal": int(np.asarray(final.environment_state.tile_animal)[0, 0, y, x]),
                "yield": int(np.asarray(final.environment_state.tile_yield)[0, 0, y, x]),
                "origin_day": int(np.asarray(final.environment_state.tile_origin_day)[0, 0, y, x]),
            }
            for y, x in np.argwhere(
                np.asarray(final.environment_state.tile_yield)[0, 0] > 0
            ).tolist()
        ],
        "last_day": [
            {
                "step": step,
                "hires": int(np.asarray(state_at(step).hires_today)[0]),
                "unit_pos": np.asarray(state_at(step).unit_pos)[0].astype(int).tolist(),
                "unit_op": np.asarray(joint.unit_op)[step, 0, 0].astype(int).tolist(),
                "task_type": np.asarray(controllers.unit_tasks.task_type)[step, 0].astype(int).tolist(),
                "task_phase": np.asarray(controllers.unit_tasks.phase)[step, 0].astype(int).tolist(),
                "task_status": np.asarray(controllers.unit_tasks.status)[step, 0].astype(int).tolist(),
                "crop_mask_count": last_day_masks[step][0],
                "animal_mask_count": last_day_masks[step][1],
            }
            for step in range(696, 719)
        ],
    }
    output = PROJECT_DIR / "receipts" / f"m35_candidate_{args.candidate}_diagnostic.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "failure_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
