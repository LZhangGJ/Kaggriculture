"""Trace one sampled M3 candidate and report exact failure steps.

This is a diagnostic tool, not an acceptance receipt.  It intentionally emits
only end-of-day animal snapshots and Full-core effect failures so a failed
search-smoke genome can be repaired without guessing at safety coefficients.
"""

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

from kaggriculture_jax.constants import FLAG_FED, NUM_ANIMALS, NUM_PRODUCTS  # noqa: E402
from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m3_rollout import (  # noqa: E402
    initialize_m3_rollout_carry_v2,
    make_m3_animal_rollout_v2,
)
from project_route_search_v2.m3_search import sample_m3_animal_genomes_v2  # noqa: E402


def _lane(tree, index: int):
    return jax.tree.map(lambda value: value[index : index + 1], tree)


def _scalar(value) -> int:
    return int(np.asarray(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--seed", type=int, default=84101)
    parser.add_argument("--sampler-seed", type=int, default=360618)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    sampled = sample_m3_animal_genomes_v2(
        max(args.candidate + 1, 64), seed=args.sampler_seed
    )
    genome = _lane(sampled, args.candidate)
    source_seeds, bank = load_event_bank(args.event_bank)
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    event_index = source_index[args.seed]
    events = Events(
        bank.weed_spawn[event_index : event_index + 1],
        bank.shop_choice[event_index : event_index + 1],
    )
    carry = initialize_m3_rollout_carry_v2(
        jnp.asarray((args.seed,), dtype=jnp.int32), genome
    )
    initial = jax.device_get(carry.environment_state)
    rollout = jax.jit(make_m3_animal_rollout_v2(trace="diagnostic"))
    final, trace = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    joint_action, next_states, effects, controllers, step_hard = jax.device_get(trace)

    rows = []
    for step in range(719):
        if step == 0:
            at = lambda value: np.asarray(value)[0]
        else:
            at = lambda value: np.asarray(value)[step - 1, 0]
        pre_animal = at(initial.tile_animal if step == 0 else next_states.tile_animal)[0]
        post_animal = np.asarray(next_states.tile_animal)[step, 0, 0]
        pre_count = int(np.sum(pre_animal >= 0))
        post_count = int(np.sum(post_animal >= 0))
        effect_values = {
            name: _scalar(value[step, 0])
            for name, value in effects._asdict().items()
            if name in (
                "effect_mismatch_count",
                "owner_inactive_count",
                "deadline_missed_count",
                "resource_unavailable_count",
            )
        }
        effect_components = {
            component: {
                name: _scalar(value[step, 0])
                for name, value in getattr(effects, component)._asdict().items()
                if name in (
                    "effect_mismatch_count",
                    "owner_inactive_count",
                    "deadline_missed_count",
                    "resource_unavailable_count",
                )
            }
            for component in ("e2", "e3", "crop_inventory")
        }
        effect_failed = any(effect_values.values())
        escaped = ((step + 1) % 24 == 0) and post_count < pre_count
        step_hard_values = {
            "feed_hard_deadline_miss": _scalar(step_hard[0][step, 0]),
            "animal_capacity_loss": _scalar(step_hard[1][step, 0]),
            "care_bonus_forfeited": _scalar(step_hard[2][step, 0]),
            "care_bonus_capacity_clipped": _scalar(step_hard[3][step, 0]),
        }
        if not (escaped or effect_failed or any(step_hard_values.values())):
            continue

        active_tasks = []
        task_status = np.asarray(controllers.unit_tasks.status)[step, 0]
        for task_slot in np.flatnonzero(task_status == 1).tolist():
            active_tasks.append(
                {
                    "unit": task_slot,
                    "task_type": int(
                        np.asarray(controllers.unit_tasks.task_type)[
                            step, 0, task_slot
                        ]
                    ),
                    "phase": int(
                        np.asarray(controllers.unit_tasks.phase)[step, 0, task_slot]
                    ),
                    "item": int(
                        np.asarray(controllers.unit_tasks.item_id)[step, 0, task_slot]
                    ),
                    "target": [
                        int(np.asarray(controllers.unit_tasks.target_x)[step, 0, task_slot]),
                        int(np.asarray(controllers.unit_tasks.target_y)[step, 0, task_slot]),
                    ],
                    "deadline": int(
                        np.asarray(controllers.unit_tasks.deadline_step)[
                            step, 0, task_slot
                        ]
                    ),
                }
            )

        flags = at(initial.tile_flags if step == 0 else next_states.tile_flags)[0]
        coords = np.argwhere(pre_animal >= 0)
        animals = []
        for y, x in coords.tolist():
            animals.append(
                {
                    "xy": [x, y],
                    "species": int(pre_animal[y, x]),
                    "neglect": int(at(initial.tile_neglect if step == 0 else next_states.tile_neglect)[0, y, x]),
                    "fed": bool(int(flags[y, x]) & int(FLAG_FED)),
                    "yield": int(at(initial.tile_yield if step == 0 else next_states.tile_yield)[0, y, x]),
                    "pending_care": int(
                        at(initial.tile_pending_care if step == 0 else next_states.tile_pending_care)[0, y, x]
                    ),
                }
            )
        unit_count = _scalar(joint_action.unit_count[step, 0, 0])
        market_count = _scalar(joint_action.market_count[step, 0, 0])
        context = []
        if escaped or any(step_hard_values.values()):
            for prior in range(max(0, step - 23), step + 1):
                if prior == 0:
                    prior_at = lambda value: np.asarray(value)[0]
                    prior_source = initial
                else:
                    prior_at = lambda value, p=prior: np.asarray(value)[p - 1, 0]
                    prior_source = next_states
                prior_unit_count = _scalar(joint_action.unit_count[prior, 0, 0])
                prior_market_count = _scalar(joint_action.market_count[prior, 0, 0])
                prior_status = np.asarray(controllers.unit_tasks.status)[prior, 0]
                prior_tasks = []
                for prior_slot in np.flatnonzero(prior_status == 1).tolist():
                    prior_tasks.append(
                        {
                            "unit": prior_slot,
                            "type": int(
                                np.asarray(controllers.unit_tasks.task_type)[
                                    prior, 0, prior_slot
                                ]
                            ),
                            "phase": int(
                                np.asarray(controllers.unit_tasks.phase)[
                                    prior, 0, prior_slot
                                ]
                            ),
                            "target": [
                                int(np.asarray(controllers.unit_tasks.target_x)[prior, 0, prior_slot]),
                                int(np.asarray(controllers.unit_tasks.target_y)[prior, 0, prior_slot]),
                            ],
                        }
                    )
                context.append(
                    {
                        "step": prior,
                        "money": int(prior_at(prior_source.money)[0]),
                        "hires": int(prior_at(prior_source.hires_today)[0]),
                        "wheat": int(prior_at(prior_source.shed)[0, 0])
                        + int(np.sum(prior_at(prior_source.unit_inventory)[0, :, 0])),
                        "unit_op": np.asarray(
                            joint_action.unit_op[
                                prior, 0, 0, :prior_unit_count
                            ]
                        ).astype(int).tolist(),
                        "market_op": np.asarray(
                            joint_action.market_op[
                                prior, 0, 0, :prior_market_count
                            ]
                        ).astype(int).tolist(),
                        "market_item": np.asarray(
                            joint_action.market_item[
                                prior, 0, 0, :prior_market_count
                            ]
                        ).astype(int).tolist(),
                        "market_amount": np.asarray(
                            joint_action.market_amount[
                                prior, 0, 0, :prior_market_count
                            ]
                        ).astype(int).tolist(),
                        "tasks": prior_tasks,
                    }
                )
        rows.append(
            {
                "step": step,
                "day": step // 24 + 1,
                "turn": step % 24 + 1,
                "money": int(at(initial.money if step == 0 else next_states.money)[0]),
                "wheat_shed": int(at(initial.shed if step == 0 else next_states.shed)[0, 0]),
                "wheat_carried": int(
                    np.sum(at(initial.unit_inventory if step == 0 else next_states.unit_inventory)[0, :, 0])
                ),
                "hires": int(at(initial.hires_today if step == 0 else next_states.hires_today)[0]),
                "active_before": pre_count,
                "active_after": post_count,
                "animals": animals,
                "unit_op": np.asarray(
                    joint_action.unit_op[step, 0, 0, :unit_count]
                ).astype(int).tolist(),
                "unit_item": np.asarray(
                    joint_action.unit_item[step, 0, 0, :unit_count]
                ).astype(int).tolist(),
                "market_op": np.asarray(
                    joint_action.market_op[step, 0, 0, :market_count]
                ).astype(int).tolist(),
                "market_item": np.asarray(
                    joint_action.market_item[step, 0, 0, :market_count]
                ).astype(int).tolist(),
                "market_amount": np.asarray(
                    joint_action.market_amount[step, 0, 0, :market_count]
                ).astype(int).tolist(),
                "effects": effect_values,
                "effect_components": effect_components,
                "step_hard": step_hard_values,
                "active_tasks": active_tasks,
                "day_context": context,
            }
        )

    result = {
        "candidate": args.candidate,
        "seed": args.seed,
        "backend": jax.default_backend(),
        "final_metrics": {
            name: _scalar(value[0]) for name, value in jax.device_get(final.metrics)._asdict().items()
        },
        "failure_steps": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
