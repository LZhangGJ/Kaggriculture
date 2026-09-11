"""Emit the exact steps behind tomato effect-mismatch diagnostics."""

from __future__ import annotations

import argparse
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

from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.crop_executor import (  # noqa: E402
    crop_player_action_dict_v2,
    crop_policy_step_v2,
    update_crop_controller_from_effects_v2,
)
from project_route_search_v2.crop_project import (  # noqa: E402
    default_crop_project_config_v2,
)
from project_route_search_v2.crop_rollout import (  # noqa: E402
    initialize_crop_rollout_carry_v2,
)
from project_route_search_v2.lifecycle import (  # noqa: E402
    snapshot_project_controller_v2,
)
from project_route_search_v2.null_opponent import (  # noqa: E402
    combine_with_null_opponent,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84101)
    parser.add_argument("--target-tiles", type=int, default=4)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    source_seeds, event_bank = load_event_bank(
        PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    )
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    index = source_index[args.seed]
    events = Events(
        weed_spawn=event_bank.weed_spawn[index : index + 1],
        shop_choice=event_bank.shop_choice[index : index + 1],
    )
    config = default_crop_project_config_v2(
        1, crop_id=2, target_tiles=args.target_tiles
    )
    initial = initialize_crop_rollout_carry_v2(
        jnp.asarray([args.seed], dtype=jnp.int32), config, args.player
    )
    tables = load_tables()

    def rollout(carry):
        def body(current, _):
            states = current.environment_state
            action, controller, _ = crop_policy_step_v2(
                states, current.controller, config, args.player
            )
            joint = combine_with_null_opponent(
                crop_player_action_dict_v2(action), player_seat=args.player
            )
            following = batched_step_sync(states, joint, events, tables)
            updated, diagnostics = update_crop_controller_from_effects_v2(
                states,
                following,
                controller,
                action,
                config,
                args.player,
            )
            updated = snapshot_project_controller_v2(
                following, updated, args.player
            )
            tomato_pre_inventory = jnp.sum(
                states.unit_inventory[:, args.player, :, 2], axis=-1
            )
            tomato_post_inventory = jnp.sum(
                following.unit_inventory[:, args.player, :, 2], axis=-1
            )
            tomato_mask_pre = states.tile_crop[:, args.player] == 2
            tomato_mask_post = following.tile_crop[:, args.player] == 2
            tomato_pre_yield = jnp.sum(
                jnp.where(
                    tomato_mask_pre, states.tile_yield[:, args.player], 0
                ),
                axis=(1, 2),
            )
            tomato_post_yield = jnp.sum(
                jnp.where(
                    tomato_mask_post,
                    following.tile_yield[:, args.player],
                    0,
                ),
                axis=(1, 2),
            )
            batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
            target_x = jnp.clip(
                controller.unit_tasks.target_x[:, 0].astype(jnp.int32), 0, 9
            )
            target_y = jnp.clip(
                controller.unit_tasks.target_y[:, 0].astype(jnp.int32), 0, 9
            )
            trace = (
                states.step,
                action.unit_op[:, 0],
                action.market_op[:, :2],
                action.market_amount[:, :2],
                controller.unit_tasks.task_type[:, 0],
                controller.unit_tasks.phase[:, 0],
                controller.market_tasks.task_type[:, :2],
                diagnostics.effect_mismatch_count,
                diagnostics.harvest_success_count,
                diagnostics.seed_purchase_success_count,
                diagnostics.sold_product_units,
                states.money[:, args.player],
                following.money[:, args.player],
                states.shed[:, args.player, 2],
                following.shed[:, args.player, 2],
                tomato_pre_inventory,
                tomato_post_inventory,
                tomato_pre_yield,
                tomato_post_yield,
                target_x,
                target_y,
                states.tile_kind[batch, args.player, target_y, target_x],
                following.tile_kind[batch, args.player, target_y, target_x],
                states.tile_crop[batch, args.player, target_y, target_x],
                following.tile_crop[batch, args.player, target_y, target_x],
                states.tile_yield[batch, args.player, target_y, target_x],
                following.tile_yield[batch, args.player, target_y, target_x],
                states.tile_flags[batch, args.player, target_y, target_x],
                following.tile_flags[batch, args.player, target_y, target_x],
                states.tile_max_lifespan[
                    batch, args.player, target_y, target_x
                ],
                following.tile_max_lifespan[
                    batch, args.player, target_y, target_x
                ],
            )
            return current._replace(
                environment_state=following, controller=updated
            ), trace

        return jax.lax.scan(body, carry, xs=None, length=719)

    _, trace = jax.jit(rollout)(initial)
    trace = jax.device_get(trace)
    mismatch = np.flatnonzero(np.asarray(trace[7])[:, 0] > 0)
    labels = (
        "step",
        "unit_op",
        "market_op",
        "market_amount",
        "unit_task_type",
        "unit_task_phase",
        "market_task_type",
        "effect_mismatch",
        "harvest_success",
        "seed_purchase_success",
        "sold_product_units",
        "money_pre",
        "money_post",
        "shed_pre",
        "shed_post",
        "inventory_pre",
        "inventory_post",
        "map_yield_pre",
        "map_yield_post",
        "target_x",
        "target_y",
        "target_kind_pre",
        "target_kind_post",
        "target_crop_pre",
        "target_crop_post",
        "target_yield_pre",
        "target_yield_post",
        "target_flags_pre",
        "target_flags_post",
        "target_lifespan_pre",
        "target_lifespan_post",
    )
    print(f"seed={args.seed} target_tiles={args.target_tiles} mismatches={len(mismatch)}")
    for time_index in mismatch:
        row = {}
        for label, values in zip(labels, trace):
            value = np.asarray(values)[time_index, 0]
            row[label] = value.tolist() if value.ndim else int(value)
        print(row)


if __name__ == "__main__":
    main()
