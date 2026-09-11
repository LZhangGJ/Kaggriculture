"""Locate exact M2.6 fertilizer effect mismatches without changing acceptance."""

from __future__ import annotations

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

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m26_candidates import m26_branch_coverage_panel_v2  # noqa: E402
from project_route_search_v2.m26_rollout import (  # noqa: E402
    initialize_m26_rollout_carry_v2,
    make_m26_crop_rollout_v2,
)


def main() -> None:
    source_seeds, bank = load_event_bank(PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz")
    genome, names = m26_branch_coverage_panel_v2()
    candidate_index = names.index("FERTILIZER_ALWAYS_STRAWBERRY")
    genome = jax.tree.map(lambda value: value[candidate_index : candidate_index + 1], genome)
    panel = json.loads((PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8"))
    seed = int(panel["panels"]["E0_AUDIT_128"]["seeds"][candidate_index])
    source_index = {int(value): index for index, value in enumerate(source_seeds)}
    event_index = source_index[seed]
    events = Events(bank.weed_spawn[event_index : event_index + 1], bank.shop_choice[event_index : event_index + 1])
    carry = initialize_m26_rollout_carry_v2(jnp.asarray((seed,), dtype=jnp.int32), genome)
    final, trace = jax.jit(make_m26_crop_rollout_v2(trace=True))(
        carry, events, load_tables(), genome
    )
    jax.block_until_ready(final)
    actions, states, effects = jax.device_get(trace)
    mismatch_steps = np.flatnonzero(np.asarray(effects.e2.effect_mismatch_count)[:, 0] > 0)
    rows = []
    for step in mismatch_steps.tolist():
        pre_state = carry.environment_state if step == 0 else jax.tree.map(lambda x: x[step - 1], states)
        post_state = jax.tree.map(lambda x: x[step], states)
        action = jax.tree.map(lambda x: x[step], actions)
        rows.append(
            {
                "step": step,
                "unit_op": np.asarray(action.unit_op[0, 0]).astype(int).tolist(),
                "unit_item": np.asarray(action.unit_item[0, 0]).astype(int).tolist(),
                "market_op": np.asarray(action.market_op[0, 0]).astype(int).tolist(),
                "market_item": np.asarray(action.market_item[0, 0]).astype(int).tolist(),
                "pre_fertilizer_shed": int(np.asarray(pre_state.shed[0, 0, 8])),
                "post_fertilizer_shed": int(np.asarray(post_state.shed[0, 0, 8])),
                "pre_fertilizer_inventory": np.asarray(pre_state.unit_inventory[0, 0, :, 8]).astype(int).tolist(),
                "post_fertilizer_inventory": np.asarray(post_state.unit_inventory[0, 0, :, 8]).astype(int).tolist(),
                "e2": {
                    field: int(np.asarray(value[step, 0]))
                    for field, value in effects.e2._asdict().items()
                },
            }
        )
    print(json.dumps({"seed": seed, "candidate": names[candidate_index], "mismatches": rows}, indent=2))


if __name__ == "__main__":
    main()
