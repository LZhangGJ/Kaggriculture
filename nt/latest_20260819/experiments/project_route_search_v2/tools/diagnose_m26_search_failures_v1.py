"""Classify hard failures from the deterministic M2.6 search smoke panel."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src", REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m26_rollout import (  # noqa: E402
    initialize_m26_rollout_carry_v2, make_m26_crop_rollout_v2,
)
from project_route_search_v2.m26_search import sample_m26_crop_genomes_v2  # noqa: E402


def _nonzero(diag, step: int, lane: int) -> dict[str, int]:
    result = {}
    for name, value in diag._asdict().items():
        if not hasattr(value, "shape"):
            continue
        scalar = np.asarray(value[step, lane])
        if scalar.ndim == 0 and int(scalar) != 0:
            result[name] = int(scalar)
    return result


def main() -> None:
    candidate_ids = np.asarray((6, 30, 31, 38, 46, 47, 52, 58, 61, 63), dtype=np.int32)
    seeds_per_candidate = 4
    all_genomes = sample_m26_crop_genomes_v2(64, 260618)
    selected = jax.tree.map(lambda value: value[candidate_ids], all_genomes)
    genome = jax.tree.map(lambda value: jnp.repeat(value, seeds_per_candidate, axis=0), selected)
    panel = json.loads((PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8"))
    base_seeds = np.asarray(panel["panels"]["E0_SEARCH_16"]["seeds"][:4], dtype=np.int64)
    seeds = np.tile(base_seeds, len(candidate_ids))
    source_seeds, bank = load_event_bank(PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz")
    lookup = {int(seed): index for index, seed in enumerate(source_seeds)}
    indices = np.asarray([lookup[int(seed)] for seed in seeds], dtype=np.int32)
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    carry = initialize_m26_rollout_carry_v2(jnp.asarray(seeds, dtype=jnp.int32), genome)
    final, trace = jax.jit(make_m26_crop_rollout_v2(trace=True))(
        carry, events, load_tables(), genome
    )
    jax.block_until_ready(final)
    actions, states, effects = jax.device_get(trace)
    total = np.asarray(effects.effect_mismatch_count) + np.asarray(effects.resource_unavailable_count)
    step_lane = np.argwhere(total > 0)
    rows = []
    for step, lane in step_lane.tolist():
        action = jax.tree.map(lambda value: value[step], actions)
        pre = carry.environment_state if step == 0 else jax.tree.map(lambda value: value[step - 1], states)
        rows.append({
            "candidate_id": int(candidate_ids[lane // seeds_per_candidate]),
            "seed": int(seeds[lane]), "step": int(step),
            "full": _nonzero(effects, step, lane),
            "e2": _nonzero(effects.e2, step, lane),
            "e3": _nonzero(effects.e3, step, lane),
            "crop": _nonzero(effects.crop_inventory, step, lane),
            "unit_op": np.asarray(action.unit_op[lane, 0]).astype(int).tolist(),
            "market_op": np.asarray(action.market_op[lane, 0]).astype(int).tolist(),
            "market_item": np.asarray(action.market_item[lane, 0]).astype(int).tolist(),
            "market_amount": np.asarray(action.market_amount[lane, 0]).astype(int).tolist(),
            "money": int(np.asarray(pre.money[lane, 0])),
            "seeds": np.asarray(pre.seeds[lane, 0]).astype(int).tolist(),
            "shed_fertilizer": int(np.asarray(pre.shed[lane, 0, 8])),
            "unit_fertilizer": np.asarray(pre.unit_inventory[lane, 0, :, 8]).astype(int).tolist(),
        })
    print(json.dumps({"failure_events": len(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
