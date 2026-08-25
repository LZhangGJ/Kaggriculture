"""Frozen high-level route calendar bank compiled from Hasegawa wins.

The bank intentionally contains no raw unit action, market action, or Replay
coordinate tensors.  It stores daily business obligations plus public/current
state reference features used by the on-device demand router.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from general_project_planner_v1.fulfillment_schema import ObligationPlanV1


Array = jax.Array


class HasegawaPlanBankV1(NamedTuple):
    plan: ObligationPlanV1
    episode_id: Array
    final_reward: Array
    ref_money: Array
    ref_hires: Array
    ref_unlocked: Array
    ref_crop: Array
    ref_animal: Array
    ref_market_price: Array
    ref_shop_demand: Array


def load_hasegawa_plan_bank_v1(path: Path) -> HasegawaPlanBankV1:
    with np.load(path, allow_pickle=False) as data:
        plan = ObligationPlanV1(
            *(jnp.asarray(data[field]) for field in ObligationPlanV1._fields)
        )
        return HasegawaPlanBankV1(
            plan=plan,
            episode_id=jnp.asarray(data["episode_id"]),
            final_reward=jnp.asarray(data["final_reward"]),
            ref_money=jnp.asarray(data["ref_money"]),
            ref_hires=jnp.asarray(data["ref_hires"]),
            ref_unlocked=jnp.asarray(data["ref_unlocked"]),
            ref_crop=jnp.asarray(data["ref_crop"]),
            ref_animal=jnp.asarray(data["ref_animal"]),
            ref_market_price=jnp.asarray(data["ref_market_price"]),
            ref_shop_demand=jnp.asarray(data["ref_shop_demand"]),
        )


__all__ = ["HasegawaPlanBankV1", "load_hasegawa_plan_bank_v1"]
