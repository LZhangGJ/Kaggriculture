"""GPU controller for public G06 V25 Meta Reset.

V25 is a frozen 719-step tape wrapped by the reusable RC5 weed-repair state
machine and Boatlee's market sell-slot ranking.  The implementation composes
the two already parity-accepted GPU primitives; formal acceptance is still
required for the complete policy.
"""

from __future__ import annotations

import jax

from kaggriculture_jax.types import Action, State, StaticTables
from strategic_v5.boatlee_v16_gpu import _rank_sell_slots
from strategic_v5.high_potential_v20_gpu import (
    HighPotentialRuntimeTablesV1,
    _rank_sell_slots_exact,
)
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    public_rc5_weed_player_action_v1,
)


def public_v25_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    skeleton_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    action, carry = public_rc5_weed_player_action_v1(
        states, bank, skeleton_id, carry, player
    )
    market_op, market_item, market_amount = _rank_sell_slots(
        states,
        tables,
        action.market_op,
        action.market_item,
        action.market_amount,
        action.market_count,
    )
    return (
        action._replace(
            market_op=market_op,
            market_item=market_item,
            market_amount=market_amount,
        ),
        carry,
    )


def public_v27_player_action_exact_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    skeleton_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    """Exact Kaito V27 controller, including its frozen legacy sell ranking.

    The public source did not use the official 1.32.7 price table when ordering
    same-step sales.  Reusing ``_rank_sell_slots`` is state-equivalent in many
    passive games but can change the emitted action tensor.  The runtime LUT
    preserves the source policy's literal ordering semantics.
    """

    action, carry = public_rc5_weed_player_action_v1(
        states, bank, skeleton_id, carry, player
    )
    return _rank_sell_slots_exact(states, runtime, action), carry
