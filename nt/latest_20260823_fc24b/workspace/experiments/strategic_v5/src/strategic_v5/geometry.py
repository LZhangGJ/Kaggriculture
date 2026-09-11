"""Small deterministic geometry helpers used by persistent executors."""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture_jax.constants import SHED_ACCESS, UnitOp


_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int8)


def manhattan_v1(first, second):
    return jnp.sum(jnp.abs(first.astype(jnp.int16) - second.astype(jnp.int16)), axis=-1)


def nearest_shed_access_v1(position):
    distance = jnp.sum(
        jnp.abs(position[..., None, :].astype(jnp.int16) - _SHED_ACCESS), axis=-1
    )
    index = jnp.argmin(distance, axis=-1)
    return _SHED_ACCESS[index], jnp.take_along_axis(distance, index[..., None], axis=-1)[..., 0]


def movement_op_toward_v1(position, target):
    dx = target[..., 0].astype(jnp.int16) - position[..., 0].astype(jnp.int16)
    dy = target[..., 1].astype(jnp.int16) - position[..., 1].astype(jnp.int16)
    return jnp.where(
        dx > 0,
        UnitOp.EAST,
        jnp.where(
            dx < 0,
            UnitOp.WEST,
            jnp.where(dy > 0, UnitOp.SOUTH, jnp.where(dy < 0, UnitOp.NORTH, UnitOp.PASS)),
        ),
    ).astype(jnp.int8)
