"""Frozen exact event panels for the route-playbook experiment."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import NUM_DAYS, SHOP_NAMES
from kaggriculture_jax.types import Events


EVENT_DRAWS = 200
WEED_CHANCE = 0.005


def build_events_v1(seeds: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the official Python MT19937 event stream exactly."""

    seed_values = np.asarray(list(seeds), dtype=np.int64)
    weed = np.zeros(
        (len(seed_values), NUM_DAYS, EVENT_DRAWS), dtype=np.bool_
    )
    choice = np.zeros(
        (len(seed_values), NUM_DAYS, EVENT_DRAWS + 1), dtype=np.int8
    )
    for seed_index, seed_value in enumerate(seed_values.tolist()):
        for day in range(NUM_DAYS):
            rng = random.Random((seed_value * 1_000_003) ^ day)
            for consumed in range(EVENT_DRAWS + 1):
                clone = random.Random()
                clone.setstate(rng.getstate())
                choice[seed_index, day, consumed] = SHOP_NAMES.index(
                    clone.choice(SHOP_NAMES)
                )
                if consumed < EVENT_DRAWS:
                    weed[seed_index, day, consumed] = rng.random() < WEED_CHANCE
    return weed, choice


def load_route_event_bank_v1(path: Path) -> tuple[np.ndarray, Events]:
    with np.load(path, allow_pickle=False) as data:
        seeds = np.asarray(data["event_seeds"], dtype=np.int64)
        weed = np.asarray(data["weed_spawn"], dtype=np.bool_)
        choice = np.asarray(data["shop_choice"], dtype=np.int8)
    return seeds, Events(jnp.asarray(weed), jnp.asarray(choice))


def select_route_events_v1(
    wanted_seeds: Iterable[int], bank_seeds: np.ndarray, bank: Events
) -> Events:
    lookup = {int(seed): index for index, seed in enumerate(bank_seeds.tolist())}
    wanted = [int(seed) for seed in wanted_seeds]
    missing = [seed for seed in wanted if seed not in lookup]
    if missing:
        raise KeyError(f"missing frozen event seeds: {missing[:8]}")
    indices = np.asarray([lookup[seed] for seed in wanted], dtype=np.int32)
    return Events(bank.weed_spawn[indices], bank.shop_choice[indices])
