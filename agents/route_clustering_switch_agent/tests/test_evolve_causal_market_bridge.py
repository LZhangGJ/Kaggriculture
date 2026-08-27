from __future__ import annotations

import random
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for value in (CODE_ROOT / "scripts", CODE_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from evolve_causal_market_bridge import (  # noqa: E402
    _crossover,
    _descriptor,
    _mutate,
)


def test_market_ga_mutation_stays_inside_option_space() -> None:
    rng = random.Random(3)
    genome = (0, 0, 0, 0)
    counts = (2, 3, 4, 2)
    for _ in range(50):
        genome = _mutate(genome, counts, rng)
        assert all(0 <= value < count for value, count in zip(genome, counts))


def test_market_ga_crossover_preserves_shape_and_parent_alleles() -> None:
    rng = random.Random(7)
    left = (0,) * 24
    right = (1,) * 24
    child = _crossover(left, right, rng)
    assert len(child) == 24
    assert set(child) <= {0, 1}


def test_market_ga_descriptor_tracks_distance_from_base() -> None:
    options = (([], [["SELL", "WHEAT", 2]]),) * 4
    assert _descriptor((0, 0, 0, 0), options, (0, 0, 0, 0))[-1] == 0
    assert _descriptor((1, 1, 1, 1), options, (0, 0, 0, 0))[-1] == 1
