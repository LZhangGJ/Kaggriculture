from __future__ import annotations

import random
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for value in (CODE_ROOT / "scripts", CODE_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from evolve_causal_route_bridges import (  # noqa: E402
    _crossover,
    _descriptor,
    _mutate,
    _splice,
)


def _tape(label: str) -> list[dict]:
    return [
        {"farmer": [label, step], "hands": [], "market": []}
        for step in range(719)
    ]


def test_genome_splice_uses_aligned_macro_blocks() -> None:
    prefix = _tape("prefix")
    donors = [_tape("left"), _tape("right")]
    route = _splice(prefix, donors, (0, 1), _tape("suffix"), 120, 124, 2)

    assert route[119]["farmer"] == ["prefix", 119]
    assert [route[i]["farmer"][0] for i in range(120, 124)] == [
        "left", "left", "right", "right"
    ]
    assert route[124]["farmer"] == ["suffix", 124]
    assert len(route) == 719


def test_mutation_and_crossover_keep_genome_shape() -> None:
    rng = random.Random(17)
    left = ((0,) * 8, (1,) * 8, 0, 0)
    right = ((1,) * 8, (0,) * 8, 1, 1)
    for _ in range(50):
        child = _crossover(left, right, rng)
        child = _mutate(child, 4, 3, 2, rng)
        assert len(child[0]) == len(child[1]) == 8
        assert all(0 <= value < 4 for value in child[0])
        assert all(0 <= value < 4 for value in child[1])
        assert 0 <= child[2] < 3
        assert 0 <= child[3] < 2


def test_descriptor_counts_action_channels_and_donor_runs() -> None:
    default = [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(719)
    ]
    branch = [dict(action) for action in default]
    for step in range(120, 128):
        default[step] = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
        branch[step] = {
            "farmer": ["PLACE", "SHEEP"],
            "hands": [],
            "market": [["BUY_ANIMAL", "SHEEP", 1]],
        }
    descriptor = _descriptor(default, branch, 120, 144, (0, 0), (1, 1))
    assert descriptor == (0, 4, 7, 4, 0, 2)
