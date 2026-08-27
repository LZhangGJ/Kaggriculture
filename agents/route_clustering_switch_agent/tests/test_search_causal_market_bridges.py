from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for value in (CODE_ROOT / "scripts", CODE_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from search_causal_market_bridges import (  # noqa: E402
    apply_causal_market_genome,
    apply_market_genome,
    market_option_space,
    splice_suffix,
)


def _tape(label: str) -> list[dict]:
    return [
        {"farmer": [label, step], "hands": [[label]], "market": []}
        for step in range(719)
    ]


def test_market_space_changes_only_variable_steps() -> None:
    left = _tape("left")
    right = _tape("right")
    right[121]["market"] = [["SELL", "WHEAT", 2]]
    steps, options = market_option_space((left, right), 120, 124)

    assert steps == (121,)
    assert len(options[0]) == 2


def test_market_genome_preserves_all_worker_paths() -> None:
    base = _tape("base")
    donor = _tape("donor")
    donor[121]["market"] = [["BUY_SEED", "WHEAT", 1]]
    steps, options = market_option_space((base, donor), 120, 124)
    result = apply_market_genome(base, steps, options, (1,))

    assert result[121]["farmer"] == base[121]["farmer"]
    assert result[121]["hands"] == base[121]["hands"]
    assert result[121]["market"] == donor[121]["market"]


def test_suffix_splice_keeps_causal_bridge() -> None:
    bridge = _tape("bridge")
    suffix = _tape("suffix")
    result = splice_suffix(bridge, suffix, 144)

    assert result[143]["farmer"] == ["bridge", 143]
    assert result[144]["farmer"] == ["suffix", 144]
    assert len(result) == 719


def test_causal_market_genome_forces_executed_prefix() -> None:
    prefix = _tape("prefix")
    base = _tape("base")
    donor = _tape("donor")
    donor[121]["market"] = [["BUY_SEED", "WHEAT", 1]]
    steps, options = market_option_space((base, donor), 120, 124)
    result = apply_causal_market_genome(
        prefix, base, steps, options, (1,), 120
    )

    assert result[119]["farmer"] == ["prefix", 119]
    assert result[120]["farmer"] == ["base", 120]
    assert result[121]["market"] == donor[121]["market"]
