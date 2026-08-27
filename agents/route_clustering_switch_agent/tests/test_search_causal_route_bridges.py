from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = CODE_ROOT / "scripts"
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from search_causal_route_bridges import splice_three  # noqa: E402


def _tape(label: str) -> list[dict]:
    return [
        {"farmer": [label, step], "hands": [], "market": []}
        for step in range(719)
    ]


def test_three_way_splice_has_exact_causal_boundaries() -> None:
    result = splice_three(_tape("prefix"), _tape("bridge"), _tape("suffix"), 120, 144)

    assert result[119]["farmer"] == ["prefix", 119]
    assert result[120]["farmer"] == ["bridge", 120]
    assert result[143]["farmer"] == ["bridge", 143]
    assert result[144]["farmer"] == ["suffix", 144]
    assert len(result) == 719
