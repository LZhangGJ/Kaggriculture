from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for value in (CODE_ROOT / "scripts", CODE_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from validate_causal_route_pairs import pair_routes  # noqa: E402


def test_pair_routes_matches_causal_leaf_names() -> None:
    routes = {
        "MGA001_DEFAULT": [{"farmer": ["PASS"]}],
        "MGA001_YARN2": [{"farmer": ["NORTH"]}],
        "MGA002_DEFAULT": [{"farmer": ["PASS"]}],
        "MGA002_YARN2": [{"farmer": ["SOUTH"]}],
    }
    pairs = pair_routes(routes)
    assert [value[0] for value in pairs] == ["MGA001", "MGA002"]
    assert pairs[0][2][0]["farmer"] == ["NORTH"]
