from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from materialize_causal_route_splices import (  # noqa: E402
    _splice,
    add_archive_routes,
    splice_tapes,
)


def _tape(label: str) -> list[dict]:
    return [
        {"farmer": [label, step], "hands": [], "market": []}
        for step in range(719)
    ]


def test_splice_parser_and_tape_boundary() -> None:
    assert _splice("H001=G001@120+NT0541") == ("H001", "G001", 120, "NT0541")
    result = splice_tapes(_tape("left"), _tape("right"), 120)
    assert result[119]["farmer"] == ["left", 119]
    assert result[120]["farmer"] == ["right", 120]
    assert result[-1]["farmer"] == ["right", 718]


def test_archive_routes_become_splice_families() -> None:
    tapes: dict[str, list[dict]] = {}
    entries: list[dict] = []
    families: dict[str, str] = {}
    imported = add_archive_routes(
        tapes, entries, families, {"MGA001_DEFAULT": _tape("ga")}, "ga.zlib"
    )
    route_id = families["MGA001_DEFAULT"]
    assert route_id.startswith("archive:")
    assert tapes[route_id][120]["farmer"] == ["ga", 120]
    assert imported == entries
    assert entries[0]["route_archive"]["source"] == "ga.zlib"
