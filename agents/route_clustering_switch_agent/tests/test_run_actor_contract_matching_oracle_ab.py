from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_actor_contract_matching_oracle_ab as actor_ab


def test_hungarian_matching_fixes_farmer_and_uses_inventory() -> None:
    runtime_positions = ((9, 9), (1, 1), (1, 1))
    donor_positions = ((1, 1), (1, 1), (1, 1))
    runtime_inventories = ({}, {"WHEAT": 8}, {"CARROT": 8})
    donor_inventories = ({"MELON": 99}, {"CARROT": 8}, {"WHEAT": 8})
    mapping = actor_ab.match_actor_contract(
        runtime_positions, runtime_inventories,
        donor_positions, donor_inventories,
    )
    assert mapping == {0: 0, 1: 2, 2: 1}


def test_rectangular_matching_leaves_extra_runtime_hand_unmatched() -> None:
    mapping = actor_ab.match_actor_contract(
        ((0, 0), (1, 0), (2, 0), (9, 9)),
        ({}, {}, {}, {}),
        ((9, 9), (2, 0), (1, 0)),
        ({}, {}, {}),
    )
    assert mapping[0] == 0
    assert mapping == {0: 0, 1: 2, 2: 1}
    assert 3 not in mapping


def test_unmatched_actor_keeps_base_units() -> None:
    base = tuple(
        (("PASS",), ("EAST",), ("WEST",), ("NORTH",)) for _ in range(4)
    )
    donor = tuple(
        (("SOUTH",), ("HARVEST",), ("DROP",)) for _ in range(4)
    )
    patched = actor_ab.patch_matched_units(base, donor, ((0, 0), (1, 2), (2, 1)))
    assert patched[0] == (("SOUTH",), ("DROP",), ("HARVEST",), ("NORTH",))
    assert all(row[3] == ("NORTH",) for row in patched)


def test_post_dedup_budget_cap_is_per_kind() -> None:
    def candidate(code: str, kind: str) -> actor_ab.path_v1.PathCandidate:
        return actor_ab.path_v1.PathCandidate(
            code=code, kind=kind, units=(((code,),),) * 4,
            assignments=tuple(), donor_id=None, donor_cluster=None,
            donor_support=0, donor_rank=0, trace_novelty=0,
            aliases=(code,), raw_sha256=code,
        )

    values = [
        candidate("KEEP", "KEEP"), candidate("s1", "SINGLE"),
        candidate("s2", "SINGLE"), candidate("p1", "PAIR"),
    ]
    assert [value.code for value in actor_ab.cap_by_kind(
        values, {"KEEP": 1, "SINGLE": 1, "PAIR": 1},
    )] == ["KEEP", "s1", "p1"]
