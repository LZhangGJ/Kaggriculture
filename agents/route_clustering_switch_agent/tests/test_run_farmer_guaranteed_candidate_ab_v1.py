from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_farmer_guaranteed_candidate_ab_v1 as farmer_ab


def _row(code: str, kind: str, assignments: tuple[tuple[int, int], ...]) -> tuple[object, ...]:
    return (
        code, kind, assignments, "d", "C00", 1, 0, 2,
        (((code,),),) * 4,
    )


def test_farmer_eligibility_requires_two_changes_and_movement() -> None:
    base = ((("PASS",),),) * 4
    donor = (
        (("EAST",),), (("WEST",),), (("PASS",),), (("PASS",),),
    )
    assert farmer_ab.farmer_eligibility(base, donor) == (True, 2)
    assert farmer_ab.farmer_eligibility(base, base) == (False, 0)


def test_guarantee_replaces_last_single_without_kind_budget_change() -> None:
    variants = [
        _row("s1", "SINGLE", ((1, 1),)),
        _row("s2", "SINGLE", ((2, 2),)),
        _row("p", "PAIR", ((1, 1), (2, 2))),
        _row("all", "ALL", ((1, 1), (2, 2), (3, 3))),
    ]
    farmer = _row("farmer", "SINGLE", ((0, 0),))
    changed, audit = farmer_ab.guarantee_farmer_single(variants, farmer, True)
    assert [row[0] for row in changed] == ["s1", "farmer", "p", "all"]
    assert audit["replacement_triggered"]
    assert farmer_ab.Counter(row[1] for row in changed) == farmer_ab.Counter(
        row[1] for row in variants
    )


def test_guarantee_skips_present_or_ineligible_farmer() -> None:
    farmer = _row("farmer", "SINGLE", ((0, 0),))
    variants = [farmer, _row("s1", "SINGLE", ((1, 1),))]
    same, audit = farmer_ab.guarantee_farmer_single(variants, farmer, True)
    assert same == variants
    assert audit["farmer_single_present_A"] and not audit["replacement_triggered"]
    same, audit = farmer_ab.guarantee_farmer_single(variants[1:], farmer, False)
    assert same == variants[1:] and not audit["replacement_triggered"]


def test_canonical_union_reuses_shared_traces() -> None:
    def candidate(code: str, units: tuple[object, ...]) -> farmer_ab.path_v1.PathCandidate:
        return farmer_ab.path_v1.PathCandidate(
            code=code, kind="KEEP" if code == "KEEP" else "SINGLE",
            units=units, assignments=tuple(), donor_id=None,
            donor_cluster=None, donor_support=0, donor_rank=0,
            trace_novelty=0, aliases=(code,), raw_sha256=code,
        )

    keep = candidate("KEEP", ((("PASS",),),) * 4)
    shared_a = candidate("a", ((("EAST",),),) * 4)
    shared_c = candidate("c", ((("EAST",),),) * 4)
    unique_c = candidate("u", ((("WEST",),),) * 4)
    union, a_indices, c_indices = farmer_ab.canonical_union(
        [keep, shared_a], [keep, shared_c, unique_c],
    )
    assert len(union) == 3
    assert a_indices == [0, 1]
    assert c_indices == [0, 1, 2]


def test_union_oracle_preserves_a_and_counts_only_incremental_vocabulary() -> None:
    def arm(rank: tuple[int, float], positive: bool, repair: bool) -> dict[str, object]:
        return {
            "candidate_count": 2,
            "best_rank": list(rank),
            "best_delta_margin": rank[1],
            "positive": positive,
            "loss_to_win_repair": repair,
        }

    a0, c0 = arm((1, 5.0), True, False), arm((1, 3.0), False, False)
    a1, c1 = arm((0, -3.0), False, False), arm((2, 2.0), True, True)
    rows = [
        {"A": a0, "C": c0, "paired": farmer_ab._paired(a0, c0),
         "union_candidate_count": 3},
        {"A": a1, "C": c1, "paired": farmer_ab._paired(a1, c1),
         "union_candidate_count": 3},
    ]
    result = farmer_ab._union_oracle_metrics(rows)
    assert result["U_positive_states"] == 2
    assert result["U_new_positive"] == 1
    assert result["U_loss_to_win_repairs"] == 1
    assert result["U_beats_A"] == 1
    assert result["U_loses_A"] == 0
    assert result["U_incremental_canonical_candidates_vs_A"] == 2
    assert result["U_incremental_canonical_candidate_rate_vs_A"] == 0.5
