from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_contract_breadth_sweep_v2 as breadth
import run_multi_farmer_path_residual_v1 as path_v1
import run_phase_breadth_h1_offsets_v1 as h1


def _candidate(code: str, sha: str, kind: str = "SINGLE") -> path_v1.PathCandidate:
    return path_v1.PathCandidate(
        code=code,
        kind=kind,
        units=(((("PASS",),)),),
        assignments=(),
        donor_id=None if kind == "KEEP" else code,
        donor_cluster=None,
        donor_support=0,
        donor_rank=0,
        trace_novelty=0,
        aliases=(code,),
        raw_sha256=sha,
    )


def test_decision_points_are_exact_anchor_plus_offsets() -> None:
    assert h1.decision_points((216, 240)) == {
        217: (216, 1), 218: (216, 2),
        219: (216, 3), 220: (216, 4),
        241: (240, 1), 242: (240, 2),
        243: (240, 3), 244: (240, 4),
    }


def test_oracle_views_keep_market_changing_candidate_but_filter_path_pure() -> None:
    candidates = [
        _candidate("KEEP", "keep", "KEEP"),
        _candidate("MARKET_BEST", "market"),
        _candidate("PURE_BEST", "pure"),
    ]
    rewards = np.asarray([[0, 1], [10, 0], [5, 0]], np.float64)
    market = np.asarray([0, 1, 0], np.int32)
    all_view = h1.oracle_view(
        candidates, rewards, market, 0, path_pure=False,
    )
    pure_view = h1.oracle_view(
        candidates, rewards, market, 0, path_pure=True,
    )
    assert all_view["best_code"] == "MARKET_BEST"
    assert all_view["best_market_diff"] == 1
    assert pure_view["best_code"] == "PURE_BEST"
    assert pure_view["best_market_diff"] == 0

    result = h1.arm_result(
        candidates,
        rewards,
        market,
        0,
        {
            "allowed": 3,
            "diversity_ids": ["a", "b", "c"],
            "pre_h1_candidate_count": 4,
            "h1_collapse_count": 1,
        },
        [0, 1, 2],
    )
    assert result["market_diff"] == [0, 1, 0]
    assert result["path_pure_mask"] == [True, False, True]
    assert result["market_diff_candidate_count"] == 1


def _variant(code: str, donor: str, first: str, second: str) -> tuple:
    units = (
        ((first,),),
        ((second,),),
        (("PASS",),),
        (("PASS",),),
    )
    return (
        code, "SINGLE", ((0, 0),), donor, "C", 0, 0, 1, units,
    )


def test_horizon_aligned_h1_diversity_can_differ_from_legacy_h4() -> None:
    rows = {
        "a": (_variant("a", "a", "NORTH", "NORTH"),),
        "b": (_variant("b", "b", "NORTH", "SOUTH"),),
        "c": (_variant("c", "c", "SOUTH", "PASS"),),
        "d": (_variant("d", "d", "EAST", "PASS"),),
    }
    donors = {
        block_id: path_v1.DonorBlock(
            block_id, 216, "C", "p", "r", 0, (), "",
        )
        for block_id in rows
    }
    payloads = {
        block_id: tuple(path_v1._canonical_bytes(row[-1]) for row in values)
        for block_id, values in rows.items()
    }
    cache = breadth.VariantCache(
        base=((("PASS",),),) * 4,
        donors=donors,
        variants=rows,
        payloads=payloads,
        payload_masks={
            "a": 0b0001, "b": 0b0010, "c": 0b0100, "d": 0b1000,
        },
        support={block_id: 0 for block_id in rows},
        build_seconds=0.0,
    )
    masks = h1.h1_payload_masks(cache)
    h1_selected, _ = breadth.select_diverse_ids(rows, masks, cache.support)
    h4_selected, _ = breadth.select_diverse_ids(
        rows, cache.payload_masks, cache.support,
    )
    assert h1_selected == ("a", "c", "d")
    assert h4_selected == ("a", "b", "c")
    legacy = h1.legacy_h4_then_h1_diagnostic(cache, rows)
    expected = path_v1.effective_candidates(
        breadth.candidates_from_cache(cache, h4_selected), 1,
    )
    assert legacy["candidate_sha256"] == [
        candidate.raw_sha256 for candidate in expected
    ]


def test_sha_slice_preserves_requested_order() -> None:
    r2 = [
        _candidate("KEEP", "keep", "KEEP"),
        _candidate("A", "a"),
        _candidate("B", "b"),
    ]
    assert h1.slice_indices([r2[0], r2[2]], r2).tolist() == [0, 2]
