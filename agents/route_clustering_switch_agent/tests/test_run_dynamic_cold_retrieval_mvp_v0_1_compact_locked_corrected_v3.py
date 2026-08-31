from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for path in (TESTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_dynamic_cold_retrieval_mvp_v0_1_compact_locked as compact
import test_run_dynamic_cold_retrieval_mvp_v0_1_compact_locked as original
import test_run_dynamic_cold_retrieval_mvp_v0_1_compact_locked_corrected_v2 as v2
from test_run_dynamic_cold_retrieval_mvp_v0_1_compact_locked import cold_index


def test_all_252_queries_are_semantically_equal(cold_index: dict) -> None:
    v2.test_all_252_queries_are_semantically_equal(cold_index)


def test_reserve_market_cash_pollution_and_coverage(cold_index: dict) -> None:
    panel = compact.build_query_panel(cold_index)
    exact = next(row for row in panel["queries"] if row["query_id"] == "exact-216-0")
    query, layout, mask = original._arrays(exact)
    for reserve in (0.0, 17.0):
        compact.assert_v0_equivalence(
            cold_index, query=query, query_layout=layout, query_mask=mask,
            anchor=216, minimum_cash_reserve=reserve,
        )

    baseline = original._result(cold_index, exact)
    market_query = query.copy()
    market_query[86:104] += 1_000_000.0
    compact.assert_v0_equivalence(
        cold_index, query=market_query, query_layout=layout, query_mask=mask,
        anchor=216,
    )
    market = compact.select_dynamic_cold(
        cold_index, query=market_query, query_layout=layout,
        query_mask=mask, anchor=216,
    )
    assert [row["block_id"] for row in baseline["dynamic_hot"]] == [
        row["block_id"] for row in market["dynamic_hot"]
    ]

    cash_case = None
    for candidate in (row for row in panel["queries"] if row["anchor"] == 216):
        candidate_query, candidate_layout, candidate_mask = original._arrays(candidate)
        candidate_baseline = original._result(cold_index, candidate)
        changed = candidate_query.copy()
        changed[0] = np.nextafter(changed[0], np.float32(np.inf))
        assert changed[0] != candidate_query[0]
        candidate_cash = compact.select_dynamic_cold(
            cold_index, query=changed, query_layout=candidate_layout,
            query_mask=candidate_mask, anchor=216,
        )
        if [row["block_id"] for row in candidate_baseline["dynamic_hot"]] == [
            row["block_id"] for row in candidate_cash["dynamic_hot"]
        ]:
            cash_case = (changed, candidate_layout, candidate_mask)
            break
    assert cash_case is not None
    cash_query, cash_layout, cash_mask = cash_case
    compact.assert_v0_equivalence(
        cold_index, query=cash_query, query_layout=cash_layout,
        query_mask=cash_mask, anchor=216,
    )

    clean = compact.semantic_view(original._result(cold_index, exact))
    polluted = original._result(cold_index, exact)
    polluted["dynamic_hot"][0]["market_variants"][0][
        "market_variant_ref"
    ]["provenance_ids"].append("pollution")
    reason = next(
        row["active_reason"] for row in polluted["active"]
        if "distance_parts" in row["active_reason"]
    )
    reason["distance_parts"]["path_action"] = -1
    assert compact.semantic_view(original._result(cold_index, exact)) == clean
    assert all(compact.cache_coverage(cold_index).values())


def test_equal_distance_ties_and_audit_gate() -> None:
    v2.test_equal_distance_ties_and_audit_gate()
