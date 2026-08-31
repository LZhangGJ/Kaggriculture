from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for path in (TESTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_offline_block_portfolio_v1 as offline
import run_dynamic_cold_retrieval_mvp_v0_1_locked as optimized
from test_build_offline_block_portfolio_v1 import _library


@pytest.fixture(scope="module")
def cold_index(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("dynamic-cold-v01")
    library_root, library_sha = _library(root, paths=40, quality=True)
    portfolio_root = root / "portfolio"
    portfolio = offline.build_portfolio(
        library_root=library_root,
        library_manifest_sha256=library_sha,
        output_root=portfolio_root,
    )
    return optimized.load_cold_index(
        portfolio_root,
        portfolio["manifest_sha256"],
        library_sha,
        portfolio["input"]["artifact_sha256s"],
    )


def _arrays(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(row["contract"], np.float32),
        np.asarray(row["layout"], np.uint8),
        np.asarray(row["mask"], np.uint8),
    )


def test_cached_adapter_is_semantically_equal(cold_index: dict) -> None:
    panel = optimized.build_query_panel(cold_index)
    chosen = [
        next(row for row in panel["queries"] if row["query_id"] == query_id)
        for query_id in ("exact-216-0", "near-216-0", "unseen-216-0")
    ]
    for row in chosen:
        query, layout, mask = _arrays(row)
        optimized.assert_v0_equivalence(
            cold_index,
            query=query,
            query_layout=layout,
            query_mask=mask,
            anchor=216,
        )

    query, layout, mask = _arrays(chosen[0])
    query[86:104] += 1_000_000.0
    optimized.assert_v0_equivalence(
        cold_index,
        query=query,
        query_layout=layout,
        query_mask=mask,
        anchor=216,
        minimum_cash_reserve=17.0,
    )
    result = optimized.select_dynamic_cold(
        cold_index,
        query=query,
        query_layout=layout,
        query_mask=mask,
        anchor=216,
        minimum_cash_reserve=17.0,
    )
    assert result["dynamic_hot_count"] == 32
    assert result["active_count"] == 8
    assert result["capital_gate_mode"] == "soft_report_only"
    assert result["capital_feasibility_proven"] is False


def test_cache_threshold_lock_and_fail_closed(cold_index: dict) -> None:
    assert optimized.THRESHOLDS == optimized.v0.THRESHOLDS
    assert len(cold_index["market_variant_reference_pool"]) > 0
    assert all(
        "overlay_groups_by_block" in cached
        for cached in cold_index["anchors"].values()
    )
    panel = optimized.build_query_panel(cold_index)
    query, layout, mask = _arrays(panel["queries"][0])
    query[0] = np.nan
    with pytest.raises(ValueError, match="invalid dynamic cold query"):
        optimized.select_dynamic_cold(
            cold_index,
            query=query,
            query_layout=layout,
            query_mask=mask,
            anchor=int(panel["queries"][0]["anchor"]),
        )
    with pytest.raises(RuntimeError, match="audit-gated"):
        optimized.main(["--output-root", "unused"])
