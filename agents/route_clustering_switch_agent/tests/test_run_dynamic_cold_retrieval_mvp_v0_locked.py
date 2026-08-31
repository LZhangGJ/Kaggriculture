from __future__ import annotations

import json
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
import retrieve_runtime_active_portfolio_v1 as runtime
import run_dynamic_cold_retrieval_mvp_v0_locked as locked
from test_build_offline_block_portfolio_v1 import _library


@pytest.fixture(scope="module")
def cold_index(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("dynamic-cold-locked")
    library_root, library_sha = _library(root, paths=40, quality=True)
    portfolio_root = root / "portfolio"
    portfolio = offline.build_portfolio(
        library_root=library_root,
        library_manifest_sha256=library_sha,
        output_root=portfolio_root,
    )
    return locked.load_cold_index(
        portfolio_root,
        portfolio["manifest_sha256"],
        library_sha,
        portfolio["input"]["artifact_sha256s"],
    )


def _variant_prototype(index: dict, anchor: int = 216) -> int:
    blocks = index["blocks"]
    target = next(
        str(block["block_id"])
        for block in blocks
        if int(block["anchor"]) == anchor
        and int(block["full_action_variant_count"]) == 2
    )
    prototypes = index["prototypes"]
    return next(
        row for row, block_id in enumerate(prototypes["block_ids"])
        if int(prototypes["anchors"][row]) == anchor and str(block_id) == target
    )


def _query(index: dict, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototypes = index["prototypes"]
    return (
        prototypes["entry_contracts"][row].copy(),
        prototypes["layouts"][row].copy(),
        prototypes["unlocked_masks"][row].copy(),
    )


def _ids(rows: list[dict]) -> list[str]:
    return [str(row["block_id"]) for row in rows]


def test_exact_market_mask_static_and_determinism(cold_index: dict) -> None:
    row = _variant_prototype(cold_index)
    query, layout, mask = _query(cold_index, row)
    first = locked.select_dynamic_cold(
        cold_index, query=query, query_layout=layout, query_mask=mask, anchor=216,
    )
    repeated = locked.select_dynamic_cold(
        cold_index, query=query, query_layout=layout, query_mask=mask, anchor=216,
    )
    expected = str(cold_index["prototypes"]["block_ids"][row])
    assert first["dynamic_hot_count"] == 32
    assert first["active_count"] == 8
    assert first["active"][0]["block_id"] == expected
    assert first["active"][0]["distance"] == 0.0
    assert first["active"][0]["active_reason"] == {
        "kind": "global_nearest_exact_contract",
    }
    assert first["capital_gate_mode"] == "soft_report_only"
    assert first["capital_feasibility_proven"] is False
    assert len(first["active"][0]["market_variants"]) == 2
    assert len(set(_ids(first["active"]))) == 8
    assert _ids(first["dynamic_hot"]) == _ids(repeated["dynamic_hot"])
    assert [x["distance"] for x in first["dynamic_hot"]] == [
        x["distance"] for x in repeated["dynamic_hot"]
    ]
    assert _ids(first["active"]) == _ids(repeated["active"])

    changed_market = query.copy()
    changed_market[86:104] += 1_000_000.0
    market = locked.select_dynamic_cold(
        cold_index,
        query=changed_market,
        query_layout=layout,
        query_mask=mask,
        anchor=216,
    )
    assert _ids(market["dynamic_hot"]) == _ids(first["dynamic_hot"])
    assert [x["distance"] for x in market["dynamic_hot"]] == [
        x["distance"] for x in first["dynamic_hot"]
    ]

    changed_mask = mask.copy()
    changed_mask[0] = 1 - changed_mask[0]
    masked = locked.select_dynamic_cold(
        cold_index,
        query=query,
        query_layout=layout,
        query_mask=changed_mask,
        anchor=216,
    )
    assert masked["dynamic_hot"][0]["distance"] > 0.0

    static = runtime.select_runtime_active(
        query=query,
        query_layout=layout,
        query_mask=mask,
        anchor=216,
        hot_rows=runtime._hot_refs(
            cold_index["portfolio"], cold_index["library"],
            cold_index["blocks"], 216,
        ),
        blocks=cold_index["blocks"],
        prototypes=cold_index["prototypes"],
        provenance=cold_index["provenance"],
        route_refs=cold_index["route_refs"],
    )
    assert first["active"][0]["distance"] <= static["selected"][0]["distance"]


def test_panel_metric_key_and_fail_closed(cold_index: dict) -> None:
    panel = locked.build_query_panel(cold_index)
    repeated = locked.build_query_panel(cold_index)
    assert panel["content_sha256"] == repeated["content_sha256"]
    assert panel["counts"] == {
        "total": 252,
        "per_kind": {"exact": 84, "near": 84, "unseen": 84},
    }
    assert panel["outcome_used_for_panel_selection"] is False
    assert not any("outcome" in json.dumps(row) for row in panel["queries"])

    mini = {
        "queries": [
            next(row for row in panel["queries"] if row["query_id"] == "exact-216-0"),
            next(row for row in panel["queries"] if row["query_id"] == "near-216-0"),
            next(row for row in panel["queries"] if row["query_id"] == "unseen-216-0"),
        ],
    }
    evaluation = locked.evaluate_panel(cold_index, mini, 0.0, 0.0)
    assert "reference_max_distance_absolute_error" in evaluation["metrics"]
    assert "reference_max_distance_absolute_error" in evaluation["checks"]
    assert "distance_absolute_tolerance" not in evaluation["checks"]

    row = _variant_prototype(cold_index)
    query, layout, mask = _query(cold_index, row)
    query[0] = np.nan
    with pytest.raises(ValueError, match="invalid dynamic cold query"):
        locked.select_dynamic_cold(
            cold_index,
            query=query,
            query_layout=layout,
            query_mask=mask,
            anchor=216,
        )
    with pytest.raises(ValueError, match="locked combined library"):
        locked.load_cold_index(
            Path(cold_index["portfolio"]["input"]["library_root"]).parent
            / "portfolio",
            cold_index["portfolio_sha256"],
            "0" * 64,
            cold_index["artifact_sha256s"],
        )
    wrong_artifacts = dict(cold_index["artifact_sha256s"])
    wrong_artifacts["vectors"] = "0" * 64
    with pytest.raises(ValueError, match="locked combined artifact"):
        locked.load_cold_index(
            Path(cold_index["portfolio"]["input"]["library_root"]).parent
            / "portfolio",
            cold_index["portfolio_sha256"],
            cold_index["library_sha256"],
            wrong_artifacts,
        )
