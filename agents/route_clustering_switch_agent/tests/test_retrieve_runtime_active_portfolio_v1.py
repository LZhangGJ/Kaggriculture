from __future__ import annotations

import hashlib
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
from test_build_offline_block_portfolio_v1 import _library


def _fixture(tmp_path: Path) -> tuple[
    Path, str, dict, dict, list[dict], dict, dict, dict, list[dict], int,
]:
    library_root, library_sha = _library(tmp_path, paths=9, quality=True)
    portfolio_root = tmp_path / "offline-portfolio"
    offline_manifest = offline.build_portfolio(
        library_root=library_root,
        library_manifest_sha256=library_sha,
        output_root=portfolio_root,
    )
    loaded = runtime._load_portfolio(
        portfolio_root, offline_manifest["manifest_sha256"],
    )
    portfolio, library, blocks, _, prototypes, provenance, route_refs, _ = loaded
    hot = runtime._hot_refs(portfolio, library, blocks, 216)
    first_block = next(
        str(row["block_id"])
        for row in hot
        if int(blocks[int(row["block_ref"]["block_index"])][
            "full_action_variant_count"
        ]) == 2
    )
    prototype_row = next(
        index for index, block_id in enumerate(prototypes["block_ids"])
        if int(prototypes["anchors"][index]) == 216 and str(block_id) == first_block
    )
    return (
        portfolio_root, offline_manifest["manifest_sha256"], portfolio, library,
        blocks, prototypes, provenance, route_refs, hot, prototype_row,
    )


def test_exact_near_mask_capital_variants_and_determinism(tmp_path: Path) -> None:
    (
        _, _, _, _, blocks, prototypes, provenance, route_refs, hot, row,
    ) = _fixture(tmp_path)
    query = prototypes["entry_contracts"][row].copy()
    layout = prototypes["layouts"][row].copy()
    mask = prototypes["unlocked_masks"][row].copy()
    exact = runtime.select_runtime_active(
        query=query, query_layout=layout, query_mask=mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
        minimum_cash_reserve=0,
    )
    expected = str(prototypes["block_ids"][row])
    assert exact["active_count"] == 8
    assert exact["selected"][0]["block_id"] == expected
    assert exact["selected"][0]["distance"] == 0.0
    assert exact["path_action_diversity"].endswith("inherited_from_hot32")
    assert exact["capital_gate_mode"] == "soft_report_only"
    assert exact["capital_feasibility_proven"] is False
    assert len(exact["selected"][0]["market_variants"]) == 2
    assert exact == runtime.select_runtime_active(
        query=query, query_layout=layout, query_mask=mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
        minimum_cash_reserve=0,
    )

    near = query.copy()
    near[86:104] = 1_000_000
    market_ignored = runtime.select_runtime_active(
        query=near, query_layout=layout, query_mask=mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
    )
    assert market_ignored["selected"][0]["block_id"] == expected
    assert market_ignored["selected"][0]["distance"] == 0.0

    changed_mask = mask.copy()
    changed_mask[0] = 1 - changed_mask[0]
    masked = runtime.select_runtime_active(
        query=query, query_layout=layout, query_mask=changed_mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
    )
    assert masked["selected"][0]["block_id"] == expected
    assert masked["selected"][0]["distance"] > 0.0
    assert masked["mask_semantics"].startswith("existing soft Hamming")

    broke = query.copy()
    broke[0] = -1
    infeasible = runtime.select_runtime_active(
        query=broke, query_layout=layout, query_mask=mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
        minimum_cash_reserve=0,
        capital_gate_mode="hard_current_cash",
    )
    assert infeasible["active_count"] == 0
    assert infeasible["hard_current_cash_rejected_path_count"] == len(hot)
    assert all(
        row["reason"] == "all_market_variants_fail_hard_current_cash_gate"
        for row in infeasible["rejected"]
    )
    report_only = runtime.select_runtime_active(
        query=broke, query_layout=layout, query_mask=mask, anchor=216,
        hot_rows=hot, blocks=blocks, prototypes=prototypes,
        provenance=provenance, route_refs=route_refs,
    )
    assert report_only["active_count"] == 8
    assert report_only["capital_feasibility_proven"] is False
    assert any(
        variant["capital_shortfall_diagnostic"] > 0
        for selected in report_only["selected"]
        for variant in selected["market_variants"]
    )


def test_cli_hash_refs_artifact_tamper_and_query_tamper(tmp_path: Path) -> None:
    (
        portfolio_root, portfolio_sha, portfolio, library, _, prototypes,
        _, _, hot, row,
    ) = _fixture(tmp_path)
    del hot
    query_path = tmp_path / "query.npz"
    np.savez_compressed(
        query_path,
        contract=prototypes["entry_contracts"][row],
        layout=prototypes["layouts"][row],
        mask=prototypes["unlocked_masks"][row],
    )
    query_raw = query_path.read_bytes()
    query_sha = hashlib.sha256(query_raw).hexdigest()
    with pytest.raises(ValueError, match="offline portfolio manifest SHA-256 mismatch"):
        runtime.retrieve(
            portfolio_root=portfolio_root,
            portfolio_manifest_sha256="0" * 64,
            query_path=query_path,
            query_sha256=query_sha,
            output_root=tmp_path / "portfolio-sha-tampered",
            anchor=216,
        )
    first = runtime.retrieve(
        portfolio_root=portfolio_root,
        portfolio_manifest_sha256=portfolio_sha,
        query_path=query_path,
        query_sha256=query_sha,
        output_root=tmp_path / "runtime-first",
        anchor=216,
        minimum_cash_reserve=17,
    )
    second = runtime.retrieve(
        portfolio_root=portfolio_root,
        portfolio_manifest_sha256=portfolio_sha,
        query_path=query_path,
        query_sha256=query_sha,
        output_root=tmp_path / "runtime-second",
        anchor=216,
        minimum_cash_reserve=17,
    )
    assert first["content_sha256"] == second["content_sha256"]
    assert first["runtime_state_conditioned"] is True
    assert first["capital_feasibility_proven"] is False
    assert first["config"]["capital_gate_mode"] == "soft_report_only"
    assert first["config"]["minimum_cash_reserve"] == 17
    assert first["storage"] == {
        "actions_copied": False,
        "contracts_copied": False,
        "query_copied": False,
        "output_contains": "refs, distances, rejection reasons, and hashes only",
    }
    assert first["input"]["validation_or_sealed_used"] is False
    assert {path.name for path in (tmp_path / "runtime-first").iterdir()} == {
        "runtime_active_manifest.json", "runtime_active_manifest.json.sha256",
    }
    changed_reserve = runtime.retrieve(
        portfolio_root=portfolio_root,
        portfolio_manifest_sha256=portfolio_sha,
        query_path=query_path,
        query_sha256=query_sha,
        output_root=tmp_path / "runtime-changed-reserve",
        anchor=216,
        minimum_cash_reserve=18,
    )
    assert changed_reserve["content_sha256"] != first["content_sha256"]

    query_path.write_bytes(query_raw + b"tampered")
    with pytest.raises(ValueError, match="runtime query SHA-256 mismatch"):
        runtime.retrieve(
            portfolio_root=portfolio_root,
            portfolio_manifest_sha256=portfolio_sha,
            query_path=query_path,
            query_sha256=query_sha,
            output_root=tmp_path / "query-tampered",
            anchor=216,
        )
    query_path.write_bytes(query_raw)
    library_root = Path(portfolio["input"]["library_root"])
    prototype_path = library_root / library["contract_prototypes_file"]
    prototype_path.write_bytes(prototype_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="contract prototypes artifact SHA-256 mismatch"):
        runtime.retrieve(
            portfolio_root=portfolio_root,
            portfolio_manifest_sha256=portfolio_sha,
            query_path=query_path,
            query_sha256=query_sha,
            output_root=tmp_path / "artifact-tampered",
            anchor=216,
        )
