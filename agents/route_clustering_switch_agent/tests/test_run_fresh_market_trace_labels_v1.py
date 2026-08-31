from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fresh_market_trace_labels_v1 as fresh


def test_shard_seed_math_and_all_reserved_ranges_are_rejected() -> None:
    assert fresh.shard_seeds(2026086400, 8, 0) == tuple(range(2026086400, 2026086408))
    assert fresh.shard_seeds(2026086400, 8, 3) == tuple(range(2026086424, 2026086432))
    with pytest.raises(ValueError, match="exactly eight"):
        fresh.shard_seeds(2026086400, 4, 0)
    for start, name in (
        (2026086100, "sealed"),
        (2026086300, "existing_train"),
        (2026086320, "validation"),
    ):
        with pytest.raises(ValueError, match=name):
            fresh.shard_seeds(start, 8, 0)


def test_fresh_formal_contract_retains_all_21_by_4_decisions() -> None:
    seeds = fresh.shard_seeds(2026086400, 8, 0)
    formal = fresh.base.scenario_contract(False, seeds)
    smoke = fresh.base.scenario_contract(True, seeds)
    assert formal["states"] == 8064
    assert len(formal["scenarios"]) == 6 * 8 * 2
    assert len(formal["anchors"]) * len(formal["offsets"]) == 84
    assert tuple(formal["seeds"]) == seeds
    assert smoke["states"] == 8
    assert tuple(smoke["seeds"]) == seeds[:1]


def test_formal_requires_explicit_nonlegacy_library(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy"
    expanded = tmp_path / "expanded"
    legacy.mkdir()
    expanded.mkdir()
    (legacy / "block_library_manifest.json").write_text("{}", encoding="utf-8")
    (expanded / "block_library_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fresh.retrieval, "DEFAULT_LIBRARY", legacy)
    assert fresh.resolve_block_library(None, True)[1] == "legacy_default_smoke_only"
    with pytest.raises(ValueError, match="explicit --block-library"):
        fresh.resolve_block_library(None, False)
    with pytest.raises(ValueError, match="reject the default"):
        fresh.resolve_block_library(legacy, False)
    assert fresh.resolve_block_library(expanded, False) == (
        expanded.resolve(), "explicit_frozen_path_base_formal",
    )


def test_formal_missing_expected_sha_stops_before_base_run(
    tmp_path, monkeypatch,
) -> None:
    library = tmp_path / "frozen_path_base"
    library.mkdir()
    (library / "block_library_manifest.json").write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(fresh.base, "run", lambda args: called.append(args))
    args = fresh.parser().parse_args([
        "--block-library", str(library), "--output-root", str(tmp_path / "out"),
    ])
    with pytest.raises(ValueError, match="--expected-library-manifest-sha256"):
        fresh.run(args)
    assert not called


def test_smoke_wrapper_records_frozen_provenance_and_candidate_counts(
    tmp_path, monkeypatch,
) -> None:
    library = tmp_path / "legacy"
    library.mkdir()
    (library / "block_library_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fresh.retrieval, "DEFAULT_LIBRARY", library)

    def fake_base_run(args):
        args.output_root.mkdir()
        decision = {
            "state_id": "O:2026086400:0:216:1",
            "arms": {
                arm: {"candidate_sha256": ["keep", f"{arm}-x"], "market_diff": [0, 0]}
                for arm in ("R0_active8", "phase_frozen", "R2_all")
            },
        }
        with (args.output_root / "h1_decisions.jsonl").open("w", encoding="utf-8") as handle:
            for _ in range(8):
                handle.write(json.dumps(decision) + "\n")
        r2_path = args.output_root / "R2_candidate_labels.jsonl"
        with r2_path.open("w", encoding="utf-8") as handle:
            for _ in range(8):
                for index in range(2):
                    handle.write(json.dumps({"candidate_index": index}) + "\n")
        return {
            "schema": fresh.SCHEMA,
            "status": "smoke_passed",
            "scenario_contract": {"states_observed": 8},
            "frozen_genome": {"sha256": "g"},
            "frozen_phase_mapping": {"216": "R0_active8"},
            "strict_library": {
                "manifest": {"sha256": "m"},
                "block_count": 168, "prototype_count": 168,
            },
            "frozen_inputs": {"mapping": {"sha256": "p"}},
            "experiment_inputs": {"route": {"sha256": "e"}},
            "implementation": {"native_extension": {"sha256": "n"}},
            "evidence_boundary": {},
            "throughput": {"physical_R2_candidate_rollouts": 16},
            "artifacts": {r2_path.name: fresh.retrieval._artifact(r2_path, 16)},
        }

    monkeypatch.setattr(fresh.base, "run", fake_base_run)
    output = tmp_path / "out"
    args = fresh.parser().parse_args(["--smoke", "--output-root", str(output)])
    report = fresh.run(args)
    provenance = json.loads((output / "shard_provenance.json").read_text(encoding="utf-8"))
    assert report["status"] == "fresh_shard_smoke_passed"
    assert report["candidate_counts"]["by_arm"]["R2_all"] == {
        "min": 2, "max": 2, "total_candidate_rows": 16,
    }
    assert provenance["input_contract_sha256"] == report["input_contract_sha256"]
    assert provenance["candidate_library"]["mode"] == "legacy_default_smoke_only"
    assert provenance["candidate_library"]["role"] == "path_base"
    assert report["candidate_execution_contract"]["market_full_action_overlay"][
        "status"] == "not_executed_in_v1"
    assert provenance["no_decision_filtering"] is True
    assert provenance["seeds"] == report["fresh_shard"]["seeds"] == [2026086400]
    assert report["fresh_shard"]["all_physical_R2_candidate_labels_retained"] is True
