from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from types import SimpleNamespace
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import merge_fresh_market_trace_label_shards_v1 as merger


def _arm(shas: list[str], markets: list[int] | None = None) -> dict:
    return {
        "candidate_sha256": shas,
        "market_diff": markets or [0] * len(shas),
        "all_candidates_oracle": {
            "best_sha256": shas[0], "best_outcome": 2, "best_margin": 1.0,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_canonical_union_keeps_r0_prefix_then_phase_only() -> None:
    decision = {
        "state_id": "O:1:0:216:1", "split": "train_proxy",
        "opponent": "O", "seed": 1, "seat": 0, "anchor": 216,
        "offset": 1, "decision_step": 217,
        "arms": {
            "R0_active8": _arm(["keep", "a"]),
            "phase_frozen": _arm(["keep", "a", "p"]),
        },
    }
    assert merger.canonical_union(decision) == ["keep", "a", "p"]


def test_report_set_requires_contiguous_disjoint_shards_and_same_input_hash() -> None:
    def report(index: int, digest: str = "same") -> dict:
        return {
            "fresh_shard": {
                "shard_index": index, "seed_start": 2026086400,
                "seeds": list(range(2026086400 + 8 * index, 2026086408 + 8 * index)),
            },
            "input_contract_sha256": digest,
            "candidate_library": {"manifest_sha256": "library"},
        }

    ordered = merger.validate_report_set([report(1), report(0)], 2)
    assert [row["fresh_shard"]["shard_index"] for row in ordered] == [0, 1]
    with pytest.raises(ValueError, match="input hash"):
        merger.validate_report_set([report(0), report(1, "different")], 2)
    with pytest.raises(ValueError, match="contiguous"):
        merger.validate_report_set([report(0), report(2)], 2)


def test_shard_validator_checks_r2_sha_order_missing_and_trajectory_counts(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(merger.fresh.base, "TRAIN_OPPONENTS", ("O",))
    monkeypatch.setattr(merger.fresh.base, "SEATS", (0,))
    monkeypatch.setattr(merger.fresh.base.path_v1, "ANCHORS", (216,))
    monkeypatch.setattr(merger.fresh.base, "OFFSETS", (1,))
    monkeypatch.setattr(merger.fresh.base, "EXPECTED_FORMAL_STATES", 1)
    state_id = "O:7:0:216:1"
    r2_shas = ["keep", "a", "p", "r2-only"]
    r2_markets = [0, 0, 0, 1]
    decision = {
        "state_id": state_id, "split": "train_proxy", "opponent": "O",
        "seed": 7, "seat": 0, "anchor": 216, "offset": 1,
        "decision_step": 217,
        "arms": {
            "R0_active8": _arm(["keep", "a"]),
            "phase_frozen": _arm(["keep", "a", "p"]),
            "R2_all": _arm(r2_shas, r2_markets),
        },
    }
    labels = [
        {
            "state_id": state_id, "union_index": index,
            "membership": "R0" if index < 2 else "phase_only",
            "candidate_sha256": sha,
        }
        for index, sha in enumerate(("keep", "a", "p"))
    ]
    r2_rows = [
        {
            "schema": merger.fresh.R2_ROW_SCHEMA,
            "state_id": state_id, "opponent": "O", "seed": 7, "seat": 0,
            "anchor": 216, "offset": 1, "step": 217,
            "candidate_index": index, "candidate_sha256": sha,
            "outcome": 2, "margin": float(index), "market_diff": market,
            "path_pure": market == 0, "code": f"C{index}",
            "donor_id": None if index == 0 else f"D{index}",
        }
        for index, (sha, market) in enumerate(zip(r2_shas, r2_markets, strict=True))
    ]
    trajectory = {
        "opponent": "O", "seed": 7, "seat": 0, "decisions": 1,
        "physical_candidate_rollouts": len(r2_rows), "completed": True,
    }
    _write_jsonl(tmp_path / "h1_decisions.jsonl", [decision])
    _write_jsonl(tmp_path / "canonical_union_labels.jsonl", labels)
    _write_jsonl(tmp_path / "R2_candidate_labels.jsonl", r2_rows)
    _write_jsonl(tmp_path / "frozen_trajectories.jsonl", [trajectory])
    provenance = {
        "input_contract_sha256": "input", "seeds": [7],
        "candidate_library": {"manifest_sha256": "library"},
    }
    (tmp_path / "shard_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8",
    )
    names = (*merger.FILES, "shard_provenance.json")
    report = {
        "_root": tmp_path, "fresh_shard": {"seeds": [7]},
        "input_contract_sha256": "input",
        "candidate_library": {"manifest_sha256": "library"},
        "artifacts": {name: merger._artifact(tmp_path / name) for name in names},
    }

    def validate() -> tuple[dict[str, int], dict[str, io.StringIO]]:
        handles = {name: io.StringIO() for name in merger.FILES}
        return merger.validate_and_copy_shard(report, handles), handles

    counts, handles = validate()
    assert counts == {
        "decisions": 1, "labels": 3, "r2_labels": 4, "trajectories": 1,
    }
    assert '"donor_id": "D3"' in handles["R2_candidate_labels.jsonl"].getvalue()
    monkeypatch.setattr(merger.fresh, "SHARD_SIZE", 1)
    formal_report = {
        "schema": merger.fresh.SCHEMA,
        "status": "fresh_shard_complete",
        "fresh_shard": {
            "shard_index": 0, "seed_start": 7, "seed_count": 1,
            "seeds": [7], "no_decision_filtering": True,
        },
        "input_contract_sha256": "input",
        "candidate_library": {
            "manifest_sha256": "library",
            "mode": "explicit_frozen_path_base_formal",
        },
        "artifacts": report["artifacts"],
    }
    (tmp_path / "FINAL_REPORT.json").write_text(
        json.dumps(formal_report), encoding="utf-8",
    )
    merged = merger.merge(SimpleNamespace(
        shard_root=[tmp_path], expected_shard_count=1,
        output_root=tmp_path / "merged",
    ))
    assert merged["artifacts"]["R2_candidate_labels.jsonl"]["rows"] == 4

    bad_sha = [dict(row) for row in r2_rows]
    bad_sha[2]["candidate_sha256"] = "wrong"
    _write_jsonl(tmp_path / "R2_candidate_labels.jsonl", bad_sha)
    report["artifacts"]["R2_candidate_labels.jsonl"] = merger._artifact(
        tmp_path / "R2_candidate_labels.jsonl",
    )
    with pytest.raises(ValueError, match="R2 candidate order/SHA/market"):
        validate()

    bad_order = [dict(row) for row in r2_rows]
    bad_order[1], bad_order[2] = bad_order[2], bad_order[1]
    _write_jsonl(tmp_path / "R2_candidate_labels.jsonl", bad_order)
    report["artifacts"]["R2_candidate_labels.jsonl"] = merger._artifact(
        tmp_path / "R2_candidate_labels.jsonl",
    )
    with pytest.raises(ValueError, match="R2 candidate order/SHA/market"):
        validate()

    _write_jsonl(tmp_path / "R2_candidate_labels.jsonl", r2_rows[:-1])
    report["artifacts"]["R2_candidate_labels.jsonl"] = merger._artifact(
        tmp_path / "R2_candidate_labels.jsonl",
    )
    with pytest.raises(ValueError, match="ended before"):
        validate()

    _write_jsonl(tmp_path / "R2_candidate_labels.jsonl", r2_rows)
    report["artifacts"]["R2_candidate_labels.jsonl"] = merger._artifact(
        tmp_path / "R2_candidate_labels.jsonl",
    )
    trajectory["physical_candidate_rollouts"] = 3
    _write_jsonl(tmp_path / "frozen_trajectories.jsonl", [trajectory])
    report["artifacts"]["frozen_trajectories.jsonl"] = merger._artifact(
        tmp_path / "frozen_trajectories.jsonl",
    )
    with pytest.raises(ValueError, match="trajectory order or completion"):
        validate()
