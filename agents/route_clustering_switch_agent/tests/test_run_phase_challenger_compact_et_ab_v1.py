from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_compact_et_ab_v1 as runner


def _full_panel(rows: int = 3) -> np.ndarray:
    values = np.arange(runner.FULL_WIDTH, dtype=np.float32)[None, :]
    panel = np.repeat(values, rows, axis=0)
    trace = panel[:, runner.TRACE_START:runner.TRACE_STOP].reshape(
        rows, *runner.TRACE_SHAPE,
    )
    for index in range(1, rows):
        trace[index, 0, index, 0] += index * 10
        trace[index, 0, index, -1] = 1
        panel[index, runner.META_START + index] += index
    return panel


def test_compact_transform_is_exact_deterministic_1078d() -> None:
    full = _full_panel()
    compact = runner.compact_feature_rows(full)
    assert compact.shape == (3, 1078)
    assert len(runner.COMPACT_FEATURE_NAMES) == 1078
    assert np.array_equal(
        compact[:, :337],
        np.concatenate((full[:, :129], full[:, 147:355]), axis=1),
    )
    assert np.array_equal(compact[:, 337:625], full[:, 355:643])
    assert np.array_equal(compact[:, 625:643], full[:, 129:147])
    trace = full[:, 643:2755].reshape(3, 4, 16, 33)
    keep = np.concatenate((trace[0].mean(axis=1), trace[0].max(axis=1)), axis=1).reshape(-1)
    assert np.array_equal(compact[0, 643:907], keep)
    assert np.array_equal(compact[2, 643:907], keep)
    current = np.concatenate((trace[:, 0].mean(axis=1), trace[:, 0].max(axis=1)), axis=1)
    assert np.array_equal(compact[:, 907:973], current)
    assert np.array_equal(compact[:, 973:989], trace[:, 0, :, -1])
    assert np.array_equal(compact[:, 989:], full[:, 2755:2844])
    assert np.array_equal(compact, runner.compact_feature_rows(full.copy()))


@pytest.mark.parametrize("column,name", [
    (0, "state"), (129, "plan summary"), (355, "actor slots"),
])
def test_compact_transform_rejects_candidate_varying_shared_context(
    column: int, name: str,
) -> None:
    full = _full_panel()
    full[1, column] += 1
    with pytest.raises(AssertionError, match=name):
        runner.compact_feature_rows(full)


def test_compact_transform_rejects_candidate_varying_future_trace() -> None:
    full = _full_panel()
    future = runner.TRACE_START + (1 * 16 * 33)
    full[1, future] += 1
    with pytest.raises(AssertionError, match="offsets 1..3"):
        runner.compact_feature_rows(full)


def _arm(shas: list[str], markets: list[int]) -> dict:
    return {
        "candidate_sha256": shas,
        "market_diff": markets,
        "all_candidates_oracle": {
            "best_sha256": shas[0], "best_outcome": 2, "best_margin": 1.0,
        },
    }


def _decision(state_id: str, offset: int) -> dict:
    r0 = [f"keep-{offset}", f"a-{offset}"]
    phase = [f"keep-{offset}", f"p-{offset}"]
    return {
        "state_id": state_id, "split": "train_proxy", "opponent": "NT0056",
        "seed": 2026086300, "seat": 0, "anchor": 216, "offset": offset,
        "decision_step": 216 + offset,
        "arms": {
            "R0_active8": _arm(r0, [0, 0]),
            "phase_frozen": _arm(phase, [0, 0]),
        },
    }


def _labels(decision: dict) -> list[dict]:
    state = runner.preflight.canonical_union_index(decision)
    return [{
        "state_id": decision["state_id"], "split": "train_proxy",
        "candidate_sha256": sha, "union_index": index,
        "market_diff": state["market_by_sha"][sha], "outcome": 2,
        "margin": float(index), "path_pure": True,
        "membership": "R0" if index < len(state["r0_sha256"]) else "phase_only",
    } for index, sha in enumerate(state["union_sha256"])]


def test_streaming_limit_keeps_complete_decisions_and_sha_order(tmp_path: Path) -> None:
    decisions = [_decision(f"s{offset}", offset) for offset in (1, 2)]
    decision_path = tmp_path / "decisions.jsonl"
    label_path = tmp_path / "labels.jsonl"
    decision_path.write_text(
        "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8",
    )
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for decision in decisions for row in _labels(decision)),
        encoding="utf-8",
    )
    panels = list(runner.iter_panels(decision_path, label_path, max_decisions=1))
    assert len(panels) == 1
    assert [row["union_index"] for row in panels[0][1]] == [0, 1, 2]
    assert runner.scan_panels(decision_path, label_path, 1)["rows"] == 3

    rows = json.loads(label_path.read_text(encoding="utf-8").splitlines()[1])
    rows["candidate_sha256"] = "wrong"
    raw = label_path.read_text(encoding="utf-8").splitlines()
    raw[1] = json.dumps(rows)
    label_path.write_text("\n".join(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA/order"):
        list(runner.iter_panels(decision_path, label_path, 1))


def _candidate(code: str, sha: str, kind: str = "SINGLE") -> runner.path_v1.PathCandidate:
    return runner.path_v1.PathCandidate(
        code=code, kind=kind, units=((('PASS',),),), assignments=tuple(),
        donor_id=None, donor_cluster=None, donor_support=0, donor_rank=-1,
        trace_novelty=0, aliases=(code,), raw_sha256=sha,
    )


def test_canonical_candidates_preserve_r0_prefix_then_phase_only() -> None:
    keep = _candidate("KEEP", "k", "KEEP")
    a = _candidate("A", "a")
    alias = _candidate("ALIAS", "a")
    p = _candidate("P", "p")
    union = runner.canonical_candidates([keep, a], [keep, alias, p])
    assert [candidate.raw_sha256 for candidate in union] == ["k", "a", "p"]


def test_formal_failed_signal_stops_before_scan_or_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "load_input_contract", lambda *_: {
        "root": tmp_path, "report": {}, "paths": {}, "signal_passed": False,
    })
    monkeypatch.setattr(
        runner, "scan_panels",
        lambda *_: (_ for _ in ()).throw(AssertionError("scan must not run")),
    )
    monkeypatch.setattr(
        runner, "materialize_features",
        lambda *_: (_ for _ in ()).throw(AssertionError("allocation must not run")),
    )
    args = runner.parser().parse_args([
        "--output-root", str(tmp_path / "out"),
    ])
    report = runner.run(args)
    assert report["status"] == "signal_insufficient_stop_before_feature_allocation"
    assert not report["feature_allocation_executed"]
    assert not report["training_executed"]
    assert not (tmp_path / "out" / "full_2844.npy").exists()


def test_cli_freezes_tree_counts_and_exposes_no_validation_or_sealed_split() -> None:
    args = runner.parser().parse_args([])
    destinations = {action.dest for action in runner.parser()._actions}
    assert (args.trees, args.cv_trees) == (96, 24)
    assert "validation_seed" not in destinations
    assert "sealed_seed" not in destinations
    assert "validation_opponent" not in destinations
    assert "direct-u-phase" in runner.SCHEMA


def test_report_declares_keep_referenced_direct_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "load_input_contract", lambda *_: {
        "root": tmp_path, "report": {}, "paths": {}, "signal_passed": False,
    })
    args = runner.parser().parse_args(["--output-root", str(tmp_path / "direct")])
    report = runner.run(args)
    objective = report["selector_objective"]
    assert objective["zero_harm_reference"] == "KEEP"
    assert objective["not_R0_residual"] is True
    assert "phase-only novelty gate" in objective["signal_gate_role"]
    markdown = (tmp_path / "direct" / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "signal_insufficient_stop_before_feature_allocation" in markdown
    assert "Evidence boundary: train only" in markdown


def test_direct_ab_diagnostic_locks_98pct_and_cannot_pass_without_seedfold() -> None:
    decisions = 4
    rows = decisions * 2
    metadata = {
        "decision": np.repeat(np.arange(decisions, dtype=np.int32), 2),
        "delta_margin": np.tile(np.asarray([0.0, 2.0]), decisions),
        "opponent": np.repeat(np.asarray([0, 0, 1, 1], np.int16), 2),
        "seed": np.repeat(np.asarray([10, 11, 10, 11], np.int64), 2),
        "edit": np.tile(np.asarray([
            runner.path_v1.KIND_INDEX["KEEP"], runner.path_v1.KIND_INDEX["SINGLE"],
        ], np.int8), decisions),
    }
    choices = np.arange(1, rows, 2, dtype=np.int64)
    gate = runner.direct_ab_gate(metadata, choices, choices, True)
    assert gate["realized_delta_retention_floor"] == 0.98
    assert all(gate["lopo_diagnostic_conditions"].values())
    assert gate["compact_fires"] == 4
    assert len(gate["compact_fire_opponents"]) == 2
    assert len(gate["compact_fire_seeds"]) == 2
    assert gate["gate_incomplete"]
    assert not gate["passed"]
    assert "leave-seed-fold-out" in gate["missing_primary_evidence"]


@pytest.mark.skipif(
    os.environ.get("RUN_PHASE_COMPACT_EXTERNAL_SMOKE") != "1",
    reason="explicit train-only native smoke",
)
def test_external_one_scenario_materialization_smoke(tmp_path: Path) -> None:
    output = tmp_path / "native_smoke"
    assert runner.main(["--smoke", "--output-root", str(output)]) == 0
    report = json.loads((output / "SMOKE_REPORT.json").read_text(encoding="utf-8"))
    assert report["status"] == "smoke_materialization_passed"
    assert report["panel"]["decisions"] == 4
    assert len(report["panel"]["scenarios"]) == 1
    assert report["materialization_audit"]["sha_order_checks"] == 4
    assert not report["training_executed"]
