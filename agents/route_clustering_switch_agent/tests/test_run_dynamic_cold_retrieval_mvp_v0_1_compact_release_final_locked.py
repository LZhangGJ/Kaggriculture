from __future__ import annotations

import sys
from pathlib import Path

import pytest


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for path in (TESTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_dynamic_cold_retrieval_mvp_v0_1_compact_release_final_locked as release


def test_preflight_source_input_and_normalized_tamper(tmp_path: Path) -> None:
    runner_path = Path(release.__file__)
    assert release._runner_normalized_sha256(
        runner_path,
    ) == release.EXPECTED_RUNNER_NORMALIZED_SHA256
    locks = release.verify_locked_environment()
    assert locks["all_locks_verified_before_index_load"] is True
    assert locks["sources"]["compact_runner"]["sha256"] == (
        release.EXPECTED_COMPACT_SHA256
    )
    assert locks["sources"]["canonical_test"]["sha256"] == (
        release.EXPECTED_CANONICAL_TEST_SHA256
    )
    tampered = tmp_path / "tampered_release.py"
    tampered.write_text(
        runner_path.read_text(encoding="utf-8").replace(
            "algorithm unchanged", "algorithm tampered", 1,
        ),
        encoding="utf-8",
    )
    assert release._runner_normalized_sha256(tampered) != (
        release.EXPECTED_RUNNER_NORMALIZED_SHA256
    )


def test_lock_failure_is_before_index_load_and_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fail_locks():
        calls.append("locks")
        raise ValueError("expected tamper")

    def forbidden_load(*_args, **_kwargs):
        calls.append("index")
        raise AssertionError("index loaded after failed source lock")

    monkeypatch.setattr(release, "verify_locked_environment", fail_locks)
    monkeypatch.setattr(release.compact, "load_cold_index", forbidden_load)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="tamper"):
        release.run(output)
    assert calls == ["locks"]
    assert not output.exists()


def test_evaluator_routes_only_to_compact_and_restores_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_selector = release.v0.select_dynamic_cold
    calls = []

    def fake_evaluate(index, panel, initialization_ms, incremental_rss_mib):
        calls.append((index, panel, initialization_ms, incremental_rss_mib))
        assert release.v0.select_dynamic_cold is release.compact.select_dynamic_cold
        return {"passed": True}

    monkeypatch.setattr(release.v0, "evaluate_panel", fake_evaluate)
    result = release.evaluate_panel({"index": 1}, {"panel": 2}, 3.0, 4.0)
    assert result == {"passed": True}
    assert calls == [({"index": 1}, {"panel": 2}, 3.0, 4.0)]
    assert release.v0.select_dynamic_cold is original_selector


