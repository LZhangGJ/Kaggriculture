from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "tools" / "sync_live_assets.py"
    spec = importlib.util.spec_from_file_location("sync_live_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parses_submission_csv_and_selects_latest_completed():
    module = _module()
    text = "warning line\nfileName,date,status,publicScore,ref\na.py,2026-08-23T10:00:00Z,complete,1.0,100\nb.py,2026-08-24T10:00:00Z,pending,,101\nc.py,2026-08-24T09:00:00Z,complete,2.0,102\n"
    rows = module._csv_rows(text, ("ref", "id"))
    submission_id, row = module.select_latest_submission(rows, allow_unscored=False)
    assert submission_id == 102
    assert row["fileName"] == "c.py"


def test_token_sanitizer_never_returns_plain_token():
    module = _module()
    token = "KGAT_" + "x" * 32
    sanitized = module._sanitize(f"Authorization: {token}")
    assert token not in sanitized
    assert "[REDACTED]" in sanitized


def test_detects_submission_download_capability_from_cli_help():
    module = _module()
    assert module._has_submission_download("episodes  replay  submission-download  logs")
    assert not module._has_submission_download("episodes  replay  logs")
