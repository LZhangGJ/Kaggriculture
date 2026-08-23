#!/usr/bin/env python3
"""Verify that one FC24B module can run many games without cross-game state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("_fc24b_reentry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_frames(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 721 or records[0].get("record_type") != "header":
        raise RuntimeError(f"invalid trace {path}")
    return records[1:]


def _observation(frame: dict, seat: int) -> dict:
    return {
        "step": frame["step"],
        "day": frame["day"],
        "hour": frame["hour"],
        "player": seat,
        "farms": frame["farms"],
        "private": frame["private"][seat],
        "market": frame["market"],
        "town": frame["town"],
    }


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    agent_path = args.agent.resolve()
    receipt_path = args.trace_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    module = _load_agent(agent_path)
    rows = []
    first_mismatch = None
    total_calls = 0
    # Deliberately use the same imported module for every game.  A second pass
    # in reverse order tests both same-seat and seat-switch reset paths.
    ordered = list(receipt["traces"]) + list(reversed(receipt["traces"]))
    for ordinal, trace in enumerate(ordered):
        frames = _load_frames(ROOT / trace["path"])
        seat = int(trace["candidate_seat"])
        exact = True
        for step in range(719):
            actual = module.agent(_observation(frames[step], seat), None)
            expected = frames[step + 1]["actions"][seat]
            total_calls += 1
            if _canonical(actual) != _canonical(expected):
                exact = False
                if first_mismatch is None:
                    first_mismatch = {
                        "ordinal": ordinal,
                        "opponent": trace["opponent"],
                        "seed": int(trace["seed"]),
                        "candidate_seat": seat,
                        "step": step,
                        "actual": actual,
                        "expected": expected,
                    }
                break
        rows.append(
            {
                "ordinal": ordinal,
                "opponent": trace["opponent"],
                "seed": int(trace["seed"]),
                "candidate_seat": seat,
                "exact": exact,
            }
        )

    strict = all(row["exact"] for row in rows)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc24b-reentry.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if strict else "FAIL",
        "strict_exact": strict,
        "games": len(rows),
        "action_calls": total_calls,
        "first_mismatch": first_mismatch,
        "inputs": {
            "agent": {"path": str(agent_path), "sha256": _sha256(agent_path)},
            "trace_receipt": {
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
            },
        },
        "rows": rows,
        "acceptance_boundary": (
            "PASS requires one imported CPU module to replay all 32 official "
            "contexts forward and backward with exact actions after every reset."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "games": payload["games"],
                "action_calls": payload["action_calls"],
                "first_mismatch": first_mismatch,
            },
            ensure_ascii=False,
        )
    )
    return 0 if strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
