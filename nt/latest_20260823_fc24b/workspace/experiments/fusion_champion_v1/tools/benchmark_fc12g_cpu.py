#!/usr/bin/env python3
"""Sequential replay/action and CPU latency acceptance for generated FC12G."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_module(path: Path, suffix: str):
    name = f"_fc12g_cpu_acceptance_{suffix}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    started = time.perf_counter()
    spec.loader.exec_module(module)
    return module, time.perf_counter() - started


def _load_trace(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 721 or records[0].get("record_type") != "header":
        raise RuntimeError(f"malformed trace: {path}")
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


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/artifacts/fc12g_cpu/main.py",
    )
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.trace_receipt.read_text(encoding="utf-8"))
    rows = []
    all_durations_ms: list[float] = []
    import_seconds = []
    for index, trace_row in enumerate(receipt["traces"]):
        trace_path = ROOT / trace_row["path"]
        frames = _load_trace(trace_path)
        seat = int(trace_row["candidate_seat"])
        module, loaded_seconds = _load_module(args.agent, str(index))
        import_seconds.append(loaded_seconds)
        mismatches = []
        durations_ms = []
        # frame t observation produces the action stored in frame t+1.
        for step in range(719):
            observation = _observation(frames[step], seat)
            started = time.perf_counter()
            actual = module.agent(observation, None)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
            expected = frames[step + 1]["actions"][seat]
            if _canonical(actual) != _canonical(expected) and len(mismatches) < 5:
                mismatches.append({"step": step, "actual": actual, "expected": expected})
        all_durations_ms.extend(durations_ms)
        rows.append(
            {
                "seed": int(trace_row["seed"]),
                "candidate_seat": seat,
                "actions": len(durations_ms),
                "exact": not mismatches,
                "mismatches": mismatches,
                "import_seconds": loaded_seconds,
                "mean_action_ms": statistics.fmean(durations_ms),
                "p95_action_ms": _percentile(durations_ms, 0.95),
                "max_action_ms": max(durations_ms),
            }
        )

    exact = all(row["exact"] for row in rows)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc12g-cpu-acceptance.v1",
        "status": "PASS" if exact and max(all_durations_ms) < 1000.0 else "FAIL",
        "sequential_action_exact": exact,
        "action_calls": len(all_durations_ms),
        "mean_import_seconds": statistics.fmean(import_seconds),
        "max_import_seconds": max(import_seconds),
        "mean_action_ms": statistics.fmean(all_durations_ms),
        "p50_action_ms": _percentile(all_durations_ms, 0.50),
        "p95_action_ms": _percentile(all_durations_ms, 0.95),
        "p99_action_ms": _percentile(all_durations_ms, 0.99),
        "max_action_ms": max(all_durations_ms),
        "acceptance_limit_ms": 1000.0,
        "inputs": {
            "agent": {"path": str(args.agent), "sha256": _sha256(args.agent)},
            "trace_receipt": {"path": str(args.trace_receipt), "sha256": _sha256(args.trace_receipt)},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("rows", "inputs")}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
