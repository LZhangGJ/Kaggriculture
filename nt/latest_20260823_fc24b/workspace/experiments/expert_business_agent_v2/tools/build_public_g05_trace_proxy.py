#!/usr/bin/env python3
"""Build an explicitly approximate two-seat medoid proxy for public G05."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_stream(path: Path, candidate_seat: int) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != 721:
        raise ValueError(f"expected header + 720 frames: {path}")
    opponent_seat = 1 - candidate_seat
    return [row["actions"][opponent_seat] for row in rows[2:]]


def action_distance(left: list[dict], right: list[dict]) -> int:
    return sum(
        json.dumps(a, sort_keys=True, separators=(",", ":"))
        != json.dumps(b, sort_keys=True, separators=(",", ":"))
        for a, b in zip(left, right, strict=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output-agent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    trace_receipt_path = resolve(args.trace_receipt)
    trace_receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    selected: dict[str, list[dict]] = {}
    analysis = []
    for candidate_seat in (0, 1):
        rows = [
            row for row in trace_receipt["traces"]
            if int(row["candidate_seat"]) == candidate_seat
        ]
        streams = [load_stream(resolve(Path(row["path"])), candidate_seat) for row in rows]
        matrix = [
            [action_distance(left, right) for right in streams]
            for left in streams
        ]
        totals = [sum(row) for row in matrix]
        medoid = min(range(len(streams)), key=lambda index: (totals[index], index))
        selected[str(1 - candidate_seat)] = streams[medoid]
        analysis.append({
            "agent_seat": 1 - candidate_seat,
            "candidate_seat": candidate_seat,
            "source_seed": int(rows[medoid]["seed"]),
            "source_trace": rows[medoid]["path"],
            "source_trace_sha256": rows[medoid]["file_sha256"],
            "pairwise_step_hamming": matrix,
            "medoid_total_distance": totals[medoid],
            "mean_pairwise_step_hamming": sum(map(sum, matrix)) / max(1, len(streams) ** 2),
        })
    payload = json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b85encode(zlib.compress(payload, 9)).decode("ascii")
    source = (
        '"""Generated approximate G05 two-seat trace-medoid proxy.\n\n'
        'NOT an exact port: runtime planning and event adaptation are absent.\n"""\n'
        "import base64\nimport json\nimport zlib\n\n"
        f"_G05_SEAT_STREAMS = json.loads(zlib.decompress(base64.b85decode({encoded!r})).decode('utf-8'))\n"
        "_CGR_STREAMS = _G05_SEAT_STREAMS\n"
        "_G05_PROXY_BOUNDARY = 'APPROXIMATE_TWO_SEAT_MEDOID_NOT_STEPWISE_EXACT'\n"
    )
    output = resolve(args.output_agent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    result = {
        "schema": "kaggriculture-public-g05-trace-medoid-proxy-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "APPROXIMATE_NOT_PARITY_ACCEPTED",
        "trace_receipt": str(trace_receipt_path),
        "trace_receipt_sha256": sha256(trace_receipt_path),
        "output_agent": str(output),
        "output_agent_sha256": sha256(output),
        "analysis": analysis,
        "acceptance_boundary": "Only a two-seat medoid of observed official actions; no claim of state/action parity outside source traces.",
    }
    receipt = resolve(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
