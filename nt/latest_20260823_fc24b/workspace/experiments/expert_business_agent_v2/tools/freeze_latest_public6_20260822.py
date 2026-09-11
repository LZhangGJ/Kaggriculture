from __future__ import annotations

import argparse
import base64
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

from freeze_latest_public8 import (
    AgentSpec,
    EXTRACTORS,
    extract_kaito_payload,
    extract_x562_source,
    find_assignments,
    freeze_one,
    safe_static_value,
)


SPECS = (
    AgentSpec(
        "boatlee_v21_latest",
        "boatlee__v21-r1-public-state-route-portfolio",
        "b64_zlib_payload",
    ),
    AgentSpec(
        "prvsiyan_soil_v26h_latest",
        "prvsiyan__soil-remembers-rain-latest",
        "soil_b85_zlib_payload",
    ),
    AgentSpec(
        "prvsiyan_moon_v92_latest",
        "prvsiyan__moon-counts-melons-latest",
        "moon_b85_zlib_payload",
    ),
    AgentSpec(
        "kaito_v39_history_gate_latest",
        "kaitofukami__v39-history-gate",
        "kaito_payload",
    ),
    AgentSpec(
        "steven_e284_hadouken_latest",
        "stevenleehans__e284-hadouken",
        "x562_source",
    ),
    AgentSpec(
        "salem_harvestforge_x_latest",
        "salemali7__3094-score-latest",
        "salem_b64_payload",
    ),
)


def extract_b64_zlib_payload(cells):
    index, assignments = find_assignments(cells, {"PAYLOAD", "EXPECTED_SOURCE_SHA"})
    payload = safe_static_value(assignments["PAYLOAD"])
    expected = str(safe_static_value(assignments["EXPECTED_SOURCE_SHA"]))
    return zlib.decompress(base64.b64decode(payload)), index, expected


def extract_soil_b85_zlib_payload(cells):
    index, assignments = find_assignments(
        cells, {"AGENT_PAYLOAD", "EXPECTED_AGENT_SHA256"}
    )
    payload = safe_static_value(assignments["AGENT_PAYLOAD"])
    expected = str(safe_static_value(assignments["EXPECTED_AGENT_SHA256"]))
    return zlib.decompress(base64.b85decode(payload.encode("ascii"))), index, expected


def extract_moon_b85_zlib_payload(cells):
    payload_index, payload_assignments = find_assignments(cells, {"PAYLOAD"})
    _, sha_assignments = find_assignments(cells, {"CANDIDATE_SHA"})
    payload = safe_static_value(payload_assignments["PAYLOAD"])
    expected = str(safe_static_value(sha_assignments["CANDIDATE_SHA"]))
    return (
        zlib.decompress(base64.b85decode(payload.encode("ascii"))),
        payload_index,
        expected,
    )


def extract_salem_b64_payload(cells):
    payload_index, payload_assignments = find_assignments(cells, {"AGENT_B64"})
    _, sha_assignments = find_assignments(cells, {"EXPECTED_MAIN_SHA"})
    payload = safe_static_value(payload_assignments["AGENT_B64"])
    expected = str(safe_static_value(sha_assignments["EXPECTED_MAIN_SHA"]))
    return base64.b64decode(payload), payload_index, expected


EXTRACTORS.update(
    {
        "b64_zlib_payload": extract_b64_zlib_payload,
        "soil_b85_zlib_payload": extract_soil_b85_zlib_payload,
        "moon_b85_zlib_payload": extract_moon_b85_zlib_payload,
        "kaito_payload": extract_kaito_payload,
        "x562_source": extract_x562_source,
        "salem_b64_payload": extract_salem_b64_payload,
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically freeze the six selected public agents from 2026-08-22."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    agents = [freeze_one(input_root, output_root, spec) for spec in SPECS]
    manifest = {
        "schema": "kaggriculture.latest_public6_20260822.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "rules": {
            "static_extraction_only": True,
            "embedded_hash_enforced": True,
            "python_compile_required": True,
            "notebook_code_executed": False,
        },
        "agents": agents,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
