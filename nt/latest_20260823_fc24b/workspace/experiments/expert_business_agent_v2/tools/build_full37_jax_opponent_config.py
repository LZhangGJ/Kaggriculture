#!/usr/bin/env python3
"""Translate the frozen official 37-pool into compiler-facing JAX names."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/expert_business_agent_v2/configs/local_representative_pool_v1.json"
OUTPUT = ROOT / "experiments/expert_business_agent_v2/configs/jax_full37_mixed_exact_proxy_opponents_v1.json"
RENAMES = {
    "public_g02_rc5_c166": "public14_rank_agent_v17",
    "public_g03_read_market": "public04_read_market",
    "public_g04_soil_rain": "public06_soil_rain",
    "local_public_opening_router_v8": "public_opening_router_v8",
}


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    opponents = []
    for row in source["opponents"]:
        name = RENAMES.get(row["name"], row["name"])
        path = row["path"]
        if row["name"] == "public_g05_structured_econ":
            path = "experiments/expert_business_agent_v2/agents/public_g05_structured_econ_trace_proxy_v1/main.py"
        opponents.append({"name": name, "path": path})
    result = {
        "schema": "kaggriculture-jax-full37-mixed-exact-proxy-opponents-v1",
        "description": (
            "Compiler-facing frozen 37 pool: 19 strict exact public/local ports, "
            "17 Replay-derived route proxies, and one explicit G05 two-seat medoid proxy."
        ),
        "source_pool": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "opponents": opponents,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "opponents": len(opponents)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
