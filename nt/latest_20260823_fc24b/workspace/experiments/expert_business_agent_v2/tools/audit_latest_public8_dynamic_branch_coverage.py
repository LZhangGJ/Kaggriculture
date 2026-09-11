from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "gpu_sim" / "src",):
    sys.path.insert(0, str(path))

from kaggriculture_jax import encode_actions, stack_actions  # noqa: E402


ALLOWED_ROUTES = {
    "deniz_v111_8c4s_latest": (10,),
    "boatlee_v20_latest": tuple(range(8)),
    "kunal_2026_v1_latest": tuple(range(8)),
    "rayk_rank_agent_latest": tuple(range(8)),
    "kaito_v36_latest": (11,),
    "x562_latest": (12, 13),
    "tetsutani_adaptive_latest": tuple(range(8)),
    "flex_multi_route_latest": (14,),
}
UNIT_FIELDS = ("unit_op", "unit_item", "unit_amount", "unit_count")
MARKET_FIELDS = ("market_op", "market_item", "market_amount", "market_count")


def load_actions(path: Path, seat: int):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    frames = [row for row in rows if row.get("record_type") == "frame"]
    encoded = stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
    return {field: np.asarray(getattr(encoded, field))[:, seat] for field in encoded._fields}, frames


def field_group_matches(source, bank, route: int, fields: tuple[str, ...]) -> np.ndarray:
    result = np.ones((719,), dtype=bool)
    for field in fields:
        expected = np.asarray(bank[field][route])
        actual = source[field]
        axes = tuple(range(1, actual.ndim))
        result &= np.all(actual == expected, axis=axes) if axes else actual == expected
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--base-traces", type=Path, required=True)
    parser.add_argument("--x562-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = np.load(args.bank.resolve())
    base = json.loads(args.base_traces.read_text(encoding="utf-8"))
    x562 = json.loads(args.x562_traces.read_text(encoding="utf-8"))
    # X562 uses the expanded eight-seed branch corpus; do not count its two
    # base seeds a second time.  The other seven identities use the base corpus.
    trace_rows = [
        row for row in base["traces"] if row["opponent"] != "x562_latest"
    ] + x562["traces"]
    results = []
    for agent, routes in ALLOWED_ROUTES.items():
        rows = [row for row in trace_rows if row["opponent"] == agent]
        contexts = []
        unit_deviation_total = 0
        market_deviation_total = 0
        any_deviation_total = 0
        high_route_contexts = 0
        low_route_contexts = 0
        for row in rows:
            seat = int(row["candidate_seat"])
            source, frames = load_actions(ROOT / row["path"], seat)
            unit_matches = np.stack(
                [field_group_matches(source, bank, route, UNIT_FIELDS) for route in routes]
            )
            market_matches = np.stack(
                [field_group_matches(source, bank, route, MARKET_FIELDS) for route in routes]
            )
            full_matches = unit_matches & market_matches
            unit_deviation = ~np.any(unit_matches, axis=0)
            market_deviation = ~np.any(market_matches, axis=0)
            any_deviation = ~np.any(full_matches, axis=0)
            shops = next(frame for frame in frames if frame.get("step") == 168)["town"][
                "unlocked_shops"
            ]
            x562_high = agent == "x562_latest" and "YARN_STORE" in shops and not (
                len(shops) >= 2
                and shops[0] == "ICE_CREAM_SHOP"
                and shops[1] == "YARN_STORE"
            )
            high_route_contexts += int(x562_high)
            low_route_contexts += int(agent == "x562_latest" and not x562_high)
            context = {
                "seed": int(row["seed"]),
                "candidate_seat": seat,
                "unit_overlay_steps": int(np.sum(unit_deviation)),
                "market_overlay_steps": int(np.sum(market_deviation)),
                "any_overlay_steps": int(np.sum(any_deviation)),
                "x562_route": "high" if x562_high else ("low" if agent == "x562_latest" else None),
            }
            contexts.append(context)
            unit_deviation_total += context["unit_overlay_steps"]
            market_deviation_total += context["market_overlay_steps"]
            any_deviation_total += context["any_overlay_steps"]
        results.append(
            {
                "agent": agent,
                "contexts": len(contexts),
                "allowed_raw_route_ids": list(routes),
                "unit_overlay_steps_total": unit_deviation_total,
                "market_overlay_steps_total": market_deviation_total,
                "any_overlay_steps_total": any_deviation_total,
                "x562_high_contexts": high_route_contexts,
                "x562_low_contexts": low_route_contexts,
                "contexts_detail": contexts,
            }
        )
    output = {
        "schema": "kaggriculture.latest_public8_dynamic_branch_coverage.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank": str(args.bank.resolve()),
        "method": (
            "For each official Python action, compare the complete unit and market "
            "tensor to every raw route allowed by that source at the same step. A "
            "non-match proves a runtime overlay changed the emitted action."
        ),
        "limitation": (
            "A zero count does not prove missing code: an implemented source branch may "
            "remain inactive in the frozen acceptance contexts. Source audit plus strict "
            "official parity remains the acceptance basis."
        ),
        "results": results,
        "status": "PASS",
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
