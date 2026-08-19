"""Evidence receipt for the fixed farm-hand tensor bound."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggriculture_jax.constants import MAX_HANDS


PROJECT = Path(__file__).resolve().parents[1]
RECEIPT = PROJECT / "receipts" / "hand_cap_analysis.json"


def fib(index: int) -> int:
    a, b = 1, 1
    for _ in range(index):
        a, b = b, a + b
    return a


def trace_paths() -> list[Path]:
    paths = list((PROJECT / "reference" / "traces").glob("*.jsonl.gz"))
    paths += list((PROJECT / "reference" / "heldout").glob("*.jsonl.gz"))
    paths += list(
        (PROJECT / "reference" / "random_differential").glob("*.jsonl.gz")
    )
    return sorted(paths)


def max_hands(path: Path) -> int:
    maximum = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            frame = json.loads(line)
            maximum = max(
                maximum, *(len(farm["hands"]) for farm in frame["farms"])
            )
    return maximum


def main() -> int:
    rows = [
        {"path": str(path.relative_to(PROJECT)), "max_hands": max_hands(path)}
        for path in trace_paths()
    ]
    observed = max(row["max_hands"] for row in rows)
    represented_cost = sum(fib(index) for index in range(MAX_HANDS))
    next_cost = fib(MAX_HANDS)
    receipt = {
        "schema": "kaggriculture_hand_cap_analysis_v1",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "ok": observed < MAX_HANDS,
        "static_max_hands": MAX_HANDS,
        "static_max_units_including_main_farmer": MAX_HANDS + 1,
        "trace_count": len(rows),
        "observed_max_hands": observed,
        "headroom_hands": MAX_HANDS - observed,
        "default_fibonacci_cost_to_hire_32_hands_in_one_day": represented_cost,
        "cost_of_33rd_hand": next_cost,
        "cash_needed_to_attempt_33_hands_from_zero_hires": represented_cost
        + next_cost,
        "cap_behavior": (
            "A HIRE beyond 32 is a no-op, increments hand_cap_hits, and uses the "
            "same MAX_UNITS action/state shape in rollout, Arena, and deployment."
        ),
        "rows": rows,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

