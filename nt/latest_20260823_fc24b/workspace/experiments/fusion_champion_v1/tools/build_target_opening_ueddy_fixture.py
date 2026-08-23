#!/usr/bin/env python3
"""Build a self-contained parity opponent with the observed opening."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OVERRIDE = r'''

# Parity fixture override: only the opening request differs from the frozen
# Ueddy controller.  The remaining 718 actions stay responsive to real state.
_ACTIONS[0] = {
    "farmer": ["BUILD_PASTURE"],
    "hands": [],
    "market": [
        ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
        ["BUY_ANIMAL", "SHEEP", 2],
        ["BUY_ANIMAL", "COW", 2],
        ["BUY_SEED", "MELON", 11],
        ["BUY_SEED", "WHEAT", 6],
        ["BUY_PRODUCT", "WHEAT", 4],
    ],
}
__version__ = "parity-fixture-target-opening-ueddy-v1"
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = (
        ROOT
        / "experiments/gold_adaptive_rule_v2/agents/"
        "gold_imitations_current_20260816_0955_v2/rank02_ueddy/main.py"
    )
    source = source_path.read_text(encoding="utf-8") + OVERRIDE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
