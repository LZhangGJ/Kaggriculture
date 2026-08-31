"""Run the hardened v2 nested residual real-feature smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import run_phase_challenger_residual_nested_smoke_v1 as v1
import train_phase_challenger_residual_nested_v2 as nested_v2


SCHEMA = "phase-challenger-residual-nested-real-feature-smoke-v2"
DEFAULT_OUTPUT = v1.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_residual_nested_real_feature_smoke_v2_20260829a"
)


def parser():
    result = v1.parser()
    result.set_defaults(output_root=DEFAULT_OUTPUT)
    return result


def run(args):
    v1.nested = nested_v2
    v1.SCHEMA = SCHEMA
    return v1.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
