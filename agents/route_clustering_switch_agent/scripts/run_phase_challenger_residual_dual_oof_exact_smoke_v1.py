"""Run compact exact-threshold dual-OOF residual calibration smoke."""

from __future__ import annotations

from typing import Sequence

import run_phase_challenger_residual_nested_smoke_v1 as runner
import train_phase_challenger_residual_dual_oof_exact_v1 as exact_nested


SCHEMA = "phase-challenger-residual-dual-oof-exact-real-feature-smoke-v1"
DEFAULT_OUTPUT = runner.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_residual_dual_oof_exact_real_feature_smoke_v1_20260829a"
)


def parser():
    result = runner.parser()
    result.set_defaults(
        output_root=DEFAULT_OUTPUT,
        representations=("compact1078",),
    )
    return result


def run(args):
    runner.nested = exact_nested
    runner.SCHEMA = SCHEMA
    return runner.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
