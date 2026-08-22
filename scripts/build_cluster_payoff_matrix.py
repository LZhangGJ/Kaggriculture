"""Build the official-replay behavior-cluster meta-game and PSRO mixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.meta_strategy import (
    build_payoff_matrix,
    payoff_matrix_report,
    payoff_records_from_official_manifest,
    write_payoff_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cluster_payoff_matrix_v1.json"),
    )
    args = parser.parse_args()
    records = payoff_records_from_official_manifest(args.manifest)
    report = payoff_matrix_report(build_payoff_matrix(records))
    write_payoff_report(report, args.output)
    print(
        f"matches={len(records)} clusters={len(report['cluster_labels'])} "
        f"coverage={report['coverage']:.3f} "
        f"worst={report['worst_cluster_value']:.4f} saved={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
