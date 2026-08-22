"""Cluster the 30 reference agents from comparable interaction evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.behavior_diagnostics import (
    build_response_features,
    cluster_behavior_profiles,
    clustering_report,
    common_opponents,
    load_panel_json,
    load_round_robin_csv,
    write_clustering_outputs,
)
from kaggriculture_lab.decision_schema import REFERENCE_STRATEGY_AGENTS


def _panel(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("panel must be AGENT=PATH")
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-results", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        type=_panel,
        default=[],
        metavar="AGENT=PATH",
        help="Add an agent panel receipt (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clusters", type=int, default=8)
    args = parser.parse_args()

    agents = list(REFERENCE_STRATEGY_AGENTS)
    records = load_round_robin_csv(args.pair_results)
    provenance: list[dict[str, object]] = [
        {
            "kind": "exact_round_robin",
            "agents": len({row.agent for row in records}),
            "source": str(args.pair_results.resolve()),
        }
    ]
    for name, path in args.panel:
        panel_records, panel_provenance = load_panel_json(path, agent_name=name)
        records.extend(panel_records)
        provenance.append({"kind": "panel_proxy", **panel_provenance})

    missing_agents = sorted(set(agents) - {row.agent for row in records})
    if missing_agents:
        parser.error("missing comparable evidence for: " + ", ".join(missing_agents))
    opponents = common_opponents(records, agents)
    if len(opponents) < 3:
        parser.error(
            f"only {len(opponents)} common opponents; at least three are required"
        )
    features, feature_names, missing = build_response_features(
        records, agents, opponents
    )
    result = cluster_behavior_profiles(
        agents,
        opponents,
        features,
        feature_names,
        missing,
        max_clusters=args.max_clusters,
    )
    report = clustering_report(result, records, provenance)
    write_clustering_outputs(report, args.output_dir)
    print(
        f"agents={report['agent_count']} common_opponents={len(opponents)} "
        f"clusters={report['cluster_count']} silhouette={report['silhouette']:.4f}"
    )
    for cluster in report["clusters"]:
        print(
            f"cluster={cluster['cluster']} size={cluster['size']} "
            f"agents={','.join(cluster['agents'])}"
        )
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
