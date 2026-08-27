#!/usr/bin/env python3
"""Export the minimal Python runtime for the searched teammate route policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ENTRYPOINT = '''from pathlib import Path
import json
import sys

if "__file__" in globals():
    _ROOT = Path(globals()["__file__"]).resolve().parent
elif Path("/kaggle_simulations/agent").is_dir():
    _ROOT = Path("/kaggle_simulations/agent")
else:
    _ROOT = Path.cwd()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from meta_agent.src.search_route_policy import SearchRouteController, SearchRoutedTeammateAgent
from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes

_SOURCE = (_ROOT / "teammate_base.py").read_text(encoding="utf-8")
_TAPES = load_action_tapes(_ROOT / "route_actions.json.zlib")
_METADATA = json.loads((_ROOT / "route_library.json").read_text(encoding="utf-8"))
_POLICY_PAYLOAD = json.loads((_ROOT / "route_policy.json").read_text(encoding="utf-8"))
_NASH = json.loads((_ROOT / "opening_nash.json").read_text(encoding="utf-8"))
_ROUTE_BY_FAMILY = {
    str(value["family"]): str(value["route_id"])
    for value in _METADATA["opponent_routes"]
}
_OPENING_WEIGHTS = [
    (str(value["family"]), float(value["weight"]))
    for value in _NASH["opening_support"]
]
_EXPANDED = TeammateExpandedRouteAgent(_SOURCE, _TAPES, "searched_teammate_submission")
_CONTROLLER = SearchRouteController(
    _POLICY_PAYLOAD, _ROUTE_BY_FAMILY, _OPENING_WEIGHTS, rng_seed=None
)
_CONTROLLER.forced_opening = __FORCED_OPENING__
_POLICY = SearchRoutedTeammateAgent(
    _EXPANDED, _CONTROLLER, _POLICY_PAYLOAD.get("targets", ())
)


def agent(observation, configuration=None):
    return _POLICY(observation, configuration)
'''


def _copyfile(source: Path, destination: Path) -> None:
    """Copy an export asset unless source and destination are the same file."""
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--nash", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--forced-opening",
        help="Use a deterministic opening while retaining trajectory-conditioned switching.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    runtime_nodes = []
    route_q_policy = "q_model_npz_base64" in policy
    exact_state_policy = "exact_state_model_npz_base64" in policy
    for wrapped in policy["nodes"]:
        selected = wrapped["selected"]
        if not selected.get("enabled", True):
            continue
        runtime_selected = {
            "opening": str(selected["opening"]),
            "checkpoint": int(selected["checkpoint"]),
            "enabled": True,
        }
        if not route_q_policy and not exact_state_policy:
            runtime_selected["tree"] = selected["tree"]
        runtime_nodes.append({"selected": runtime_selected})
    runtime_policy = {
        "schema_version": 1,
        "kind": "searched_one_switch_route_sequence_runtime",
        "feature_schema": policy["feature_schema"],
        "targets": policy["targets"],
        "nodes": runtime_nodes,
        "sequence": policy.get("sequence_selection", {}).get(
            "checkpoints", [value["selected"]["checkpoint"] for value in runtime_nodes]
        ),
    }
    if route_q_policy:
        runtime_policy["kind"] = "backward_counterfactual_route_q_runtime"
        runtime_policy["q_model_npz_base64"] = policy["q_model_npz_base64"]
        runtime_policy["q_model"] = policy.get("q_model", {})
        if "fallback_nodes" in policy:
            runtime_policy["kind"] = "gated_backward_counterfactual_route_q_runtime"
            runtime_policy["fallback_nodes"] = policy["fallback_nodes"]
        if "mode_nodes" in policy:
            runtime_policy["kind"] = "multimode_gated_backward_route_q_runtime"
            runtime_policy["mode_nodes"] = policy["mode_nodes"]
        if "gate" in policy:
            runtime_policy["gate"] = policy["gate"]
    if exact_state_policy:
        runtime_policy["kind"] = "two_stage_exact_public_state_route_runtime"
        runtime_policy["exact_state_model_npz_base64"] = policy[
            "exact_state_model_npz_base64"
        ]
        runtime_policy["exact_state_model"] = policy.get("exact_state_model", {})
    nash = json.loads(args.nash.read_text(encoding="utf-8"))
    runtime_nash = {
        "schema_version": 1,
        "source": str(args.nash),
        "value": nash["value"],
        "opening_support": nash["opening_support"],
    }
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))

    args.output.mkdir(parents=True, exist_ok=True)
    entrypoint = ENTRYPOINT.replace("__FORCED_OPENING__", repr(args.forced_opening))
    (args.output / "main.py").write_text(entrypoint, encoding="utf-8")
    _copyfile(args.base, args.output / "teammate_base.py")
    _copyfile(args.actions, args.output / "route_actions.json.zlib")
    _copyfile(args.metadata, args.output / "route_library.json")
    (args.output / "route_policy.json").write_text(
        json.dumps(runtime_policy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (args.output / "opening_nash.json").write_text(
        json.dumps(runtime_nash, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    modules = (
        "__init__.py", "src/__init__.py", "src/search_route_policy.py",
        "src/teammate_expanded_routes.py", "src/route_switch_features.py",
    )
    project = Path(__file__).resolve().parents[1]
    for relative in modules:
        source = project / "src" / "meta_agent" / relative
        destination = args.output / "meta_agent" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = {
        "schema_version": 1,
        "description": "Teammate execution stack plus expanded genetic-route meta controller",
        "runtime_routes": len(json.loads(args.metadata.read_text(encoding="utf-8"))["opponent_routes"]),
        "forced_opening": args.forced_opening,
        "opening_support": runtime_nash["opening_support"],
        "switch_nodes": [
            {
                "opening": value["selected"]["opening"],
                "checkpoint": value["selected"]["checkpoint"],
            }
            for value in runtime_nodes
        ],
        "final_holdout": holdout.get(
            "best", holdout.get("metrics", holdout.get("selector", {}))
        ),
        "sources": {
            "base": str(args.base), "actions": str(args.actions),
            "metadata": str(args.metadata), "policy": str(args.policy),
            "nash": str(args.nash), "holdout": str(args.holdout),
        },
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "files": sorted(str(value.relative_to(args.output)) for value in args.output.rglob("*") if value.is_file()),
        "bytes": sum(value.stat().st_size for value in args.output.rglob("*") if value.is_file()),
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
