#!/usr/bin/env python3
"""Pack the searched route agent into one zlib/Base85 Python submission."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path


def _module_body(path: Path, *, remove: tuple[str, ...] = ()) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    for line in lines:
        if line == "from __future__ import annotations" or line in remove:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _compressed_literal(data: bytes) -> str:
    return repr(base64.b85encode(zlib.compress(data, level=9)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("teammate_meta_route_submission_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("teammate_meta_route_175_single.py")
    )
    parser.add_argument(
        "--forced-opening", choices=("G001", "G136"), default=None,
        help="Fix the root route while retaining all learned switch nodes.",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()

    feature_source = _module_body(
        source_dir / "meta_agent/src/route_switch_features.py"
    )
    controller_source = _module_body(
        source_dir / "meta_agent/src/search_route_policy.py",
        remove=(
            "from .route_switch_features import RouteSwitchHistory, route_switch_vector",
        ),
    )
    adapter_source = _module_body(
        source_dir / "meta_agent/src/teammate_expanded_routes.py"
    )

    base = (source_dir / "teammate_base.py").read_bytes()
    # route_actions is already zlib-compressed; embedding its bytes avoids a
    # second compression layer in the expanded runtime.
    actions = (source_dir / "route_actions.json.zlib").read_bytes()
    metadata = json.loads((source_dir / "route_library.json").read_text(encoding="utf-8"))
    policy = json.loads((source_dir / "route_policy.json").read_text(encoding="utf-8"))
    nash = json.loads((source_dir / "opening_nash.json").read_text(encoding="utf-8"))
    forced_opening_line = (
        f'_S_CONTROLLER.forced_opening = {args.forced_opening!r}'
        if args.forced_opening else ""
    )

    runtime = f'''# Generated searched-route runtime.  Do not edit this expanded payload directly.
import base64 as _bundle_b64
import zlib as _bundle_zlib

{feature_source}
{controller_source}
{adapter_source}

_S_BASE = _bundle_zlib.decompress(_bundle_b64.b85decode({_compressed_literal(base)})).decode("utf-8")
_S_ACTIONS = json.loads(_bundle_zlib.decompress(_bundle_b64.b85decode({repr(base64.b85encode(actions))})))
_S_METADATA = {repr(metadata)}
_S_POLICY_PAYLOAD = {repr(policy)}
_S_NASH = {repr(nash)}
_S_ROUTE_BY_FAMILY = {{
    str(value["family"]): str(value["route_id"])
    for value in _S_METADATA["opponent_routes"]
}}
_S_OPENING_WEIGHTS = [
    (str(value["family"]), float(value["weight"]))
    for value in _S_NASH["opening_support"]
]
_S_EXPANDED = TeammateExpandedRouteAgent(
    _S_BASE, _S_ACTIONS, "searched_teammate_single_file"
)
_S_CONTROLLER = SearchRouteController(
    _S_POLICY_PAYLOAD, _S_ROUTE_BY_FAMILY, _S_OPENING_WEIGHTS, rng_seed=None
)
{forced_opening_line}
_S_POLICY = SearchRoutedTeammateAgent(
    _S_EXPANDED, _S_CONTROLLER, _S_POLICY_PAYLOAD.get("targets", ())
)

def agent(observation, configuration=None):
    return _S_POLICY(observation, configuration)
'''
    compressed = zlib.compress(runtime.encode("utf-8"), level=9)
    encoded = base64.b85encode(compressed)
    wrapper = (
        "# Single-file Kaggriculture agent: zlib + Base85 payload.\n"
        "import base64 as _b,zlib as _z\n"
        f"exec(_z.decompress(_b.b85decode({encoded!r})).decode('utf-8'))\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(wrapper, encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output.resolve()),
        "source_bytes": len(runtime.encode("utf-8")),
        "compressed_bytes": len(compressed),
        "file_bytes": args.output.stat().st_size,
        "sha256": digest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
