#!/usr/bin/env python3
"""Audit exact action-prefix compatibility inside a route manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_actions(path: Path, label: str) -> list[Any]:
    name = "prefix_group_" + hashlib.sha256((label + str(path)).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "_CGR_STREAMS"):
        streams = dict(module._CGR_STREAMS)
        route = getattr(module, "_FCGR_ROUTE", None) or getattr(module, "_FRP_ROUTE", None)
        return list(streams[route] if route in streams else next(iter(streams.values())))
    if hasattr(module, "_ACTIONS"):
        return list(module._ACTIONS)
    raise ValueError(f"No static action stream: {path}")


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steps", default="1,5,10,20,48,63,72,96,120,168,216")
    parser.add_argument("--select", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = {value.strip() for value in args.select.split(",") if value.strip()}
    entries = [
        item for item in manifest["candidates"]
        if not selected or str(item["name"]) in selected
    ]
    streams = {
        str(item["name"]): load_actions(resolve(str(item["path"])), str(item["name"]))
        for item in entries
    }
    steps = sorted({max(1, int(value)) for value in args.steps.split(",") if value.strip()})
    groups: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        members: dict[str, list[str]] = {}
        for name, actions in streams.items():
            members.setdefault(digest(actions[:step]), []).append(name)
        groups[str(step)] = sorted(
            ({"size": len(names), "routes": sorted(names)} for names in members.values()),
            key=lambda item: (-item["size"], item["routes"]),
        )

    pairwise = []
    names = sorted(streams)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise.append({
                "left": left,
                "right": right,
                "common_prefix": common_prefix(streams[left], streams[right]),
            })
    pairwise.sort(key=lambda item: (-item["common_prefix"], item["left"], item["right"]))

    output = {
        "schema": "route-manifest-prefix-groups-v1",
        "manifest": str(args.manifest),
        "route_count": len(streams),
        "steps": steps,
        "groups": groups,
        "pairwise": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
