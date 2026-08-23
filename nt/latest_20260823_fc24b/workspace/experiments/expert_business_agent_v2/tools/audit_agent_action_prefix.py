#!/usr/bin/env python3
"""Audit exact static action-stream prefixes exposed by local Agents.

This is intentionally a structural audit: it does not claim that two Agents
remain semantically compatible after a switch.  It only records whether their
frozen choreography arrays are byte-for-byte equal up to the first mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load(path: Path, label: str):
    name = "eba2_prefix_" + hashlib.sha256((label + str(path)).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streams(module: Any) -> dict[str, list[Any]]:
    if hasattr(module, "_CGR_STREAMS"):
        return {str(key): list(value) for key, value in module._CGR_STREAMS.items()}
    if hasattr(module, "_ACTIONS"):
        return {"_ACTIONS": list(module._ACTIONS)}
    result: dict[str, list[Any]] = {}
    for embedded_name in ("_PRT", "_RANK1", "_ALT", "_BASE"):
        embedded = getattr(module, embedded_name, None)
        if isinstance(embedded, dict):
            if "_CGR_STREAMS" in embedded:
                for key, value in embedded["_CGR_STREAMS"].items():
                    result[f"{embedded_name}:{key}"] = list(value)
            elif "_ACTIONS" in embedded:
                result[f"{embedded_name}:_ACTIONS"] = list(embedded["_ACTIONS"])
    for namespace_name in ("_MODAL_NS", "_RC5_NS"):
        namespace = getattr(module, namespace_name, None)
        if not isinstance(namespace, dict):
            continue
        if "_CGR_STREAMS" in namespace:
            for key, value in namespace["_CGR_STREAMS"].items():
                result[f"{namespace_name}:{key}"] = list(value)
        if "_ACTIONS" in namespace:
            result[f"{namespace_name}:_ACTIONS"] = list(namespace["_ACTIONS"])
        # Some public wrappers use a differently named 719/720-step list.
        for key, value in namespace.items():
            if (
                isinstance(value, list)
                and len(value) >= 700
                and (not value or isinstance(value[0], dict))
            ):
                result.setdefault(f"{namespace_name}:{key}", list(value))
    if not result:
        raise ValueError("No exposed static action stream")
    return result


def common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def digest(actions: list[Any]) -> str:
    payload = json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left_path = resolve(args.left)
    right_path = resolve(args.right)
    left_streams = streams(load(left_path, "left"))
    right_streams = streams(load(right_path, "right"))
    comparisons = []
    for left_name, left_actions in left_streams.items():
        for right_name, right_actions in right_streams.items():
            prefix = common_prefix(left_actions, right_actions)
            comparisons.append({
                "left_stream": left_name,
                "right_stream": right_name,
                "common_prefix": prefix,
                "left_length": len(left_actions),
                "right_length": len(right_actions),
                "left_sha256": digest(left_actions),
                "right_sha256": digest(right_actions),
                "left_first_mismatch": left_actions[prefix] if prefix < len(left_actions) else None,
                "right_first_mismatch": right_actions[prefix] if prefix < len(right_actions) else None,
            })
    result = {
        "schema": "eba2-agent-action-prefix-audit-v1",
        "left": str(left_path),
        "right": str(right_path),
        "comparisons": sorted(comparisons, key=lambda row: row["common_prefix"], reverse=True),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
