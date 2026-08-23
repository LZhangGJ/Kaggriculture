#!/usr/bin/env python3
"""Summarize selected atomic operations in a generated route action stream."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_route_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streams(module: Any) -> dict[str, list[dict[str, Any]]]:
    if hasattr(module, "_CGR_STREAMS"):
        values = dict(module._CGR_STREAMS)
        fixed = getattr(module, "_FCGR_ROUTE", None) or getattr(module, "_FRP_ROUTE", None)
        if fixed in values:
            return {str(fixed): list(values[fixed])}
        return {str(key): list(value) for key, value in values.items()}
    if hasattr(module, "_ACTIONS"):
        return {"_ACTIONS": list(module._ACTIONS)}
    raise KeyError("module has neither _CGR_STREAMS nor _ACTIONS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=719)
    parser.add_argument("--verbs", default="BUILD_PASTURE,BUY_ANIMAL,PICKUP,PLACE,SELL")
    parser.add_argument("--items", default="COW,SHEEP,MILK,WOOL")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    module = load_module(args.agent.resolve())
    wanted_verbs = {value.strip() for value in args.verbs.split(",") if value.strip()}
    wanted_items = {value.strip() for value in args.items.split(",") if value.strip()}
    result: dict[str, list[dict[str, Any]]] = {}
    for name, actions in streams(module).items():
        rows = []
        for step, action in enumerate(actions):
            if not args.start <= step <= args.stop:
                continue
            units = [("farmer", action.get("farmer", []))]
            units += [(f"hand{index}", value) for index, value in enumerate(action.get("hands", []) or [])]
            atoms = [(label, list(value or [])) for label, value in units]
            atoms += [(f"market{index}", list(value or [])) for index, value in enumerate(action.get("market", []) or [])]
            selected = []
            for label, atom in atoms:
                verb = str(atom[0]) if atom else ""
                item = str(atom[1]) if len(atom) >= 2 else ""
                if verb not in wanted_verbs:
                    continue
                if wanted_items and verb != "BUILD_PASTURE" and item not in wanted_items:
                    continue
                selected.append({"slot": label, "action": atom})
            if selected:
                rows.append({"step": step, "selected": selected})
        result[name] = rows

    payload = {
        "schema": "route-stream-selected-operations-v1",
        "agent": str(args.agent.resolve()),
        "start": args.start,
        "stop": args.stop,
        "verbs": sorted(wanted_verbs),
        "items": sorted(wanted_items),
        "streams": result,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
