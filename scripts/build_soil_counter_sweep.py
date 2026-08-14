"""Build V17 variants with a sparse future-sale counter for the Soil route."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import zlib
from pathlib import Path


VARIANTS = (
    {"name": "soil_l1_f025", "lookahead": 1, "fraction": 0.25},
    {"name": "soil_l1_f050", "lookahead": 1, "fraction": 0.50},
    {"name": "soil_l1_f075", "lookahead": 1, "fraction": 0.75},
    {"name": "soil_l1_f100", "lookahead": 1, "fraction": 1.00},
    {"name": "soil_l2_f025", "lookahead": 2, "fraction": 0.25},
    {"name": "soil_l2_f050", "lookahead": 2, "fraction": 0.50},
    {"name": "soil_l2_f075", "lookahead": 2, "fraction": 0.75},
    {"name": "soil_l2_f100", "lookahead": 2, "fraction": 1.00},
)


def _replace(source: str, name: str, value: str) -> str:
    rendered, count = re.subn(
        rf"(?m)^{re.escape(name)}\s*=\s*.*$", f"{name} = {value}", source, count=1
    )
    if count != 1:
        raise ValueError(f"expected one {name}, found {count}")
    return rendered


def _soil_markets(path: Path) -> list[list[list[object]]]:
    spec = importlib.util.spec_from_file_location("soil_counter_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    route = module._MODAL_NS["_ROUTE"]
    return [[list(order) for order in (action.get("market") or [])] for action in route]


COUNTER_TEMPLATE = r'''
_SOIL_COUNTER_MARKETS = json.loads(zlib.decompress(base64.b85decode({payload!r})).decode("utf-8"))
_SOIL_COUNTER_LOOKAHEAD = {lookahead}
_SOIL_COUNTER_FRACTION = {fraction!r}
_SOIL_COUNTER_ITEMS = ("MELON", "MILK", "STRAWBERRY", "WOOL", "FERTILIZER")


def _soil_route_counter(obs, action, step):
    if not _v17_is_r5_family(obs, step):
        return action
    future = step + _SOIL_COUNTER_LOOKAHEAD
    if future >= len(_SOIL_COUNTER_MARKETS):
        return action
    targets = {{}}
    for order in _SOIL_COUNTER_MARKETS[future]:
        if len(order) >= 3 and order[0] == "SELL" and order[1] in _SOIL_COUNTER_ITEMS:
            targets[order[1]] = targets.get(order[1], 0) + max(0, int(order[2] or 0))
    if not targets:
        return action
    action = _copy_action(action)
    market = [list(order) for order in action.get("market", []) or []]
    remaining = _projected_shed(obs, action)
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            remaining[order[1]] = max(0, int(remaining.get(order[1], 0) or 0) - max(0, int(order[2] or 0)))
    for item in _SOIL_COUNTER_ITEMS:
        target = targets.get(item, 0)
        if target <= 0 or len(market) >= 10:
            continue
        available = max(0, int(remaining.get(item, 0) or 0) - _v17_pickup_reserve(action, item))
        quantity = min(available, max(1, int(round(target * _SOIL_COUNTER_FRACTION))))
        if quantity <= 0:
            continue
        current = next(
            (order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
            None,
        )
        if current is None:
            market.append(["SELL", item, quantity])
        else:
            current[2] = max(0, int(current[2] or 0)) + quantity
        remaining[item] = max(0, available - quantity)
    action["market"] = market[:10]
    return action
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17-source", type=Path, required=True)
    parser.add_argument("--soil-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = args.v17_source.read_text(encoding="utf-8")
    source = _replace(source, "_PREEMPT_ENABLED", "True")
    source = _replace(source, "_PREEMPT_FRACTION", "1.0")
    source = _replace(source, "_PREEMPT_MAX_CLONE_DISTANCE", "0")
    source = _replace(source, "_PREEMPT_MIN_FUTURE_QUANTITY", "1")
    source = _replace(source, "_PREEMPT_START", "24")
    source = _replace(source, "_PREEMPT_STOP", "716")
    markets = _soil_markets(args.soil_source)
    payload = base64.b85encode(zlib.compress(json.dumps(markets, separators=(",", ":")).encode("utf-8"))).decode("ascii")
    marker = "\ndef agent(obs):\n"
    if marker not in source:
        raise ValueError("V17 agent marker not found")
    entries = []
    for variant in VARIANTS:
        counter = COUNTER_TEMPLATE.format(payload=payload, **variant)
        rendered = source.replace(marker, counter + marker, 1)
        rendered = rendered.replace(
            "        action = _v17_r5_counter(obs, action, step)",
            "        action = _soil_route_counter(obs, action, step)\n        action = _v17_r5_counter(obs, action, step)",
            1,
        )
        rendered = rendered.replace(
            "__version__ = 'BL-V17-R1-RC2'",
            f"__version__ = 'BL-V17-SOIL-{variant['name']}'",
            1,
        )
        candidate_dir = args.output_dir / variant["name"]
        candidate_dir.mkdir(parents=True, exist_ok=False)
        path = candidate_dir / "main.py"
        path.write_text(rendered, encoding="utf-8")
        compile(rendered, str(path), "exec")
        entries.append(
            {
                **variant,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "kind": "v17_soil_counter_sweep",
        "v17_source": str(args.v17_source.resolve()),
        "soil_source": str(args.soil_source.resolve()),
        "variants": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "variants": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
