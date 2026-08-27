#!/usr/bin/env python3
"""Import fixed action tapes from the NT 68-agent assets without using JAX.

The source ``.npz`` files contain plain NumPy arrays produced by the separate
simulator.  This script decodes those arrays directly into the replay-tape
format consumed by ``NativeTeammateExecutor`` and the compiled C++ simulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
UNIT_OPS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
)
MARKET_OPS = (
    "NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL",
)
ACTION_ARRAYS = (
    "unit_op", "unit_item", "unit_amount", "unit_count",
    "market_op", "market_item", "market_amount", "market_count",
)
MEMBER_GROUPS = (
    "top40_members", "recent_public_members",
    "current_top20_reconstruction_members",
)
ROUTING_FILES = ("route_map", "second_route_map", "route_tree")


def _unit_action(op: int, item: int, amount: int) -> list[Any]:
    if not 0 <= op < len(UNIT_OPS):
        raise ValueError(f"unknown unit op {op}")
    name = UNIT_OPS[op]
    if name == "PLANT":
        if not 0 <= item < len(CROPS):
            raise ValueError(f"invalid crop id {item}")
        return [name, CROPS[item]]
    if name in {"PICKUP", "PLACE"}:
        if not 0 <= item < len(SHED_ITEMS):
            raise ValueError(f"invalid shed item id {item}")
        return [name, SHED_ITEMS[item], int(amount)]
    return [name]


def _market_action(op: int, item: int, amount: int) -> list[Any] | None:
    if not 0 <= op < len(MARKET_OPS):
        raise ValueError(f"unknown market op {op}")
    name = MARKET_OPS[op]
    if name == "NONE":
        return None
    if name in {"HIRE", "BUY_LAND"}:
        return [name]
    if name == "BUY_SEED":
        if not 0 <= item < len(CROPS):
            raise ValueError(f"invalid crop id {item}")
        value = CROPS[item]
    elif name == "BUY_ANIMAL":
        animal = item - len(PRODUCTS)
        if not 0 <= animal < len(ANIMALS):
            raise ValueError(f"invalid animal id {item}")
        value = ANIMALS[animal]
    else:
        if not 0 <= item < len(PRODUCTS):
            raise ValueError(f"invalid product id {item}")
        value = PRODUCTS[item]
    return [name, value, int(amount)]


def decode_route(bank: Mapping[str, np.ndarray], route_index: int) -> list[dict[str, Any]]:
    """Decode one tensor route into 719 official action dictionaries."""

    missing = [name for name in ACTION_ARRAYS if name not in bank]
    if missing:
        raise KeyError(f"action bank is missing {missing[0]}")
    horizon = int(bank["unit_op"].shape[1])
    if horizon != 719:
        raise ValueError(f"expected a 719-step bank, got {horizon}")
    route_count = int(bank["unit_op"].shape[0])
    if not 0 <= route_index < route_count:
        raise IndexError(f"route {route_index} outside [0, {route_count})")

    tape: list[dict[str, Any]] = []
    for step in range(horizon):
        unit_count = max(1, min(int(bank["unit_count"][route_index, step]), 33))
        units = [
            _unit_action(
                int(bank["unit_op"][route_index, step, unit]),
                int(bank["unit_item"][route_index, step, unit]),
                int(bank["unit_amount"][route_index, step, unit]),
            )
            for unit in range(unit_count)
        ]
        market_count = max(0, min(int(bank["market_count"][route_index, step]), 10))
        market = []
        for order in range(market_count):
            decoded = _market_action(
                int(bank["market_op"][route_index, step, order]),
                int(bank["market_item"][route_index, step, order]),
                int(bank["market_amount"][route_index, step, order]),
            )
            if decoded is not None:
                market.append(decoded)
        tape.append({"farmer": units[0], "hands": units[1:], "market": market})
    return tape


def _integers(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, np.integer)):
        yield int(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _integers(item)


RUNTIME_ROUTE_KEYS = {
    "route_id", "route_ids", "first_route_ids", "base_route", "current_route",
    "selected_route", "second_route_ids_by_first_shop",
    "third_route_ids_by_shop", "second_routes", "third_routes",
}


def collect_route_ids(value: Any) -> set[int]:
    """Collect route indices while ignoring scores, seeds, and shop indices."""

    result: set[int] = set()
    if isinstance(value, Mapping):
        first_routes = list(_integers(value.get("first_route_ids")))
        legacy_second = value.get("second_route_ids")
        explicit_second = value.get("second_route_ids_by_first_shop")
        # Older two-shop maps are shaped [route_capacity, second_shop].  Only
        # the rows reached by first_route_ids are runtime routes; flattening
        # the full table would incorrectly import every padded identity row.
        if explicit_second is not None:
            result.update(_integers(explicit_second))
        elif first_routes and isinstance(legacy_second, Sequence):
            for route in first_routes:
                if 0 <= route < len(legacy_second):
                    result.update(_integers(legacy_second[route]))
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized == "second_route_ids":
                continue
            if normalized in RUNTIME_ROUTE_KEYS:
                result.update(_integers(child))
            elif isinstance(child, Mapping):
                result.update(collect_route_ids(child))
            elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                for item in child:
                    if isinstance(item, Mapping):
                        result.update(collect_route_ids(item))
    return result


def _canonical_tape(tape: Sequence[Mapping[str, Any]]) -> bytes:
    normalized = [
        {
            "farmer": list(step.get("farmer") or ["PASS"]),
            "hands": [list(action or ["PASS"]) for action in step.get("hands", ()) or ()],
            "market": [list(order) for order in step.get("market", ()) or ()],
        }
        for step in tape
    ]
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _scalar(bank: Mapping[str, np.ndarray], key: str, index: int) -> int | None:
    if key not in bank or bank[key].ndim != 1:
        return None
    return int(bank[key][index])


def _shop_sequence(bank: Mapping[str, np.ndarray], index: int) -> list[int] | None:
    if "source_shop_sequence" not in bank:
        return None
    return [int(value) for value in bank["source_shop_sequence"][index].tolist()]


def _quality_extra_indices(bank: Mapping[str, np.ndarray], count: int) -> list[int]:
    if count <= 0 or "source_margin" not in bank:
        return []
    route_count = int(bank["unit_op"].shape[0])
    margin = bank["source_margin"]
    reward = bank.get("source_reward", np.zeros(route_count, dtype=np.int64))
    order = sorted(
        range(route_count),
        key=lambda index: (int(margin[index]), int(reward[index]), -index),
        reverse=True,
    )
    # First cover distinct public market trajectories, then fill by quality.
    selected: list[int] = []
    seen_prefixes: set[tuple[int, ...]] = set()
    if "source_shop_sequence" in bank:
        for index in order:
            prefix = tuple(int(value) for value in bank["source_shop_sequence"][index, :3])
            if prefix not in seen_prefixes:
                selected.append(index)
                seen_prefixes.add(prefix)
                if len(selected) >= count:
                    return selected
    for index in order:
        if index not in selected:
            selected.append(index)
            if len(selected) >= count:
                break
    return selected


def _load_tapes(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        str(key): list(value)
        for key, value in json.loads(zlib.decompress(path.read_bytes())).items()
    }


def _bank_specs(config: Mapping[str, Any], source_root: Path) -> list[dict[str, Any]]:
    grouped: dict[Path, dict[str, Any]] = {}
    for group in MEMBER_GROUPS:
        for member in config.get(group, ()):
            if "bank" not in member:
                continue
            path = (source_root / str(member["bank"])).resolve()
            spec = grouped.setdefault(path, {
                "path": path,
                "members": [],
                "referenced": set(),
                "all_routes": str(member.get("implementation_kind")) == "native_kaito_v48",
            })
            spec["members"].append(str(member["name"]))
            for field in ROUTING_FILES:
                if field not in member:
                    continue
                routing_path = (source_root / str(member[field])).resolve()
                routing = json.loads(routing_path.read_text(encoding="utf-8"))
                spec["referenced"].update(collect_route_ids(routing))

    historical = (
        source_root
        / "experiments/expert_business_agent_v2/artifacts/"
        "jax_full37_mixed_exact_proxy_bank_v1.npz"
    ).resolve()
    grouped[historical] = {
        "path": historical,
        "members": [
            str(value["name"]) for value in config.get("historical_public_members", ())
        ],
        "referenced": set(),
        "all_routes": True,
    }
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--top-per-bank", type=int, default=8)
    parser.add_argument("--all-bank-routes", action="store_true")
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.top_per_bank < 0:
        parser.error("--top-per-bank must be non-negative")

    source_root = args.source_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    tapes = _load_tapes(args.base_actions)
    entries = list(base_metadata["opponent_routes"])
    digest_to_route: dict[str, str] = {}
    for route_id, tape in tapes.items():
        digest_to_route[hashlib.sha256(_canonical_tape(tape)).hexdigest()] = route_id

    imported_entries: list[dict[str, Any]] = []
    provenance_by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bank_summaries = []
    duplicate_base = 0
    duplicate_import = 0
    next_family = 1

    for spec in _bank_specs(config, source_root):
        path = Path(spec["path"])
        with np.load(path, allow_pickle=False) as loaded:
            bank = {name: loaded[name] for name in loaded.files}
        route_count = int(bank["unit_op"].shape[0])
        raw_referenced = {int(value) for value in spec["referenced"]}
        invalid = sorted(value for value in raw_referenced if not 0 <= value < route_count)
        # Some routing tables use ``len(bank)`` as the frozen-baseline sentinel.
        # It is not an action-bank row and the existing base library already
        # supplies that baseline, so retain it in the audit trail but do not
        # attempt to decode it.
        if any(value < 0 for value in invalid):
            raise ValueError(f"{path.name}: invalid negative route {invalid[0]}")
        referenced = raw_referenced - set(invalid)
        quality = set(_quality_extra_indices(bank, args.top_per_bank))
        if args.all_bank_routes or bool(spec["all_routes"]):
            selected = set(range(route_count))
            selection_mode = "all"
        else:
            selected = referenced | quality
            selection_mode = "routing_plus_market_diverse_quality"
        new_count = 0
        for route_index in sorted(selected):
            tape = decode_route(bank, route_index)
            canonical = _canonical_tape(tape)
            digest = hashlib.sha256(canonical).hexdigest()
            reason = []
            if route_index in referenced:
                reason.append("runtime_routing_table")
            if route_index in quality:
                reason.append("market_diverse_quality")
            if selection_mode == "all" and not reason:
                reason.append("complete_constituent_bank")
            source = {
                "bank": str(path.relative_to(source_root)).replace("\\", "/"),
                "route_index": route_index,
                "member_names": list(spec["members"]),
                "selection_reasons": reason,
                "source_episode_id": _scalar(bank, "source_episode_id", route_index),
                "source_reward": _scalar(bank, "source_reward", route_index),
                "source_opponent_reward": _scalar(
                    bank, "source_opponent_reward", route_index
                ),
                "source_margin": _scalar(bank, "source_margin", route_index),
                "source_seat": _scalar(bank, "source_seat", route_index),
                "source_shop_sequence": _shop_sequence(bank, route_index),
            }
            provenance_by_digest[digest].append(source)
            if digest in digest_to_route:
                if digest_to_route[digest].startswith("nt:"):
                    duplicate_import += 1
                else:
                    duplicate_base += 1
                continue
            family = f"NT{next_family:04d}"
            next_family += 1
            route_id = f"nt:{path.stem}:{route_index}:{digest[:16]}"
            digest_to_route[digest] = route_id
            tapes[route_id] = tape
            entry = {
                "family": family,
                "alias": f"nt68-{path.stem}-route-{route_index}",
                "route_id": route_id,
                "team": "nt_68_route_pool",
                "support": 1,
                "selected": True,
                "drop_reason": None,
                "source_execution_hard_failures": 0,
                "provenance": {
                    "schema": "nt-numpy-action-bank-route-v1",
                    "tape_sha256": digest,
                    "sources": provenance_by_digest[digest],
                    "jax_required": False,
                    "cpp_compatible": True,
                },
            }
            entries.append(entry)
            imported_entries.append(entry)
            new_count += 1
        bank_summaries.append({
            "bank": str(path.relative_to(source_root)).replace("\\", "/"),
            "members": list(spec["members"]),
            "bank_route_count": route_count,
            "referenced_route_count": len(referenced),
            "external_or_sentinel_route_ids": invalid,
            "quality_extra_count": len(quality),
            "selected_route_count": len(selected),
            "new_unique_route_count": new_count,
            "selection_mode": selection_mode,
        })

    # A duplicate can acquire provenance after its entry was created.
    by_digest = {
        str(value.get("provenance", {}).get("tape_sha256")): value
        for value in imported_entries
    }
    for digest, sources in provenance_by_digest.items():
        if digest in by_digest:
            by_digest[digest]["support"] = len(sources)
            by_digest[digest]["provenance"]["sources"] = sources

    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    actions_sha256 = hashlib.sha256(packed).hexdigest()
    output_metadata = copy.deepcopy(base_metadata)
    output_metadata["opponent_routes"] = entries
    output_metadata["selected"] = [
        *list(base_metadata.get("selected", base_metadata["opponent_routes"])),
        *imported_entries,
    ]
    output_metadata["actions_file"] = args.output_actions.name
    output_metadata["actions_sha256"] = actions_sha256
    output_metadata["nt_route_extension"] = {
        "schema": "nt-68-cpp-route-extension-v1",
        "source_config": str(args.config.resolve()),
        "jax_used": False,
        "decoder": "direct NumPy tensor to official action dictionary",
        "base_route_count": len(base_metadata["opponent_routes"]),
        "imported_unique_route_count": len(imported_entries),
        "total_route_count": len(entries),
    }
    manifest = {
        "schema": "nt-68-cpp-route-import-v1",
        "source_root": str(source_root),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "jax_used": False,
        "cpp_engine_target": True,
        "selection": {
            "all_bank_routes": args.all_bank_routes,
            "top_per_bank": args.top_per_bank,
            "policy": (
                "all fixed constituents for non-map banks; runtime routing-table routes "
                "plus high-margin routes covering distinct first-three-shop trajectories"
            ),
        },
        "base_route_count": len(base_metadata["opponent_routes"]),
        "imported_unique_route_count": len(imported_entries),
        "duplicate_base_count": duplicate_base,
        "duplicate_import_count": duplicate_import,
        "total_route_count": len(entries),
        "actions_sha256": actions_sha256,
        "banks": bank_summaries,
        "routes": imported_entries,
    }
    for path in (args.output_actions, args.output_metadata, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    args.output_metadata.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "banks": len(bank_summaries),
        "base_routes": len(base_metadata["opponent_routes"]),
        "imported_unique_routes": len(imported_entries),
        "duplicate_base": duplicate_base,
        "duplicate_import": duplicate_import,
        "total_routes": len(entries),
        "actions_bytes": len(packed),
        "jax_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
