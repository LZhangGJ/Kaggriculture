"""Shared loading and causal traversal helpers for multi-shop route trees."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _key(prefix: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(int(value)) for value in prefix)


def load_resolver(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "previous_tree" in payload:
        resolver = load_resolver(Path(payload["previous_tree"]))
    else:
        base = np.asarray(payload["route_ids"], dtype=np.int32)
        if "second_route_ids_by_first_shop" in payload:
            second = np.asarray(payload["second_route_ids_by_first_shop"], dtype=np.int32)
        else:
            legacy = np.asarray(payload["second_route_ids"], dtype=np.int32)
            second = np.stack([legacy[int(base[first])] for first in range(8)], axis=0)
        resolver = {
            "base": base,
            "second": second,
            "third": None,
            "sparse": {},
            "shop_count": 2,
        }
    if "third_route_ids_by_shop" in payload:
        resolver["third"] = np.asarray(payload["third_route_ids_by_shop"], dtype=np.int32)
        resolver["shop_count"] = max(int(resolver["shop_count"]), 3)
    if "route_by_prefix" in payload:
        stage = int(payload["shop_count"])
        sparse = dict(resolver["sparse"])
        sparse[stage] = {str(key): int(value) for key, value in payload["route_by_prefix"].items()}
        resolver["sparse"] = sparse
        resolver["shop_count"] = max(int(resolver["shop_count"]), stage)
    return resolver


def follow(
    resolver: dict[str, object],
    observed: np.ndarray,
    lookup: dict[int, int],
    seat: int,
    seed: int,
    target_shop_count: int,
) -> tuple[tuple[int, ...], int]:
    if target_shop_count < 1:
        raise ValueError("target shop count must be positive")
    first = int(observed[seat, seed, 0, 0])
    if not 0 <= first < 8:
        raise RuntimeError("first shop is not observable")
    base = np.asarray(resolver["base"], dtype=np.int32)
    second_map = np.asarray(resolver["second"], dtype=np.int32)
    third_map = resolver.get("third")
    sparse = resolver["sparse"]
    prefix = [first]
    current = int(base[first])
    for position in range(1, target_shop_count):
        if current not in lookup:
            raise RuntimeError(f"route {current} is absent from screen")
        shop = int(observed[seat, seed, lookup[current], position])
        if not 0 <= shop < 8:
            break
        prefix.append(shop)
        stage = position + 1
        if stage == 2:
            current = int(second_map[prefix[0], prefix[1]])
        elif stage == 3 and third_map is not None:
            current = int(np.asarray(third_map)[prefix[0], prefix[1], prefix[2]])
        elif stage in sparse:
            current = int(sparse[stage].get(_key(prefix), current))
    return tuple(prefix), current
