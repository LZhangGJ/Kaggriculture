#!/usr/bin/env python3
"""Build a runnable manifest for NT's replay-reconstructed leaderboard top 40."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SHOP_COUNT = 8


def _map_values(path: Path, *keys: str) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"none of {keys!r} found in {path}")


def _source_reward_map(bank: Path) -> list[int]:
    with np.load(bank, allow_pickle=False) as payload:
        shops = np.asarray(payload["source_shop_sequence"])
        rewards = np.asarray(payload["source_reward"])
    global_best = int(np.argmax(rewards))
    result: list[int] = []
    for shop_id in range(SHOP_COUNT):
        eligible = np.flatnonzero(shops[:, 0] == shop_id)
        result.append(
            int(eligible[int(np.argmax(rewards[eligible]))]) if eligible.size else global_best
        )
    return result


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=project / "external/kaggriculture-nt-front40-opponents/manifest.json",
    )
    args = parser.parse_args()
    fusion = args.fusion_root.resolve()
    workspace = fusion.parents[1]
    artifacts = fusion / "artifacts"
    configs = fusion / "configs"

    frozen: dict[int, dict[str, Any]] = {}
    for config_path in sorted(configs.glob("*frozen_best*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = int(config.get("source", {}).get("leaderboard_rank", 1))
        runtime = config.get("runtime", {})
        bank_rel = runtime.get("trace_bank", config.get("trace_bank"))
        bank = workspace / str(bank_rel)
        route_ids = config.get("route_ids", {})
        first = config.get("first_shop_routes") or route_ids.get("first_shop_routes")
        if first is None and "fixed_route" in route_ids:
            first = [int(route_ids["fixed_route"])] * SHOP_COUNT
        second = None
        tree_rel = runtime.get("route_tree")
        if tree_rel:
            tree = workspace / str(tree_rel)
            first = _map_values(tree, "route_ids")
            second = _map_values(tree, "second_route_ids_by_first_shop")
        second_rel = runtime.get("second_shop_map", config.get("second_shop_route_map"))
        if second_rel:
            second_path = workspace / str(second_rel)
            second = _map_values(
                second_path, "second_route_ids", "second_route_ids_by_first_shop"
            )
        frozen[rank] = {
            "bank": bank,
            "first_route_ids": [int(value) for value in first],
            "second_route_ids": second,
            "selection": "archive_frozen_public_shop_router",
            "frozen_config": _relative(config_path, project),
            "team": config.get("source", {}).get("team_name", config.get("player", "hasegawa")),
        }

    banks: dict[int, Path] = {1: artifacts / "hasegawa_current_trace_bank_v1.npz"}
    for bank in sorted(artifacts.glob("rank*_trace_bank_v1.npz")):
        match = re.match(r"rank(\d+)_", bank.name)
        if match:
            banks[int(match.group(1))] = bank

    agents: list[dict[str, Any]] = []
    for rank in range(1, 41):
        if rank == 30:
            agents.append({
                "rank": rank,
                "slug": "rank30_fc24b",
                "team": "FC24B",
                "type": "source",
                "source": "experiments/teammate-route-meta/decoded/_fc24b_base.py",
                "selection": "local_decoded_frozen_source",
            })
            continue
        bank = banks[rank]
        slug = bank.stem.removesuffix("_trace_bank_v1")
        spec = frozen.get(rank)
        if spec is None:
            spec = {
                "bank": bank,
                "first_route_ids": _source_reward_map(bank),
                "second_route_ids": None,
                "selection": "highest_source_reward_per_first_public_shop",
                "team": slug.split("_", 1)[1],
            }
        row: dict[str, Any] = {
            "rank": rank,
            "slug": slug,
            "team": spec["team"],
            "type": "trace_bank",
            "bank": _relative(Path(spec["bank"]), project),
            "first_route_ids": spec["first_route_ids"],
            "selection": spec["selection"],
        }
        if spec.get("second_route_ids") is not None:
            row["second_route_ids"] = spec["second_route_ids"]
        if spec.get("frozen_config"):
            row["frozen_config"] = spec["frozen_config"]
        agents.append(row)

    manifest = {
        "schema": "kaggriculture.nt-front40-official-eval-manifest.v1",
        "leaderboard_snapshot": "replay/top40_live_20260823",
        "official_package_version": "1.32.7",
        "provenance": "Kaggriculture_nt.zip latest_20260825_front40_public_v48",
        "warning": (
            "Ranks other than 30 are public-state replay-trace reconstructions, "
            "not the teams' original private submission source."
        ),
        "agents": agents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"agents": len(agents), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
