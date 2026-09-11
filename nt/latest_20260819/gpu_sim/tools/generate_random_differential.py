"""Generate deterministic state-aware random/invalid-action official traces."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import io
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_environments import make

from generate_reference_traces import canonical_frame


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "reference" / "random_differential"
RECEIPT = PROJECT / "receipts" / "random_differential_reference.json"
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
MOVES = ("NORTH", "SOUTH", "EAST", "WEST", "PASS")


def value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_action(obs: Any, unit_index: int, pos: Any, rng: random.Random) -> list:
    player = int(value(obs, "player"))
    farm = value(obs, "farms")[player]
    private = value(obs, "private")
    x, y = int(pos[0]), int(pos[1])
    tile = value(farm, "tiles")[y][x]
    inventory = value(private, "inventories")[unit_index]
    seeds = value(private, "seeds")
    shed = value(private, "shed")

    # Inject malformed/illegal requests at a fixed low rate.
    roll = rng.random()
    if roll < 0.025:
        return ["TELEPORT", 99]
    if roll < 0.05:
        return ["PLANT", "NOT_A_CROP"]

    if tile is None:
        available = [crop for crop in CROPS if seeds.get(crop, 0) > 0]
        if available and rng.random() < 0.55:
            return ["PLANT", rng.choice(available)]
        if rng.random() < 0.08:
            return [rng.choice(("BUILD_COOP", "BUILD_PASTURE"))]
    elif tile == "LOCKED":
        return [rng.choice(MOVES)]
    elif isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "WEED":
            return ["DIG"] if rng.random() < 0.75 else [rng.choice(MOVES)]
        if kind == "PLANT":
            choices = ["WATER", "HARVEST", "DIG"]
            if inventory.get("FERTILIZER", 0) > 0:
                choices.append("FERTILIZE")
            return [rng.choice(choices)]
        if "animal" in tile:
            choices = ["CARE", "COLLECT_FERTILIZER", "HARVEST"]
            if inventory.get("WHEAT", 0) > 0:
                choices.extend(("FEED", "FEED"))
            return [rng.choice(choices)]
        if kind in ("COOP", "PASTURE"):
            matching = [
                animal
                for animal in ANIMALS
                if shed.get(animal, 0) > 0
                and ((kind == "COOP" and animal == "GOOSE") or kind == "PASTURE")
            ]
            carried = [animal for animal in matching if inventory.get(animal, 0) > 0]
            if carried:
                return ["PLACE", rng.choice(carried)]

    if (x, y) in ((4, 4), (5, 4), (4, 5), (5, 5)):
        carried = [item for item in SHED_ITEMS if inventory.get(item, 0) > 0]
        if carried and rng.random() < 0.45:
            return ["DROP"]
        available = [item for item in SHED_ITEMS if shed.get(item, 0) > 0]
        if available and rng.random() < 0.45:
            return ["PICKUP", rng.choice(available), rng.randint(1, 3)]
    return [rng.choice(MOVES)]


def make_agent(episode_seed: int):
    def agent(obs: Any) -> dict:
        player = int(value(obs, "player"))
        step = int(value(obs, "step"))
        rng = random.Random(
            (episode_seed * 1_000_003) ^ (step * 9_176) ^ (player * 104_729)
        )
        farm = value(obs, "farms")[player]
        private = value(obs, "private")
        positions = [value(farm, "farmer"), *value(farm, "hands", [])]
        units = [
            unit_action(obs, index, pos, rng) for index, pos in enumerate(positions)
        ]

        shed = value(private, "shed")
        seeds = value(private, "seeds")
        money = int(value(farm, "money"))
        orders: list[list] = []
        if step % 24 == 0:
            crop = rng.choice(CROPS)
            if seeds.get(crop, 0) < 2:
                orders.append(["BUY_SEED", crop, rng.randint(1, 3)])
        for _ in range(rng.randint(0, 3)):
            kind = rng.randrange(8)
            if kind == 0:
                orders.append(["HIRE"])
            elif kind == 1:
                orders.append(["BUY_LAND"])
            elif kind == 2:
                orders.append(["BUY_SEED", rng.choice(CROPS), rng.randint(1, 3)])
            elif kind == 3:
                orders.append(["BUY_PRODUCT", rng.choice(("WHEAT", "FERTILIZER")), rng.randint(1, 2)])
            elif kind == 4:
                orders.append(["BUY_ANIMAL", rng.choice(ANIMALS), 1])
            elif kind == 5:
                sellable = [item for item in PRODUCTS if shed.get(item, 0) > 0]
                orders.append(["SELL", rng.choice(sellable or PRODUCTS), rng.randint(1, 3)])
            elif kind == 6:
                orders.append(["BUY_SEED", "NOT_A_CROP", 1])
            else:
                orders.append(["SELL", "WHEAT", 0 if money >= 0 else -1])
        return {"farmer": units[0], "hands": units[1:], "market": orders}

    return agent


def write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def generate(seed: int, output_dir: Path) -> dict:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run([make_agent(seed), make_agent(seed)])
    frames = [canonical_frame(i, states) for i, states in enumerate(env.steps)]
    if len(frames) != 720 or frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"seed={seed}: incomplete trace")
    header = {
        "record_type": "header",
        "schema": "kaggriculture_official_trace_v1",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "scenario": "state_aware_random_with_invalid",
        "configuration": {"episodeSteps": 720, "seed": seed},
        "frame_count": len(frames),
    }
    path = output_dir / f"random_seed{seed}.jsonl.gz"
    write_trace(path, [header, *({"record_type": "frame", **frame} for frame in frames)])
    max_hands = max(len(farm["hands"]) for frame in frames for farm in frame["farms"])
    return {
        "seed": seed,
        "path": str(path.relative_to(PROJECT)),
        "frames": len(frames),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "max_hands": max_hands,
        "terminal_reward": frames[-1]["reward"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=32)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    rows = []
    for index, seed in enumerate(range(args.first_seed, args.first_seed + args.count), 1):
        row = generate(seed, args.output_dir)
        rows.append(row)
        print(f"[{index}/{args.count}] seed={seed} max_hands={row['max_hands']}", flush=True)
    receipt = {
        "schema": "kaggriculture_random_differential_reference_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": importlib.metadata.version("kaggle-environments"),
        "scenario": "state_aware_random_with_invalid",
        "count": len(rows),
        "total_frames": sum(row["frames"] for row in rows),
        "max_hands": max(row["max_hands"] for row in rows),
        "rows": rows,
        "ok": len(rows) == args.count and all(row["frames"] == 720 for row in rows),
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

