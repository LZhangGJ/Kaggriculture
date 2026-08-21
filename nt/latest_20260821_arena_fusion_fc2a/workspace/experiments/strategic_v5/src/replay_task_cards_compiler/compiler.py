"""Lossless offline compiler from official replays to fixed-shape task cards.

The JSON ``templates`` section preserves every official action family.  The
``runtime`` section is a dense 720 x 21 adapter for the current V5 market/build
candidate slots.  Unsupported runtime cards remain visible in the coverage
report instead of being silently discarded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = "replay_task_cards_v1"
EPISODE_STEPS = 720
MARKET_CARD_SLOTS = 21
PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
UNIT_OPS = {
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "DROP",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}
MARKET_OPS = {
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
}
_TEAM_NAMES_RE = re.compile(rb'"TeamNames"\s*:\s*(\[[^\]]*\])')


def _unit_action(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return ["PASS"]
    op = value[0] if isinstance(value[0], str) else "PASS"
    if op not in UNIT_OPS:
        return ["PASS"]
    if op in {"PLANT", "PICKUP", "PLACE"} and len(value) >= 2:
        result: list[Any] = [op, str(value[1])]
        if op in {"PICKUP", "PLACE"} and len(value) >= 3:
            try:
                result.append(int(value[2]))
            except (TypeError, ValueError, OverflowError):
                return ["PASS"]
        return result
    return [op]


def _market_action(value: Any) -> list[Any] | None:
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        return None
    op = value[0]
    if op not in MARKET_OPS:
        return None
    if op in {"HIRE", "BUY_LAND"}:
        return [op]
    if len(value) < 3:
        return None
    try:
        amount = int(value[2])
    except (TypeError, ValueError, OverflowError):
        return None
    return [op, str(value[1]), amount]


def normalize_action(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    hands = raw.get("hands", [])
    hands = hands if isinstance(hands, list) else []
    market = raw.get("market", [])
    market = market if isinstance(market, list) else []
    return {
        "farmer": _unit_action(raw.get("farmer", ["PASS"])),
        "hands": [_unit_action(value) for value in hands],
        "market": [
            order
            for order in (_market_action(value) for value in market[:10])
            if order is not None
        ],
    }


def extract_bundle(raw: Any) -> dict[str, Any]:
    """Compress only lossless patterns; currently consecutive atomic orders."""

    action = normalize_action(raw)
    cards: list[dict[str, Any]] = []
    units = [action["farmer"], *action["hands"]]
    for actor_index, unit in enumerate(units):
        if unit[0] == "PASS":
            continue
        card = {
            "scope": "unit",
            "actor_index": actor_index,
            "op": unit[0],
            "item": unit[1] if len(unit) >= 2 else None,
            "quantity": int(unit[2]) if len(unit) >= 3 else 1,
            "explicit_quantity": len(unit) >= 3,
            "ordinal": actor_index,
        }
        cards.append(card)

    orders = action["market"]
    index = 0
    while index < len(orders):
        order = orders[index]
        op = order[0]
        if op in {"HIRE", "BUY_LAND"}:
            end = index + 1
            while end < len(orders) and orders[end][0] == op:
                end += 1
            cards.append(
                {
                    "scope": "market",
                    "op": op,
                    "item": None,
                    "quantity": end - index,
                    "repeat": end - index,
                    "ordinal": index,
                    "fill_policy": "atomic",
                }
            )
            index = end
            continue
        cards.append(
            {
                "scope": "market",
                "op": op,
                "item": order[1],
                "quantity": int(order[2]),
                "repeat": 1,
                "ordinal": index,
                "fill_policy": "up_to" if op == "BUY_PRODUCT" else "exact",
            }
        )
        index += 1
    return {
        "unit_count": len(units),
        "market_count": len(orders),
        "cards": cards,
    }


def expand_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    unit_count = max(int(bundle.get("unit_count", 1)), 1)
    units: list[list[Any]] = [["PASS"] for _ in range(unit_count)]
    market_with_ordinal: list[tuple[int, int, list[Any]]] = []
    serial = 0
    for card in bundle.get("cards", []):
        if card.get("scope") == "unit":
            actor = int(card["actor_index"])
            if actor >= len(units):
                units.extend([["PASS"] for _ in range(actor + 1 - len(units))])
            action = [str(card["op"])]
            if card.get("item") is not None:
                action.append(str(card["item"]))
            if action[0] in {"PICKUP", "PLACE"} and card.get("explicit_quantity", False):
                action.append(int(card["quantity"]))
            units[actor] = action
            continue
        if card.get("scope") != "market":
            continue
        op = str(card["op"])
        repeat = int(card.get("repeat", 1))
        ordinal = int(card.get("ordinal", 0))
        for offset in range(repeat):
            if op in {"HIRE", "BUY_LAND"}:
                action = [op]
            else:
                action = [op, str(card["item"]), int(card["quantity"])]
            market_with_ordinal.append((ordinal + offset, serial, action))
            serial += 1
    market = [value for _, _, value in sorted(market_with_ordinal)]
    return {"farmer": units[0], "hands": units[1:], "market": market}


def _bundle_signature(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_team_names_header(path: Path, limit: int = 1_048_576) -> list[str]:
    with path.open("rb") as stream:
        prefix = stream.read(limit)
    match = _TEAM_NAMES_RE.search(prefix)
    if match is None:
        return []
    try:
        value = json.loads(match.group(1).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _name_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def discover_replay_files(
    replay_dirs: Iterable[str | os.PathLike[str]],
    player_names: Iterable[str],
    *,
    episode_ids: Iterable[str] = (),
    max_episodes: int | None = None,
) -> list[Path]:
    wanted_ids = {str(value) for value in episode_ids}
    wanted_names = {_name_key(value) for value in player_names}
    files: list[Path] = []
    for directory in replay_dirs:
        for path in Path(directory).glob("*.json"):
            if wanted_ids and path.stem not in wanted_ids:
                continue
            names = _read_team_names_header(path)
            if wanted_names and not any(_name_key(name) in wanted_names for name in names):
                continue
            files.append(path.resolve())
    files.sort(
        key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
        reverse=True,
    )
    if wanted_ids:
        found = {path.stem for path in files}
        missing = sorted(wanted_ids - found)
        if missing:
            raise FileNotFoundError(f"requested replay ids not found for player: {missing}")
    return files if max_episodes is None else files[:max_episodes]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_slot(card: dict[str, Any]) -> int | None:
    if card.get("scope") != "market":
        return None
    op = card.get("op")
    item = card.get("item")
    if op == "BUY_LAND":
        return 0
    if op == "BUY_PRODUCT" and item == "FERTILIZER":
        return 1
    if op == "BUY_PRODUCT" and item == "WHEAT":
        return 2
    if op == "BUY_ANIMAL" and item in ANIMALS:
        return 3 + ANIMALS.index(item)
    if op == "BUY_SEED" and item in CROPS:
        return 6 + CROPS.index(item)
    if op == "HIRE":
        return 11
    if op == "SELL" and item in PRODUCTS:
        return 12 + PRODUCTS.index(item)
    return None


def _empty_runtime() -> dict[str, Any]:
    return {
        "enabled": [False] * EPISODE_STEPS,
        "support": [0] * EPISODE_STEPS,
        "consensus": [0.0] * EPISODE_STEPS,
        "market_quantity": [
            [0] * MARKET_CARD_SLOTS for _ in range(EPISODE_STEPS)
        ],
        "market_priority": [
            [0.0] * MARKET_CARD_SLOTS for _ in range(EPISODE_STEPS)
        ],
        "build_animal_id": [-1] * EPISODE_STEPS,
        "build_priority": [0.0] * EPISODE_STEPS,
    }


def _map_template_to_runtime(
    runtime: dict[str, Any], template: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step = int(template["source_step"])
    mapped: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    used_slots: set[int] = set()
    for card in template["bundle"]["cards"]:
        priority = 20_000.0 - float(card.get("ordinal", 0)) * 100.0
        slot = _runtime_slot(card)
        if slot is not None:
            if slot in used_slots:
                unsupported.append({"reason": "duplicate_runtime_slot", "card": card})
                continue
            used_slots.add(slot)
            runtime["market_quantity"][step][slot] = int(card["quantity"])
            runtime["market_priority"][step][slot] = priority
            mapped.append(card)
            continue
        if card.get("scope") == "unit" and card.get("op") in {
            "BUILD_COOP",
            "BUILD_PASTURE",
        }:
            if runtime["build_animal_id"][step] >= 0:
                unsupported.append({"reason": "multiple_build_cards", "card": card})
                continue
            runtime["build_animal_id"][step] = (
                0 if card["op"] == "BUILD_COOP" else 1
            )
            runtime["build_priority"][step] = priority
            mapped.append(card)
            continue
        unsupported.append({"reason": "not_mapped_by_v5_runtime", "card": card})
    runtime["enabled"][step] = bool(mapped)
    runtime["support"][step] = int(template["support"])
    runtime["consensus"][step] = float(template["consensus"])
    return mapped, unsupported


def compile_profile(
    replay_files: Iterable[str | os.PathLike[str]],
    player_names: Iterable[str],
    *,
    profile_name: str,
    min_support: int = 2,
    min_consensus: float = 0.8,
    alternatives: int = 3,
) -> dict[str, Any]:
    wanted_names = {_name_key(value) for value in player_names}
    samples: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    roundtrip_failures = 0

    for replay_value in replay_files:
        path = Path(replay_value).resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        names = [str(value) for value in document.get("info", {}).get("TeamNames", [])]
        seats = [index for index, name in enumerate(names) if _name_key(name) in wanted_names]
        if len(seats) != 1:
            raise ValueError(f"expected one matching player in {path.name}; got {names}")
        seat = seats[0]
        episode_id = path.stem
        steps = document.get("steps", [])
        source_samples = 0
        for action_index in range(1, len(steps)):
            if seat >= len(steps[action_index]) or seat >= len(steps[action_index - 1]):
                continue
            action = steps[action_index][seat].get("action")
            observation = steps[action_index - 1][seat].get("observation", {})
            if not isinstance(observation, dict):
                continue
            source_step = int(
                observation.get(
                    "step",
                    int(observation.get("day", 0)) * 24
                    + int(observation.get("hour", 0)),
                )
            )
            if not 0 <= source_step < EPISODE_STEPS:
                continue
            bundle = extract_bundle(action)
            if expand_bundle(bundle) != normalize_action(action):
                roundtrip_failures += 1
                continue
            samples[source_step].append(
                {
                    "episode_id": episode_id,
                    "source_step": source_step,
                    "action_index": action_index,
                    "day": int(observation.get("day", source_step // 24)),
                    "hour": int(observation.get("hour", source_step % 24)),
                    "bundle": bundle,
                    "signature": _bundle_signature(bundle),
                }
            )
            source_samples += 1
        sources.append(
            {
                "episode_id": episode_id,
                "document_id": str(document.get("id", "")),
                "path": str(path),
                "sha256": _file_sha256(path),
                "seat": seat,
                "team_name": names[seat],
                "samples": source_samples,
            }
        )

    templates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    runtime = _empty_runtime()
    mapped_count = 0
    unsupported_count = 0
    for source_step in sorted(samples):
        values = samples[source_step]
        counter = Counter(value["signature"] for value in values)
        ranked = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        signature, support = ranked[0]
        total = len(values)
        consensus = support / total
        representative = next(value for value in values if value["signature"] == signature)
        alternative_rows = []
        for alternative_signature, count in ranked[: max(alternatives, 1)]:
            example = next(
                value for value in values if value["signature"] == alternative_signature
            )
            alternative_rows.append(
                {
                    "support": count,
                    "consensus": count / total,
                    "bundle": example["bundle"],
                }
            )
        row = {
            "source_step": source_step,
            "day": representative["day"],
            "hour": representative["hour"],
            "sample_count": total,
            "support": support,
            "consensus": consensus,
            "bundle": representative["bundle"],
            "alternatives": alternative_rows,
        }
        if support < min_support or consensus < min_consensus:
            rejected.append(
                {
                    "source_step": source_step,
                    "sample_count": total,
                    "support": support,
                    "consensus": consensus,
                    "reason": "below_acceptance_threshold",
                }
            )
            continue
        mapped, unsupported = _map_template_to_runtime(runtime, row)
        row["runtime_mapped_cards"] = mapped
        row["runtime_unsupported_cards"] = unsupported
        mapped_count += len(mapped)
        unsupported_count += len(unsupported)
        templates.append(row)

    activation_reasons = []
    if roundtrip_failures:
        activation_reasons.append("roundtrip_failures_nonzero")
    if not templates:
        activation_reasons.append("no_accepted_templates")
    if not mapped_count:
        activation_reasons.append("no_runtime_mapped_cards")
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_name": profile_name,
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "player_names": sorted(str(value) for value in player_names),
        "compiler": {
            "action_alignment": "observation[t-1] -> action[t]",
            "min_support": min_support,
            "min_consensus": min_consensus,
            "alternatives": alternatives,
            "fixed_runtime_shapes": {
                "episode_steps": EPISODE_STEPS,
                "market_card_slots": MARKET_CARD_SLOTS,
            },
        },
        "sources": sources,
        "coverage": {
            "episodes": len(sources),
            "aligned_samples": sum(source["samples"] for source in sources),
            "roundtrip_failures": roundtrip_failures,
            "accepted_templates": len(templates),
            "rejected_steps": len(rejected),
            "runtime_mapped_cards": mapped_count,
            "runtime_unsupported_cards": unsupported_count,
        },
        "activation_gate": {
            "passed": not activation_reasons,
            "reasons": activation_reasons,
        },
        "templates": templates,
        "rejected": rejected,
        "runtime": runtime,
    }


def write_profile(document: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile official Kaggriculture replays into parameterized task cards."
    )
    parser.add_argument("--replay-dir", action="append", required=True)
    parser.add_argument("--player-name", action="append", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-consensus", type=float, default=0.8)
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="also atomically write active_profile.json beside --output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files = discover_replay_files(
        args.replay_dir,
        args.player_name,
        episode_ids=args.episode_id,
        max_episodes=args.max_episodes,
    )
    if not files:
        raise SystemExit("no matching replay files")
    document = compile_profile(
        files,
        args.player_name,
        profile_name=args.profile_name,
        min_support=args.min_support,
        min_consensus=args.min_consensus,
        alternatives=args.alternatives,
    )
    output = write_profile(document, args.output)
    active = None
    if args.activate:
        gate = document["activation_gate"]
        if not gate["passed"]:
            raise SystemExit(
                "profile was written for review but activation failed: "
                + ", ".join(gate["reasons"])
            )
        active = write_profile(deepcopy(document), output.parent / "active_profile.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "active": str(active) if active else None,
                "coverage": document["coverage"],
                "episode_ids": [source["episode_id"] for source in document["sources"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
