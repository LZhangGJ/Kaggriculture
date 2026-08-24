"""Kaggriculture replay reconstruction and behavior feature extraction.

Kaggle simulation replays can store the second seat observation as a public-state
increment rather than a standalone full observation.  The helpers below rebuild a
usable per-player observation stream before extracting route-family descriptors.
The implementation is intentionally tolerant: malformed files are reported in the
manifest instead of aborting the complete corpus scan.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .schema import PlayerBehavior, ReplaySummary

_PUBLIC_KEYS = ("step", "day", "hour", "farms", "market", "town")
_ANIMAL_PRODUCTS = ("EGG", "MILK", "WOOL")
_CROP_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deep_merge(base: Any, delta: Any) -> Any:
    """Recursively overlay ``delta`` onto ``base`` without mutating either."""

    if isinstance(base, Mapping) and isinstance(delta, Mapping):
        result: dict[str, Any] = {str(key): deepcopy(value) for key, value in base.items()}
        for key, value in delta.items():
            key = str(key)
            if key in result:
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
    # Lists in observations are snapshots, not sparse list patches.
    return deepcopy(delta)


def _unwrap_replay(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("steps"), list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("result", "episode", "replay", "data"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and isinstance(nested.get("steps"), list):
                return nested
    raise ValueError("JSON payload does not contain a Kaggle replay 'steps' array")


def _as_state_list(step: Any) -> list[Mapping[str, Any]]:
    if not isinstance(step, Sequence) or isinstance(step, (str, bytes, bytearray)):
        return []
    return [state if isinstance(state, Mapping) else {} for state in step]


def reconstruct_observations(
    steps: Sequence[Any],
) -> tuple[list[list[dict[str, Any]]], tuple[str, ...]]:
    """Reconstruct full observations for all players in every replay step.

    Player 0 is normally the full public snapshot.  Other seats are first merged
    with their previous observation and then receive any still-missing public
    fields from player 0.  ``private`` is never copied between players.
    """

    player_count = max((len(_as_state_list(step)) for step in steps), default=0)
    previous: list[dict[str, Any]] = [{} for _ in range(player_count)]
    reconstructed: list[list[dict[str, Any]]] = []
    warnings: set[str] = set()

    for step_index, raw_step in enumerate(steps):
        states = _as_state_list(raw_step)
        current: list[dict[str, Any]] = [{} for _ in range(player_count)]

        for player in range(player_count):
            state = states[player] if player < len(states) else {}
            raw_obs = state.get("observation", {})
            if not isinstance(raw_obs, Mapping):
                raw_obs = {}
                warnings.add("non_mapping_observation")
            current[player] = _deep_merge(previous[player], raw_obs)

        public_source = current[0] if current else {}
        for player in range(player_count):
            obs = current[player]
            raw_state = states[player] if player < len(states) else {}
            raw_obs = raw_state.get("observation", {})
            raw_obs = raw_obs if isinstance(raw_obs, Mapping) else {}
            used_public_fill = False
            if player > 0:
                # Previous-seat merge alone leaves stale public values (notably
                # ``step``) when the current replay frame stores only a delta.
                # Rebase every public field on player 0's current full snapshot,
                # then overlay any explicit seat-local public delta.
                for key in _PUBLIC_KEYS:
                    if key in public_source:
                        obs[key] = _deep_merge(public_source[key], raw_obs[key]) if key in raw_obs else deepcopy(public_source[key])
                        if key not in raw_obs:
                            used_public_fill = True
            else:
                for key in _PUBLIC_KEYS:
                    if key not in obs and key in public_source:
                        obs[key] = deepcopy(public_source[key])
                        used_public_fill = True
            if "player" not in obs:
                obs["player"] = player
            if "step" not in obs:
                obs["step"] = step_index
                warnings.add("synthetic_step_index")
            # A seat delta can omit public fields even after prior-state merge when
            # a replay starts mid-stream.  Record that reconstruction was needed.
            if player > 0 and (used_public_fill or any(key not in raw_obs for key in _PUBLIC_KEYS)):
                warnings.add("seat_delta_public_fields_reconstructed")
            current[player] = obs

        previous = [deepcopy(obs) for obs in current]
        reconstructed.append(current)

    return reconstructed, tuple(sorted(warnings))


def _op_name(action: Any) -> str:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
        if action:
            return str(action[0]).upper()
    if isinstance(action, str):
        return action.upper()
    return "NONE"


def _quantity(action: Any) -> int:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
        if len(action) >= 3:
            try:
                return max(0, int(action[2]))
            except (TypeError, ValueError):
                return 0
    return 1


def _item(action: Any) -> str:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
        if len(action) >= 2:
            return str(action[1]).upper()
    return "UNKNOWN"


def _tile_snapshot(tiles: Any) -> dict[str, dict[str, int] | int]:
    kinds: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    locked = 0
    empty = 0
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes, bytearray)):
        return {
            "kinds": {},
            "crops": {},
            "animals": {},
            "locked": 0,
            "empty": 0,
        }
    for row in tiles:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            continue
        for tile in row:
            if tile is None:
                empty += 1
            elif tile == "LOCKED":
                locked += 1
            elif isinstance(tile, Mapping):
                kind = str(tile.get("kind", "UNKNOWN")).upper()
                kinds[kind] += 1
                crop = tile.get("crop")
                animal = tile.get("animal")
                if crop is not None:
                    crops[str(crop).upper()] += 1
                if animal is not None:
                    animals[str(animal).upper()] += 1
            else:
                kinds["OTHER"] += 1
    return {
        "kinds": dict(sorted(kinds.items())),
        "crops": dict(sorted(crops.items())),
        "animals": dict(sorted(animals.items())),
        "locked": locked,
        "empty": empty,
    }


def _public_snapshot(obs: Mapping[str, Any], player: int) -> dict[str, Any]:
    farms = obs.get("farms")
    farm: Mapping[str, Any] = {}
    if isinstance(farms, Sequence) and not isinstance(farms, (str, bytes, bytearray)):
        if 0 <= player < len(farms) and isinstance(farms[player], Mapping):
            farm = farms[player]
    tiles = _tile_snapshot(farm.get("tiles"))
    market = obs.get("market") if isinstance(obs.get("market"), Mapping) else {}
    town = obs.get("town") if isinstance(obs.get("town"), Mapping) else {}
    shops = town.get("unlocked_shops", [])
    shops = shops if isinstance(shops, Sequence) and not isinstance(shops, (str, bytes, bytearray)) else []
    prices = market.get("prices", {})
    prices = prices if isinstance(prices, Mapping) else {}
    unlocked = farm.get("unlocked_quadrants", [])
    unlocked = unlocked if isinstance(unlocked, Sequence) and not isinstance(unlocked, (str, bytes, bytearray)) else []
    hands = farm.get("hands", [])
    hands = hands if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes, bytearray)) else []
    return {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", int(obs.get("step", 0) or 0) // 24) or 0),
        "money": int(farm.get("money", 0) or 0),
        "hands": len(hands),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "unlocked_count": len(unlocked),
        "tiles": tiles,
        "prices": {str(key).upper(): int(value) for key, value in sorted(prices.items()) if isinstance(value, (int, float))},
        "shops": dict(sorted(Counter(str(shop).upper() for shop in shops).items())),
    }


def _coarse_family(
    sold: Mapping[str, int],
    bought: Mapping[str, int],
    snapshots: Sequence[Mapping[str, Any]],
) -> str:
    animal_sales = sum(int(sold.get(name, 0)) for name in _ANIMAL_PRODUCTS)
    crop_sales = sum(int(sold.get(name, 0)) for name in _CROP_PRODUCTS)
    dominant_animal = max(_ANIMAL_PRODUCTS, key=lambda name: int(sold.get(name, 0)), default="")
    dominant_crop = max(_CROP_PRODUCTS, key=lambda name: int(sold.get(name, 0)), default="")
    final_animals: Counter[str] = Counter()
    final_crops: Counter[str] = Counter()
    if snapshots:
        tiles = snapshots[-1].get("tiles", {})
        if isinstance(tiles, Mapping):
            animals = tiles.get("animals", {})
            crops = tiles.get("crops", {})
            if isinstance(animals, Mapping):
                final_animals.update({str(k): int(v) for k, v in animals.items()})
            if isinstance(crops, Mapping):
                final_crops.update({str(k): int(v) for k, v in crops.items()})

    animal_assets = sum(final_animals.values()) + sum(
        int(bought.get(name, 0)) for name in ("GOOSE", "COW", "SHEEP")
    )
    crop_assets = sum(final_crops.values())

    if animal_sales > 0 and crop_sales > 0:
        if animal_sales >= crop_sales * 2:
            return f"{dominant_animal}_ANIMAL_HEAVY"
        if crop_sales >= animal_sales * 2:
            return f"{dominant_crop}_CROP_HEAVY"
        return "HYBRID"
    if animal_sales > 0:
        return f"{dominant_animal}_PRIMARY"
    if crop_sales > 0:
        return f"{dominant_crop}_PRIMARY"
    if animal_assets > 0 and crop_assets > 0:
        return "HYBRID_UNSOLD"
    if animal_assets > 0:
        dominant_asset = max(final_animals, key=final_animals.get, default="ANIMAL")
        return f"{dominant_asset}_ASSET"
    if crop_assets > 0:
        dominant_asset = max(final_crops, key=final_crops.get, default="CROP")
        return f"{dominant_asset}_ASSET"
    return "UNCLASSIFIED"


def _extract_identifiers(replay: Mapping[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    episode_id: str | None = None
    for key in ("id", "episodeId", "episode_id"):
        value = replay.get(key)
        if value is not None:
            episode_id = str(value)
            break
    info = replay.get("info")
    if episode_id is None and isinstance(info, Mapping):
        for key in ("id", "episodeId", "episode_id"):
            if info.get(key) is not None:
                episode_id = str(info[key])
                break

    submission_ids: list[str] = []
    agents = replay.get("agents")
    if isinstance(agents, Sequence) and not isinstance(agents, (str, bytes, bytearray)):
        for agent in agents:
            identifier = ""
            if isinstance(agent, Mapping):
                for key in ("submissionId", "submission_id", "submission", "id"):
                    value = agent.get(key)
                    if value is not None:
                        identifier = str(value)
                        break
            submission_ids.append(identifier)
    return episode_id, tuple(submission_ids)


def _extract_seed(replay: Mapping[str, Any]) -> int | None:
    for container in (replay.get("configuration"), replay.get("info"), replay):
        if not isinstance(container, Mapping):
            continue
        for key in ("seed", "episodeSeed", "episode_seed", "randomSeed"):
            value = container.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
    return None


def summarize_replay_payload(
    payload: Any,
    *,
    source_path: str = "<memory>",
    replay_sha256: str | None = None,
) -> ReplaySummary:
    replay = _unwrap_replay(payload)
    steps = replay.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("replay 'steps' must be a list")
    observations, reconstruction_warnings = reconstruct_observations(steps)
    player_count = max((len(_as_state_list(step)) for step in steps), default=0)
    players: list[PlayerBehavior] = []
    warnings = set(reconstruction_warnings)

    for player in range(player_count):
        farmer_ops: Counter[str] = Counter()
        hand_ops: Counter[str] = Counter()
        market_ops: Counter[str] = Counter()
        bought: Counter[str] = Counter()
        sold: Counter[str] = Counter()
        first_op_step: dict[str, int] = {}
        snapshots: list[dict[str, Any]] = []

        for step_index, raw_step in enumerate(steps):
            states = _as_state_list(raw_step)
            state = states[player] if player < len(states) else {}
            action = state.get("action")
            if action is None:
                action = {}
            if not isinstance(action, Mapping):
                warnings.add("non_mapping_action")
                action = {}

            farmer = action.get("farmer", ["PASS"])
            farmer_name = _op_name(farmer)
            farmer_ops[farmer_name] += 1
            first_op_step.setdefault(f"farmer:{farmer_name}", step_index)

            hands = action.get("hands", [])
            if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes, bytearray)):
                for hand in hands:
                    name = _op_name(hand)
                    hand_ops[name] += 1
                    first_op_step.setdefault(f"hand:{name}", step_index)

            market = action.get("market", [])
            if isinstance(market, Sequence) and not isinstance(market, (str, bytes, bytearray)):
                for order in market:
                    name = _op_name(order)
                    market_ops[name] += 1
                    first_op_step.setdefault(f"market:{name}", step_index)
                    item = _item(order)
                    quantity = _quantity(order)
                    if name == "SELL":
                        sold[item] += quantity
                    elif name in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                        bought[item] += quantity

            obs = observations[step_index][player] if step_index < len(observations) else {}
            step_value = int(obs.get("step", step_index) or step_index)
            hour = obs.get("hour")
            is_day_boundary = hour == 0 if isinstance(hour, int) else step_value % 24 == 0
            if is_day_boundary:
                snapshots.append(_public_snapshot(obs, player))

        final_reward: float | None = None
        final_status: str | None = None
        if steps:
            final_states = _as_state_list(steps[-1])
            if player < len(final_states):
                reward = final_states[player].get("reward")
                if isinstance(reward, (int, float)):
                    final_reward = float(reward)
                status = final_states[player].get("status")
                if status is not None:
                    final_status = str(status)

        family = _coarse_family(sold, bought, snapshots)
        signature_payload = {
            "farmer_ops": dict(sorted(farmer_ops.items())),
            "hand_ops": dict(sorted(hand_ops.items())),
            "market_ops": dict(sorted(market_ops.items())),
            "bought": dict(sorted(bought.items())),
            "sold": dict(sorted(sold.items())),
            "first_op_step": dict(sorted(first_op_step.items())),
            "daily": snapshots,
            "coarse_family": family,
        }
        players.append(
            PlayerBehavior(
                player_index=player,
                final_reward=final_reward,
                final_status=final_status,
                farmer_ops=dict(sorted(farmer_ops.items())),
                hand_ops=dict(sorted(hand_ops.items())),
                market_ops=dict(sorted(market_ops.items())),
                market_items_bought=dict(sorted(bought.items())),
                market_items_sold=dict(sorted(sold.items())),
                first_op_step=dict(sorted(first_op_step.items())),
                daily_public_snapshots=tuple(snapshots),
                behavior_signature=_json_hash(signature_payload),
                coarse_family=family,
            )
        )

    episode_id, submission_ids = _extract_identifiers(replay)
    if replay_sha256 is None:
        replay_sha256 = _json_hash(replay)
    return ReplaySummary(
        source_path=source_path,
        replay_sha256=replay_sha256,
        episode_id=episode_id,
        submission_ids=submission_ids,
        step_count=len(steps),
        player_count=player_count,
        episode_seed=_extract_seed(replay),
        players=tuple(players),
        parse_warnings=tuple(sorted(warnings)),
    )


def summarize_replay_file(path: str | Path) -> ReplaySummary:
    replay_path = Path(path)
    raw = replay_path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    return summarize_replay_payload(
        payload,
        source_path=replay_path.as_posix(),
        replay_sha256=hashlib.sha256(raw).hexdigest(),
    )


def looks_like_replay(payload: Any) -> bool:
    try:
        replay = _unwrap_replay(payload)
    except ValueError:
        return False
    steps = replay.get("steps")
    return isinstance(steps, list) and bool(steps)


def iter_player_rows(summary: ReplaySummary) -> Iterable[dict[str, Any]]:
    for player in summary.players:
        yield {
            "source_path": summary.source_path,
            "replay_sha256": summary.replay_sha256,
            "episode_id": summary.episode_id,
            "episode_seed": summary.episode_seed,
            "player_index": player.player_index,
            "behavior_signature": player.behavior_signature,
            "coarse_family": player.coarse_family,
            "final_reward": player.final_reward,
            "final_status": player.final_status,
            "submission_ids": list(summary.submission_ids),
        }
