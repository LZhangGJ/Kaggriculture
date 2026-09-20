"""Causal recurrent context for route selection and storage trading.

The module deliberately separates information by visibility:

* both farms pass through the *same* public encoder and GRU weights;
* the two public recurrent states are independent;
* only the acting player's private inventory and route plan enter the private GRU;
* market and town history have their own GRU.

This makes the self/opponent representations comparable without pretending that
the opponent's private inventory is observable.  The route head is a residual
on top of the existing Tree-Q score; its final layer is zero-initialised so an
untrained model cannot damage the existing selector.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from .fingerprints import BASE_PRICES, ITEMS, QUADRANTS
from .tree_q import FEATURE_DIM, numpy_forward


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
TRADE_ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
STRUCTURES = ("SOIL", "COOP", "PASTURE")
PLAN_HORIZONS = (24, 48, 72)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _log_scale(value: Any, maximum: float) -> float:
    return math.log1p(max(0.0, _number(value))) / math.log1p(maximum)


def _position(value: Any) -> tuple[int, int] | None:
    try:
        return int(value[0]), int(value[1])
    except (IndexError, TypeError, ValueError):
        return None


def observation_step(observation: Any) -> int:
    explicit = _get(observation, "step", None)
    if explicit is not None:
        return int(_number(explicit))
    return int(_number(_get(observation, "day", 0))) * 24 + int(
        _number(_get(observation, "hour", 0))
    )


def public_farm_vector(farm: Any) -> np.ndarray:
    """Encode only fields publicly visible for either player.

    The exact same function is used for self and opponent.  No ``private``
    observation field is accepted here by construction.
    """
    values: list[float] = []
    hands = list(_get(farm, "hands", []) or [])
    unlocked = {str(item) for item in (_get(farm, "unlocked_quadrants", []) or [])}
    positions = [_get(farm, "farmer", None), *hands]
    values.extend(
        [
            _log_scale(_get(farm, "money", 0), 200_000),
            min(1.0, len(hands) / 12.0),
            min(1.0, len(unlocked) / 4.0),
            min(1.0, _number(_get(farm, "hires_today", 0)) / 12.0),
        ]
    )
    farmer = _position(_get(farm, "farmer", None)) or (0, 0)
    values.extend([farmer[0] / 9.0, farmer[1] / 9.0])

    workers_by_quadrant = {name: 0 for name in QUADRANTS}
    for raw in positions:
        point = _position(raw)
        if point is None:
            continue
        x, y = point
        name = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
        workers_by_quadrant[name] += 1

    crop_stats = {item: np.zeros(6, dtype=np.float32) for item in CROPS}
    animal_stats = {item: np.zeros(6, dtype=np.float32) for item in ANIMALS}
    structure_counts = {item: 0 for item in STRUCTURES}
    quadrant_stats = {name: np.zeros(5, dtype=np.float32) for name in QUADRANTS}
    tiles = list(_get(farm, "tiles", []) or [])
    for y, row in enumerate(tiles):
        for x, tile in enumerate(list(row or [])):
            if x >= 10 or y >= 10:
                continue
            quadrant = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
            if tile is None:
                quadrant_stats[quadrant][1] += 1
                continue
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind") or "")
            if kind in structure_counts:
                structure_counts[kind] += 1
            crop = str(tile.get("crop") or "")
            if crop in crop_stats:
                row_stats = crop_stats[crop]
                row_stats[0] += 1
                row_stats[1] += max(0.0, _number(tile.get("yield_units", 0)))
                row_stats[2] += float(not bool(tile.get("watered_today", False)))
                row_stats[3] += max(0.0, _number(tile.get("consecutive_unwatered", 0)))
                row_stats[4] += float(_number(tile.get("yield_units", 0)) > 0)
                row_stats[5] += float(bool(tile.get("fertilized", False)))
                quadrant_stats[quadrant][2] += 1
            animal = str(tile.get("animal") or "")
            if animal in animal_stats:
                row_stats = animal_stats[animal]
                row_stats[0] += 1
                row_stats[1] += max(0.0, _number(tile.get("yield_units", 0)))
                row_stats[2] += float(not bool(tile.get("fed_today", False)))
                row_stats[3] += float(not bool(tile.get("cared_today", False)))
                row_stats[4] += max(0.0, _number(tile.get("consecutive_unfed", 0)))
                row_stats[5] += float(_number(tile.get("yield_units", 0)) > 0)
                quadrant_stats[quadrant][3] += 1

    for item in CROPS:
        row = crop_stats[item]
        values.extend((row / np.asarray([25, 100, 25, 100, 25, 25])).tolist())
    for item in ANIMALS:
        row = animal_stats[item]
        values.extend((row / np.asarray([25, 100, 25, 25, 100, 25])).tolist())
    values.extend(structure_counts[item] / 100.0 for item in STRUCTURES)
    for name in QUADRANTS:
        row = quadrant_stats[name]
        row[0] = float(name in unlocked)
        row[4] = workers_by_quadrant[name]
        values.extend((row / np.asarray([1, 25, 25, 25, 12])).tolist())
    return np.asarray(values, dtype=np.float32)


PUBLIC_DIM = len(public_farm_vector({}))


def _route_plan_vector(actions: Sequence[Mapping[str, Any]] | None) -> np.ndarray:
    """Summarise the already selected route's next 24/48/72 turns."""
    result: list[float] = []
    rows = list(actions or [])
    for horizon in PLAN_HORIZONS:
        expenses = {"HIRE": 0.0, "BUY_LAND": 0.0, "BUY_SEED": 0.0, "BUY_PRODUCT": 0.0}
        sells = {item: 0.0 for item in TRADE_ITEMS}
        for action in rows[:horizon]:
            for order in list(_get(action, "market", []) or []):
                if not order:
                    continue
                op = str(order[0])
                quantity = max(1.0, _number(order[2], 1.0)) if len(order) >= 3 else 1.0
                if op in expenses:
                    expenses[op] += quantity
                if op == "SELL" and len(order) >= 2 and str(order[1]) in sells:
                    sells[str(order[1])] += quantity
        result.extend(
            [
                min(1.0, expenses["HIRE"] / 12.0),
                min(1.0, expenses["BUY_LAND"] / 4.0),
                min(1.0, expenses["BUY_SEED"] / 100.0),
                min(1.0, expenses["BUY_PRODUCT"] / 100.0),
            ]
        )
        result.extend(min(1.0, sells[item] / 100.0) for item in TRADE_ITEMS)
    return np.asarray(result, dtype=np.float32)


def route_plan_matrix(actions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Compile all rolling plan summaries once instead of rescanning 72 turns."""
    rows = list(actions)
    raw = np.zeros((len(rows), 4 + len(TRADE_ITEMS)), dtype=np.float32)
    expense_index = {"HIRE": 0, "BUY_LAND": 1, "BUY_SEED": 2, "BUY_PRODUCT": 3}
    sell_index = {item: 4 + index for index, item in enumerate(TRADE_ITEMS)}
    for step, action in enumerate(rows):
        for order in list(_get(action, "market", []) or []):
            if not order:
                continue
            op = str(order[0])
            quantity = max(1.0, _number(order[2], 1.0)) if len(order) >= 3 else 1.0
            if op in expense_index:
                raw[step, expense_index[op]] += quantity
            if op == "SELL" and len(order) >= 2 and str(order[1]) in sell_index:
                raw[step, sell_index[str(order[1])]] += quantity
    prefix = np.concatenate([np.zeros((1, raw.shape[1]), dtype=np.float32), raw.cumsum(axis=0)])
    result = np.empty((len(rows), len(PLAN_HORIZONS) * raw.shape[1]), dtype=np.float32)
    scales = np.asarray([12, 4, 100, 100, *([100] * len(TRADE_ITEMS))], dtype=np.float32)
    for step in range(len(rows)):
        pieces = []
        for horizon in PLAN_HORIZONS:
            stop = min(len(rows), step + horizon)
            pieces.append(np.minimum(1.0, (prefix[stop] - prefix[step]) / scales))
        result[step] = np.concatenate(pieces)
    return result


def private_plan_vector(
    observation: Any,
    future_route_actions: Sequence[Mapping[str, Any]] | None = None,
    compiled_route_plan: np.ndarray | None = None,
) -> np.ndarray:
    private = dict(_get(observation, "private", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    carried = {item: 0.0 for item in ITEMS}
    for inventory in list(private.get("inventories", []) or []):
        for item, quantity in dict(inventory or {}).items():
            if str(item) in carried:
                carried[str(item)] += max(0.0, _number(quantity))
    values: list[float] = []
    values.extend(_log_scale(shed.get(item, 0), 100) for item in ITEMS)
    values.extend(_log_scale(carried.get(item, 0), 100) for item in ITEMS)
    values.extend(_log_scale(seeds.get(item, 0), 100) for item in CROPS)
    shed_total = sum(max(0.0, _number(value)) for value in shed.values())
    values.extend(
        [
            min(1.0, shed_total / 100.0),
            min(1.0, sum(carried.values()) / 100.0),
            float(shed_total >= 90),
        ]
    )
    plan = (
        np.asarray(compiled_route_plan, dtype=np.float32)
        if compiled_route_plan is not None
        else _route_plan_vector(future_route_actions)
    )
    values.extend(plan.tolist())
    return np.asarray(values, dtype=np.float32)


PRIVATE_DIM = len(private_plan_vector({}))


def market_town_vector(observation: Any) -> np.ndarray:
    market = dict(_get(observation, "market", {}) or {})
    inventory = dict(market.get("inventory", {}) or {})
    prices = dict(market.get("prices", {}) or {})
    values: list[float] = []
    for item in TRADE_ITEMS:
        values.append((_number(inventory.get(item, 10_000)) - 10_000.0) / 2_000.0)
        values.append(_number(prices.get(item, BASE_PRICES[item])) / BASE_PRICES[item] - 1.0)
    shops = list(_get(_get(observation, "town", {}) or {}, "unlocked_shops", []) or [])
    shop_hash = np.zeros(16, dtype=np.float32)
    for shop in shops:
        # Python's hash is process-randomised; this byte sum is submission-stable.
        encoded = str(shop).encode("utf-8")
        shop_hash[sum((index + 1) * byte for index, byte in enumerate(encoded)) % 16] += 1.0
    values.extend(np.tanh(shop_hash).tolist())
    step = observation_step(observation)
    day = int(_number(_get(observation, "day", step // 24)))
    hour = int(_number(_get(observation, "hour", step % 24)))
    values.extend([step / 719.0, day / 29.0, hour / 23.0])
    return np.asarray(values, dtype=np.float32)


MARKET_DIM = len(market_town_vector({}))


@dataclass(slots=True)
class EncodedMetaObservation:
    self_public: np.ndarray
    opponent_public: np.ndarray
    private_plan: np.ndarray
    market_town: np.ndarray

    def validate(self) -> None:
        expected = (
            (self.self_public, PUBLIC_DIM),
            (self.opponent_public, PUBLIC_DIM),
            (self.private_plan, PRIVATE_DIM),
            (self.market_town, MARKET_DIM),
        )
        for value, size in expected:
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid recurrent feature {value.shape}, expected {(size,)}")


def encode_meta_observation(
    observation: Any,
    future_route_actions: Sequence[Mapping[str, Any]] | None = None,
    compiled_route_plan: np.ndarray | None = None,
) -> EncodedMetaObservation:
    player = int(_number(_get(observation, "player", 0)))
    farms = list(_get(observation, "farms", []) or [])
    own = farms[player] if 0 <= player < len(farms) else {}
    opponent_index = 1 - player
    opponent = farms[opponent_index] if 0 <= opponent_index < len(farms) else {}
    cached_public = _get(observation, "_meta_public_vectors", None)
    cached_market = _get(observation, "_meta_market_vector", None)
    result = EncodedMetaObservation(
        self_public=(
            np.asarray(cached_public[player], dtype=np.float32)
            if cached_public is not None else public_farm_vector(own)
        ),
        opponent_public=(
            np.asarray(cached_public[opponent_index], dtype=np.float32)
            if cached_public is not None else public_farm_vector(opponent)
        ),
        private_plan=private_plan_vector(
            observation, future_route_actions, compiled_route_plan
        ),
        market_town=(
            np.asarray(cached_market, dtype=np.float32)
            if cached_market is not None else market_town_vector(observation)
        ),
    )
    result.validate()
    return result


class MetaHiddenState(NamedTuple):
    self_public: Any
    opponent_public: Any
    private_plan: Any
    market_town: Any


def build_torch_model():
    """Return the model class lazily so submission feature code imports without torch."""
    import torch
    from torch import nn

    class SymmetricHistoryModel(nn.Module):
        """Shared event encoders plus causal multi-window history summaries."""

        windows = (1, 8, 24, 72, 168)
        public_hidden = 32
        private_hidden = 32
        market_hidden = 32
        context_dim = 128

        def __init__(self) -> None:
            super().__init__()
            self.public_input = nn.Sequential(
                nn.Linear(PUBLIC_DIM, self.public_hidden),
                nn.LayerNorm(self.public_hidden), nn.SiLU()
            )
            self.private_input = nn.Sequential(
                nn.Linear(PRIVATE_DIM, self.private_hidden),
                nn.LayerNorm(self.private_hidden), nn.SiLU()
            )
            self.market_input = nn.Sequential(
                nn.Linear(MARKET_DIM, self.market_hidden),
                nn.LayerNorm(self.market_hidden), nn.SiLU()
            )
            scales = len(self.windows)
            fusion_dim = (
                self.public_hidden * scales * 4
                + self.private_hidden * scales
                + self.market_hidden * scales
            )
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 192), nn.LayerNorm(192), nn.SiLU(),
                nn.Linear(192, self.context_dim), nn.SiLU(),
            )
            self.branch_encoder = nn.Sequential(
                nn.Linear(FEATURE_DIM, 192), nn.SiLU(),
                nn.Linear(192, 96), nn.SiLU(),
            )
            self.route_residual = nn.Sequential(
                nn.Linear(self.context_dim + 96, 96), nn.SiLU(), nn.Linear(96, 1)
            )
            nn.init.zeros_(self.route_residual[-1].weight)
            nn.init.zeros_(self.route_residual[-1].bias)
            # Per item: sell gate, inventory fraction, and execution priority.
            self.trade_head = nn.Linear(self.context_dim, len(TRADE_ITEMS) * 3 + 3)
            # Trade logits are interpreted as HOLD / KEEP_BASELINE / LIQUIDATE.
            # A freshly created trading policy must be an exact no-op on top of
            # StorageMarketManager instead of deleting its valid SELL orders.
            nn.init.zeros_(self.trade_head.weight)
            nn.init.zeros_(self.trade_head.bias)
            with torch.no_grad():
                self.trade_head.bias[: len(TRADE_ITEMS) * 3].view(
                    len(TRADE_ITEMS), 3
                )[:, 1] = 2.0
            self.value_head = nn.Linear(self.context_dim, 1)
            # Auxiliary target keeps the recurrent state tied to observable dynamics.
            self.next_public_head = nn.Linear(self.context_dim, PUBLIC_DIM * 2 + MARKET_DIM)

        def initial_state(self, batch_size: int, device=None, dtype=None) -> MetaHiddenState:
            reference = next(self.parameters())
            device = reference.device if device is None else device
            dtype = reference.dtype if dtype is None else dtype
            return MetaHiddenState(
                torch.zeros(batch_size, 0, self.public_hidden, device=device, dtype=dtype),
                torch.zeros(batch_size, 0, self.public_hidden, device=device, dtype=dtype),
                torch.zeros(batch_size, 0, self.private_hidden, device=device, dtype=dtype),
                torch.zeros(batch_size, 0, self.market_hidden, device=device, dtype=dtype),
            )

        def _online_summary(self, history, current):
            history = torch.cat([history, current.unsqueeze(1)], dim=1)
            history = history[:, -self.windows[-1] :]
            summaries = [history[:, -min(window, history.shape[1]) :].mean(dim=1)
                         for window in self.windows]
            return torch.cat(summaries, dim=-1), history

        def step(self, self_public, opponent_public, private_plan, market_town, state):
            # public_input is shared; histories are deliberately not shared.
            own, own_state = self._online_summary(
                state.self_public, self.public_input(self_public)
            )
            opponent, opponent_state = self._online_summary(
                state.opponent_public, self.public_input(opponent_public)
            )
            private, private_state = self._online_summary(
                state.private_plan, self.private_input(private_plan)
            )
            market, market_state = self._online_summary(
                state.market_town, self.market_input(market_town)
            )
            fused = torch.cat(
                [own, opponent, own - opponent, own * opponent, private, market], dim=-1
            )
            context = self.fusion(fused)
            return context, MetaHiddenState(
                own_state, opponent_state, private_state, market_state
            )

        def _sequence_summary(self, encoded):
            # AvgPool1d is a single fused causal kernel per time scale.  Correct
            # the left zero padding so early turns use their actual prefix size.
            transposed = encoded.transpose(1, 2)
            length = encoded.shape[1]
            time = torch.arange(1, length + 1, device=encoded.device, dtype=encoded.dtype)
            summaries = []
            for window in self.windows:
                pooled = torch.nn.functional.avg_pool1d(
                    torch.nn.functional.pad(transposed, (window - 1, 0)),
                    kernel_size=window,
                    stride=1,
                ).transpose(1, 2)
                correction = window / torch.clamp(time, max=window)
                summaries.append(pooled * correction.view(1, -1, 1))
            return torch.cat(summaries, dim=-1)

        def forward_sequence(self, self_public, opponent_public, private_plan, market_town):
            """Fused causal sequence path used by BC and recurrent RL updates."""
            own = self._sequence_summary(self.public_input(self_public))
            opponent = self._sequence_summary(self.public_input(opponent_public))
            private = self._sequence_summary(self.private_input(private_plan))
            market = self._sequence_summary(self.market_input(market_town))
            fused = torch.cat(
                [own, opponent, own - opponent, own * opponent, private, market], dim=-1
            )
            return self.fusion(fused)

        def forward_window(self, self_public, opponent_public, private_plan, market_town):
            """Return only the final context for an already-cropped history window.

            Route inference consumes a context only at sparse branch checkpoints.
            Computing contexts for every prefix position would repeat most of the
            sequence work at every checkpoint, while the final context depends
            only on the suffix means below.
            """
            def final_summary(encoded):
                return torch.cat(
                    [encoded[:, -min(window, encoded.shape[1]) :].mean(dim=1)
                     for window in self.windows],
                    dim=-1,
                )

            own = final_summary(self.public_input(self_public))
            opponent = final_summary(self.public_input(opponent_public))
            private = final_summary(self.private_input(private_plan))
            market = final_summary(self.market_input(market_town))
            fused = torch.cat(
                [own, opponent, own - opponent, own * opponent, private, market], dim=-1
            )
            return self.fusion(fused)

        def score_routes(self, context, candidate_features):
            """Score ``[...,K,FEATURE_DIM]`` candidates with a learned residual."""
            encoded = self.branch_encoder(candidate_features)
            expanded = context.unsqueeze(-2).expand(*encoded.shape[:-1], context.shape[-1])
            return self.route_residual(torch.cat([expanded, encoded], dim=-1)).squeeze(-1)

        def trading(self, context):
            raw = self.trade_head(context)
            item = raw[..., : len(TRADE_ITEMS) * 3].reshape(
                *raw.shape[:-1], len(TRADE_ITEMS), 3
            )
            wheat_mode = raw[..., len(TRADE_ITEMS) * 3 :]
            return item, wheat_mode

    return SymmetricHistoryModel


# Submission rollout uses the NumPy implementation below and must not pay the
# cold-start cost of importing Torch.  Training keeps the historical default.
SymmetricHistoryModel = (
    None if os.environ.get("KAGG_META_NUMPY_ONLY") == "1" else build_torch_model()
)


def export_recurrent_numpy(torch_checkpoint: str, output: str) -> dict[str, Any]:
    """Export inference weights without a torch dependency in rollout workers."""
    import pickle
    import torch

    payload = torch.load(torch_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    arrays = {key: value.detach().float().cpu().numpy() for key, value in state.items()}
    result = {
        "architecture": "symmetric-window-v1",
        "windows": tuple(SymmetricHistoryModel.windows),
        "state": arrays,
    }
    with open(output, "wb") as handle:
        pickle.dump(result, handle, protocol=5)
    return {"path": output, "bytes": __import__("os").path.getsize(output)}


def _np_silu(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -20.0, 20.0)
    return value / (1.0 + np.exp(-clipped))


def _np_linear(value: np.ndarray, state: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    return value @ state[f"{prefix}.weight"].T + state[f"{prefix}.bias"]


def _np_layer_norm(value: np.ndarray, state: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (value - mean) / np.sqrt(variance + 1e-5)
    return normalized * state[f"{prefix}.weight"] + state[f"{prefix}.bias"]


class NumpySymmetricHistoryModel:
    """Low-overhead rollout implementation numerically equivalent to the torch model."""

    def __init__(self, path: str) -> None:
        import pickle

        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("architecture") != "symmetric-window-v1":
            raise ValueError(f"unsupported recurrent export {payload.get('architecture')!r}")
        self.state = payload["state"]
        self.windows = tuple(int(value) for value in payload["windows"])
        self.histories: dict[str, list[np.ndarray]] = {}
        self.reset()

    def reset(self) -> None:
        self.histories = {"own": [], "opponent": [], "private": [], "market": []}

    def _input(self, value: np.ndarray, prefix: str) -> np.ndarray:
        result = _np_linear(np.asarray(value, dtype=np.float32), self.state, f"{prefix}.0")
        result = _np_layer_norm(result, self.state, f"{prefix}.1")
        return _np_silu(result).astype(np.float32, copy=False)

    def _summary(self, name: str, current: np.ndarray) -> np.ndarray:
        # Store cumulative sums, not individual vectors: every window mean is
        # one subtraction instead of repeatedly stacking Python lists.
        history = self.histories[name]
        cumulative = current if not history else history[-1] + current
        history.append(cumulative)
        if len(history) > self.windows[-1] + 1:
            del history[0]
        summaries = []
        for window in self.windows:
            count = min(window, len(history))
            total = history[-1] - (history[-count - 1] if len(history) > count else 0.0)
            summaries.append(total / count)
        return np.concatenate(summaries).astype(np.float32, copy=False)

    def step(self, encoded: EncodedMetaObservation) -> np.ndarray:
        own = self._summary("own", self._input(encoded.self_public, "public_input"))
        opponent = self._summary(
            "opponent", self._input(encoded.opponent_public, "public_input")
        )
        private = self._summary("private", self._input(encoded.private_plan, "private_input"))
        market = self._summary("market", self._input(encoded.market_town, "market_input"))
        fused = np.concatenate([own, opponent, own - opponent, own * opponent, private, market])
        hidden = _np_linear(fused, self.state, "fusion.0")
        hidden = _np_silu(_np_layer_norm(hidden, self.state, "fusion.1"))
        return _np_silu(_np_linear(hidden, self.state, "fusion.3")).astype(np.float32)

    def score_routes(self, context: np.ndarray, features: np.ndarray) -> np.ndarray:
        branch = _np_silu(_np_linear(features, self.state, "branch_encoder.0"))
        branch = _np_silu(_np_linear(branch, self.state, "branch_encoder.2"))
        expanded = np.broadcast_to(context, (len(branch), len(context)))
        hidden = _np_silu(
            _np_linear(np.concatenate([expanded, branch], axis=-1), self.state, "route_residual.0")
        )
        return _np_linear(hidden, self.state, "route_residual.2").reshape(-1)

    def trading(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = _np_linear(context, self.state, "trade_head")
        split = len(TRADE_ITEMS) * 3
        return raw[:split].reshape(len(TRADE_ITEMS), 3), raw[split:]


class NumpyRecurrentTreeQSelector:
    """Torch-free rollout selector with raw sequence traces for off-policy RL."""

    def __init__(
        self,
        root: str,
        descriptor_path: str,
        *,
        tree_q_model_path: str | None,
        recurrent_model_path: str,
        residual_weight: float = 1.0,
        epsilon: float = 0.10,
        top_k: int = 8,
        prior_weight: float = 0.05,
        seed: int = 0,
    ) -> None:
        from .tree_q import ExploringShardedRouteSelector
        from .tree_q import load_numpy_model

        class _Selector(ExploringShardedRouteSelector):
            def __init__(inner, owner, *args, **kwargs):
                inner.owner = owner
                super().__init__(*args, **kwargs)

            def route_adjustment(inner, features: np.ndarray) -> np.ndarray:
                return inner.owner._route_adjustment(features)

            def observe(inner, observation: Any, configuration: Any = None) -> None:
                inner.owner.observe(observation, configuration)

        self.model = NumpySymmetricHistoryModel(recurrent_model_path)
        self.residual_weight = float(residual_weight)
        self.base_model = (
            load_numpy_model(tree_q_model_path)
            if tree_q_model_path is not None else None
        )
        self.prior_weight = float(prior_weight)
        self.context: np.ndarray | None = None
        self.last_step = -1
        self.encoded_history: list[EncodedMetaObservation] = []
        self.rl_decisions: list[dict[str, Any]] = []
        self._candidate_features: np.ndarray | None = None
        self._plan_cache: dict[str, np.ndarray] = {}
        self.selector = _Selector(
            self,
            root,
            descriptor_path,
            model_path=tree_q_model_path,
            epsilon=epsilon,
            top_k=top_k,
            prior_weight=prior_weight,
            seed=seed,
        )

    def __getattr__(self, name: str):
        if name != "selector":
            return getattr(self.selector, name)
        raise AttributeError(name)

    @property
    def current(self):
        return self.selector.current

    @current.setter
    def current(self, value):
        self.selector.current = value

    def reset(self) -> None:
        self.selector.reset()
        self.model.reset()
        self.context = None
        self.last_step = -1
        self.encoded_history.clear()
        self.rl_decisions.clear()
        self._candidate_features = None

    def observe(self, observation: Any, configuration: Any = None) -> None:
        step = observation_step(observation)
        if step == 0 and self.last_step >= 0:
            self.reset()
        if step == self.last_step:
            return
        if self.last_step >= 0 and step != self.last_step + 1:
            raise ValueError(f"non-contiguous recurrent stream: {self.last_step} -> {step}")
        future = None
        compiled_plan = None
        if self.selector.current_actions is not None:
            route_id = str(self.selector.current_route_id)
            if route_id not in self._plan_cache:
                self._plan_cache[route_id] = route_plan_matrix(self.selector.current_actions)
            compiled_plan = self._plan_cache[route_id][step]
        encoded = encode_meta_observation(observation, future, compiled_plan)
        self.encoded_history.append(
            EncodedMetaObservation(
                *(np.asarray(value, dtype=np.float16) for value in (
                    encoded.self_public,
                    encoded.opponent_public,
                    encoded.private_plan,
                    encoded.market_town,
                ))
            )
        )
        self.context = self.model.step(encoded)
        self.last_step = step

    def _route_adjustment(self, features: np.ndarray) -> np.ndarray:
        self._candidate_features = np.asarray(features, dtype=np.float16).copy()
        if self.context is None:
            return np.zeros(len(features), dtype=np.float32)
        return self.model.score_routes(self.context, features) * self.residual_weight

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        decision = self.selector.select(fingerprint, checkpoint)
        sampled = self.selector.training_decisions[-1]
        if self._candidate_features is None:
            raise RuntimeError("route candidates were not captured")
        self.rl_decisions.append(
            {
                "checkpoint": int(checkpoint),
                "features": self._candidate_features,
                "selected": int(sampled.selected_rank),
                "explored": bool(sampled.explored),
                "opponent_candidates": int(sampled.candidates),
                "base_scores": (
                    (
                        numpy_forward(
                            self.base_model, self._candidate_features.astype(np.float32)
                        )
                        if self.base_model is not None
                        else np.zeros(len(self._candidate_features), dtype=np.float32)
                    )
                    + np.linspace(
                        0.0, -self.prior_weight, len(self._candidate_features), dtype=np.float32
                    )
                ).astype(np.float16),
            }
        )
        return decision

    def trace_arrays(self, top_k: int = 8) -> dict[str, np.ndarray]:
        if not self.encoded_history or not self.rl_decisions:
            raise RuntimeError("cannot export an empty rollout trace")
        result = {
            "self_public": np.stack([row.self_public for row in self.encoded_history]),
            "opponent_public": np.stack([row.opponent_public for row in self.encoded_history]),
            "private_plan": np.stack([row.private_plan for row in self.encoded_history]),
            "market_town": np.stack([row.market_town for row in self.encoded_history]),
            "checkpoints": np.asarray([row["checkpoint"] for row in self.rl_decisions], dtype=np.int16),
            "selected": np.asarray([row["selected"] for row in self.rl_decisions], dtype=np.int8),
            "explored": np.asarray([row["explored"] for row in self.rl_decisions], dtype=np.bool_),
        }
        padded = np.zeros((len(self.rl_decisions), top_k, FEATURE_DIM), dtype=np.float16)
        base_scores = np.full((len(self.rl_decisions), top_k), -20.0, dtype=np.float16)
        counts = np.empty(len(self.rl_decisions), dtype=np.int8)
        for index, row in enumerate(self.rl_decisions):
            count = min(top_k, len(row["features"]))
            padded[index, :count] = row["features"][:count]
            base_scores[index, :count] = row["base_scores"][:count]
            counts[index] = count
        result["route_features"] = padded
        result["candidate_counts"] = counts
        result["base_scores"] = base_scores
        return result


def trade_targets(observation: Any, action: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Executable expert SELL labels plus WHEAT buy-hold-sell mode.

    This deliberately does not label an empty requested SELL as a sale.  Older
    versions used the raw request and taught the policy to emit SELL even when
    the official environment would execute zero units.
    """
    from .market_manager import executable_sell_labels

    shed, sells, sell_order, _ = executable_sell_labels(
        observation, action, TRADE_ITEMS, 100
    )
    targets = np.zeros((len(TRADE_ITEMS), 3), dtype=np.float32)
    wheat_mode = np.asarray(1, dtype=np.int64)  # BUY, HOLD, SELL => 0,1,2
    orders = list(_get(action, "market", []) or [])
    for order in orders:
        if not order:
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) >= 2 else ""
        if item == "WHEAT" and op == "BUY_PRODUCT":
            wheat_mode = np.asarray(0, dtype=np.int64)
    if sells.get("WHEAT", 0) > 0:
        wheat_mode = np.asarray(2, dtype=np.int64)
    for index, item in enumerate(TRADE_ITEMS):
        quantity = sells.get(item, 0)
        targets[index, 0] = float(quantity > 0)
        targets[index, 1] = min(1.0, quantity / max(1.0, shed.get(item, 0.0)))
        targets[index, 2] = (
            1.0 - sell_order.index(item) / max(1, len(sell_order)) if item in sell_order else 0.0
        )
    return targets, wheat_mode


def apply_trading_output(
    observation: Any,
    route_action: Mapping[str, Any],
    item_logits: np.ndarray,
    wheat_mode_logits: np.ndarray,
    *,
    gate_threshold: float = 0.55,
    reserve_ratio: float = 0.10,
    max_market_slots: int = 10,
) -> dict[str, Any]:
    """Safely replace SELL intents while preserving route purchases and hires."""
    import copy

    action = copy.deepcopy(dict(route_action or {}))
    market = [list(row) for row in list(action.get("market", []) or []) if row]
    market = [row for row in market if str(row[0]) != "SELL"]
    shed = dict(_get(_get(observation, "private", {}) or {}, "shed", {}) or {})
    raw = np.asarray(item_logits, dtype=np.float32).reshape(len(TRADE_ITEMS), 3)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
    wheat_mode = int(np.argmax(np.asarray(wheat_mode_logits)))
    candidates: list[tuple[float, list[Any]]] = []
    for index, item in enumerate(TRADE_ITEMS):
        gate, fraction, priority = sigmoid[index]
        if item == "WHEAT" and wheat_mode != 2:
            continue
        inventory = max(0, int(_number(shed.get(item, 0))))
        reserve = int(math.ceil(inventory * max(0.0, min(1.0, reserve_ratio))))
        sellable = max(0, inventory - reserve)
        quantity = min(sellable, int(math.floor(fraction * inventory)))
        if gate >= gate_threshold and quantity > 0:
            candidates.append((float(priority), ["SELL", item, quantity]))
    candidates.sort(key=lambda row: (-row[0], row[1][1]))
    room = max(0, max_market_slots - len(market))
    market.extend(row for _, row in candidates[:room])
    action["market"] = market
    return action


def apply_trading_residual(
    observation: Any,
    baseline_action: Mapping[str, Any],
    item_modes: np.ndarray,
    configuration: Any = None,
    *,
    max_market_slots: int = 10,
) -> dict[str, Any]:
    """Apply HOLD / KEEP_BASELINE / LIQUIDATE as a safe rule residual.

    Mode 1 is a strict identity operation.  The official last-four-turn
    liquidation produced by StorageMarketManager is never overridden.
    """
    from .market_manager import (
        _canonical_action, _is_sell, _merge_or_insert_sell,
        _positive_int, _project_market_shed, _rank_sell_slots,
    )

    action = _canonical_action(baseline_action)
    episode_steps = max(4, int(_number(_get(configuration, "episodeSteps", 720), 720)))
    step = observation_step(observation)
    if step >= episode_steps - 4:
        return action
    modes = np.asarray(item_modes, dtype=np.int64).reshape(len(TRADE_ITEMS))
    modes = np.clip(modes, 0, 2)
    capacity = max(1, int(_number(_get(configuration, "shedCapacity", 100), 100)))
    projected = _project_market_shed(observation, action, capacity)
    market: list[list[Any]] = []
    remaining = dict(projected)

    # Preserve non-SELL orders exactly.  Existing SELLs are held, retained, or
    # replaced by an aggressive order according to the per-item daily mode.
    for raw in action["market"]:
        order = list(raw)
        if not _is_sell(order):
            market.append(order)
            continue
        item = str(order[1])
        if item not in TRADE_ITEMS:
            market.append(order)
            continue
        mode = int(modes[TRADE_ITEMS.index(item)])
        if mode == 0:  # HOLD
            continue
        if mode == 2:  # LIQUIDATE is inserted once after all baseline orders.
            continue
        quantity = min(_positive_int(order[2]), max(0, remaining.get(item, 0)))
        if quantity > 0:
            market.append(["SELL", item, quantity])
            remaining[item] = max(0, remaining.get(item, 0) - quantity)

    for index, item in enumerate(TRADE_ITEMS):
        if int(modes[index]) != 2:
            continue
        quantity = max(0, remaining.get(item, 0))
        if quantity <= 0:
            continue
        if _merge_or_insert_sell(market, item, quantity, max_market_slots):
            remaining[item] = 0

    action["market"] = _rank_sell_slots(
        observation, market[:max_market_slots], configuration, demand_alpha=0.25
    )
    return action


def apply_direct_trading_output(
    observation: Any,
    baseline_action: Mapping[str, Any],
    item_logits: np.ndarray,
    configuration: Any = None,
    *,
    gate_threshold: float = 0.80,
    max_market_slots: int = 10,
    replace_baseline_sells: bool = False,
) -> dict[str, Any]:
    """Apply per-turn BC sales, optionally replacing route SELL requests.

    ``replace_baseline_sells`` is the standalone intraday policy used for BC
    fidelity evaluation and subsequent PPO.  The default remains the safer
    additive mode for old checkpoints.
    """
    from .market_manager import (
        _canonical_action, _is_sell, _merge_or_insert_sell,
        _project_market_shed, _rank_sell_slots,
    )

    action = _canonical_action(baseline_action)
    episode_steps = max(4, int(_number(_get(configuration, "episodeSteps", 720), 720)))
    if observation_step(observation) >= episode_steps - 4:
        return action
    capacity = max(1, int(_number(_get(configuration, "shedCapacity", 100), 100)))
    available = _project_market_shed(observation, action, capacity)
    market = [
        list(order) for order in action["market"]
        if not (replace_baseline_sells and _is_sell(order))
    ]
    remaining = dict(available)
    baseline_sell_items = set()
    for order in market:
        if not _is_sell(order):
            continue
        item = str(order[1])
        baseline_sell_items.add(item)
        quantity = int(_number(order[2], 1)) if len(order) >= 3 else 1
        remaining[item] = max(0, remaining.get(item, 0) - max(0, quantity))
    raw = np.asarray(item_logits, dtype=np.float32).reshape(len(TRADE_ITEMS), 3)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
    candidates: list[tuple[float, str, int]] = []
    for index, item in enumerate(TRADE_ITEMS):
        # In additive compatibility mode WHEAT and existing expert sales remain
        # under the baseline manager.  Standalone mode learned both from the
        # expert's causal history and is allowed to decide them itself.
        if (not replace_baseline_sells and item == "WHEAT") or item in baseline_sell_items:
            continue
        gate, fraction, priority = probabilities[index]
        inventory = max(0, int(remaining.get(item, 0)))
        quantity = min(inventory, int(math.floor(float(fraction) * inventory)))
        if gate >= gate_threshold and quantity > 0:
            candidates.append((float(priority), item, quantity))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    for _, item, quantity in candidates:
        if not _merge_or_insert_sell(market, item, quantity, max_market_slots):
            break
    action["market"] = _rank_sell_slots(
        observation, market[:max_market_slots], configuration, demand_alpha=0.25
    )
    return action


class RecurrentTreeQSelector:
    """Factory-compatible history wrapper around ``ExploringShardedRouteSelector``.

    Torch is only used for the small recurrent model.  The route library and
    baseline Tree-Q retain their existing NumPy implementation.  Calling
    ``observe`` exactly once per turn is enforced, which protects recurrent
    state from accidental double advancement by wrappers.
    """

    def __init__(
        self,
        root: str,
        descriptor_path: str,
        *,
        tree_q_model_path: str | None = None,
        recurrent_model_path: str | None = None,
        residual_weight: float = 1.0,
        enable_trading: bool = False,
        epsilon: float = 0.0,
        top_k: int = 8,
        prior_weight: float = 0.05,
        seed: int = 0,
        stochastic_softmax: bool = False,
        temperature: float = 1.0,
        uniform_mix: float = 0.0,
        stochastic_trading: bool = False,
        trade_temperature: float = 1.0,
        trade_uniform_mix: float = 0.0,
        trade_start_step: int = 0,
        direct_trade_model_path: str | None = None,
        direct_trade_gate_threshold: float = 0.55,
        direct_trade_replace_baseline: bool = False,
    ) -> None:
        import torch

        from .tree_q import ExploringShardedRouteSelector, load_numpy_model

        class _Selector(ExploringShardedRouteSelector):
            def __init__(inner, owner, *args, **kwargs):
                inner.owner = owner
                super().__init__(*args, **kwargs)

            def route_adjustment(inner, features: np.ndarray) -> np.ndarray:
                return inner.owner._route_adjustment(features)

            def choose_candidate(inner, scores: np.ndarray) -> tuple[int, bool]:
                if inner.owner.stochastic_softmax:
                    return inner.owner._choose_candidate(scores)
                return super().choose_candidate(scores)

            def observe(inner, observation: Any, configuration: Any = None) -> None:
                inner.owner.observe(observation, configuration)

            def adjust_market(inner, observation: Any, action: Mapping[str, Any], configuration=None):
                return inner.owner.adjust_market(observation, action, configuration)

        self.torch = torch
        # Tiny per-turn inference is latency-bound; large OpenMP pools make it slower.
        if torch.get_num_threads() > 4:
            torch.set_num_threads(1)
        self.model = SymmetricHistoryModel().cpu().eval()
        self.loaded = False
        if recurrent_model_path:
            payload = torch.load(recurrent_model_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("model", payload)
            self.model.load_state_dict(state_dict)
            self.loaded = True
        self.residual_weight = float(residual_weight)
        self.base_model = (
            load_numpy_model(tree_q_model_path) if tree_q_model_path is not None else None
        )
        self.prior_weight = float(prior_weight)
        self.stochastic_softmax = bool(stochastic_softmax)
        self.temperature = max(1e-4, float(temperature))
        self.uniform_mix = max(0.0, min(1.0, float(uniform_mix)))
        self.enable_trading = bool(enable_trading and self.loaded)
        self.stochastic_trading = bool(stochastic_trading)
        self.trade_temperature = max(1e-4, float(trade_temperature))
        self.trade_uniform_mix = max(0.0, min(1.0, float(trade_uniform_mix)))
        self.trade_start_step = max(0, int(trade_start_step))
        self.direct_trade_model = (
            NumpySymmetricHistoryModel(direct_trade_model_path)
            if direct_trade_model_path else None
        )
        self.direct_trade_context: np.ndarray | None = None
        self.direct_trade_gate_threshold = max(
            0.0, min(1.0, float(direct_trade_gate_threshold))
        )
        self.direct_trade_replace_baseline = bool(direct_trade_replace_baseline)
        self.hidden: MetaHiddenState | None = None
        self.context = None
        self.context_step = -1
        self.last_trade = None
        self.trade_modes = np.ones(len(TRADE_ITEMS), dtype=np.int8)
        self.last_step = -1
        self.encoded_history: list[EncodedMetaObservation] = []
        self.rl_decisions: list[dict[str, Any]] = []
        self.trade_decisions: list[dict[str, Any]] = []
        self._candidate_features: np.ndarray | None = None
        self._last_sample: dict[str, float] | None = None
        self._plan_cache: dict[str, np.ndarray] = {}
        self.selector = _Selector(
            self,
            root,
            descriptor_path,
            model_path=tree_q_model_path,
            epsilon=epsilon,
            top_k=top_k,
            prior_weight=prior_weight,
            seed=seed,
        )

    def __getattr__(self, name: str):
        # ReplayTrieAgent expects a normal selector interface.
        if name != "selector":
            return getattr(self.selector, name)
        raise AttributeError(name)

    @property
    def current(self):
        return self.selector.current

    @current.setter
    def current(self, value):
        self.selector.current = value

    def reset(self) -> None:
        self.selector.reset()
        self.hidden = None
        self.context = None
        self.context_step = -1
        self.last_trade = None
        self.trade_modes.fill(1)
        self.last_step = -1
        self.encoded_history.clear()
        self.rl_decisions.clear()
        self.trade_decisions.clear()
        if self.direct_trade_model is not None:
            self.direct_trade_model.reset()
        self.direct_trade_context = None
        self._candidate_features = None
        self._last_sample = None

    def observe(self, observation: Any, configuration: Any = None) -> None:
        step = observation_step(observation)
        if step == 0 and self.last_step >= 0:
            self.reset()
        if step == self.last_step:
            return
        if self.last_step >= 0 and step != self.last_step + 1:
            # A skipped or rewound turn is a new causal stream, not a valid GRU prefix.
            self.hidden = None
            self.context = None
            self.context_step = -1
            self.last_trade = None
            self.trade_modes.fill(1)
            self.encoded_history.clear()
            self.rl_decisions.clear()
            self.trade_decisions.clear()
            if self.direct_trade_model is not None:
                self.direct_trade_model.reset()
            self.direct_trade_context = None
            self._candidate_features = None
            self._last_sample = None
        compiled_plan = None
        if self.selector.current_actions is not None:
            route_id = str(self.selector.current_route_id)
            if route_id not in self._plan_cache:
                self._plan_cache[route_id] = route_plan_matrix(self.selector.current_actions)
            compiled_plan = self._plan_cache[route_id][step]
        encoded = encode_meta_observation(observation, compiled_route_plan=compiled_plan)
        self.encoded_history.append(
            EncodedMetaObservation(
                *(np.asarray(value, dtype=np.float16) for value in (
                    encoded.self_public,
                    encoded.opponent_public,
                    encoded.private_plan,
                    encoded.market_town,
                ))
            )
        )
        if self.direct_trade_model is not None:
            self.direct_trade_context = self.direct_trade_model.step(encoded)
        self.last_step = step
        # Trading makes one persistent decision per in-game day.  Recomputing
        # dozens of tiny Torch kernels on all 720 turns is both slower and much
        # higher variance than a daily inventory policy.
        if self.enable_trading and step % 24 == 0:
            self._refresh_route_context()
            self._sample_trade_modes(observation, compiled_plan)

    def _refresh_route_context(self) -> None:
        if self.context_step == self.last_step or not self.encoded_history:
            return
        torch = self.torch
        dtype = next(self.model.parameters()).dtype
        tensors = []
        recent = self.encoded_history[-self.model.windows[-1] :]
        for name in ("self_public", "opponent_public", "private_plan", "market_town"):
            values = np.stack([getattr(row, name) for row in recent])
            tensors.append(torch.from_numpy(values).to(dtype=dtype).unsqueeze(0))
        with torch.inference_mode():
            self.context = self.model.forward_window(*tensors)
        self.context_step = self.last_step

    def _trade_active_mask(
        self, observation: Any, compiled_plan: np.ndarray | None
    ) -> np.ndarray:
        if self.last_step < self.trade_start_step:
            return np.zeros(len(TRADE_ITEMS), dtype=np.bool_)
        private = dict(_get(observation, "private", {}) or {})
        shed = dict(private.get("shed", {}) or {})
        active = np.asarray(
            [max(0.0, _number(shed.get(item, 0))) > 0 for item in TRADE_ITEMS],
            dtype=np.bool_,
        )
        if compiled_plan is not None and len(compiled_plan) >= 4 + len(TRADE_ITEMS):
            active |= np.asarray(
                compiled_plan[4 : 4 + len(TRADE_ITEMS)] > 0,
                dtype=np.bool_,
            )
        farms = list(_get(observation, "farms", []) or [])
        player = int(_number(_get(observation, "player", 0)))
        farm = farms[player] if 0 <= player < len(farms) else {}
        animal_products = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
        visible = set()
        for row in list(_get(farm, "tiles", []) or []):
            for tile in list(row or []):
                if not isinstance(tile, Mapping):
                    continue
                crop = str(tile.get("crop") or "")
                animal = animal_products.get(str(tile.get("animal") or ""), "")
                if crop in TRADE_ITEMS:
                    visible.add(crop)
                if animal:
                    visible.add(animal)
        active |= np.asarray([item in visible for item in TRADE_ITEMS], dtype=np.bool_)
        return active

    def _sample_trade_modes(
        self, observation: Any, compiled_plan: np.ndarray | None
    ) -> None:
        if self.context is None:
            return
        with self.torch.inference_mode():
            item_logits, _ = self.model.trading(self.context)
            logits = item_logits[0].float() / self.trade_temperature
            probabilities = self.torch.softmax(logits, dim=-1)
            if self.trade_uniform_mix > 0.0:
                probabilities = (
                    (1.0 - self.trade_uniform_mix) * probabilities
                    + self.trade_uniform_mix / probabilities.shape[-1]
                )
            probabilities_np = probabilities.cpu().numpy()
            value = float(self.model.value_head(self.context).item())
        active = self._trade_active_mask(observation, compiled_plan)
        modes = np.ones(len(TRADE_ITEMS), dtype=np.int8)
        logprobs = np.zeros(len(TRADE_ITEMS), dtype=np.float32)
        entropies = np.zeros(len(TRADE_ITEMS), dtype=np.float32)
        for index in range(len(TRADE_ITEMS)):
            if not active[index]:
                continue
            probs = probabilities_np[index]
            if self.stochastic_trading:
                draw = self.selector.rng.random()
                mode = int(np.searchsorted(np.cumsum(probs), draw, side="right"))
                mode = min(mode, len(probs) - 1)
            else:
                mode = int(np.argmax(probs))
            modes[index] = mode
            logprobs[index] = float(np.log(max(float(probs[mode]), 1e-12)))
            entropies[index] = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        self.trade_modes = modes
        self.last_trade = modes
        self.trade_decisions.append({
            "checkpoint": int(self.last_step),
            "context": self.context[0].float().cpu().numpy().astype(np.float16),
            "modes": modes.copy(),
            "active": active,
            "old_logprobs": logprobs,
            "entropies": entropies,
            "old_value": value,
        })

    def _route_adjustment(self, features: np.ndarray) -> np.ndarray:
        self._candidate_features = np.asarray(features, dtype=np.float16).copy()
        if self.loaded and not self.enable_trading:
            self._refresh_route_context()
        if not self.loaded or self.context is None:
            return np.zeros(len(features), dtype=np.float32)
        torch = self.torch
        dtype = next(self.model.parameters()).dtype
        candidates = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(dtype=dtype)
        with torch.inference_mode():
            values = self.model.score_routes(self.context, candidates.unsqueeze(0))[0]
        return values.float().cpu().numpy() * self.residual_weight

    def _choose_candidate(self, scores: np.ndarray) -> tuple[int, bool]:
        scaled = np.asarray(scores, dtype=np.float64) / self.temperature
        scaled -= float(np.max(scaled))
        probabilities = np.exp(np.clip(scaled, -60.0, 0.0))
        probabilities /= probabilities.sum()
        if self.uniform_mix > 0.0:
            probabilities = (
                (1.0 - self.uniform_mix) * probabilities
                + self.uniform_mix / len(probabilities)
            )
        draw = self.selector.rng.random()
        cumulative = 0.0
        selected = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if draw <= cumulative:
                selected = index
                break
        self._last_sample = {
            "logprob": float(np.log(max(probabilities[selected], 1e-12))),
            "entropy": float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))),
        }
        return selected, len(probabilities) > 1

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        decision = self.selector.select(fingerprint, checkpoint)
        sampled = self.selector.training_decisions[-1]
        if self._candidate_features is None:
            raise RuntimeError("route candidates were not captured")
        count = len(self._candidate_features)
        old_value = 0.0
        if self.loaded and self.context is not None:
            with self.torch.inference_mode():
                old_value = float(self.model.value_head(self.context).item())
        self.rl_decisions.append(
            {
                "checkpoint": int(checkpoint),
                "features": self._candidate_features,
                "base_scores": (
                    (
                        numpy_forward(self.base_model, self._candidate_features.astype(np.float32))
                        if self.base_model is not None
                        else np.zeros(count, dtype=np.float32)
                    )
                    + np.linspace(0.0, -self.prior_weight, count, dtype=np.float32)
                ).astype(np.float16),
                "selected": int(sampled.selected_rank),
                "explored": bool(sampled.explored),
                "old_logprob": float(
                    self._last_sample["logprob"] if self._last_sample is not None else 0.0
                ),
                "old_value": old_value,
                "entropy": float(
                    self._last_sample["entropy"] if self._last_sample is not None else 0.0
                ),
            }
        )
        self._last_sample = None
        return decision

    def trace_arrays(self, top_k: int = 8) -> dict[str, np.ndarray]:
        if not self.encoded_history or not self.rl_decisions:
            raise RuntimeError("cannot export an empty rollout trace")
        result = {
            "self_public": np.stack([row.self_public for row in self.encoded_history]),
            "opponent_public": np.stack([row.opponent_public for row in self.encoded_history]),
            "private_plan": np.stack([row.private_plan for row in self.encoded_history]),
            "market_town": np.stack([row.market_town for row in self.encoded_history]),
            "checkpoints": np.asarray([row["checkpoint"] for row in self.rl_decisions], dtype=np.int16),
            "selected": np.asarray([row["selected"] for row in self.rl_decisions], dtype=np.int8),
            "explored": np.asarray([row["explored"] for row in self.rl_decisions], dtype=np.bool_),
            "old_logprobs": np.asarray(
                [row["old_logprob"] for row in self.rl_decisions], dtype=np.float32
            ),
            "old_values": np.asarray(
                [row["old_value"] for row in self.rl_decisions], dtype=np.float32
            ),
            "policy_entropies": np.asarray(
                [row["entropy"] for row in self.rl_decisions], dtype=np.float32
            ),
        }
        features = np.zeros((len(self.rl_decisions), top_k, FEATURE_DIM), dtype=np.float16)
        base = np.full((len(self.rl_decisions), top_k), -20.0, dtype=np.float16)
        counts = np.empty(len(self.rl_decisions), dtype=np.int8)
        for index, row in enumerate(self.rl_decisions):
            count = min(top_k, len(row["features"]))
            features[index, :count] = row["features"][:count]
            base[index, :count] = row["base_scores"][:count]
            counts[index] = count
        result["route_features"] = features
        result["base_scores"] = base
        result["candidate_counts"] = counts
        if self.trade_decisions:
            result["trade_checkpoints"] = np.asarray(
                [row["checkpoint"] for row in self.trade_decisions], dtype=np.int16
            )
            result["trade_contexts"] = np.stack(
                [row["context"] for row in self.trade_decisions]
            )
            result["trade_modes"] = np.stack(
                [row["modes"] for row in self.trade_decisions]
            )
            result["trade_active"] = np.stack(
                [row["active"] for row in self.trade_decisions]
            )
            result["trade_old_logprobs"] = np.stack(
                [row["old_logprobs"] for row in self.trade_decisions]
            )
            result["trade_entropies"] = np.stack(
                [row["entropies"] for row in self.trade_decisions]
            )
            result["trade_old_values"] = np.asarray(
                [row["old_value"] for row in self.trade_decisions], dtype=np.float32
            )
        return result

    def adjust_market(
        self,
        observation: Any,
        route_action: Mapping[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        if (
            self.direct_trade_model is not None
            and self.direct_trade_context is not None
            and self.last_step >= self.trade_start_step
        ):
            item_logits, _ = self.direct_trade_model.trading(self.direct_trade_context)
            return apply_direct_trading_output(
                observation, route_action, item_logits, configuration,
                gate_threshold=self.direct_trade_gate_threshold,
                replace_baseline_sells=self.direct_trade_replace_baseline,
            )
        if not self.enable_trading or self.last_trade is None:
            return dict(route_action)
        return apply_trading_residual(
            observation, route_action, self.trade_modes, configuration
        )
