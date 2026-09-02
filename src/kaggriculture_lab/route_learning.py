"""Route-level counterfactual data, value model, and lexicographic losses.

This module learns ``Q(public state, candidate plan)``.  It intentionally does
not learn low-level farmer movement.  Candidate agents remain responsible for
hiring, task assignment, legal actions, and path execution; their complete
outcomes become supervision for a small strategic selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - only without the gpu extra
    raise ImportError("Route training requires the `gpu` extra (NumPy and PyTorch)") from exc

from .route_planner import PLAN_FEATURE_NAMES, PlanSpec


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
ITEMS = PRODUCTS + ANIMALS
SHOP_NAMES = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
_TILE_BUCKETS = (
    "EMPTY",
    "LOCKED",
    "WEED",
    *CROPS,
    "COOP",
    "PASTURE",
    *ANIMALS,
    "OTHER",
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(value, key, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _tile_bucket(tile: Any) -> str:
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    if not isinstance(tile, Mapping):
        return "OTHER"
    kind = str(tile.get("kind", "OTHER"))
    if kind == "PLANT":
        crop = str(tile.get("crop", "OTHER"))
        return crop if crop in CROPS else "OTHER"
    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        return str(animal) if animal in ANIMALS else kind
    return kind if kind in _TILE_BUCKETS else "OTHER"


def _farm_public_features(farm: Any) -> list[float]:
    tiles = list(_get(farm, "tiles", []) or [])
    counts = {bucket: 0 for bucket in _TILE_BUCKETS}
    for row in tiles:
        for tile in list(row or []):
            counts[_tile_bucket(tile)] += 1
    unlocked = list(_get(farm, "unlocked_quadrants", []) or [])
    hands = list(_get(farm, "hands", []) or [])
    values = [
        float(_get(farm, "money", 0.0)) / 100_000.0,
        min(len(hands) / 16.0, 2.0),
        min(float(_get(farm, "hires_today", 0)) / 16.0, 2.0),
        len(unlocked) / 4.0,
    ]
    values.extend(counts[bucket] / 100.0 for bucket in _TILE_BUCKETS)
    return values


ROUTE_CONTEXT_FEATURE_NAMES = (
    "step",
    "day",
    "hour",
    *tuple(f"market_inventory_{item}" for item in PRODUCTS),
    *tuple(f"market_price_{item}" for item in PRODUCTS),
    *tuple(f"shop_{shop}" for shop in SHOP_NAMES),
    *tuple(f"own_{name}" for name in ("money", "hands", "hires_today", "unlocked")),
    *tuple(f"own_tile_{bucket}" for bucket in _TILE_BUCKETS),
    *tuple(f"opponent_{name}" for name in ("money", "hands", "hires_today", "unlocked")),
    *tuple(f"opponent_tile_{bucket}" for bucket in _TILE_BUCKETS),
    *tuple(f"shed_{item}" for item in ITEMS),
    *tuple(f"carried_{item}" for item in ITEMS),
    *tuple(f"seed_{crop}" for crop in CROPS),
)


def encode_route_context(observation: Any) -> np.ndarray:
    """Encode only information legally visible to the acting player."""

    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    hour = int(_get(observation, "hour", step % 24) or 0)
    values: list[float] = [step / 719.0, day / 30.0, hour / 24.0]

    market = _get(observation, "market", {}) or {}
    inventory = _mapping(_get(market, "inventory", {}) or {})
    prices = _mapping(_get(market, "prices", {}) or {})
    values.extend(float(inventory.get(item, 10_000)) / 10_000.0 for item in PRODUCTS)
    values.extend(float(prices.get(item, 0)) / 500.0 for item in PRODUCTS)

    town = _get(observation, "town", {}) or {}
    unlocked_shops = list(_get(town, "unlocked_shops", []) or [])
    values.extend(unlocked_shops.count(shop) / 8.0 for shop in SHOP_NAMES)

    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    while len(farms) < 2:
        farms.append({})
    own = farms[player] if player in (0, 1) else farms[0]
    opponent = farms[1 - player] if player in (0, 1) else farms[1]
    values.extend(_farm_public_features(own))
    values.extend(_farm_public_features(opponent))

    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}) or {})
    seeds = _mapping(_get(private, "seeds", {}) or {})
    inventories = list(_get(private, "inventories", []) or [])
    carried = {item: 0 for item in ITEMS}
    for unit_inventory in inventories:
        row = _mapping(unit_inventory)
        for item in ITEMS:
            carried[item] += int(row.get(item, 0) or 0)
    values.extend(min(float(shed.get(item, 0)) / 100.0, 5.0) for item in ITEMS)
    values.extend(min(float(carried[item]) / 100.0, 5.0) for item in ITEMS)
    values.extend(min(float(seeds.get(crop, 0)) / 100.0, 5.0) for crop in CROPS)

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(ROUTE_CONTEXT_FEATURE_NAMES),):
        raise AssertionError(
            f"Route context has shape {vector.shape}, expected "
            f"({len(ROUTE_CONTEXT_FEATURE_NAMES)},)"
        )
    return vector


@dataclass(frozen=True)
class CounterfactualRecord:
    scenario_id: str
    context_hash: str
    route_name: str
    seed: int
    seat: int
    decision_step: int
    opponent: str
    prefix_agent: str
    context: tuple[float, ...]
    plan: tuple[float, ...]
    completed: float
    win: float
    margin: float
    terminal_cash: float
    opponent_cash: float
    steps: int
    candidate_status: str
    opponent_status: str
    error: str | None = None

    def __post_init__(self) -> None:
        if len(self.context) != len(ROUTE_CONTEXT_FEATURE_NAMES):
            raise ValueError("CounterfactualRecord context dimension mismatch")
        if len(self.plan) != len(PLAN_FEATURE_NAMES):
            raise ValueError("CounterfactualRecord plan dimension mismatch")
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if not 0.0 <= self.completed <= 1.0:
            raise ValueError("completed must be in [0, 1]")
        if not 0.0 <= self.win <= 1.0:
            raise ValueError("win must be in [0, 1]")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["context"] = list(self.context)
        payload["plan"] = list(self.plan)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CounterfactualRecord":
        data = dict(payload)
        data["context"] = tuple(float(value) for value in data["context"])
        data["plan"] = tuple(float(value) for value in data["plan"])
        return cls(**data)


def write_counterfactual_records(
    path: str | Path,
    records: Iterable[CounterfactualRecord],
    *,
    append: bool = False,
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")
            count += 1
    return count


def read_counterfactual_records(path: str | Path) -> list[CounterfactualRecord]:
    records: list[CounterfactualRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(CounterfactualRecord.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid record at line {line_number}: {exc}") from exc
    return records


def validate_counterfactual_groups(
    records: Sequence[CounterfactualRecord],
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[CounterfactualRecord]] = {}
    for record in records:
        groups.setdefault(record.scenario_id, []).append(record)
    result: dict[str, tuple[str, ...]] = {}
    for scenario_id, rows in groups.items():
        hashes = {row.context_hash for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"Scenario {scenario_id} contains different decision states")
        routes = [row.route_name for row in rows]
        if len(routes) != len(set(routes)):
            raise ValueError(f"Scenario {scenario_id} contains duplicate routes")
        result[scenario_id] = tuple(routes)
    return result


@dataclass(frozen=True)
class RouteNormalizer:
    context_mean: tuple[float, ...]
    context_std: tuple[float, ...]
    plan_mean: tuple[float, ...]
    plan_std: tuple[float, ...]
    margin_scale: float
    cash_scale: float

    @classmethod
    def fit(cls, records: Sequence[CounterfactualRecord]) -> "RouteNormalizer":
        if not records:
            raise ValueError("Cannot fit a normalizer without records")
        context = np.asarray([record.context for record in records], dtype=np.float32)
        plan = np.asarray([record.plan for record in records], dtype=np.float32)
        context_std = np.maximum(context.std(axis=0), 1e-4)
        plan_std = np.maximum(plan.std(axis=0), 1e-4)
        margins = np.asarray([record.margin for record in records], dtype=np.float32)
        cash = np.asarray([record.terminal_cash for record in records], dtype=np.float32)
        margin_scale = float(max(np.percentile(np.abs(margins), 90), 1.0))
        cash_scale = float(max(np.percentile(np.abs(cash), 90), 1.0))
        return cls(
            context_mean=tuple(float(value) for value in context.mean(axis=0)),
            context_std=tuple(float(value) for value in context_std),
            plan_mean=tuple(float(value) for value in plan.mean(axis=0)),
            plan_std=tuple(float(value) for value in plan_std),
            margin_scale=margin_scale,
            cash_scale=cash_scale,
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RouteNormalizer":
        data = dict(values)
        for key in ("context_mean", "context_std", "plan_mean", "plan_std"):
            data[key] = tuple(float(value) for value in data[key])
        return cls(**data)


class RouteBatch(NamedTuple):
    context: torch.Tensor
    plan: torch.Tensor
    completed: torch.Tensor
    win: torch.Tensor
    margin: torch.Tensor
    cash: torch.Tensor
    group: torch.Tensor


class RoutePredictions(NamedTuple):
    completion_logit: torch.Tensor
    win_logit: torch.Tensor
    margin: torch.Tensor
    cash: torch.Tensor


class RouteValueNetwork(nn.Module):
    """Small route-value model for CPU deployment and large offline batches."""

    def __init__(
        self,
        context_dim: int = len(ROUTE_CONTEXT_FEATURE_NAMES),
        plan_dim: int = len(PLAN_FEATURE_NAMES),
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.plan_dim = plan_dim
        self.hidden_size = hidden_size
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.plan_encoder = nn.Sequential(
            nn.Linear(plan_dim, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.SiLU(),
        )
        fusion_size = hidden_size + hidden_size // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.completion_head = nn.Linear(hidden_size, 1)
        self.win_head = nn.Linear(hidden_size, 1)
        self.margin_head = nn.Linear(hidden_size, 1)
        self.cash_head = nn.Linear(hidden_size, 1)

    def forward(self, context: torch.Tensor, plan: torch.Tensor) -> RoutePredictions:
        latent = self.fusion(
            torch.cat((self.context_encoder(context), self.plan_encoder(plan)), dim=-1)
        )
        return RoutePredictions(
            completion_logit=self.completion_head(latent).squeeze(-1),
            win_logit=self.win_head(latent).squeeze(-1),
            margin=self.margin_head(latent).squeeze(-1),
            cash=self.cash_head(latent).squeeze(-1),
        )


@dataclass(frozen=True)
class RouteLossConfig:
    completion_weight: float = 2.0
    win_weight: float = 2.0
    margin_weight: float = 1.0
    cash_weight: float = 0.25
    ranking_weight: float = 1.0


class RouteLosses(NamedTuple):
    total: torch.Tensor
    completion: torch.Tensor
    win: torch.Tensor
    margin: torch.Tensor
    cash: torch.Tensor
    ranking: torch.Tensor
    pair_count: int


def records_to_batch(
    records: Sequence[CounterfactualRecord],
    indices: Sequence[int],
    normalizer: RouteNormalizer,
    *,
    device: str | torch.device,
) -> RouteBatch:
    selected = [records[index] for index in indices]
    if not selected:
        raise ValueError("Cannot build an empty route batch")
    context = np.asarray([record.context for record in selected], dtype=np.float32)
    plan = np.asarray([record.plan for record in selected], dtype=np.float32)
    context = (context - np.asarray(normalizer.context_mean)) / np.asarray(
        normalizer.context_std
    )
    plan = (plan - np.asarray(normalizer.plan_mean)) / np.asarray(normalizer.plan_std)
    group_names: dict[str, int] = {}
    groups = []
    for record in selected:
        group_names.setdefault(record.scenario_id, len(group_names))
        groups.append(group_names[record.scenario_id])
    return RouteBatch(
        context=torch.as_tensor(context, dtype=torch.float32, device=device),
        plan=torch.as_tensor(plan, dtype=torch.float32, device=device),
        completed=torch.tensor(
            [record.completed for record in selected], dtype=torch.float32, device=device
        ),
        win=torch.tensor([record.win for record in selected], dtype=torch.float32, device=device),
        margin=torch.tensor(
            [record.margin / normalizer.margin_scale for record in selected],
            dtype=torch.float32,
            device=device,
        ),
        cash=torch.tensor(
            [record.terminal_cash / normalizer.cash_scale for record in selected],
            dtype=torch.float32,
            device=device,
        ),
        group=torch.tensor(groups, dtype=torch.long, device=device),
    )


def _target_priority(
    batch: RouteBatch, left: int, right: int, tolerance: float = 1e-6
) -> tuple[int, float]:
    fields = (batch.completed, batch.win, batch.margin, batch.cash)
    for priority, values in enumerate(fields):
        delta = float((values[left] - values[right]).detach().cpu())
        if abs(delta) > tolerance:
            return priority, 1.0 if delta > 0 else -1.0
    return -1, 0.0


def lexicographic_pairwise_loss(
    predictions: RoutePredictions,
    batch: RouteBatch,
) -> tuple[torch.Tensor, int]:
    heads = (
        predictions.completion_logit,
        predictions.win_logit,
        predictions.margin,
        predictions.cash,
    )
    losses: list[torch.Tensor] = []
    pair_count = 0
    for group_id in torch.unique(batch.group).tolist():
        indices = torch.nonzero(batch.group == group_id, as_tuple=False).flatten().tolist()
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                priority, sign = _target_priority(batch, left, right)
                if priority < 0:
                    continue
                difference = heads[priority][left] - heads[priority][right]
                losses.append(F.softplus(-difference * sign))
                pair_count += 1
    if not losses:
        return predictions.margin.sum() * 0.0, 0
    return torch.stack(losses).mean(), pair_count


def route_value_loss(
    predictions: RoutePredictions,
    batch: RouteBatch,
    config: RouteLossConfig | None = None,
) -> RouteLosses:
    cfg = config or RouteLossConfig()
    completion = F.binary_cross_entropy_with_logits(
        predictions.completion_logit, batch.completed
    )
    complete_mask = batch.completed > 0.5
    if bool(complete_mask.any()):
        win = F.binary_cross_entropy_with_logits(
            predictions.win_logit[complete_mask], batch.win[complete_mask]
        )
        margin = F.smooth_l1_loss(
            predictions.margin[complete_mask], batch.margin[complete_mask]
        )
        cash = F.smooth_l1_loss(
            predictions.cash[complete_mask], batch.cash[complete_mask]
        )
    else:
        zero = predictions.margin.sum() * 0.0
        win = margin = cash = zero
    ranking, pair_count = lexicographic_pairwise_loss(predictions, batch)
    total = (
        cfg.completion_weight * completion
        + cfg.win_weight * win
        + cfg.margin_weight * margin
        + cfg.cash_weight * cash
        + cfg.ranking_weight * ranking
    )
    return RouteLosses(total, completion, win, margin, cash, ranking, pair_count)


def predicted_lexicographic_score(predictions: RoutePredictions) -> torch.Tensor:
    """Scalar used only to choose among routes after multi-head training."""
    return (
        8.0 * torch.sigmoid(predictions.completion_logit)
        + 4.0 * torch.sigmoid(predictions.win_logit)
        + torch.tanh(predictions.margin)
        + 0.1 * torch.tanh(predictions.cash)
    )


def target_lexicographic_key(record: CounterfactualRecord) -> tuple[float, ...]:
    return (record.completed, record.win, record.margin, record.terminal_cash)


def scenario_batches(
    records: Sequence[CounterfactualRecord],
    *,
    scenarios_per_batch: int,
    shuffle: bool,
    seed: int,
) -> Iterator[list[int]]:
    if scenarios_per_batch <= 0:
        raise ValueError("scenarios_per_batch must be positive")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.scenario_id, []).append(index)
    scenario_ids = list(groups)
    if shuffle:
        random.Random(seed).shuffle(scenario_ids)
    for start in range(0, len(scenario_ids), scenarios_per_batch):
        batch_ids = scenario_ids[start : start + scenarios_per_batch]
        yield [index for scenario_id in batch_ids for index in groups[scenario_id]]


def split_records_by_scenario(
    records: Sequence[CounterfactualRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[list[CounterfactualRecord], list[CounterfactualRecord]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    scenario_ids = sorted({record.scenario_id for record in records})
    random.Random(seed).shuffle(scenario_ids)
    validation_count = int(round(len(scenario_ids) * validation_fraction))
    if validation_fraction > 0 and len(scenario_ids) > 1:
        validation_count = max(1, min(validation_count, len(scenario_ids) - 1))
    validation_ids = set(scenario_ids[:validation_count])
    training = [record for record in records if record.scenario_id not in validation_ids]
    validation = [record for record in records if record.scenario_id in validation_ids]
    return training, validation


@torch.no_grad()
def evaluate_route_model(
    model: RouteValueNetwork,
    records: Sequence[CounterfactualRecord],
    normalizer: RouteNormalizer,
    *,
    device: str | torch.device,
    scenarios_per_batch: int = 64,
) -> dict[str, float]:
    if not records:
        return {
            "loss": math.nan,
            "top1": math.nan,
            "completion_accuracy": math.nan,
            "win_accuracy": math.nan,
            "margin_mae": math.nan,
            "cash_mae": math.nan,
        }
    model.eval()
    losses: list[float] = []
    completion_correct = 0
    win_correct = 0
    completed_count = 0
    margin_error = 0.0
    cash_error = 0.0
    predictions_by_scenario: dict[str, list[tuple[float, CounterfactualRecord]]] = {}
    for indices in scenario_batches(
        records,
        scenarios_per_batch=scenarios_per_batch,
        shuffle=False,
        seed=0,
    ):
        batch = records_to_batch(records, indices, normalizer, device=device)
        output = model(batch.context, batch.plan)
        losses.append(float(route_value_loss(output, batch).total.cpu()))
        completion_prediction = torch.sigmoid(output.completion_logit) >= 0.5
        completion_correct += int((completion_prediction == (batch.completed >= 0.5)).sum())
        mask = batch.completed > 0.5
        if bool(mask.any()):
            win_prediction = torch.sigmoid(output.win_logit[mask]) >= 0.5
            win_correct += int((win_prediction == (batch.win[mask] >= 0.5)).sum())
            completed_count += int(mask.sum())
            margin_error += float(
                (output.margin[mask] - batch.margin[mask]).abs().sum().cpu()
            )
            cash_error += float((output.cash[mask] - batch.cash[mask]).abs().sum().cpu())
        scores = predicted_lexicographic_score(output).cpu().tolist()
        for index, score in zip(indices, scores, strict=True):
            record = records[index]
            predictions_by_scenario.setdefault(record.scenario_id, []).append((score, record))

    top1_correct = 0
    for rows in predictions_by_scenario.values():
        predicted = max(rows, key=lambda row: row[0])[1]
        target = max((row[1] for row in rows), key=target_lexicographic_key)
        top1_correct += int(predicted.route_name == target.route_name)
    return {
        "loss": float(np.mean(losses)),
        "top1": top1_correct / max(1, len(predictions_by_scenario)),
        "completion_accuracy": completion_correct / len(records),
        "win_accuracy": win_correct / max(1, completed_count),
        "margin_mae": margin_error / max(1, completed_count) * normalizer.margin_scale,
        "cash_mae": cash_error / max(1, completed_count) * normalizer.cash_scale,
    }


@dataclass
class RouteSelector:
    model: RouteValueNetwork
    normalizer: RouteNormalizer
    device: torch.device

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "RouteSelector":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RouteValueNetwork(
            context_dim=int(checkpoint["context_dim"]),
            plan_dim=int(checkpoint["plan_dim"]),
            hidden_size=int(checkpoint["hidden_size"]),
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        return cls(
            model=model,
            normalizer=RouteNormalizer.from_mapping(checkpoint["normalizer"]),
            device=torch.device(device),
        )

    @torch.no_grad()
    def score(
        self,
        observation: Any,
        plans: Sequence[PlanSpec],
    ) -> list[dict[str, float | str]]:
        if not plans:
            raise ValueError("At least one plan is required")
        context = encode_route_context(observation)
        records = [
            CounterfactualRecord(
                scenario_id="inference",
                context_hash="inference",
                route_name=plan.name,
                seed=0,
                seat=int(_get(observation, "player", 0) or 0),
                decision_step=int(_get(observation, "step", 0) or 0),
                opponent="inference",
                prefix_agent="inference",
                context=tuple(float(value) for value in context),
                plan=plan.normalized_features(),
                completed=1.0,
                win=0.5,
                margin=0.0,
                terminal_cash=0.0,
                opponent_cash=0.0,
                steps=0,
                candidate_status="INFERENCE",
                opponent_status="INFERENCE",
            )
            for plan in plans
        ]
        batch = records_to_batch(
            records,
            list(range(len(records))),
            self.normalizer,
            device=self.device,
        )
        predictions = self.model(batch.context, batch.plan)
        score = predicted_lexicographic_score(predictions)
        result = []
        for index, plan in enumerate(plans):
            result.append(
                {
                    "route": plan.name,
                    "score": float(score[index].cpu()),
                    "completion_probability": float(
                        torch.sigmoid(predictions.completion_logit[index]).cpu()
                    ),
                    "win_probability": float(
                        torch.sigmoid(predictions.win_logit[index]).cpu()
                    ),
                    "predicted_margin": float(
                        predictions.margin[index].cpu() * self.normalizer.margin_scale
                    ),
                    "predicted_cash": float(
                        predictions.cash[index].cpu() * self.normalizer.cash_scale
                    ),
                }
            )
        return sorted(result, key=lambda row: float(row["score"]), reverse=True)
