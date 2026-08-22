"""V7 expert injection, phase labels, and replayable proposal utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp
from typing import Any, Sequence

import numpy as np

from .board_policy import BOARD_SIZE, _canonical_farms
from .decision_schema import (
    TASK_PRIOR,
    CandidateSource,
    DecisionCandidate,
    TaskType,
    _tile_at,
)
from .gpu_policy import ANIMALS, CROPS, ITEMS, MAX_UNITS, _get, _mapping
from .hierarchical_schema import (
    MAX_TASK_CARDS,
    CandidateOrigin,
    EncodedTaskSet,
    TaskCard,
)
from .hierarchical_trace import (
    INTENT_PHASE_INDEX,
    INTENT_PHASES,
    SoftIntentDistribution,
)


PROPOSAL_TASK_TYPES = (
    TaskType.SAFE_RECOVERY,
    TaskType.CROP_PRODUCTION,
    TaskType.WATER_CROP,
    TaskType.CLEAR_OR_REMOVE_TILE,
    TaskType.BUILD_ANIMAL_STRUCTURE,
    TaskType.ANIMAL_PLACE,
    TaskType.ANIMAL_FEED,
    TaskType.ANIMAL_CARE,
    TaskType.ANIMAL_COLLECT_PRODUCT,
    TaskType.ANIMAL_COLLECT_FERTILIZER,
    TaskType.APPLY_FERTILIZER,
    TaskType.SHED_PICKUP,
    TaskType.SHED_DEPOSIT,
)
PROPOSAL_TASK_INDEX = {
    task: index for index, task in enumerate(PROPOSAL_TASK_TYPES)
}
PROPOSAL_AUXILIARIES = ("urgency", "eta", "value", "slack")


@dataclass(frozen=True)
class SoftAssignmentTargets:
    probabilities: np.ndarray
    active: np.ndarray
    closed_mass: np.ndarray


@dataclass(frozen=True)
class DenseProposalTargets:
    task_probabilities: np.ndarray
    item_targets: np.ndarray
    item_weights: np.ndarray
    auxiliary_targets: np.ndarray
    spatial_weights: np.ndarray


@dataclass(frozen=True)
class PhaseTargets:
    probabilities: np.ndarray
    active: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class SampledProposalCards:
    cards: list[tuple[TaskCard, ...]]
    log_probs: Any
    entropies: Any
    sampled_indices: tuple[tuple[int, ...], ...]
    drawn_indices: Any


def phase_targets(
    distributions: Sequence[SoftIntentDistribution],
) -> PhaseTargets:
    probabilities = np.zeros((MAX_UNITS, len(INTENT_PHASES)), dtype=np.float32)
    active = np.zeros(MAX_UNITS, dtype=np.bool_)
    confidence = np.zeros(MAX_UNITS, dtype=np.float32)
    for distribution in distributions:
        unit = distribution.unit
        if not 0 <= unit < MAX_UNITS:
            continue
        active[unit] = True
        confidence[unit] = distribution.confidence
        for hypothesis in distribution.hypotheses:
            probabilities[unit, INTENT_PHASE_INDEX[hypothesis.phase]] += (
                hypothesis.probability
            )
        total = float(probabilities[unit].sum())
        if total > 0:
            probabilities[unit] /= total
    return PhaseTargets(probabilities, active, confidence)


def _source_for_task(task: TaskType) -> CandidateSource:
    if task in (TaskType.CROP_PRODUCTION, TaskType.CLEAR_OR_REMOVE_TILE):
        return CandidateSource.CROP
    if task in (
        TaskType.BUILD_ANIMAL_STRUCTURE,
        TaskType.ANIMAL_PLACE,
        TaskType.ANIMAL_COLLECT_PRODUCT,
        TaskType.ANIMAL_COLLECT_FERTILIZER,
    ):
        return CandidateSource.ANIMAL
    if task in (TaskType.ANIMAL_FEED, TaskType.ANIMAL_CARE, TaskType.WATER_CROP):
        return CandidateSource.MAINTENANCE
    if task == TaskType.APPLY_FERTILIZER:
        return CandidateSource.FERTILIZER_APPLY
    return CandidateSource.INVENTORY


def _coerce_item(task: TaskType, item_id: int) -> int:
    item = ITEMS[item_id] if 0 <= item_id < len(ITEMS) else None
    if task == TaskType.ANIMAL_FEED:
        return ITEMS.index("WHEAT")
    if task in (TaskType.APPLY_FERTILIZER, TaskType.ANIMAL_COLLECT_FERTILIZER):
        return ITEMS.index("FERTILIZER")
    if task == TaskType.CROP_PRODUCTION and item not in CROPS:
        return ITEMS.index("WHEAT")
    if task in (TaskType.BUILD_ANIMAL_STRUCTURE, TaskType.ANIMAL_PLACE) and item not in ANIMALS:
        return ITEMS.index("GOOSE")
    if task in (
        TaskType.WATER_CROP,
        TaskType.CLEAR_OR_REMOVE_TILE,
        TaskType.SAFE_RECOVERY,
        TaskType.SHED_DEPOSIT,
    ):
        return -1
    return item_id


def expert_cards_from_soft_intents(
    distributions: Sequence[SoftIntentDistribution],
    *,
    max_cards: int = 8,
    min_probability: float = 0.02,
    dropout: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[TaskCard, ...]:
    """Turn Top-M labels into owner-bound oracle cards for BC only."""

    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    ranked: list[tuple[float, TaskCard]] = []
    for distribution in distributions:
        for hypothesis in distribution.hypotheses:
            intent = hypothesis.intent
            if (
                hypothesis.probability < min_probability
                or intent.task_type in (TaskType.NONE, TaskType.IDLE_OR_PASS)
                or rng.random() < dropout
            ):
                continue
            item_id = _coerce_item(intent.task_type, intent.item_id)
            candidate = DecisionCandidate(
                task_type=intent.task_type,
                target_x=intent.target_x,
                target_y=intent.target_y,
                item_id=item_id,
                source=_source_for_task(intent.task_type),
                prior=8_000.0 + 4_000.0 * hypothesis.probability,
                expected_value=TASK_PRIOR.get(intent.task_type, 0.0),
            )
            ranked.append(
                (
                    hypothesis.probability,
                    TaskCard(
                        candidate=candidate,
                        preferred_owner=distribution.unit,
                        origin=CandidateOrigin.EXPERT,
                    ),
                )
            )
    ranked.sort(key=lambda value: value[0], reverse=True)
    return tuple(card for _, card in ranked[: max(max_cards, 0)])


def soft_assignment_targets(
    encoded: EncodedTaskSet,
    distributions: Sequence[SoftIntentDistribution],
) -> SoftAssignmentTargets:
    """Project inverse Top-M distributions onto the current candidate union."""

    probabilities = np.zeros((len(encoded.pair_mask), MAX_TASK_CARDS), dtype=np.float32)
    active = np.zeros(len(encoded.pair_mask), dtype=np.bool_)
    closed_mass = np.zeros(len(encoded.pair_mask), dtype=np.float32)
    idle = next(
        index
        for index, card in enumerate(encoded.tasks)
        if card.task_type == TaskType.IDLE_OR_PASS
    )
    for distribution in distributions:
        unit = distribution.unit
        if not 0 <= unit < probabilities.shape[0]:
            continue
        active[unit] = True
        unmatched = 0.0
        for hypothesis in distribution.hypotheses:
            intent = hypothesis.intent
            matches: list[int] = []
            for index, card in enumerate(encoded.tasks):
                candidate = card.candidate
                if not encoded.pair_mask[unit, index]:
                    continue
                if candidate.task_type != intent.task_type:
                    continue
                if intent.task_type != TaskType.IDLE_OR_PASS and (
                    candidate.target_x != intent.target_x
                    or candidate.target_y != intent.target_y
                ):
                    continue
                if intent.item_id >= 0 and candidate.item_id != _coerce_item(
                    intent.task_type, intent.item_id
                ):
                    continue
                matches.append(index)
            if matches:
                share = hypothesis.probability / len(matches)
                probabilities[unit, matches] += share
                closed_mass[unit] += hypothesis.probability
            else:
                unmatched += hypothesis.probability
        probabilities[unit, idle] += unmatched
        total = float(probabilities[unit].sum())
        if total > 0:
            probabilities[unit] /= total
    return SoftAssignmentTargets(probabilities, active, closed_mass)


def dense_proposal_targets(
    batch: Sequence[Sequence[SoftIntentDistribution]],
) -> DenseProposalTargets:
    """Rasterize Top-M labels for focal/KL/ETA-value-slack supervision."""

    size = len(batch)
    task = np.zeros(
        (size, len(PROPOSAL_TASK_TYPES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32
    )
    item = np.zeros((size, BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    item_weights = np.zeros((size, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    auxiliaries = np.zeros(
        (size, len(PROPOSAL_AUXILIARIES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32
    )
    spatial_weights = np.zeros((size, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row, distributions in enumerate(batch):
        item_best = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for distribution in distributions:
            for hypothesis in distribution.hypotheses:
                intent = hypothesis.intent
                task_index = PROPOSAL_TASK_INDEX.get(intent.task_type)
                x, y = intent.target_x, intent.target_y
                if task_index is None or not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                    continue
                weight = float(hypothesis.probability)
                task[row, task_index, y, x] = min(
                    task[row, task_index, y, x] + weight, 1.0
                )
                eta = max(hypothesis.eta_steps, 0)
                auxiliary = np.asarray(
                    (
                        exp(-eta / 8.0),
                        min(eta / 48.0, 2.0),
                        min(TASK_PRIOR.get(intent.task_type, 0.0) / 12_000.0, 1.0),
                        max((24.0 - eta) / 24.0, 0.0),
                    ),
                    dtype=np.float32,
                )
                previous = spatial_weights[row, y, x]
                total = previous + weight
                auxiliaries[row, :, y, x] = (
                    auxiliaries[row, :, y, x] * previous + auxiliary * weight
                ) / max(total, 1e-12)
                spatial_weights[row, y, x] = total
                coerced = _coerce_item(intent.task_type, intent.item_id)
                if coerced >= 0 and weight > item_best[y, x]:
                    item[row, y, x] = coerced + 1
                    item_best[y, x] = weight
                    item_weights[row, y, x] = weight
    return DenseProposalTargets(task, item, item_weights, auxiliaries, spatial_weights)


def _cards_from_ranked_indices(
    probability: Any,
    item_logits: Any,
    auxiliary_predictions: Any,
    observation: Any,
    ranked_indices: Sequence[int],
    *,
    top_k: int,
    score_threshold: float,
) -> tuple[tuple[TaskCard, ...], tuple[int, ...]]:
    own, _ = _canonical_farms(observation)
    per_family = max(2, ceil(max(top_k, 1) / 4))
    family_counts: dict[TaskType, int] = {}
    cards: list[TaskCard] = []
    accepted: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_index in ranked_indices:
        raw_score = float(probability.flatten()[raw_index])
        if len(cards) >= top_k or raw_score < score_threshold:
            continue
        task_index, spatial = divmod(raw_index, BOARD_SIZE * BOARD_SIZE)
        y, x = divmod(spatial, BOARD_SIZE)
        task_type = PROPOSAL_TASK_TYPES[task_index]
        if family_counts.get(task_type, 0) >= per_family:
            continue
        raw_tile = _tile_at(own, x, y)
        tile = _mapping(raw_tile)
        if raw_tile == "LOCKED" or tile.get("kind") == "LOCKED":
            continue
        predicted_item = int(item_logits[:, y, x].argmax().item()) - 1
        item_id = _coerce_item(task_type, predicted_item)
        if task_type == TaskType.SHED_PICKUP and item_id < 0:
            continue
        key = (int(task_type), x, y, item_id)
        if key in seen:
            continue
        seen.add(key)
        family_counts[task_type] = family_counts.get(task_type, 0) + 1
        urgency, eta, value, slack = auxiliary_predictions[:, y, x].tolist()
        step = int(_get(observation, "step", 0) or 0)
        deadline = min(
            719,
            step
            + max(
                int(
                    round(
                        48.0 * max(eta, 0.0)
                        + 24.0 * max(slack, 0.0)
                    )
                ),
                1,
            ),
        )
        candidate = DecisionCandidate(
            task_type=task_type,
            target_x=x,
            target_y=y,
            item_id=item_id,
            source=_source_for_task(task_type),
            mandatory=False,
            prior=(
                2_000.0
                + 8_000.0 * raw_score
                + 1_000.0 * max(urgency, 0.0)
            ),
            deadline_step=deadline if urgency > 0.5 else 719,
            expected_value=max(value, 0.0) * 12_000.0,
        )
        cards.append(TaskCard(candidate, origin=CandidateOrigin.LEARNED))
        accepted.append(raw_index)
    return tuple(cards), tuple(accepted)


def proposal_cards_from_logits(
    task_logits: Any,
    item_logits: Any,
    auxiliary_predictions: Any,
    observations: Sequence[Any],
    *,
    top_k: int = 24,
    score_threshold: float = 0.05,
) -> list[tuple[TaskCard, ...]]:
    """Decode non-differentiable Top-K CNN heatmaps into deployable task cards."""

    import torch

    probabilities = torch.sigmoid(task_logits).detach().cpu()
    items = item_logits.detach().cpu()
    auxiliaries = auxiliary_predictions.detach().cpu()
    output: list[tuple[TaskCard, ...]] = []
    for row, observation in enumerate(observations):
        flat = probabilities[row].flatten()
        count = min(max(top_k * 8, top_k), flat.numel())
        _, indices = torch.topk(flat, count)
        cards, _ = _cards_from_ranked_indices(
            probabilities[row],
            items[row],
            auxiliaries[row],
            observation,
            indices.tolist(),
            top_k=top_k,
            score_threshold=score_threshold,
        )
        output.append(cards)
    return output


def sample_proposal_cards(
    task_logits: Any,
    item_logits: Any,
    auxiliary_predictions: Any,
    observations: Sequence[Any],
    *,
    top_k: int = 24,
    deterministic: bool = False,
    locked_mask: Any | None = None,
) -> SampledProposalCards:
    """Sample Top-K proposal events and return a REINFORCE surrogate log-prob."""

    import torch

    batch, _, height, width = task_logits.shape
    flat_logits = task_logits.flatten(1)
    if locked_mask is not None:
        locked = torch.as_tensor(
            locked_mask, dtype=torch.bool, device=task_logits.device
        )
        if locked.shape != (batch, height, width):
            raise ValueError("locked_mask must have shape [batch, height, width]")
        legal = (~locked[:, None]).expand_as(task_logits).flatten(1)
    else:
        legal = torch.ones_like(flat_logits, dtype=torch.bool)
        for row, observation in enumerate(observations):
            own, _ = _canonical_farms(observation)
            locked = np.asarray(
                [
                    [
                        _tile_at(own, x, y) == "LOCKED"
                        or _mapping(_tile_at(own, x, y)).get("kind") == "LOCKED"
                        for x in range(width)
                    ]
                    for y in range(height)
                ],
                dtype=np.bool_,
            )
            spatial_legal = torch.as_tensor(
                ~locked, device=task_logits.device
            )
            legal[row] = spatial_legal[None].expand(
                task_logits.shape[1], -1, -1
            ).flatten()
    floor = torch.finfo(flat_logits.dtype).min
    masked_logits = flat_logits.masked_fill(~legal, floor)
    probabilities = torch.softmax(masked_logits, dim=-1)
    entropies = torch.distributions.Categorical(logits=masked_logits).entropy()
    draw_count = min(max(top_k * 3, top_k), flat_logits.shape[1])
    if deterministic:
        sampled = masked_logits.topk(draw_count, dim=-1).indices
    else:
        sampled = torch.multinomial(
            probabilities, draw_count, replacement=False
        )
    cpu_probability = torch.sigmoid(task_logits).detach().cpu()
    cpu_items = item_logits.detach().cpu()
    cpu_auxiliary = auxiliary_predictions.detach().cpu()
    cards: list[tuple[TaskCard, ...]] = []
    accepted_rows: list[tuple[int, ...]] = []
    for row, observation in enumerate(observations):
        decoded, accepted = _cards_from_ranked_indices(
            cpu_probability[row],
            cpu_items[row],
            cpu_auxiliary[row],
            observation,
            sampled[row].detach().cpu().tolist(),
            top_k=top_k,
            score_threshold=0.0,
        )
        cards.append(decoded)
        accepted_rows.append(accepted)
    if deterministic:
        log_probs = task_logits.flatten(1).sum(dim=-1) * 0.0
    else:
        # torch.multinomial samples without replacement.  This normalized
        # Plackett-Luce surrogate preserves useful score-function gradients
        # without letting K dominate the other policy factors.
        working = masked_logits
        log_probs = torch.zeros(batch, device=task_logits.device)
        for slot in range(draw_count):
            choice = sampled[:, slot]
            log_probs = log_probs + torch.log_softmax(working, dim=-1).gather(
                1, choice[:, None]
            ).squeeze(1)
            working = working.scatter(1, choice[:, None], floor)
        log_probs = log_probs / float(draw_count)
    return SampledProposalCards(
        cards=cards,
        log_probs=log_probs,
        entropies=entropies,
        sampled_indices=tuple(accepted_rows),
        drawn_indices=sampled,
    )


def evaluate_proposal_draws(
    task_logits: Any,
    observations: Sequence[Any],
    drawn_indices: Any,
    *,
    locked_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Recompute the normalized Plackett-Luce proposal probability for PPO."""

    import torch

    batch, _, height, width = task_logits.shape
    flat_logits = task_logits.flatten(1)
    if locked_mask is not None:
        locked = torch.as_tensor(
            locked_mask, dtype=torch.bool, device=task_logits.device
        )
        if locked.shape != (batch, height, width):
            raise ValueError("locked_mask must have shape [batch, height, width]")
        legal = (~locked[:, None]).expand_as(task_logits).flatten(1)
    else:
        legal = torch.ones_like(flat_logits, dtype=torch.bool)
        for row, observation in enumerate(observations):
            own, _ = _canonical_farms(observation)
            locked = np.asarray(
                [
                    [
                        _tile_at(own, x, y) == "LOCKED"
                        or _mapping(_tile_at(own, x, y)).get("kind") == "LOCKED"
                        for x in range(width)
                    ]
                    for y in range(height)
                ],
                dtype=np.bool_,
            )
            spatial_legal = torch.as_tensor(
                ~locked, device=task_logits.device
            )
            legal[row] = spatial_legal[None].expand(
                task_logits.shape[1], -1, -1
            ).flatten()
    floor = torch.finfo(flat_logits.dtype).min
    working = flat_logits.masked_fill(~legal, floor)
    draws = torch.as_tensor(
        drawn_indices, dtype=torch.long, device=task_logits.device
    )
    if draws.ndim != 2 or draws.shape[0] != batch:
        raise ValueError("drawn proposal indices must have shape [batch, draws]")
    log_probs = torch.zeros(batch, device=task_logits.device)
    for slot in range(draws.shape[1]):
        choice = draws[:, slot]
        log_probs = log_probs + torch.log_softmax(working, dim=-1).gather(
            1, choice[:, None]
        ).squeeze(1)
        working = working.scatter(1, choice[:, None], floor)
    log_probs = log_probs / float(max(draws.shape[1], 1))
    entropies = torch.distributions.Categorical(
        logits=flat_logits.masked_fill(~legal, floor)
    ).entropy()
    return log_probs, entropies


def soft_closure_summary(targets: Sequence[SoftAssignmentTargets]) -> dict[str, float | int]:
    active = sum(int(value.active.sum()) for value in targets)
    closed_mass = sum(float(value.closed_mass[value.active].sum()) for value in targets)
    return {
        "active_units": active,
        "closed_probability_mass": closed_mass,
        "soft_closure_rate": closed_mass / max(active, 1),
    }
