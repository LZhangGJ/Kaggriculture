"""V7 persistent task-chain policy with joint resources and exact PPO replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import log
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .board_policy import (
    BOARD_CHANNELS,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    encode_board_batch,
)
from .candidate_proposal import (
    PROPOSAL_AUXILIARIES,
    PROPOSAL_TASK_TYPES,
    SampledProposalCards,
    sample_proposal_cards,
)
from .decision_schema import CANDIDATE_FEATURES, CandidateSource, TaskType
from .gpu_policy import ITEMS, MARKET_ACTIONS, MAX_UNITS, _get
from .hierarchical_trace import INTENT_PHASES
from .hierarchical_schema import (
    BUDGET_CATEGORIES,
    BUDGET_FEATURES,
    MAX_MARKET_ORDERS,
    MAX_TASK_CARDS,
    MODE_INDEX,
    PAIR_FEATURES,
    STRATEGY_MODES,
    BudgetPlan,
    EncodedTaskSet,
    HierarchicalMemory,
    MarketBudget,
    PlannedMarketBudget,
    TaskCard,
    compile_hierarchical_action,
    generate_task_cards,
    market_prior_features,
)
from .opponent_model import (
    OPPONENT_FEATURES,
    OPPONENT_HISTORY_LAGS,
    runtime_opponent_history,
)


@dataclass(frozen=True)
class HierarchicalTensorBatch:
    boards: np.ndarray
    global_features: np.ndarray
    unit_features: np.ndarray
    unit_mask: np.ndarray
    task_features: np.ndarray
    task_ids: np.ndarray
    item_ids: np.ndarray
    source_ids: np.ndarray
    target_xy: np.ndarray
    preferred_owner_ids: np.ndarray
    task_mask: np.ndarray
    pair_features: np.ndarray
    eta_steps: np.ndarray
    pair_mask: np.ndarray
    task_capacity: np.ndarray
    market_priors: np.ndarray
    required_market_indices: np.ndarray
    opponent_history: np.ndarray
    opponent_history_mask: np.ndarray
    encoded_sets: tuple[EncodedTaskSet, ...]


@dataclass(frozen=True)
class TaskMatchingTrace:
    choices: torch.Tensor


@dataclass(frozen=True)
class HierarchicalReplayBatch:
    observations: tuple[Any, ...]
    tensors: HierarchicalTensorBatch
    proposal_draws: torch.Tensor
    matching: TaskMatchingTrace
    assignments: torch.Tensor
    market_indices: torch.Tensor
    strategy_modes: torch.Tensor
    mode_switches: torch.Tensor
    initial_mode_active: torch.Tensor
    daily_mode_active: torch.Tensor
    previous_modes: torch.Tensor
    budget_plan_features: torch.Tensor
    budget_action_active: torch.Tensor
    budget_cash_at_plan: torch.Tensor
    budget_days: torch.Tensor
    budget_spent_before: torch.Tensor
    budget_emergency_before: torch.Tensor
    budget_minimum_reserve: torch.Tensor


@dataclass
class HierarchicalPolicyBatch:
    actions: list[dict[str, Any]]
    assignments: torch.Tensor
    market_indices: torch.Tensor
    strategy_modes: torch.Tensor
    mode_switches: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor
    memories: list[HierarchicalMemory]
    proposal_log_probs: torch.Tensor
    proposal_entropies: torch.Tensor
    budget_log_probs: torch.Tensor
    budget_entropies: torch.Tensor
    budget_action_active: torch.Tensor
    replay: HierarchicalReplayBatch | None = None


@dataclass(frozen=True)
class HierarchicalContext:
    state: torch.Tensor
    board_maps: torch.Tensor
    board_tokens: torch.Tensor
    unit_tokens: torch.Tensor
    mode_logits: torch.Tensor
    switch_logits: torch.Tensor
    values: torch.Tensor
    opponent_embedding: torch.Tensor
    opponent_belief_logits: torch.Tensor
    opponent_belief_probabilities: torch.Tensor
    opponent_uncertainty: torch.Tensor
    horizon_values: torch.Tensor


@dataclass(frozen=True)
class ProposalOutput:
    task_logits: torch.Tensor
    item_logits: torch.Tensor
    auxiliary_predictions: torch.Tensor


def load_compatible_hierarchical_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load shape-compatible legacy weights and report reinitialized tensors."""

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(set(state_dict) - set(compatible))


@dataclass(frozen=True)
class BudgetHeadOutput:
    reserve_fraction: torch.Tensor
    category_fractions: torch.Tensor
    emergency_fraction: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        return torch.cat(
            (
                self.reserve_fraction[:, None],
                self.category_fractions,
                self.emergency_fraction[:, None],
            ),
            dim=-1,
        )


_BUDGET_RESERVE_CONCENTRATION = 16.0
_BUDGET_CATEGORY_CONCENTRATION = 20.0
_BUDGET_EMERGENCY_CONCENTRATION = 20.0
_MAX_EMERGENCY_FRACTION = 0.25


def _budget_action_distributions(
    output: BudgetHeadOutput,
) -> tuple[
    torch.distributions.Beta,
    torch.distributions.Dirichlet,
    torch.distributions.Beta,
]:
    epsilon = 1e-4
    reserve = output.reserve_fraction.clamp(epsilon, 1.0 - epsilon)
    emergency = (
        output.emergency_fraction / _MAX_EMERGENCY_FRACTION
    ).clamp(epsilon, 1.0 - epsilon)
    categories = output.category_fractions.clamp_min(epsilon)
    categories = categories / categories.sum(dim=-1, keepdim=True)
    return (
        torch.distributions.Beta(
            reserve * _BUDGET_RESERVE_CONCENTRATION,
            (1.0 - reserve) * _BUDGET_RESERVE_CONCENTRATION,
        ),
        torch.distributions.Dirichlet(
            categories * _BUDGET_CATEGORY_CONCENTRATION
        ),
        torch.distributions.Beta(
            emergency * _BUDGET_EMERGENCY_CONCENTRATION,
            (1.0 - emergency) * _BUDGET_EMERGENCY_CONCENTRATION,
        ),
    )


def evaluate_budget_action_features(
    output: BudgetHeadOutput,
    features: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the exact density of sampled daily budget actions."""

    if features.shape != (output.reserve_fraction.shape[0], BUDGET_FEATURES):
        raise ValueError("budget action features have an incompatible shape")
    active = active.to(device=features.device, dtype=torch.bool)
    reserve_distribution, category_distribution, emergency_distribution = (
        _budget_action_distributions(output)
    )
    epsilon = 1e-6
    reserve = features[:, 0].clamp(epsilon, 1.0 - epsilon)
    categories = features[:, 1 : 1 + len(BUDGET_CATEGORIES)].clamp_min(epsilon)
    categories = categories / categories.sum(dim=-1, keepdim=True)
    emergency_unit = (
        features[:, -1] / _MAX_EMERGENCY_FRACTION
    ).clamp(epsilon, 1.0 - epsilon)
    log_prob = (
        reserve_distribution.log_prob(reserve)
        + category_distribution.log_prob(categories)
        + emergency_distribution.log_prob(emergency_unit)
        - log(_MAX_EMERGENCY_FRACTION)
    )
    entropy = (
        reserve_distribution.entropy()
        + category_distribution.entropy()
        + emergency_distribution.entropy()
        + log(_MAX_EMERGENCY_FRACTION)
    )
    zeros = torch.zeros_like(log_prob)
    return torch.where(active, log_prob, zeros), torch.where(active, entropy, zeros)


def encode_hierarchical_batch(
    observations: Sequence[Any],
    memories: Sequence[HierarchicalMemory] | None = None,
    *,
    extra_cards: Sequence[Sequence[TaskCard]] | None = None,
    expert_quota: int = 8,
    learned_quota: int = 24,
    state_encoding: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
    opponent_histories: np.ndarray | None = None,
    opponent_history_masks: np.ndarray | None = None,
) -> HierarchicalTensorBatch:
    if memories is None:
        memories = [HierarchicalMemory() for _ in observations]
    if len(memories) != len(observations):
        raise ValueError("memories must align one-to-one with observations")
    if extra_cards is None:
        extra_cards = [() for _ in observations]
    if len(extra_cards) != len(observations):
        raise ValueError("extra_cards must align one-to-one with observations")
    if state_encoding is None:
        boards, global_features, unit_features, unit_mask = encode_board_batch(
            observations
        )
    else:
        boards, global_features, unit_features, unit_mask = state_encoding
        if boards.shape[0] != len(observations):
            raise ValueError("state_encoding must align with observations")
    if opponent_histories is None or opponent_history_masks is None:
        runtime_rows = [
            runtime_opponent_history(memory, observation)
            for memory, observation in zip(memories, observations, strict=True)
        ]
        opponent_histories = np.stack([row[0] for row in runtime_rows])
        opponent_history_masks = np.stack([row[1] for row in runtime_rows])
    if opponent_histories.shape != (
        len(observations),
        len(OPPONENT_HISTORY_LAGS),
        OPPONENT_FEATURES,
    ) or opponent_history_masks.shape != (
        len(observations),
        len(OPPONENT_HISTORY_LAGS),
    ):
        raise ValueError("opponent histories have incompatible shapes")
    encoded = tuple(
        generate_task_cards(
            observation,
            memory,
            extra_cards=cards,
            expert_quota=expert_quota,
            learned_quota=learned_quota,
        )
        for observation, memory, cards in zip(
            observations, memories, extra_cards, strict=True
        )
    )
    return HierarchicalTensorBatch(
        boards=boards,
        global_features=global_features,
        unit_features=unit_features,
        unit_mask=unit_mask,
        task_features=np.stack([value.features for value in encoded]),
        task_ids=np.stack([value.task_ids for value in encoded]),
        item_ids=np.stack([value.item_ids for value in encoded]),
        source_ids=np.stack([value.source_ids for value in encoded]),
        target_xy=np.stack([value.target_xy for value in encoded]),
        preferred_owner_ids=np.stack(
            [value.preferred_owner_ids for value in encoded]
        ),
        task_mask=np.stack([value.task_mask for value in encoded]),
        pair_features=np.stack([value.pair_features for value in encoded]),
        eta_steps=np.stack([value.eta_steps for value in encoded]),
        pair_mask=np.stack([value.pair_mask for value in encoded]),
        task_capacity=np.stack([value.capacity for value in encoded]),
        market_priors=np.stack(
            [market_prior_features(observation) for observation in observations]
        ),
        required_market_indices=np.stack(
            [value.required_market_indices for value in encoded]
        ),
        opponent_history=np.asarray(opponent_histories, dtype=np.float32),
        opponent_history_mask=np.asarray(opponent_history_masks, dtype=np.bool_),
        encoded_sets=encoded,
    )


class HierarchicalKaggriculturePolicy(nn.Module):
    """Shared encoder plus proposal, task, market, mode, and value heads."""

    def __init__(
        self,
        *,
        hidden_size: int = 384,
        board_width: int = 64,
        d_model: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        proposal_top_k: int = 24,
        opponent_clusters: int = 8,
    ) -> None:
        super().__init__()
        if min(hidden_size, board_width, d_model, transformer_layers, transformer_heads) <= 0:
            raise ValueError("network sizes must be positive")
        if not 0 < proposal_top_k < MAX_TASK_CARDS:
            raise ValueError("proposal_top_k must be between 1 and MAX_TASK_CARDS - 1")
        if board_width < 2 or d_model % transformer_heads:
            raise ValueError("board_width must be >=2 and d_model divisible by heads")
        if opponent_clusters <= 1:
            raise ValueError("opponent_clusters must be greater than one")
        self.hidden_size = hidden_size
        self.board_width = board_width
        self.d_model = d_model
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.proposal_top_k = proposal_top_k
        self.opponent_clusters = opponent_clusters

        self.board_encoder = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, board_width // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(board_width // 2, board_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(board_width, board_width, 3, padding=1),
            nn.SiLU(),
        )
        self.board_projection = nn.Sequential(
            nn.Linear(2 * board_width, 128), nn.SiLU()
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(UNIT_FEATURES, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.unit_projection = nn.Sequential(nn.Linear(128, 64), nn.SiLU())
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.SiLU(),
        )
        fusion_width = 4 * 128 + 2 * 64 + 128
        self.state_trunk = nn.Sequential(
            nn.Linear(fusion_width, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.opponent_feature_encoder = nn.Sequential(
            nn.Linear(OPPONENT_FEATURES, d_model), nn.SiLU()
        )
        self.opponent_history_encoder = nn.GRU(
            d_model, d_model, batch_first=True
        )
        self.opponent_state_projection = nn.Linear(d_model, hidden_size)
        self.opponent_belief_head = nn.Linear(d_model, opponent_clusters)
        self.mode_head = nn.Linear(hidden_size, len(STRATEGY_MODES))
        self.switch_head = nn.Linear(hidden_size, 2)
        self.mode_embedding = nn.Embedding(len(STRATEGY_MODES), d_model)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.horizon_value_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, 3)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.SiLU(),
            nn.Linear(128, len(INTENT_PHASES)),
        )

        self.proposal_fusion = nn.Sequential(
            nn.Conv2d(4 * board_width, 2 * board_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * board_width, board_width, 3, padding=1),
            nn.SiLU(),
        )
        self.proposal_task_head = nn.Conv2d(
            board_width, len(PROPOSAL_TASK_TYPES), 1
        )
        self.proposal_item_head = nn.Conv2d(board_width, len(ITEMS) + 1, 1)
        self.proposal_auxiliary_head = nn.Conv2d(
            board_width, len(PROPOSAL_AUXILIARIES), 1
        )
        self.proposal_belief_projection = nn.Linear(d_model, board_width)

        self.task_feature_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURES, 128), nn.SiLU()
        )
        self.task_embedding = nn.Embedding(len(TaskType), 32)
        self.item_embedding = nn.Embedding(len(ITEMS) + 1, 16)
        self.source_embedding = nn.Embedding(len(CandidateSource), 8)
        self.preferred_owner_embedding = nn.Embedding(MAX_UNITS + 1, 16)
        task_width = 128 + board_width + 32 + 16 + 8 + 16
        self.task_projection = nn.Sequential(
            nn.Linear(task_width, d_model), nn.SiLU(), nn.LayerNorm(d_model)
        )
        self.state_projection = nn.Linear(hidden_size, d_model)
        self.board_token_projection = nn.Linear(128, d_model)
        self.prefix_type = nn.Embedding(4, d_model)
        relation_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.task_relation = nn.TransformerEncoder(
            relation_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(d_model + 64 + PAIR_FEATURES + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.continue_head = nn.Linear(d_model, 1)
        self.task_prior_scale = nn.Parameter(torch.tensor(4.0))

        self.market_action_embedding = nn.Embedding(len(MARKET_ACTIONS), d_model)
        self.required_market_embedding = nn.Embedding(len(MARKET_ACTIONS), 16)
        self.task_plan_encoder = nn.Sequential(
            nn.Linear(32 + 16 + 16 + PAIR_FEATURES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.budget_encoder = nn.Sequential(
            nn.Linear(hidden_size + d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.budget_reserve_head = nn.Linear(d_model, 1)
        self.budget_category_head = nn.Linear(d_model, len(BUDGET_CATEGORIES))
        self.budget_emergency_head = nn.Linear(d_model, 1)
        self.budget_context_projection = nn.Sequential(
            nn.Linear(BUDGET_FEATURES, d_model), nn.GELU()
        )
        self.market_initial = nn.Linear(hidden_size + d_model, d_model)
        self.market_plan_projection = nn.Linear(d_model, d_model, bias=False)
        self.market_plan_head = nn.Linear(
            d_model, len(MARKET_ACTIONS), bias=False
        )
        self.market_gru = nn.GRUCell(2 * d_model, d_model)
        self.market_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(MARKET_ACTIONS))
        )
        self.market_prior_scale = nn.Parameter(torch.tensor(4.0))

    def encode_state(
        self,
        boards: torch.Tensor,
        global_features: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
        opponent_history: torch.Tensor | None = None,
        opponent_history_mask: torch.Tensor | None = None,
    ) -> HierarchicalContext:
        batch = boards.shape[0]
        board_maps = self.board_encoder(boards.flatten(0, 1)).view(
            batch, 2, self.board_width, boards.shape[-2], boards.shape[-1]
        )
        pooled = torch.cat(
            (board_maps.mean(dim=(-2, -1)), board_maps.amax(dim=(-2, -1))),
            dim=-1,
        )
        board_tokens = self.board_projection(pooled)
        unit_tokens = self.unit_encoder(unit_features)
        weights = unit_mask.unsqueeze(-1).to(unit_tokens.dtype)
        mean = (unit_tokens * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        maximum = unit_tokens.masked_fill(
            ~unit_mask.unsqueeze(-1), -torch.inf
        ).amax(dim=2)
        maximum = torch.where(
            unit_mask.any(dim=2, keepdim=True), maximum, 0.0
        )
        unit_summary = self.unit_projection(torch.cat((mean, maximum), dim=-1))
        global_hidden = self.global_encoder(global_features)
        fused = torch.cat(
            (
                board_tokens[:, 0],
                board_tokens[:, 1],
                board_tokens[:, 0] - board_tokens[:, 1],
                (board_tokens[:, 0] - board_tokens[:, 1]).abs(),
                unit_summary[:, 0],
                unit_summary[:, 1],
                global_hidden,
            ),
            dim=-1,
        )
        if opponent_history is None:
            opponent_history = torch.zeros(
                (
                    batch,
                    len(OPPONENT_HISTORY_LAGS),
                    OPPONENT_FEATURES,
                ),
                device=boards.device,
                dtype=boards.dtype,
            )
            opponent_history_mask = torch.zeros(
                (batch, len(OPPONENT_HISTORY_LAGS)),
                device=boards.device,
                dtype=torch.bool,
            )
        if opponent_history_mask is None:
            raise ValueError("opponent history mask is required with history features")
        opponent_tokens = self.opponent_feature_encoder(opponent_history)
        opponent_tokens = opponent_tokens * opponent_history_mask.unsqueeze(-1).to(
            opponent_tokens.dtype
        )
        opponent_encoded, _ = self.opponent_history_encoder(opponent_tokens)
        opponent_embedding = opponent_encoded[:, -1]
        belief_logits = self.opponent_belief_head(opponent_embedding)
        belief_probabilities = torch.softmax(belief_logits, dim=-1)
        uncertainty = -(
            belief_probabilities
            * torch.log(belief_probabilities.clamp_min(1e-8))
        ).sum(dim=-1) / log(float(self.opponent_clusters))
        state = self.state_trunk(fused) + self.opponent_state_projection(
            opponent_embedding
        )
        return HierarchicalContext(
            state=state,
            board_maps=board_maps,
            board_tokens=board_tokens,
            unit_tokens=unit_tokens,
            mode_logits=self.mode_head(state),
            switch_logits=self.switch_head(state),
            values=self.value_head(state).squeeze(-1),
            opponent_embedding=opponent_embedding,
            opponent_belief_logits=belief_logits,
            opponent_belief_probabilities=belief_probabilities,
            opponent_uncertainty=uncertainty,
            horizon_values=self.horizon_value_head(state),
        )

    def score_assignments(
        self,
        context: HierarchicalContext,
        task_features: torch.Tensor,
        task_ids: torch.Tensor,
        item_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_xy: torch.Tensor,
        preferred_owner_ids: torch.Tensor,
        task_mask: torch.Tensor,
        pair_features: torch.Tensor,
        strategy_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, task_count = task_ids.shape
        batch_index = torch.arange(
            batch, device=task_ids.device
        )[:, None].expand(-1, task_count)
        x = target_xy[..., 0].clamp(0, context.board_maps.shape[-1] - 1)
        y = target_xy[..., 1].clamp(0, context.board_maps.shape[-2] - 1)
        own_map = context.board_maps[:, 0].permute(0, 2, 3, 1)
        target_hidden = own_map[batch_index, y, x]
        target_hidden = target_hidden * (target_xy >= 0).all(
            dim=-1, keepdim=True
        ).to(target_hidden.dtype)
        task_hidden = self.task_projection(
            torch.cat(
                (
                    self.task_feature_encoder(task_features),
                    target_hidden,
                    self.task_embedding(task_ids),
                    self.item_embedding(item_ids),
                    self.source_embedding(source_ids),
                    self.preferred_owner_embedding(preferred_owner_ids),
                ),
                dim=-1,
            )
        )
        mode_hidden = self.mode_embedding(strategy_ids)
        prefix = torch.stack(
            (
                self.state_projection(context.state),
                self.board_token_projection(context.board_tokens[:, 0]),
                self.board_token_projection(context.board_tokens[:, 1]),
                mode_hidden,
            ),
            dim=1,
        )
        prefix = prefix + self.prefix_type(
            torch.arange(4, device=task_ids.device)
        )[None]
        padding = torch.cat(
            (
                torch.zeros((batch, 4), dtype=torch.bool, device=task_ids.device),
                ~task_mask,
            ),
            dim=1,
        )
        related = self.task_relation(
            torch.cat((prefix, task_hidden), dim=1),
            src_key_padding_mask=padding,
        )[:, 4:]
        workers = context.unit_tokens[:, 0]
        expanded_tasks = related[:, None].expand(-1, MAX_UNITS, -1, -1)
        expanded_workers = workers[:, :, None].expand(-1, -1, task_count, -1)
        expanded_mode = mode_hidden[:, None, None].expand(
            -1, MAX_UNITS, task_count, -1
        )
        logits = self.pair_scorer(
            torch.cat(
                (expanded_tasks, expanded_workers, pair_features, expanded_mode),
                dim=-1,
            )
        ).squeeze(-1)
        logits = logits + self.task_prior_scale * task_features[:, None, :, 8]
        continue_logit = self.continue_head(prefix[:, 0]).squeeze(-1)
        logits = logits + continue_logit[:, None, None] * (
            task_ids[:, None] != int(TaskType.IDLE_OR_PASS)
        ).to(logits.dtype)
        return logits

    def propose(self, context: HierarchicalContext) -> ProposalOutput:
        """Predict task-family heatmaps, items, urgency, ETA, value, and slack."""

        own = context.board_maps[:, 0]
        opponent = context.board_maps[:, 1]
        hidden = self.proposal_fusion(
            torch.cat((own, opponent, own - opponent, (own - opponent).abs()), dim=1)
        )
        hidden = hidden + self.proposal_belief_projection(
            context.opponent_embedding
        )[:, :, None, None]
        raw_auxiliary = self.proposal_auxiliary_head(hidden)
        auxiliary = torch.stack(
            (
                torch.sigmoid(raw_auxiliary[:, 0]),
                torch.nn.functional.softplus(raw_auxiliary[:, 1]),
                torch.sigmoid(raw_auxiliary[:, 2]),
                torch.sigmoid(raw_auxiliary[:, 3]),
            ),
            dim=1,
        )
        return ProposalOutput(
            task_logits=self.proposal_task_head(hidden),
            item_logits=self.proposal_item_head(hidden),
            auxiliary_predictions=auxiliary,
        )

    def predict_phases(self, context: HierarchicalContext) -> torch.Tensor:
        state = context.state[:, None].expand(-1, MAX_UNITS, -1)
        return self.phase_head(
            torch.cat((state, context.unit_tokens[:, 0]), dim=-1)
        )

    def predict_budget(
        self,
        context: HierarchicalContext,
        strategy_ids: torch.Tensor,
    ) -> BudgetHeadOutput:
        """Predict a reserve, spending simplex, and bounded emergency reserve."""

        hidden = self.budget_encoder(
            torch.cat((context.state, self.mode_embedding(strategy_ids)), dim=-1)
        )
        return BudgetHeadOutput(
            reserve_fraction=torch.sigmoid(self.budget_reserve_head(hidden)).squeeze(-1),
            category_fractions=torch.softmax(
                self.budget_category_head(hidden), dim=-1
            ),
            # Emergency capital is deliberately capped at 25% of day-start cash.
            emergency_fraction=(
                0.25 * torch.sigmoid(self.budget_emergency_head(hidden)).squeeze(-1)
            ),
        )

    def budget_plan_context(self, output: BudgetHeadOutput) -> torch.Tensor:
        return self.budget_context_projection(output.features)

    def initial_market_hidden(
        self,
        context: HierarchicalContext,
        strategy_ids: torch.Tensor,
        plan_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.market_initial(
            torch.cat((context.state, self.mode_embedding(strategy_ids)), dim=-1)
        )
        if plan_context is not None:
            hidden = hidden + self.market_plan_projection(plan_context)
        return torch.tanh(hidden)

    def assignment_plan_context(
        self,
        task_ids: torch.Tensor,
        item_ids: torch.Tensor,
        pair_features: torch.Tensor,
        required_market_indices: torch.Tensor,
        assignment_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Encode selected tasks and resource deficits for the market decoder."""

        batch, workers, tasks = assignment_weights.shape
        task_hidden = self.task_embedding(task_ids)[:, None].expand(
            -1, workers, -1, -1
        )
        item_hidden = self.item_embedding(item_ids)[:, None].expand(
            -1, workers, -1, -1
        )
        resource_hidden = self.required_market_embedding(required_market_indices)
        encoded = self.task_plan_encoder(
            torch.cat(
                (task_hidden, item_hidden, resource_hidden, pair_features), dim=-1
            )
        )
        weights = assignment_weights.to(encoded.dtype)
        worker_plans = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        worker_active = (weights.sum(dim=2) > 0).to(encoded.dtype)
        return (worker_plans * worker_active.unsqueeze(-1)).sum(dim=1) / (
            worker_active.sum(dim=1, keepdim=True).clamp_min(1.0)
        )

    def market_step(
        self,
        hidden: torch.Tensor,
        previous_indices: torch.Tensor,
        strategy_ids: torch.Tensor,
        market_priors: torch.Tensor,
        plan_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.market_gru(
            torch.cat(
                (
                    self.market_action_embedding(previous_indices),
                    self.mode_embedding(strategy_ids),
                ),
                dim=-1,
            ),
            hidden,
        )
        logits = self.market_head(hidden)
        logits = logits + self.market_prior_scale * market_priors
        if plan_context is not None:
            logits = logits + self.market_plan_head(plan_context)
        return logits, hidden

    def market_teacher_logits(
        self,
        context: HierarchicalContext,
        strategy_ids: torch.Tensor,
        market_history: torch.Tensor,
        market_priors: torch.Tensor,
        plan_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.initial_market_hidden(
            context, strategy_ids, plan_context=plan_context
        )
        previous = torch.zeros(
            context.state.shape[0], dtype=torch.long, device=context.state.device
        )
        logits: list[torch.Tensor] = []
        for slot in range(market_history.shape[1]):
            current, hidden = self.market_step(
                hidden,
                previous,
                strategy_ids,
                market_priors,
                plan_context=plan_context,
            )
            logits.append(current)
            previous = market_history[:, slot]
        return torch.stack(logits, dim=1)

    def forward(
        self,
        boards: torch.Tensor,
        global_features: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
        task_features: torch.Tensor,
        task_ids: torch.Tensor,
        item_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_xy: torch.Tensor,
        preferred_owner_ids: torch.Tensor,
        task_mask: torch.Tensor,
        pair_features: torch.Tensor,
        strategy_ids: torch.Tensor,
        market_history: torch.Tensor,
        market_priors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_state(
            boards, global_features, unit_features, unit_mask
        )
        assignments = self.score_assignments(
            context,
            task_features,
            task_ids,
            item_ids,
            source_ids,
            target_xy,
            preferred_owner_ids,
            task_mask,
            pair_features,
            strategy_ids,
        )
        market = self.market_teacher_logits(
            context, strategy_ids, market_history, market_priors
        )
        return (
            assignments,
            market,
            context.mode_logits,
            context.switch_logits,
            context.values,
        )


def _as_tensors(
    batch: HierarchicalTensorBatch,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(batch.boards, device=device),
        torch.as_tensor(batch.global_features, device=device),
        torch.as_tensor(batch.unit_features, device=device),
        torch.as_tensor(batch.unit_mask, device=device),
        torch.as_tensor(batch.task_features, device=device),
        torch.as_tensor(batch.task_ids, device=device),
        torch.as_tensor(batch.item_ids, device=device),
        torch.as_tensor(batch.source_ids, device=device),
        torch.as_tensor(batch.target_xy, device=device),
        torch.as_tensor(batch.preferred_owner_ids, device=device),
        torch.as_tensor(batch.task_mask, device=device),
        torch.as_tensor(batch.pair_features, device=device),
        torch.as_tensor(batch.pair_mask, device=device),
        torch.as_tensor(batch.task_capacity, device=device),
        torch.as_tensor(batch.market_priors, device=device),
        torch.as_tensor(batch.required_market_indices, device=device),
        torch.as_tensor(batch.opponent_history, device=device),
        torch.as_tensor(batch.opponent_history_mask, device=device),
    )


def prepare_hierarchical_batch(
    model: HierarchicalKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: Sequence[HierarchicalMemory] | None = None,
    *,
    expert_cards: Sequence[Sequence[TaskCard]] | None = None,
    proposal_top_k: int | None = None,
    deterministic: bool = True,
) -> tuple[
    HierarchicalTensorBatch,
    tuple[torch.Tensor, ...],
    HierarchicalContext,
    ProposalOutput,
    SampledProposalCards,
]:
    """Run the differentiable CNN proposal, then encode its discrete Top-K union."""

    if memories is None:
        memories = [HierarchicalMemory() for _ in observations]
    if expert_cards is None:
        expert_cards = [() for _ in observations]
    if len(memories) != len(observations) or len(expert_cards) != len(observations):
        raise ValueError("memories and expert_cards must align with observations")
    boards, global_features, unit_features, unit_mask = encode_board_batch(observations)
    opponent_rows = [
        runtime_opponent_history(memory, observation)
        for memory, observation in zip(memories, observations, strict=True)
    ]
    opponent_histories = np.stack([row[0] for row in opponent_rows])
    opponent_history_masks = np.stack([row[1] for row in opponent_rows])
    state_tensors = (
        torch.as_tensor(boards, device=device),
        torch.as_tensor(global_features, device=device),
        torch.as_tensor(unit_features, device=device),
        torch.as_tensor(unit_mask, device=device),
        torch.as_tensor(opponent_histories, device=device),
        torch.as_tensor(opponent_history_masks, device=device),
    )
    context = model.encode_state(*state_tensors)
    proposal = model.propose(context)
    top_k = model.proposal_top_k if proposal_top_k is None else proposal_top_k
    selection = sample_proposal_cards(
        proposal.task_logits,
        proposal.item_logits,
        proposal.auxiliary_predictions,
        observations,
        top_k=top_k,
        deterministic=deterministic,
        locked_mask=state_tensors[0][:, 0, 1],
    )
    combined = [
        tuple(learned) + tuple(expert)
        for learned, expert in zip(selection.cards, expert_cards, strict=True)
    ]
    batch = encode_hierarchical_batch(
        observations,
        memories,
        extra_cards=combined,
        expert_quota=8,
        learned_quota=top_k,
        state_encoding=(boards, global_features, unit_features, unit_mask),
        opponent_histories=opponent_histories,
        opponent_history_masks=opponent_history_masks,
    )
    return batch, _as_tensors(batch, device), context, proposal, selection


def _select_task_matching_with_trace(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    capacities: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TaskMatchingTrace]:
    """Greedily sample a constrained global worker-task matching."""

    batch, workers, tasks = logits.shape
    selected = torch.full(
        (batch, workers), -1, dtype=torch.long, device=logits.device
    )
    unassigned = pair_mask.any(dim=-1)
    remaining = capacities.clone()
    log_probs = torch.zeros(batch, device=logits.device)
    entropies = torch.zeros_like(log_probs)
    factors = torch.zeros_like(log_probs)
    trace_choices: list[torch.Tensor] = []
    floor = torch.finfo(logits.dtype).min
    batch_index = torch.arange(batch, device=logits.device)
    for _ in range(workers):
        available_tasks = remaining > 0
        mask = pair_mask & unassigned[:, :, None] & available_tasks[:, None, :]
        active = mask.flatten(1).any(dim=-1)
        safe_mask = mask.flatten(1).clone()
        safe_mask[~active, 0] = True
        distribution = torch.distributions.Categorical(
            logits=logits.flatten(1).masked_fill(~safe_mask, floor)
        )
        choice = (
            distribution.logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        trace_choices.append(choice)
        worker = choice // tasks
        task = choice % tasks
        selected[batch_index, worker] = torch.where(
            active, task, selected[batch_index, worker]
        )
        unassigned[batch_index, worker] &= ~active
        decrement = torch.zeros_like(remaining)
        decrement[batch_index, task] = active.to(remaining.dtype)
        remaining = torch.clamp(remaining - decrement, min=0)
        log_probs = log_probs + distribution.log_prob(choice) * active
        entropies = entropies + distribution.entropy() * active
        factors = factors + active
    return (
        selected,
        log_probs,
        entropies / factors.clamp_min(1.0),
        TaskMatchingTrace(torch.stack(trace_choices, dim=1)),
    )


def select_task_matching(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    capacities: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected, log_probs, entropies, _ = _select_task_matching_with_trace(
        logits, pair_mask, capacities, deterministic=deterministic
    )
    return selected, log_probs, entropies


def evaluate_task_matching(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    capacities: torch.Tensor,
    trace: TaskMatchingTrace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the exact constrained matching draw probability."""

    batch, workers, _ = logits.shape
    choices = torch.as_tensor(trace.choices, device=logits.device)
    if choices.shape != (batch, workers):
        raise ValueError("matching replay choices have an incompatible shape")
    unassigned = pair_mask.any(dim=-1)
    remaining = capacities.clone()
    log_probs = torch.zeros(batch, device=logits.device)
    entropies = torch.zeros_like(log_probs)
    factors = torch.zeros_like(log_probs)
    floor = torch.finfo(logits.dtype).min
    batch_index = torch.arange(batch, device=logits.device)
    task_count = logits.shape[-1]
    for slot in range(workers):
        available_tasks = remaining > 0
        mask = pair_mask & unassigned[:, :, None] & available_tasks[:, None, :]
        active = mask.flatten(1).any(dim=-1)
        safe_mask = mask.flatten(1).clone()
        safe_mask[~active, 0] = True
        distribution = torch.distributions.Categorical(
            logits=logits.flatten(1).masked_fill(~safe_mask, floor)
        )
        choice = choices[:, slot]
        worker = choice // task_count
        task = choice % task_count
        chosen_legal = safe_mask.gather(1, choice[:, None]).squeeze(1)
        if not bool(chosen_legal.all()):
            raise ValueError("matching replay selected an illegal edge")
        unassigned[batch_index, worker] &= ~active
        decrement = torch.zeros_like(remaining)
        decrement[batch_index, task] = active.to(remaining.dtype)
        remaining = torch.clamp(remaining - decrement, min=0)
        log_probs = log_probs + distribution.log_prob(choice) * active
        entropies = entropies + distribution.entropy() * active
        factors = factors + active
    return log_probs, entropies / factors.clamp_min(1.0)


def opponent_conditioned_mode_logits(
    mode_logits: torch.Tensor,
    opponent_uncertainty: torch.Tensor,
    observations: Sequence[Any],
) -> torch.Tensor:
    """Apply the same uncertainty-aware opening prior in acting and PPO replay."""

    adjusted = mode_logits.clone()
    early = torch.tensor(
        [int(_get(row, "step", 0) or 0) < 168 for row in observations],
        dtype=mode_logits.dtype,
        device=mode_logits.device,
    )
    uncertainty = opponent_uncertainty.to(
        device=mode_logits.device, dtype=mode_logits.dtype
    ).clamp(0.0, 1.0)
    robust_index = MODE_INDEX["ROBUST_OPENING"]
    if robust_index < adjusted.shape[-1]:
        adjusted[:, robust_index] += early * (0.5 + 1.5 * uncertainty)
    adjusted[:, MODE_INDEX["CONSERVATIVE"]] += early * 0.5 * uncertainty
    return adjusted


def uncertainty_reserve_floors(
    opponent_uncertainty: torch.Tensor,
    observations: Sequence[Any],
) -> torch.Tensor:
    """Minimum cash reserve used by the hard decoder, outside PPO density."""

    early = torch.tensor(
        [int(_get(row, "step", 0) or 0) < 168 for row in observations],
        dtype=opponent_uncertainty.dtype,
        device=opponent_uncertainty.device,
    )
    uncertainty = opponent_uncertainty.clamp(0.0, 1.0)
    return early * (0.10 + 0.35 * uncertainty) + (1.0 - early) * (
        0.05 + 0.15 * uncertainty
    )


def _select_modes(
    mode_logits: torch.Tensor,
    switch_logits: torch.Tensor,
    observations: Sequence[Any],
    memories: Sequence[HierarchicalMemory],
    opponent_uncertainty: torch.Tensor | None = None,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose an initial mode, then expose KEEP/SWITCH only once per new day."""

    if opponent_uncertainty is None:
        opponent_uncertainty = torch.zeros(
            mode_logits.shape[0], device=mode_logits.device, dtype=mode_logits.dtype
        )
    mode_logits = opponent_conditioned_mode_logits(
        mode_logits, opponent_uncertainty, observations
    )
    mode_distribution = torch.distributions.Categorical(logits=mode_logits)
    switch_distribution = torch.distributions.Categorical(logits=switch_logits)
    sampled_modes = (
        mode_logits.argmax(dim=-1)
        if deterministic
        else mode_distribution.sample()
    )
    sampled_switches = (
        switch_logits.argmax(dim=-1)
        if deterministic
        else switch_distribution.sample()
    )
    chosen = sampled_modes.clone()
    switches = torch.full(
        (len(memories),), -1, dtype=torch.long, device=mode_logits.device
    )
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for row, (observation, memory) in enumerate(
        zip(observations, memories, strict=True)
    ):
        day = int(_get(observation, "day", int(_get(observation, "step", 0) or 0) // 24) or 0)
        if not 0 <= memory.strategy_mode < mode_logits.shape[-1]:
            selected = sampled_modes[row]
            chosen[row] = selected
            memory.strategy_mode = int(selected.detach().cpu())
            memory.strategy_day = day
            memory.mode_age = 0
            memory.mode_history.append(
                {
                    "day": day,
                    "decision": "INITIAL",
                    "previous_mode": -1,
                    "mode": memory.strategy_mode,
                }
            )
            log_probs.append(mode_distribution.log_prob(sampled_modes)[row])
            entropies.append(mode_distribution.entropy()[row])
            continue

        current = memory.strategy_mode
        chosen[row] = current
        if day <= memory.strategy_day:
            log_probs.append(mode_logits[row].sum() * 0.0)
            entropies.append(mode_logits[row].sum() * 0.0)
            continue

        decision = sampled_switches[row]
        switches[row] = decision
        row_log_prob = switch_distribution.log_prob(sampled_switches)[row]
        row_entropy = switch_distribution.entropy()[row]
        switched = int(decision.detach().cpu()) == 1
        previous = current
        if switched:
            allowed = torch.ones(
                mode_logits.shape[-1], dtype=torch.bool, device=mode_logits.device
            )
            allowed[current] = False
            floor = torch.finfo(mode_logits.dtype).min
            alternate_logits = mode_logits[row].masked_fill(~allowed, floor)
            alternate_distribution = torch.distributions.Categorical(
                logits=alternate_logits
            )
            selected = (
                alternate_logits.argmax(dim=-1)
                if deterministic
                else alternate_distribution.sample()
            )
            current = int(selected.detach().cpu())
            chosen[row] = selected
            row_log_prob = row_log_prob + alternate_distribution.log_prob(selected)
            row_entropy = row_entropy + alternate_distribution.entropy()
            memory.strategy_mode = current
            memory.mode_age = 0
        memory.strategy_day = day
        memory.record_mode_transition(
            day=day,
            previous_mode=previous,
            mode=current,
            switched=switched,
        )
        log_probs.append(row_log_prob)
        entropies.append(row_entropy)
    return chosen, switches, torch.stack(log_probs), torch.stack(entropies)


def budget_output_from_plans(
    plans: Sequence[BudgetPlan],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BudgetHeadOutput:
    features = torch.as_tensor(
        [plan.features for plan in plans], device=device, dtype=dtype
    )
    return BudgetHeadOutput(
        reserve_fraction=features[:, 0],
        category_fractions=features[:, 1 : 1 + len(BUDGET_CATEGORIES)],
        emergency_fraction=features[:, -1],
    )


def _select_daily_budget_plans(
    model: HierarchicalKaggriculturePolicy,
    context: HierarchicalContext,
    modes: torch.Tensor,
    observations: Sequence[Any],
    memories: Sequence[HierarchicalMemory],
    *,
    deterministic: bool,
) -> tuple[
    tuple[BudgetPlan, ...],
    BudgetHeadOutput,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create a plan at a day boundary and otherwise KEEP the stored plan."""

    predicted = model.predict_budget(context, modes)
    reserve_distribution, category_distribution, emergency_distribution = (
        _budget_action_distributions(predicted)
    )
    proposed_features = (
        predicted.features
        if deterministic
        else torch.cat(
            (
                reserve_distribution.sample()[:, None],
                category_distribution.sample(),
                (
                    _MAX_EMERGENCY_FRACTION * emergency_distribution.sample()
                )[:, None],
            ),
            dim=-1,
        )
    )
    plans: list[BudgetPlan] = []
    active_rows: list[bool] = []
    reserve_floors = uncertainty_reserve_floors(
        context.opponent_uncertainty, observations
    )
    for row, (observation, memory) in enumerate(
        zip(observations, memories, strict=True)
    ):
        step = int(_get(observation, "step", 0) or 0)
        day = int(_get(observation, "day", step // 24) or 0)
        active = memory.budget_plan is None or day != memory.budget_day
        active_rows.append(active)
        if active:
            money = MarketBudget.from_observation(observation).money
            plan = BudgetPlan.from_fractions(
                day=day,
                money=money,
                reserve_fraction=float(proposed_features[row, 0].detach().cpu()),
                category_fractions=(
                    proposed_features[
                        row, 1 : 1 + len(BUDGET_CATEGORIES)
                    ].detach().cpu().tolist()
                ),
                emergency_fraction=float(
                    proposed_features[row, -1].detach().cpu()
                ),
                minimum_reserve_fraction=float(
                    reserve_floors[row].detach().cpu()
                ),
            )
            memory.budget_day = day
            memory.budget_plan = plan
            memory.budget_spent_by_category[:] = [0.0] * len(BUDGET_CATEGORIES)
            memory.budget_emergency_spent = 0.0
        plans.append(memory.budget_plan)
    selected = budget_output_from_plans(
        plans, device=context.state.device, dtype=context.state.dtype
    )
    active_tensor = torch.tensor(
        active_rows, dtype=torch.bool, device=context.state.device
    )
    log_prob, entropy = evaluate_budget_action_features(
        predicted, selected.features, active_tensor
    )
    return tuple(plans), selected, log_prob, entropy, active_tensor


def _budget_legal_mask(
    budget: MarketBudget | PlannedMarketBudget, *, emergency: bool = False
) -> np.ndarray:
    if isinstance(budget, PlannedMarketBudget):
        return budget.legal_mask(emergency=emergency)
    return budget.legal_mask()


def _budget_apply(
    budget: MarketBudget | PlannedMarketBudget,
    action_index: int,
    *,
    emergency: bool = False,
) -> None:
    if isinstance(budget, PlannedMarketBudget):
        budget.apply(action_index, emergency=emergency)
    else:
        budget.apply(action_index)


def _decode_market(
    model: HierarchicalKaggriculturePolicy,
    context: HierarchicalContext,
    modes: torch.Tensor,
    priors: torch.Tensor,
    observations: Sequence[Any],
    plan_context: torch.Tensor | None = None,
    required_orders: torch.Tensor | None = None,
    budget_ledgers: Sequence[MarketBudget | PlannedMarketBudget] | None = None,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(observations)
    budgets = (
        [MarketBudget.from_observation(value) for value in observations]
        if budget_ledgers is None
        else list(budget_ledgers)
    )
    if len(budgets) != batch:
        raise ValueError("budget ledgers must align with observations")
    hidden = model.initial_market_hidden(
        context, modes, plan_context=plan_context
    )
    pending_orders = (
        [
            [int(value) for value in row if int(value) > 0]
            for row in required_orders.detach().cpu().tolist()
        ]
        if required_orders is not None
        else [[] for _ in observations]
    )
    previous = torch.zeros(batch, dtype=torch.long, device=context.state.device)
    selected = torch.zeros(
        (batch, MAX_MARKET_ORDERS),
        dtype=torch.long,
        device=context.state.device,
    )
    active = torch.ones(batch, dtype=torch.bool, device=context.state.device)
    log_probs = torch.zeros(batch, device=context.state.device)
    entropies = torch.zeros_like(log_probs)
    factors = torch.zeros_like(log_probs)
    floor = torch.finfo(context.state.dtype).min
    for slot in range(MAX_MARKET_ORDERS):
        slot_active = active.clone()
        logits, hidden = model.market_step(
            hidden,
            previous,
            modes,
            priors,
            plan_context=plan_context,
        )
        legal_np = np.stack([_budget_legal_mask(budget) for budget in budgets])
        forced_indices = [0] * batch
        for row in range(batch):
            if not bool(slot_active[row]):
                continue
            while pending_orders[row]:
                required = pending_orders[row][0]
                if bool(_budget_legal_mask(budgets[row], emergency=True)[required]):
                    forced_indices[row] = pending_orders[row].pop(0)
                    legal_np[row, required] = True
                    break
                pending_orders[row].pop(0)
        legal = torch.as_tensor(legal_np, device=context.state.device)
        legal[~slot_active] = False
        legal[~slot_active, 0] = True
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~legal, floor)
        )
        sampled = (
            logits.masked_fill(~legal, floor).argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        choice = sampled.clone()
        forced = torch.zeros(batch, dtype=torch.bool, device=context.state.device)
        for row, required in enumerate(forced_indices):
            if required > 0:
                choice[row] = required
                forced[row] = True
        selected[:, slot] = torch.where(slot_active, choice, 0)
        learned = slot_active & ~forced
        log_probs = log_probs + distribution.log_prob(choice) * learned
        entropies = entropies + distribution.entropy() * learned
        factors = factors + learned
        for row, budget in enumerate(budgets):
            if bool(slot_active[row]) and int(choice[row].detach().cpu()) != 0:
                _budget_apply(
                    budget,
                    int(choice[row].detach().cpu()),
                    emergency=bool(forced[row]),
                )
        active = slot_active & (choice != 0)
        previous = choice
    return selected, log_probs, entropies / factors.clamp_min(1.0)


def evaluate_market_sequence(
    model: HierarchicalKaggriculturePolicy,
    context: HierarchicalContext,
    modes: torch.Tensor,
    priors: torch.Tensor,
    observations: Sequence[Any],
    selected: torch.Tensor,
    plan_context: torch.Tensor | None = None,
    required_orders: torch.Tensor | None = None,
    budget_ledgers: Sequence[MarketBudget | PlannedMarketBudget] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute learned market-order probabilities, excluding forced orders."""

    batch = len(observations)
    choices = torch.as_tensor(selected, dtype=torch.long, device=context.state.device)
    if choices.shape != (batch, MAX_MARKET_ORDERS):
        raise ValueError("market replay choices have an incompatible shape")
    budgets = (
        [MarketBudget.from_observation(value) for value in observations]
        if budget_ledgers is None
        else list(budget_ledgers)
    )
    if len(budgets) != batch:
        raise ValueError("budget ledgers must align with observations")
    hidden = model.initial_market_hidden(context, modes, plan_context=plan_context)
    pending_orders = (
        [
            [int(value) for value in row if int(value) > 0]
            for row in required_orders.detach().cpu().tolist()
        ]
        if required_orders is not None
        else [[] for _ in observations]
    )
    previous = torch.zeros(batch, dtype=torch.long, device=context.state.device)
    active = torch.ones(batch, dtype=torch.bool, device=context.state.device)
    log_probs = torch.zeros(batch, device=context.state.device)
    entropies = torch.zeros_like(log_probs)
    factors = torch.zeros_like(log_probs)
    floor = torch.finfo(context.state.dtype).min
    for slot in range(MAX_MARKET_ORDERS):
        slot_active = active.clone()
        logits, hidden = model.market_step(
            hidden, previous, modes, priors, plan_context=plan_context
        )
        legal_np = np.stack([_budget_legal_mask(budget) for budget in budgets])
        forced_indices = [0] * batch
        for row in range(batch):
            if not bool(slot_active[row]):
                continue
            while pending_orders[row]:
                required = pending_orders[row][0]
                if bool(_budget_legal_mask(budgets[row], emergency=True)[required]):
                    forced_indices[row] = pending_orders[row].pop(0)
                    legal_np[row, required] = True
                    break
                pending_orders[row].pop(0)
        legal = torch.as_tensor(legal_np, device=context.state.device)
        legal[~slot_active] = False
        legal[~slot_active, 0] = True
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~legal, floor)
        )
        choice = choices[:, slot]
        forced = torch.zeros(batch, dtype=torch.bool, device=context.state.device)
        for row, required in enumerate(forced_indices):
            if required <= 0:
                continue
            if int(choice[row].detach().cpu()) != required:
                raise ValueError("market replay disagrees with forced resource order")
            forced[row] = True
        chosen_legal = legal.gather(1, choice[:, None]).squeeze(1)
        if not bool(chosen_legal.all()):
            raise ValueError("market replay selected an illegal order")
        learned = slot_active & ~forced
        log_probs = log_probs + distribution.log_prob(choice) * learned
        entropies = entropies + distribution.entropy() * learned
        factors = factors + learned
        for row, budget in enumerate(budgets):
            if bool(slot_active[row]) and int(choice[row].detach().cpu()) != 0:
                _budget_apply(
                    budget,
                    int(choice[row].detach().cpu()),
                    emergency=bool(forced[row]),
                )
        active = slot_active & (choice != 0)
        previous = choice
    return log_probs, entropies / factors.clamp_min(1.0)


def _reconcile_resource_budget(
    assignments: torch.Tensor,
    required_market_indices: torch.Tensor,
    encoded_sets: Sequence[EncodedTaskSet],
    observations: Sequence[Any],
    budget_plans: Sequence[BudgetPlan] | None = None,
    spent_by_category: Sequence[Sequence[float]] | None = None,
    emergency_spent: Sequence[float] | None = None,
) -> torch.Tensor:
    """Replace jointly unaffordable resource tasks with the safe idle card."""

    reconciled = assignments.clone()
    cpu_assignments = assignments.detach().cpu().tolist()
    for row, (choices, encoded, observation) in enumerate(
        zip(cpu_assignments, encoded_sets, observations, strict=True)
    ):
        budget: MarketBudget | PlannedMarketBudget
        if budget_plans is None:
            budget = MarketBudget.from_observation(observation)
        else:
            budget = PlannedMarketBudget.from_observation(
                observation,
                budget_plans[row],
                spent_by_category=(
                    spent_by_category[row]
                    if spent_by_category is not None
                    else None
                ),
                emergency_spent=(
                    emergency_spent[row] if emergency_spent is not None else 0.0
                ),
            )
        idle = next(
            index
            for index, card in enumerate(encoded.tasks)
            if card.task_type == TaskType.IDLE_OR_PASS
        )
        for unit, task in enumerate(choices):
            if task < 0:
                continue
            required = int(required_market_indices[row, unit, task])
            if required <= 0:
                continue
            legal = _budget_legal_mask(budget, emergency=True)
            if required >= len(legal) or not bool(legal[required]):
                reconciled[row, unit] = idle
                continue
            _budget_apply(budget, required, emergency=True)
    return reconciled


def _policy_batch_impl(
    model: HierarchicalKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: list[HierarchicalMemory],
    *,
    deterministic: bool,
) -> HierarchicalPolicyBatch:
    batch, tensors, context, _, proposal_selection = prepare_hierarchical_batch(
        model,
        observations,
        device,
        memories,
        deterministic=deterministic,
    )
    previous_modes = torch.tensor(
        [memory.strategy_mode for memory in memories],
        dtype=torch.long,
        device=context.state.device,
    )
    initial_mode_active = previous_modes < 0
    daily_mode_active = torch.tensor(
        [
            not bool(initial_mode_active[row])
            and int(
                _get(
                    observation,
                    "day",
                    int(_get(observation, "step", 0) or 0) // 24,
                )
                or 0
            )
            > memory.strategy_day
            for row, (observation, memory) in enumerate(
                zip(observations, memories, strict=True)
            )
        ],
        dtype=torch.bool,
        device=context.state.device,
    )
    modes, switches, mode_log_prob, mode_entropy = _select_modes(
        context.mode_logits,
        context.switch_logits,
        observations,
        memories,
        context.opponent_uncertainty,
        deterministic=deterministic,
    )
    (
        budget_plans,
        selected_budget_output,
        budget_log_prob,
        budget_entropy,
        budget_action_active,
    ) = _select_daily_budget_plans(
        model,
        context,
        modes,
        observations,
        memories,
        deterministic=deterministic,
    )
    budget_spent_before = [
        list(memory.budget_spent_by_category) for memory in memories
    ]
    budget_emergency_before = [
        float(memory.budget_emergency_spent) for memory in memories
    ]
    budget_context = model.budget_plan_context(selected_budget_output)
    assignment_logits = model.score_assignments(
        context,
        *tensors[4:12],
        modes,
    )
    (
        assignments,
        assignment_log_prob,
        assignment_entropy,
        matching_trace,
    ) = _select_task_matching_with_trace(
        assignment_logits,
        tensors[12],
        tensors[13],
        deterministic=deterministic,
    )
    assignments = _reconcile_resource_budget(
        assignments,
        tensors[15],
        batch.encoded_sets,
        observations,
        budget_plans=budget_plans,
        spent_by_category=budget_spent_before,
        emergency_spent=budget_emergency_before,
    )
    valid_assignments = assignments >= 0
    assignment_weights = torch.zeros_like(
        assignment_logits, dtype=context.state.dtype
    )
    assignment_weights.scatter_(
        2,
        assignments.clamp_min(0).unsqueeze(-1),
        valid_assignments.to(context.state.dtype).unsqueeze(-1),
    )
    plan_context = model.assignment_plan_context(
        tensors[5],
        tensors[6],
        tensors[11],
        tensors[15],
        assignment_weights,
    )
    market_context = plan_context + budget_context
    required_orders = tensors[15].gather(
        2, assignments.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    required_orders = torch.where(
        valid_assignments, required_orders, torch.zeros_like(required_orders)
    )
    budget_ledgers = [
        PlannedMarketBudget.from_observation(
            observation,
            plan,
            spent_by_category=spent,
            emergency_spent=emergency,
        )
        for observation, plan, spent, emergency in zip(
            observations,
            budget_plans,
            budget_spent_before,
            budget_emergency_before,
            strict=True,
        )
    ]
    market_indices, market_log_prob, market_entropy = _decode_market(
        model,
        context,
        modes,
        tensors[14],
        observations,
        plan_context=market_context,
        required_orders=required_orders,
        budget_ledgers=budget_ledgers,
        deterministic=deterministic,
    )
    for memory, budget in zip(memories, budget_ledgers, strict=True):
        memory.budget_spent_by_category[:] = budget.spent_by_category.tolist()
        memory.budget_emergency_spent = float(budget.emergency_spent)
    actions = [
        compile_hierarchical_action(
            observation,
            encoded,
            assignment,
            market,
            memory,
        )
        for observation, encoded, assignment, market, memory in zip(
            observations,
            batch.encoded_sets,
            assignments.detach().cpu().numpy(),
            market_indices.detach().cpu().numpy(),
            memories,
            strict=True,
        )
    ]
    return HierarchicalPolicyBatch(
        actions=actions,
        assignments=assignments,
        market_indices=market_indices,
        strategy_modes=modes,
        mode_switches=switches,
        log_probs=(
            proposal_selection.log_probs
            + mode_log_prob
            + budget_log_prob
            + assignment_log_prob
            + market_log_prob
        ),
        entropies=(
            proposal_selection.entropies
            + mode_entropy
            + budget_entropy
            + assignment_entropy
            + market_entropy
        )
        / (4.0 + budget_action_active.to(context.state.dtype)),
        values=context.values,
        memories=memories,
        proposal_log_probs=proposal_selection.log_probs,
        proposal_entropies=proposal_selection.entropies,
        budget_log_probs=budget_log_prob,
        budget_entropies=budget_entropy,
        budget_action_active=budget_action_active,
        replay=HierarchicalReplayBatch(
            observations=tuple(copy.deepcopy(list(observations))),
            tensors=batch,
            proposal_draws=proposal_selection.drawn_indices.detach().cpu(),
            matching=TaskMatchingTrace(matching_trace.choices.detach().cpu()),
            assignments=assignments.detach().cpu(),
            market_indices=market_indices.detach().cpu(),
            strategy_modes=modes.detach().cpu(),
            mode_switches=switches.detach().cpu(),
            initial_mode_active=initial_mode_active.detach().cpu(),
            daily_mode_active=daily_mode_active.detach().cpu(),
            previous_modes=previous_modes.detach().cpu(),
            budget_plan_features=selected_budget_output.features.detach().cpu(),
            budget_action_active=budget_action_active.detach().cpu(),
            budget_cash_at_plan=torch.tensor(
                [plan.cash_at_plan for plan in budget_plans], dtype=torch.float32
            ),
            budget_days=torch.tensor(
                [plan.day for plan in budget_plans], dtype=torch.long
            ),
            budget_spent_before=torch.tensor(
                budget_spent_before, dtype=torch.float32
            ),
            budget_emergency_before=torch.tensor(
                budget_emergency_before, dtype=torch.float32
            ),
            budget_minimum_reserve=torch.tensor(
                [plan.minimum_reserve_fraction for plan in budget_plans],
                dtype=torch.float32,
            ),
        ),
    )


@torch.no_grad()
def hierarchical_policy_batch(
    model: HierarchicalKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    *,
    memories: list[HierarchicalMemory] | None = None,
    deterministic: bool = False,
) -> HierarchicalPolicyBatch:
    memories = (
        [HierarchicalMemory() for _ in observations]
        if memories is None
        else memories
    )
    return _policy_batch_impl(
        model,
        observations,
        device,
        memories,
        deterministic=deterministic,
    )
    BudgetPlan,
