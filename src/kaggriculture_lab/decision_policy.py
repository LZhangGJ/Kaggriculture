"""Candidate-selection network for strategy-derived Kaggriculture decisions."""

from __future__ import annotations

from dataclasses import dataclass
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
from .decision_schema import (
    CANDIDATE_FEATURES,
    DECISION_GROUPS,
    MAX_DECISION_CANDIDATES,
    CandidateSource,
    DecisionMemory,
    EncodedCandidateSet,
    TaskType,
    compile_decision_action,
    generate_decision_candidates,
)
from .gpu_policy import ITEMS, MAX_UNITS


@dataclass(frozen=True)
class DecisionTensorBatch:
    boards: np.ndarray
    global_features: np.ndarray
    unit_features: np.ndarray
    unit_mask: np.ndarray
    candidate_features: np.ndarray
    task_ids: np.ndarray
    owner_ids: np.ndarray
    item_ids: np.ndarray
    source_ids: np.ndarray
    target_xy: np.ndarray
    group_ids: np.ndarray
    candidate_mask: np.ndarray
    encoded_sets: tuple[EncodedCandidateSet, ...]


@dataclass
class DecisionPolicyBatch:
    actions: list[dict[str, Any]]
    selected_indices: torch.Tensor
    values: torch.Tensor
    memories: list[DecisionMemory]


def encode_decision_batch(
    observations: Sequence[Any],
    memories: Sequence[DecisionMemory] | None = None,
) -> DecisionTensorBatch:
    if memories is None:
        memories = [DecisionMemory() for _ in observations]
    if len(memories) != len(observations):
        raise ValueError("memories must align one-to-one with observations")
    boards, global_features, unit_features, unit_mask = encode_board_batch(observations)
    encoded = tuple(
        generate_decision_candidates(observation, memory)
        for observation, memory in zip(observations, memories, strict=True)
    )
    return DecisionTensorBatch(
        boards=boards,
        global_features=global_features,
        unit_features=unit_features,
        unit_mask=unit_mask,
        candidate_features=np.stack([value.features for value in encoded]),
        task_ids=np.stack([value.task_ids for value in encoded]),
        owner_ids=np.stack([value.owner_ids for value in encoded]),
        item_ids=np.stack([value.item_ids for value in encoded]),
        source_ids=np.stack([value.source_ids for value in encoded]),
        target_xy=np.stack([value.target_xy for value in encoded]),
        group_ids=np.stack([value.group_ids for value in encoded]),
        candidate_mask=np.stack([value.mask for value in encoded]),
        encoded_sets=encoded,
    )


class DecisionKaggriculturePolicy(nn.Module):
    """Shared spatial encoder plus a permutation-equivariant candidate scorer."""

    def __init__(
        self,
        *,
        hidden_size: int = 384,
        board_width: int = 64,
        d_model: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        if board_width < 2 or hidden_size <= 0 or d_model <= 0:
            raise ValueError("network widths must be positive")
        if d_model % transformer_heads:
            raise ValueError("d_model must be divisible by transformer_heads")
        self.hidden_size = hidden_size
        self.board_width = board_width
        self.d_model = d_model
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
            nn.Linear(UNIT_FEATURES, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
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

        self.candidate_feature_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURES, 128), nn.SiLU()
        )
        self.task_embedding = nn.Embedding(len(TaskType), 32)
        self.owner_embedding = nn.Embedding(MAX_UNITS + 1, 16)
        self.item_embedding = nn.Embedding(len(ITEMS) + 1, 16)
        self.source_embedding = nn.Embedding(len(CandidateSource), 8)
        candidate_input = 128 + board_width + 64 + 32 + 16 + 16 + 8
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_input, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.state_projection = nn.Linear(hidden_size, d_model)
        self.board_token_projection = nn.Linear(128, d_model)
        self.prefix_type = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, norm=nn.LayerNorm(d_model)
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.stop_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.prior_scale = nn.Parameter(torch.tensor(4.0))

    def _encode_state(
        self,
        boards: torch.Tensor,
        global_features: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = boards.shape[0]
        board_maps = self.board_encoder(boards.flatten(0, 1)).view(
            batch, 2, self.board_width, boards.shape[-2], boards.shape[-1]
        )
        pooled = torch.cat(
            (
                board_maps.mean(dim=(-2, -1)),
                board_maps.amax(dim=(-2, -1)),
            ),
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
        return self.state_trunk(fused), board_maps, board_tokens, unit_tokens

    def forward(
        self,
        boards: torch.Tensor,
        global_features: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        task_ids: torch.Tensor,
        owner_ids: torch.Tensor,
        item_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_xy: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, board_maps, board_tokens, unit_tokens = self._encode_state(
            boards, global_features, unit_features, unit_mask
        )
        batch, candidates = task_ids.shape
        batch_index = torch.arange(batch, device=boards.device)[:, None].expand(
            -1, candidates
        )
        x = target_xy[..., 0].clamp(0, boards.shape[-1] - 1)
        y = target_xy[..., 1].clamp(0, boards.shape[-2] - 1)
        own_map = board_maps[:, 0].permute(0, 2, 3, 1)
        target_hidden = own_map[batch_index, y, x]
        target_valid = (target_xy >= 0).all(dim=-1, keepdim=True)
        target_hidden = target_hidden * target_valid.to(target_hidden.dtype)

        owner_index = (owner_ids - 1).clamp(0, MAX_UNITS - 1)
        owner_hidden = unit_tokens[:, 0][batch_index, owner_index]
        owner_hidden = owner_hidden * (owner_ids > 0).unsqueeze(-1).to(
            owner_hidden.dtype
        )
        candidate_hidden = self.candidate_projection(
            torch.cat(
                (
                    self.candidate_feature_encoder(candidate_features),
                    target_hidden,
                    owner_hidden,
                    self.task_embedding(task_ids),
                    self.owner_embedding(owner_ids),
                    self.item_embedding(item_ids),
                    self.source_embedding(source_ids),
                ),
                dim=-1,
            )
        )
        prefix = torch.stack(
            (
                self.state_projection(state),
                self.board_token_projection(board_tokens[:, 0]),
                self.board_token_projection(board_tokens[:, 1]),
            ),
            dim=1,
        )
        prefix = prefix + self.prefix_type(
            torch.arange(3, device=boards.device)
        )[None]
        tokens = torch.cat((prefix, candidate_hidden), dim=1)
        padding = torch.cat(
            (
                torch.zeros((batch, 3), dtype=torch.bool, device=boards.device),
                ~candidate_mask,
            ),
            dim=1,
        )
        encoded = self.relation_encoder(tokens, src_key_padding_mask=padding)
        candidate_encoded = encoded[:, 3:]
        logits = self.candidate_head(candidate_encoded).squeeze(-1)
        logits = logits + self.prior_scale * candidate_features[..., 8]
        # The shared stop/continue signal controls how readily every assignment
        # group leaves its idle candidate.  Unlike a detached auxiliary head, it
        # is therefore trained directly by the policy-gradient objective.
        continue_logits = self.stop_head(encoded[:, 0]).squeeze(-1)
        logits = logits + continue_logits[:, None] * (
            task_ids != int(TaskType.IDLE_OR_PASS)
        ).to(logits.dtype)
        return (
            logits,
            continue_logits,
            self.value_head(encoded[:, 0]).squeeze(-1),
        )


def _as_tensors(
    batch: DecisionTensorBatch, device: torch.device | str
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(batch.boards, device=device),
        torch.as_tensor(batch.global_features, device=device),
        torch.as_tensor(batch.unit_features, device=device),
        torch.as_tensor(batch.unit_mask, device=device),
        torch.as_tensor(batch.candidate_features, device=device),
        torch.as_tensor(batch.task_ids, device=device),
        torch.as_tensor(batch.owner_ids, device=device),
        torch.as_tensor(batch.item_ids, device=device),
        torch.as_tensor(batch.source_ids, device=device),
        torch.as_tensor(batch.target_xy, device=device),
        torch.as_tensor(batch.candidate_mask, device=device),
    )


def select_candidate_groups(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one candidate per active group and return log-prob/entropy."""

    selected = torch.full(
        (logits.shape[0], DECISION_GROUPS),
        -1,
        dtype=torch.long,
        device=logits.device,
    )
    log_probs = torch.zeros(logits.shape[0], device=logits.device)
    entropies = torch.zeros_like(log_probs)
    factors = torch.zeros_like(log_probs)
    floor = torch.finfo(logits.dtype).min
    for group in range(DECISION_GROUPS):
        mask = candidate_mask & (group_ids == group)
        active = mask.any(dim=-1)
        safe_mask = mask.clone()
        safe_mask[~active, 0] = True
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~safe_mask, floor)
        )
        choice = (
            distribution.logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        selected[:, group] = torch.where(active, choice, -1)
        log_probs = log_probs + distribution.log_prob(choice) * active
        entropies = entropies + distribution.entropy() * active
        factors = factors + active
    return selected, log_probs, entropies / factors.clamp_min(1.0)


@torch.no_grad()
def decision_policy_batch(
    model: DecisionKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    *,
    memories: list[DecisionMemory] | None = None,
    deterministic: bool = False,
) -> DecisionPolicyBatch:
    memories = (
        [DecisionMemory() for _ in observations] if memories is None else memories
    )
    batch = encode_decision_batch(observations, memories)
    tensors = _as_tensors(batch, device)
    logits, _, values = model(*tensors)
    selected, _, _ = select_candidate_groups(
        logits,
        torch.as_tensor(batch.group_ids, device=device),
        tensors[-1],
        deterministic=deterministic,
    )
    selected_cpu = selected.cpu().numpy()
    actions = [
        compile_decision_action(observation, encoded, row, memory)
        for observation, encoded, row, memory in zip(
            observations,
            batch.encoded_sets,
            selected_cpu,
            memories,
            strict=True,
        )
    ]
    return DecisionPolicyBatch(actions, selected, values, memories)
