"""On-policy helpers for the V2 candidate-decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .decision_policy import (
    DecisionKaggriculturePolicy,
    _as_tensors,
    encode_decision_batch,
    select_candidate_groups,
)
from .decision_schema import (
    DecisionMemory,
    compile_decision_action,
)


@dataclass
class DecisionActorCriticBatch:
    actions: list[dict[str, Any]]
    selected_indices: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor
    memories: list[DecisionMemory]


def decision_actor_critic_batch(
    model: DecisionKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: list[DecisionMemory],
) -> DecisionActorCriticBatch:
    batch = encode_decision_batch(observations, memories)
    tensors = _as_tensors(batch, device)
    logits, _, values = model(*tensors)
    selected, log_probs, entropies = select_candidate_groups(
        logits,
        torch.as_tensor(batch.group_ids, device=device),
        tensors[-1],
        deterministic=False,
    )
    selected_cpu = selected.detach().cpu().numpy()
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
    return DecisionActorCriticBatch(
        actions=actions,
        selected_indices=selected,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
        memories=memories,
    )


@torch.no_grad()
def decision_values(
    model: DecisionKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: list[DecisionMemory],
) -> torch.Tensor:
    batch = encode_decision_batch(observations, memories)
    _, _, values = model(*_as_tensors(batch, device))
    return values
