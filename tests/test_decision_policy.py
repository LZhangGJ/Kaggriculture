from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.decision_policy import (
    DecisionKaggriculturePolicy,
    decision_policy_batch,
    encode_decision_batch,
)
from kaggriculture_lab.decision_schema import (
    CANDIDATE_FEATURES,
    DECISION_GROUPS,
    MAX_DECISION_CANDIDATES,
    REFERENCE_STRATEGY_AGENTS,
    DecisionMemory,
    TaskType,
    compile_decision_action,
    generate_decision_candidates,
)


def _tensors(batch: object) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(batch.boards),
        torch.as_tensor(batch.global_features),
        torch.as_tensor(batch.unit_features),
        torch.as_tensor(batch.unit_mask),
        torch.as_tensor(batch.candidate_features),
        torch.as_tensor(batch.task_ids),
        torch.as_tensor(batch.owner_ids),
        torch.as_tensor(batch.item_ids),
        torch.as_tensor(batch.source_ids),
        torch.as_tensor(batch.target_xy),
        torch.as_tensor(batch.candidate_mask),
    )


def _small_model() -> DecisionKaggriculturePolicy:
    return DecisionKaggriculturePolicy(
        hidden_size=64,
        board_width=16,
        d_model=64,
        transformer_layers=1,
        transformer_heads=4,
    )


def test_strategy_taxonomy_and_candidate_shapes() -> None:
    assert len(REFERENCE_STRATEGY_AGENTS) == 30
    assert len(TaskType) == 21

    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=31))
    encoded = generate_decision_candidates(observations[0])
    count = len(encoded.candidates)

    assert 0 < count <= MAX_DECISION_CANDIDATES
    assert encoded.features.shape == (
        MAX_DECISION_CANDIDATES,
        CANDIDATE_FEATURES,
    )
    assert encoded.target_xy.shape == (MAX_DECISION_CANDIDATES, 2)
    assert encoded.mask.sum() == count
    assert np.isfinite(encoded.features).all()
    assert 0 in encoded.group_ids[encoded.mask]
    assert 1 in encoded.group_ids[encoded.mask]


def test_decision_network_shapes_gradients_and_legal_step() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=32))
    batch = encode_decision_batch(observations)
    model = _small_model()
    logits, stop_logits, values = model(*_tensors(batch))

    assert logits.shape == (2, MAX_DECISION_CANDIDATES)
    assert stop_logits.shape == (2,)
    assert values.shape == (2,)
    mask = torch.as_tensor(batch.candidate_mask)
    loss = logits[mask].square().mean() + stop_logits.square().mean() + values.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    policy = decision_policy_batch(
        model.eval(), observations, "cpu", deterministic=True
    )
    result = env.step(policy.actions)
    assert result.step == 1
    assert all(status != "ERROR" for status in result.statuses)


def test_candidate_scorer_is_permutation_equivariant() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=33))
    batch = encode_decision_batch(observations)
    tensors = list(_tensors(batch))
    model = _small_model().eval()

    with torch.no_grad():
        original, _, _ = model(*tensors)
        permutation = torch.randperm(MAX_DECISION_CANDIDATES)
        permuted = tensors[:4] + [value[:, permutation] for value in tensors[4:]]
        reordered, _, _ = model(*permuted)

    torch.testing.assert_close(reordered, original[:, permutation], atol=1e-5, rtol=1e-5)


def test_executor_persists_selected_worker_plan() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=34)[0]
    memory = DecisionMemory()
    encoded = generate_decision_candidates(observation, memory)
    market_idle = next(
        index
        for index, candidate in enumerate(encoded.candidates)
        if candidate.group_id == 0 and candidate.task_type == TaskType.IDLE_OR_PASS
    )
    worker_task = next(
        index
        for index, candidate in enumerate(encoded.candidates)
        if candidate.group_id == 1 and candidate.task_type != TaskType.IDLE_OR_PASS
    )
    selected = [-1] * DECISION_GROUPS
    selected[0] = market_idle
    selected[1] = worker_task

    action = compile_decision_action(observation, encoded, selected, memory)
    assert set(action) == {"farmer", "hands", "market"}
    assert memory.unit_plans[0] is not None
    assert memory.unit_plans[0].task_type == encoded.candidates[worker_task].task_type
